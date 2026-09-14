"""Audit mask-derived QA stability under the frozen preprocessing resolutions."""

import argparse
import csv
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as functional
from torch import Tensor

from mri_vlm.data.msd import discover_training_cases
from mri_vlm.real_qa import ENHANCING_PRESENCE_ML, MIDLINE_RELATIVE_MARGIN
from mri_vlm.schema import Split
from mri_vlm.split import assign_subject


@dataclass(frozen=True, slots=True)
class LabelMeasurements:
    whole_ml: float
    edema_ml: float
    core_ml: float
    enhancing_ml: float
    left_ml: float
    right_ml: float

    @property
    def enhancing_fraction(self) -> float:
        return self.enhancing_ml / self.whole_ml

    def categorical_answers(self) -> dict[str, str]:
        laterality_ratio = abs(self.left_ml - self.right_ml) / self.whole_ml
        laterality = (
            "midline"
            if laterality_ratio <= MIDLINE_RELATIVE_MARGIN
            else "left"
            if self.left_ml > self.right_ml
            else "right"
        )
        return {
            "presence": "yes" if self.enhancing_ml >= ENHANCING_PRESENCE_ML else "no",
            "laterality": laterality,
            "relative_volume": "edema" if self.edema_ml >= self.core_ml else "tumor core",
            "cross_region_comparison": "yes"
            if self.core_ml > 0.5 * self.whole_ml
            else "no",
        }

    def boundary_margins(self) -> dict[str, float]:
        laterality_ratio = abs(self.left_ml - self.right_ml) / self.whole_ml
        return {
            "presence_margin_ml": abs(self.enhancing_ml - ENHANCING_PRESENCE_ML),
            "laterality_margin_fraction": abs(laterality_ratio - MIDLINE_RELATIVE_MARGIN),
            "relative_volume_margin_fraction": abs(self.edema_ml - self.core_ml)
            / self.whole_ml,
            "cross_region_margin_fraction": abs(self.core_ml / self.whole_ml - 0.5),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit real QA targets after label resampling")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--resolutions", type=int, nargs="+", default=(16, 24, 32, 48))
    parser.add_argument("--seed", type=int, default=20260914)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    resolutions = tuple(sorted(set(args.resolutions)))
    if not resolutions or any(size <= 0 or size % 2 for size in resolutions):
        raise ValueError("resolutions must be positive even integers")
    rows = audit_resolutions(args.dataset_root.resolve(), resolutions, seed=args.seed)
    summary = summarize(rows, resolutions=resolutions, seed=args.seed)
    prefix = args.output_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(prefix.with_suffix(".csv"), rows)
    prefix.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"cases": summary["cases"], "output_prefix": str(prefix)}))


def audit_resolutions(
    dataset_root: Path, resolutions: tuple[int, ...], *, seed: int
) -> list[dict[str, object]]:
    try:
        nib: Any = importlib.import_module("nibabel")
        np: Any = importlib.import_module("numpy")
    except ImportError as error:
        raise RuntimeError("install the project with the 'data' extra") from error

    rows: list[dict[str, object]] = []
    for case in discover_training_cases(dataset_root):
        split = assign_subject(case.case_id, seed=seed)
        if split is Split.TEST:
            continue
        image: Any = nib.load(str(case.label))
        original_array = np.asanyarray(image.dataobj, dtype=np.int64)
        label = torch.from_numpy(np.ascontiguousarray(original_array)).permute(2, 1, 0)
        voxel_ml = float(np.prod(image.header.get_zooms()[:3])) / 1000.0
        original = measure_label(label, voxel_ml=voxel_ml)
        original_answers = original.categorical_answers()
        margins = original.boundary_margins()
        physical_fov_ml = float(label.numel()) * voxel_ml
        for size in resolutions:
            resized = functional.interpolate(
                label[None, None].float(), size=(size,) * 3, mode="nearest"
            ).squeeze(0).squeeze(0).long()
            measured = measure_label(resized, voxel_ml=physical_fov_ml / resized.numel())
            resized_answers = measured.categorical_answers()
            row: dict[str, object] = {
                "case_id": case.case_id,
                "split": split.value,
                "resolution": size,
                "enhancing_fraction_absolute_error": abs(
                    measured.enhancing_fraction - original.enhancing_fraction
                ),
                **margins,
            }
            for question, answer in original_answers.items():
                row[f"{question}_reference"] = answer
                row[f"{question}_resampled"] = resized_answers[question]
                row[f"{question}_stable"] = answer == resized_answers[question]
            rows.append(row)
    return rows


def measure_label(label: Tensor, *, voxel_ml: float) -> LabelMeasurements:
    if label.ndim != 3 or voxel_ml <= 0.0:
        raise ValueError("label must be 3D and voxel_ml must be positive")
    whole = label > 0
    whole_count = int(whole.sum().item())
    if whole_count == 0:
        raise ValueError("label must contain tumor")
    midpoint = label.shape[-1] // 2
    return LabelMeasurements(
        whole_ml=whole_count * voxel_ml,
        edema_ml=int((label == 1).sum().item()) * voxel_ml,
        core_ml=int(((label == 2) | (label == 3)).sum().item()) * voxel_ml,
        enhancing_ml=int((label == 3).sum().item()) * voxel_ml,
        left_ml=int(whole[..., :midpoint].sum().item()) * voxel_ml,
        right_ml=int(whole[..., midpoint:].sum().item()) * voxel_ml,
    )


def summarize(
    rows: list[dict[str, object]], *, resolutions: tuple[int, ...], seed: int
) -> dict[str, object]:
    questions = (
        "presence",
        "laterality",
        "relative_volume",
        "cross_region_comparison",
    )
    by_resolution: dict[str, object] = {}
    for resolution in resolutions:
        selected = [row for row in rows if row["resolution"] == resolution]
        by_split: dict[str, object] = {}
        for split in (Split.TRAIN, Split.VALIDATION):
            subset = [row for row in selected if row["split"] == split.value]
            by_split[split.value] = _aggregate(subset, questions)
        by_resolution[str(resolution)] = {
            "all_development": _aggregate(selected, questions),
            "by_split": by_split,
        }
    return {
        "status": "development_target_audit_only",
        "seed": seed,
        "test_cases_read": 0,
        "cases": len({str(row["case_id"]) for row in rows}),
        "resolutions": list(resolutions),
        "by_resolution": by_resolution,
        "train_derived_ambiguity_screen": {
            str(resolution): _ambiguity_screen(rows, resolution=resolution)
            for resolution in resolutions
        },
        "structural_warning": (
            "relative_volume and cross_region_comparison are algebraically redundant "
            "because whole volume equals edema plus tumor core"
        ),
    }


def _aggregate(rows: list[dict[str, object]], questions: tuple[str, ...]) -> dict[str, object]:
    if not rows:
        return {"cases": 0}
    stability = {
        question: sum(bool(row[f"{question}_stable"]) for row in rows) / len(rows)
        for question in questions
    }
    errors = sorted(
        cast(float, row["enhancing_fraction_absolute_error"]) for row in rows
    )
    return {
        "cases": len(rows),
        "categorical_stability": stability,
        "all_categorical_stable": sum(
            all(bool(row[f"{question}_stable"]) for question in questions) for row in rows
        )
        / len(rows),
        "enhancing_fraction_mae": sum(errors) / len(errors),
        "enhancing_fraction_p95_absolute_error": _quantile(errors, 0.95),
    }


def _ambiguity_screen(
    rows: list[dict[str, object]], *, resolution: int
) -> dict[str, object]:
    question_margins = {
        "presence": "presence_margin_ml",
        "laterality": "laterality_margin_fraction",
        "relative_volume": "relative_volume_margin_fraction",
        "cross_region_comparison": "cross_region_margin_fraction",
    }
    selected = [row for row in rows if row["resolution"] == resolution]
    output: dict[str, object] = {}
    for question, margin_name in question_margins.items():
        train = [row for row in selected if row["split"] == Split.TRAIN.value]
        validation = [row for row in selected if row["split"] == Split.VALIDATION.value]
        unstable_train_margins = [
            cast(float, row[margin_name])
            for row in train
            if not bool(row[f"{question}_stable"])
        ]
        cutoff = max(unstable_train_margins, default=0.0)
        retained = [row for row in validation if cast(float, row[margin_name]) > cutoff]
        answer_counts: dict[str, int] = {}
        for row in retained:
            answer = str(row[f"{question}_reference"])
            answer_counts[answer] = answer_counts.get(answer, 0) + 1
        output[question] = {
            "train_unstable_cases": len(unstable_train_margins),
            "train_derived_exclusive_margin": cutoff,
            "validation_retained_cases": len(retained),
            "validation_stability": (
                sum(bool(row[f"{question}_stable"]) for row in retained) / len(retained)
                if retained
                else None
            ),
            "validation_reference_answer_counts": dict(sorted(answer_counts.items())),
        }
    return output


def _quantile(ordered: list[float], fraction: float) -> float:
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty stability audit")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
