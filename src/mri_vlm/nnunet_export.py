"""Leakage-safe Phase II export from MSD Task01 to nnU-Net v2 format."""

import importlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from mri_vlm.data.msd import RawCasePaths, discover_training_cases, load_modality_mapping
from mri_vlm.schema import Split
from mri_vlm.split import assign_subject


@dataclass(frozen=True, slots=True)
class Phase2ExportPlan:
    dataset_name: str
    training_subjects: tuple[str, ...]
    validation_subjects: tuple[str, ...]
    retired_subjects: tuple[str, ...]
    cases: tuple[RawCasePaths, ...]

    @property
    def development_subjects(self) -> tuple[str, ...]:
        return self.training_subjects + self.validation_subjects

    def summary(self) -> dict[str, object]:
        return {
            "dataset_name": self.dataset_name,
            "training_subject_count": len(self.training_subjects),
            "validation_subject_count": len(self.validation_subjects),
            "retired_subject_count": len(self.retired_subjects),
            "development_subject_count": len(self.development_subjects),
            "retired_subjects_exported": 0,
        }


def build_phase2_plan(
    root: Path,
    retired_manifest: Path,
    *,
    dataset_name: str = "Dataset501_MSDTask01Phase2Dev",
    split_seed: int = 20260914,
) -> Phase2ExportPlan:
    modalities = load_modality_mapping(root)
    if len(modalities) != 4:
        raise ValueError("Phase II nnU-Net export requires exactly four contrasts")
    cases = discover_training_cases(root)
    retired = _load_retired_subjects(retired_manifest, expected_seed=split_seed)
    by_id = {case.case_id: case for case in cases}
    if len(by_id) != len(cases):
        raise ValueError("duplicate case identifiers in source dataset")
    missing = sorted(retired - set(by_id))
    if missing:
        raise ValueError(f"retired manifest contains unknown subjects: {missing}")

    assigned_test = {
        case.case_id
        for case in cases
        if assign_subject(case.case_id, seed=split_seed) is Split.TEST
    }
    if assigned_test != retired:
        raise ValueError("retired manifest does not exactly match deterministic test assignment")

    training = tuple(
        sorted(
            case.case_id
            for case in cases
            if assign_subject(case.case_id, seed=split_seed) is Split.TRAIN
        )
    )
    validation = tuple(
        sorted(
            case.case_id
            for case in cases
            if assign_subject(case.case_id, seed=split_seed) is Split.VALIDATION
        )
    )
    development = set(training) | set(validation)
    if development & retired:
        raise ValueError("retired subjects overlap Phase II development export")
    selected_cases = tuple(by_id[case_id] for case_id in sorted(development))
    return Phase2ExportPlan(
        dataset_name=dataset_name,
        training_subjects=training,
        validation_subjects=validation,
        retired_subjects=tuple(sorted(retired)),
        cases=selected_cases,
    )


def nnunet_dataset_json(plan: Phase2ExportPlan) -> dict[str, object]:
    return {
        "channel_names": {"0": "FLAIR", "1": "T1", "2": "T1GD", "3": "T2"},
        "labels": {
            "background": 0,
            "whole_tumor": [1, 2, 3],
            "tumor_core": [2, 3],
            "enhancing_tumor": [3],
        },
        "regions_class_order": [1, 2, 3],
        "numTraining": len(plan.development_subjects),
        "file_ending": ".nii.gz",
        "name": plan.dataset_name,
        "description": "MSD Task01 Phase II development only; Phase I test subjects excluded",
        "licence": "CC-BY-SA 4.0",
    }


def nnunet_split_json(plan: Phase2ExportPlan) -> list[dict[str, list[str]]]:
    return [
        {
            "train": list(plan.training_subjects),
            "val": list(plan.validation_subjects),
        }
    ]


def materialize_phase2_export(plan: Phase2ExportPlan, output_root: Path) -> Path:
    """Write four 3D channels per development case; never export retired subjects."""
    dataset_root = output_root / plan.dataset_name
    if dataset_root.exists() and any(dataset_root.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty export: {dataset_root}")
    images = dataset_root / "imagesTr"
    labels = dataset_root / "labelsTr"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    try:
        nib = importlib.import_module("nibabel")
    except ImportError as error:
        raise RuntimeError("install the project with the 'data' extra") from error

    retired = set(plan.retired_subjects)
    for case in plan.cases:
        if case.case_id in retired:
            raise RuntimeError("retired subject reached materialization boundary")
        source: Any = nib.load(str(case.image))
        if len(source.shape) != 4 or int(source.shape[3]) != 4:
            raise ValueError(f"case {case.case_id} does not contain four channels")
        for channel in range(4):
            destination = images / f"{case.case_id}_{channel:04d}.nii.gz"
            nib.save(source.slicer[..., channel], str(destination))
        shutil.copy2(case.label, labels / f"{case.case_id}.nii.gz")

    (dataset_root / "dataset.json").write_text(
        json.dumps(nnunet_dataset_json(plan), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    (dataset_root / "phase2_split.json").write_text(
        json.dumps(nnunet_split_json(plan), indent=2) + "\n",
        encoding="utf-8",
    )
    (dataset_root / "export_summary.json").write_text(
        json.dumps(plan.summary(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return dataset_root


def _load_retired_subjects(path: Path, *, expected_seed: int) -> set[str]:
    raw = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    if raw.get("seed") != expected_seed or raw.get("split") != "test":
        raise ValueError("retired manifest seed or split does not match Phase II protocol")
    subjects = raw.get("subjects")
    if not isinstance(subjects, list) or not all(isinstance(value, str) for value in subjects):
        raise ValueError("retired manifest must contain a string subject list")
    retired = set(cast(list[str], subjects))
    if len(retired) != raw.get("subject_count"):
        raise ValueError("retired manifest count is inconsistent or contains duplicates")
    return retired
