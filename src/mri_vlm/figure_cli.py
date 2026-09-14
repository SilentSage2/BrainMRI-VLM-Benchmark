"""Generate the real-data cohort figure and its auditable source table."""

import argparse
import csv
import importlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from mri_vlm.cohort import TumorBurden, burden_from_counts, percentile
from mri_vlm.data.msd import _strip_nifti_suffix, discover_training_cases
from mri_vlm.schema import Split
from mri_vlm.split import assign_subject


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate audited MSD cohort figure")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260914)
    return parser


def load_burdens(root: Path, *, seed: int) -> tuple[TumorBurden, ...]:
    try:
        nib = importlib.import_module("nibabel")
        np = importlib.import_module("numpy")
    except ImportError as error:
        raise RuntimeError("install the project with the 'figures' extra") from error

    rows: list[TumorBurden] = []
    for case in discover_training_cases(root):
        image: Any = nib.load(str(case.label))
        array: Any = np.asanyarray(image.dataobj)
        labels, counts = np.unique(array, return_counts=True)
        label_counts = {
            int(label): int(count)
            for label, count in zip(labels.tolist(), counts.tolist(), strict=True)
        }
        spacing = image.header.get_zooms()[:3]
        voxel_volume_mm3 = float(spacing[0] * spacing[1] * spacing[2])
        rows.append(
            burden_from_counts(
                case_id=_strip_nifti_suffix(case.label.name),
                split=assign_subject(case.case_id, seed=seed),
                label_counts=label_counts,
                voxel_volume_mm3=voxel_volume_mm3,
            )
        )
    return tuple(rows)


def write_source_table(rows: tuple[TumorBurden, ...], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "case_id",
                "split",
                "edema_ml",
                "non_enhancing_ml",
                "enhancing_ml",
                "whole_tumor_ml",
                "enhancing_fraction",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "case_id": row.case_id,
                    "split": row.split.value,
                    "edema_ml": f"{row.edema_ml:.6f}",
                    "non_enhancing_ml": f"{row.non_enhancing_ml:.6f}",
                    "enhancing_ml": f"{row.enhancing_ml:.6f}",
                    "whole_tumor_ml": f"{row.whole_tumor_ml:.6f}",
                    "enhancing_fraction": f"{row.enhancing_fraction:.8f}",
                }
            )


def write_summary(rows: tuple[TumorBurden, ...], path: Path) -> None:
    split_counts = Counter(row.split.value for row in rows)
    summary = {
        "cases": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "whole_tumor_ml": _five_number([row.whole_tumor_ml for row in rows]),
        "enhancing_fraction": _five_number([row.enhancing_fraction for row in rows]),
        "cases_without_enhancing_label": sum(row.enhancing_ml == 0.0 for row in rows),
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_figure(rows: tuple[TumorBurden, ...], output_prefix: Path) -> None:
    try:
        matplotlib: Any = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        plt: Any = importlib.import_module("matplotlib.pyplot")
    except ImportError as error:
        raise RuntimeError("install the project with the 'figures' extra") from error

    colors = ("#355C7D", "#6C5B7B", "#C06C84")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), constrained_layout=True)

    split_counts = Counter(row.split for row in rows)
    split_order = (Split.TRAIN, Split.VALIDATION, Split.TEST)
    bars = axes[0].bar(
        ["Train", "Validation", "Test"],
        [split_counts[item] for item in split_order],
        color=colors,
    )
    axes[0].bar_label(bars, padding=3, fontsize=9)
    axes[0].set_ylabel("Subjects")
    axes[0].set_title("A  Locked subject split", loc="left", fontweight="bold")
    axes[0].spines[["top", "right"]].set_visible(False)

    region_values = (
        [row.edema_ml for row in rows],
        [row.non_enhancing_ml for row in rows],
        [row.enhancing_ml for row in rows],
    )
    positive_values = [[value for value in values if value > 0.0] for values in region_values]
    violins = axes[1].violinplot(positive_values, showmedians=True, showextrema=False)
    for body, color in zip(violins["bodies"], colors, strict=True):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.85)
    violins["cmedians"].set_color("black")
    axes[1].set_yscale("log")
    axes[1].set_xticks((1, 2, 3), ("Edema", "Non-enhancing", "Enhancing"))
    axes[1].tick_params(axis="x", labelrotation=18)
    axes[1].set_ylabel("Mask-derived volume (mL, log scale)")
    axes[1].set_title("B  Tumor subregion burden", loc="left", fontweight="bold")
    axes[1].spines[["top", "right"]].set_visible(False)
    upper_limit = axes[1].get_ylim()[1]
    for position, values in enumerate(region_values, start=1):
        absent = sum(value == 0.0 for value in values)
        axes[1].text(position, upper_limit / 1.3, f"absent: {absent}", ha="center", fontsize=8)

    fractions = [row.enhancing_fraction for row in rows]
    axes[2].hist(fractions, bins=20, range=(0.0, 1.0), color=colors[2], edgecolor="white")
    median = percentile(fractions, 0.5)
    axes[2].axvline(median, color="black", linestyle="--", linewidth=1.4)
    axes[2].text(
        median,
        axes[2].get_ylim()[1] * 0.92,
        f" median={median:.2f}",
        va="top",
        fontsize=9,
    )
    axes[2].set_xlabel("Enhancing / whole-tumor volume")
    axes[2].set_ylabel("Subjects")
    axes[2].set_title("C  Quantitative QA target", loc="left", fontweight="bold")
    axes[2].spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"MSD Task01 BrainTumour: audited cohort and mask-derived targets (n={len(rows)})",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.02,
        "Reference masks supervise and verify questions; they are not model inputs.",
        ha="center",
        fontsize=9,
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _five_number(values: list[float]) -> dict[str, float]:
    return {
        "minimum": percentile(values, 0.0),
        "q1": percentile(values, 0.25),
        "median": percentile(values, 0.5),
        "q3": percentile(values, 0.75),
        "maximum": percentile(values, 1.0),
    }


def main() -> None:
    args = build_parser().parse_args()
    prefix = args.output_prefix.resolve()
    rows = load_burdens(args.dataset_root.resolve(), seed=args.seed)
    write_source_table(rows, prefix.with_name(f"{prefix.name}_source.csv"))
    write_summary(rows, prefix.with_name(f"{prefix.name}_summary.json"))
    render_figure(rows, prefix)
    print(json.dumps({"status": "pass", "cases": len(rows), "output_prefix": str(prefix)}))


if __name__ == "__main__":
    main()
