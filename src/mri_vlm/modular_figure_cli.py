"""Render modular MR robustness and frozen failure-profile Figure 5."""

import argparse
import csv
import importlib
import statistics
from pathlib import Path
from typing import Any, cast

from mri_vlm.conditions import condition_id, modality_conditions
from mri_vlm.matched_figure_cli import load_test_sealed
from mri_vlm.schema import Modality


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render modular MR V4 Figure 5")
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = load_test_sealed(args.summary)
    seed_dir = args.summary.parent / "seeds"
    seeds = cast(list[int], summary["seeds"])
    payloads = {
        mode: [
            load_test_sealed(seed_dir / f"seed-{seed}-{mode}.json") for seed in seeds
        ]
        for mode in ("dropout", "no_dropout")
    }
    render(summary, payloads, output_prefix=args.output_prefix)


def render(
    summary: dict[str, Any],
    payloads: dict[str, list[dict[str, Any]]],
    *,
    output_prefix: Path,
) -> None:
    matplotlib: Any = importlib.import_module("matplotlib")
    matplotlib.use("Agg")
    plt: Any = importlib.import_module("matplotlib.pyplot")
    conditions = [condition_id(item) for item in modality_conditions()]
    full_id = condition_id(frozenset(Modality))
    missing = [item for item in conditions if item != full_id]
    modes = ("dropout", "no_dropout")
    region_names = ("whole_tumor", "tumor_core", "enhancing_tumor")
    region_labels = ("WT", "TC", "ET")
    full_dice = {
        mode: [
            statistics.mean(
                payload["evaluation"][full_id]["mean_region_dice"][region]
                for payload in payloads[mode]
            )
            for region in region_names
        ]
        for mode in modes
    }
    condition_accuracy = {
        mode: [
            statistics.mean(
                payload["evaluation"][condition]["symbolic_balanced_accuracy"]
                for payload in payloads[mode]
            )
            for condition in conditions
        ]
        for mode in modes
    }
    subject_profiles = {
        mode: subject_missing_profiles(payloads[mode], missing) for mode in modes
    }
    selected = select_failure_profiles(subject_profiles["dropout"], count=3)
    effect = summary["aggregate"]["dropout_minus_no_dropout_missing_symbolic_accuracy"]

    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    figure, axes = plt.subplots(
        1,
        4,
        figsize=(15, 3.8),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1.0, 2.0, 1.35, 1.1)},
    )
    positions = range(3)
    axes[0].bar(
        [item - 0.18 for item in positions],
        full_dice["dropout"],
        width=0.36,
        label="Dropout",
        color="#3182bd",
    )
    axes[0].bar(
        [item + 0.18 for item in positions],
        full_dice["no_dropout"],
        width=0.36,
        label="No dropout",
        color="#9ecae1",
    )
    axes[0].set_xticks(list(positions), region_labels)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Dice")
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].set_title("A  Full-input segmentation", loc="left", fontweight="bold")

    matrix = [condition_accuracy[mode] for mode in modes]
    image = axes[1].imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    axes[1].set_yticks(range(2), ("Dropout", "No dropout"))
    abbreviations = {"flair": "F", "t1": "T1", "t1gd": "G", "t2": "T2"}
    condition_labels = [
        "/".join(abbreviations[token] for token in item.split("+")) for item in conditions
    ]
    axes[1].set_xticks(
        range(15), condition_labels, rotation=60, ha="right", fontsize=6.5
    )
    axes[1].set_title("B  QA across 15 contrast subsets", loc="left", fontweight="bold")
    figure.colorbar(image, ax=axes[1], label="Balanced accuracy", fraction=0.05)

    y = range(len(selected))
    axes[2].scatter(
        [subject_profiles["dropout"][subject] for subject in selected],
        y,
        label="Dropout",
        color="#3182bd",
    )
    axes[2].scatter(
        [subject_profiles["no_dropout"][subject] for subject in selected],
        y,
        label="No dropout",
        color="#9ecae1",
        marker="s",
    )
    axes[2].set_yticks(list(y), selected, fontsize=6.5)
    axes[2].set_xlim(0.0, 1.0)
    axes[2].set_xlabel("Mean missing-contrast accuracy")
    axes[2].set_title("C  Frozen extreme profiles", loc="left", fontweight="bold")
    axes[2].legend(frameon=False, fontsize=7)
    axes[2].invert_yaxis()

    mean = effect["mean"]
    interval = effect["hierarchical_subject_bootstrap_95ci"]
    axes[3].errorbar(
        mean,
        0,
        xerr=([mean - interval[0]], [interval[1] - mean]),
        fmt="o",
        color="#3182bd",
        capsize=4,
    )
    axes[3].axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[3].set_yticks([0], ["Dropout minus\nno dropout"])
    axes[3].set_xlabel("Missing-contrast accuracy effect")
    axes[3].set_title("D  Dropout effect (95% CI)", loc="left", fontweight="bold")

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_prefix.with_suffix(".png"), dpi=300)
    figure.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(figure)
    write_source(
        output_prefix.with_name(f"{output_prefix.name}_source.csv"),
        full_dice,
        condition_accuracy,
        subject_profiles,
        selected,
        effect,
    )


def subject_missing_profiles(
    payloads: list[dict[str, Any]], conditions: list[str]
) -> dict[str, float]:
    reference = payloads[0]["evaluation"][conditions[0]]["subject_ids"]
    output: dict[str, float] = {}
    for index, subject in enumerate(reference):
        values = [
            payload["evaluation"][condition]["subject_symbolic_scores"][index]
            for payload in payloads
            for condition in conditions
        ]
        output[subject] = statistics.mean(values)
    return output


def select_failure_profiles(profiles: dict[str, float], *, count: int) -> tuple[str, ...]:
    if len(profiles) < 2 * count:
        raise ValueError("not enough subjects for disjoint failure and success profiles")
    ordered = sorted(profiles, key=lambda subject: (profiles[subject], subject))
    return tuple(ordered[:count] + ordered[-count:])


def write_source(
    path: Path,
    full_dice: dict[str, list[float]],
    condition_accuracy: dict[str, list[float]],
    subject_profiles: dict[str, dict[str, float]],
    selected: tuple[str, ...],
    effect: dict[str, Any],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("panel", "mode", "item", "estimate", "ci_low", "ci_high"))
        for mode, values in full_dice.items():
            for region, value in zip(("WT", "TC", "ET"), values, strict=True):
                writer.writerow(("A", mode, region, value, "", ""))
        for mode, values in condition_accuracy.items():
            for condition, value in zip(
                [condition_id(item) for item in modality_conditions()], values, strict=True
            ):
                writer.writerow(("B", mode, condition, value, "", ""))
        for mode, profiles in subject_profiles.items():
            for subject in selected:
                writer.writerow(("C", mode, subject, profiles[subject], "", ""))
        interval = effect["hierarchical_subject_bootstrap_95ci"]
        writer.writerow(("D", "dropout_minus_no_dropout", "missing", effect["mean"], *interval))


if __name__ == "__main__":
    main()
