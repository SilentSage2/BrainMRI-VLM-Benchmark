"""Render the locked V3 model-comparison figure from test-sealed result artifacts."""

import argparse
import csv
import importlib
import json
import statistics
from pathlib import Path
from typing import Any, cast

from mri_vlm.conditions import condition_id, modality_conditions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render matched MRI-VLM V3 Figure 4")
    parser.add_argument("summary", type=Path)
    parser.add_argument("--question-only", type=Path, required=True)
    parser.add_argument("--slice-baseline", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = load_test_sealed(args.summary)
    question_only = load_test_sealed(args.question_only)
    slice_baseline = load_test_sealed(args.slice_baseline)
    seed_dir = args.summary.parent / "seeds"
    seeds = cast(list[int], summary["seeds"])
    seed_payloads = [load_test_sealed(seed_dir / f"seed-{seed}.json") for seed in seeds]
    render_figure(
        summary,
        seed_payloads,
        question_only,
        slice_baseline,
        output_prefix=args.output_prefix,
    )


def load_test_sealed(path: Path) -> dict[str, Any]:
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    if payload.get("test_cases_read") != 0:
        raise ValueError(f"figure input is not test-sealed: {path}")
    return payload


def render_figure(
    summary: dict[str, Any],
    seeds: list[dict[str, Any]],
    question_only: dict[str, Any],
    slice_baseline: dict[str, Any],
    *,
    output_prefix: Path,
) -> None:
    plt: Any = importlib.import_module("matplotlib.pyplot")

    conditions = [condition_id(item) for item in modality_conditions()]
    roles = ("answer_only", "unconditional_auxiliary", "question_grounded")
    all_modalities = {item for condition in modality_conditions() for item in condition}
    full_id = condition_id(frozenset(all_modalities))
    full_balanced = [
        question_only["evaluation"]["balanced_answer_accuracy"],
        slice_baseline["evaluation"][full_id]["balanced_answer_accuracy"],
        *[
            statistics.mean(
                seed["evaluation"][role][full_id]["balanced_answer_accuracy"]
                for seed in seeds
            )
            for role in roles
        ],
    ]
    heatmap = [
        [
            statistics.mean(
                seed["evaluation"][role][condition]["balanced_answer_accuracy"]
                for seed in seeds
            )
            for condition in conditions
        ]
        for role in roles
    ]
    effects = summary["aggregate"]["full_input_paired_effects"]
    effect_items = [effects[f"grounded_minus_{role}_answer_accuracy"] for role in roles[:2]]

    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(13.5, 3.8),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1.15, 2.25, 1.35)},
    )
    labels = ("Question\nonly", "2D slice", "3D answer", "3D auxiliary", "3D grounded")
    colors = ("#8c8c8c", "#6baed6", "#9ecae1", "#3182bd", "#de2d26")
    axes[0].bar(range(len(labels)), full_balanced, color=colors)
    axes[0].set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Balanced answer accuracy")
    axes[0].set_title("A  Full-input development performance", loc="left", fontweight="bold")

    image = axes[1].imshow(heatmap, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    axes[1].set_yticks(range(3), ("Answer", "Auxiliary", "Grounded"))
    abbreviations = {"flair": "F", "t1": "T1", "t1gd": "G", "t2": "T2"}
    condition_labels = [
        "/".join(abbreviations[token] for token in item.split("+")) for item in conditions
    ]
    axes[1].set_xticks(
        range(len(conditions)), condition_labels, rotation=60, ha="right", fontsize=6.5
    )
    axes[1].set_title("B  Missing-contrast matrix", loc="left", fontweight="bold")
    figure.colorbar(image, ax=axes[1], label="Balanced accuracy", fraction=0.05)

    means = [item["mean"] for item in effect_items]
    intervals = [item["hierarchical_subject_bootstrap_95ci"] for item in effect_items]
    lower = [mean - interval[0] for mean, interval in zip(means, intervals, strict=True)]
    upper = [interval[1] - mean for mean, interval in zip(means, intervals, strict=True)]
    axes[2].errorbar(means, range(2), xerr=(lower, upper), fmt="o", color="#de2d26", capsize=4)
    axes[2].axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[2].axvline(0.05, color="#636363", linewidth=0.8, linestyle=":")
    axes[2].set_yticks(range(2), ("vs answer-only", "vs auxiliary"))
    axes[2].set_xlabel("Grounded-minus-comparator accuracy")
    axes[2].set_title("C  Hierarchical bootstrap effects", loc="left", fontweight="bold")
    axes[2].invert_yaxis()

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_prefix.with_suffix(".png"), dpi=300)
    figure.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(figure)
    _write_source_csv(
        output_prefix.with_name(f"{output_prefix.name}_source.csv"),
        labels,
        full_balanced,
        conditions,
        roles,
        heatmap,
        effect_items,
    )


def _write_source_csv(
    path: Path,
    labels: tuple[str, ...],
    full_balanced: list[float],
    conditions: list[str],
    roles: tuple[str, ...],
    heatmap: list[list[float]],
    effects: list[dict[str, Any]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("panel", "system", "condition", "estimate", "ci_low", "ci_high"))
        for label, value in zip(labels, full_balanced, strict=True):
            writer.writerow(("A", label.replace("\n", " "), "all", value, "", ""))
        for role, values in zip(roles, heatmap, strict=True):
            for condition, value in zip(conditions, values, strict=True):
                writer.writerow(("B", role, condition, value, "", ""))
        comparators = ("answer_only", "unconditional_auxiliary")
        for comparator, effect in zip(comparators, effects, strict=True):
            interval = effect["hierarchical_subject_bootstrap_95ci"]
            writer.writerow(("C", f"grounded_minus_{comparator}", "all", effect["mean"], *interval))


if __name__ == "__main__":
    main()
