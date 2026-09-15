"""Render the frozen held-out reliability figure from a completed one-shot summary."""

import argparse
import csv
import importlib
import json
from pathlib import Path
from typing import Any, cast

EXPECTED_SCHEMA = "heldout-v1-complete-aggregate-20260914"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the frozen held-out reliability figure")
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser


def load_heldout_summary(path: Path) -> dict[str, Any]:
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    if payload.get("status") != "complete-one-shot-heldout-v1":
        raise ValueError("held-out figure requires a completed one-shot result")
    if payload.get("result_schema") != EXPECTED_SCHEMA:
        raise ValueError("held-out figure result schema mismatch")
    if payload.get("test_cases_read") != 66 or payload.get("subject_count") != 66:
        raise ValueError("held-out figure requires exactly 66 evaluated subjects")
    aggregate_value = payload.get("aggregate")
    if not isinstance(aggregate_value, dict):
        raise ValueError("held-out aggregate is missing")
    aggregate = cast(dict[str, Any], aggregate_value)
    if aggregate.get("test_cases_read") != 66 or len(aggregate.get("conditions", [])) != 15:
        raise ValueError("held-out aggregate is incomplete")
    return payload


def render(payload: dict[str, Any], *, output_prefix: Path) -> None:
    matplotlib: Any = importlib.import_module("matplotlib")
    matplotlib.use("Agg")
    plt: Any = importlib.import_module("matplotlib.pyplot")
    aggregate = cast(dict[str, Any], payload["aggregate"])
    systems = cast(dict[str, Any], aggregate["systems"])
    conditions = cast(list[str], aggregate["conditions"])
    system_order = ("question_grounded", "modular_dropout", "modular_no_dropout")
    system_labels = ("Grounded VLM", "Modular dropout", "Modular no dropout")
    balanced_keys = {
        "question_grounded": "balanced_answer_accuracy",
        "modular_dropout": "symbolic_balanced_accuracy",
        "modular_no_dropout": "symbolic_balanced_accuracy",
    }
    heatmap = [
        [
            _mean(systems[system]["by_condition"][condition][balanced_keys[system]])
            for condition in conditions
        ]
        for system in system_order
    ]
    full_missing = [
        (
            _mean(systems[system]["full_input"][balanced_keys[system]]),
            _mean(
                systems[system]["mean_across_incomplete_conditions"][balanced_keys[system]]
            ),
        )
        for system in system_order
    ]
    dice = {
        system: [
            _mean(systems[system]["full_input"]["mean_region_dice"][region])
            for region in ("whole_tumor", "tumor_core", "enhancing_tumor")
        ]
        for system in ("modular_dropout", "modular_no_dropout")
    }
    comparisons = cast(dict[str, Any], aggregate["primary_and_secondary_comparisons"])
    comparison_keys = (
        "modular_dropout_minus_question_grounded_missing_accuracy",
        "modular_no_dropout_minus_question_grounded_missing_accuracy",
        "modular_dropout_minus_no_dropout_missing_accuracy",
    )
    comparison_labels = (
        "Dropout - grounded",
        "No dropout - grounded",
        "Dropout - no dropout",
    )

    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    figure, axes = plt.subplots(
        1,
        4,
        figsize=(15, 3.8),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1.2, 2.2, 1.1, 1.5)},
    )
    x = range(3)
    axes[0].bar(
        [value - 0.18 for value in x],
        [value[0] for value in full_missing],
        width=0.36,
        label="Full input",
        color="#3182bd",
    )
    axes[0].bar(
        [value + 0.18 for value in x],
        [value[1] for value in full_missing],
        width=0.36,
        label="14 incomplete",
        color="#9ecae1",
    )
    axes[0].set_xticks(list(x), ("Grounded", "Dropout", "No dropout"), rotation=20)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Balanced QA accuracy")
    axes[0].set_title("A  Held-out QA", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=7)

    image = axes[1].imshow(heatmap, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    axes[1].set_yticks(range(3), system_labels)
    abbreviations = {"flair": "F", "t1": "T1", "t1gd": "G", "t2": "T2"}
    labels = [
        "/".join(abbreviations[token] for token in condition.split("+"))
        for condition in conditions
    ]
    axes[1].set_xticks(range(15), labels, rotation=60, ha="right", fontsize=6.5)
    axes[1].set_title("B  All 15 contrast subsets", loc="left", fontweight="bold")
    figure.colorbar(image, ax=axes[1], label="Balanced accuracy", fraction=0.05)

    positions = range(3)
    axes[2].bar(
        [value - 0.18 for value in positions],
        dice["modular_dropout"],
        width=0.36,
        label="Dropout",
        color="#3182bd",
    )
    axes[2].bar(
        [value + 0.18 for value in positions],
        dice["modular_no_dropout"],
        width=0.36,
        label="No dropout",
        color="#9ecae1",
    )
    axes[2].set_xticks(list(positions), ("WT", "TC", "ET"))
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_ylabel("Dice")
    axes[2].set_title("C  Full-input segmentation", loc="left", fontweight="bold")
    axes[2].legend(frameon=False, fontsize=7)

    means = [float(comparisons[key]["mean"]) for key in comparison_keys]
    intervals = [comparisons[key]["hierarchical_subject_bootstrap_95ci"] for key in comparison_keys]
    lower = [mean - interval[0] for mean, interval in zip(means, intervals, strict=True)]
    upper = [interval[1] - mean for mean, interval in zip(means, intervals, strict=True)]
    axes[3].errorbar(means, range(3), xerr=(lower, upper), fmt="o", color="#de2d26", capsize=4)
    axes[3].axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[3].set_yticks(range(3), comparison_labels)
    axes[3].set_xlabel("Raw incomplete-condition accuracy effect")
    axes[3].set_title("D  Paired effects (95% CI)", loc="left", fontweight="bold")
    axes[3].invert_yaxis()

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_prefix.with_suffix(".png"), dpi=300)
    figure.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(figure)
    _write_source(output_prefix.with_name(f"{output_prefix.name}_source.csv"), aggregate)


def _mean(summary: dict[str, Any]) -> float:
    value = summary.get("mean")
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError("held-out figure requires every displayed metric")
    return float(value)


def _write_source(path: Path, aggregate: dict[str, Any]) -> None:
    systems = cast(dict[str, Any], aggregate["systems"])
    conditions = cast(list[str], aggregate["conditions"])
    balanced_keys = {
        "question_grounded": "balanced_answer_accuracy",
        "modular_dropout": "symbolic_balanced_accuracy",
        "modular_no_dropout": "symbolic_balanced_accuracy",
    }
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("panel", "system", "item", "estimate", "ci_low", "ci_high", "schema")
        )
        for system, key in balanced_keys.items():
            writer.writerow(
                (
                    "A",
                    system,
                    "full_input_balanced_qa",
                    _mean(systems[system]["full_input"][key]),
                    "",
                    "",
                    EXPECTED_SCHEMA,
                )
            )
            writer.writerow(
                (
                    "A",
                    system,
                    "incomplete_mean_balanced_qa",
                    _mean(systems[system]["mean_across_incomplete_conditions"][key]),
                    "",
                    "",
                    EXPECTED_SCHEMA,
                )
            )
            for condition in conditions:
                writer.writerow(
                    (
                        "B",
                        system,
                        condition,
                        _mean(systems[system]["by_condition"][condition][key]),
                        "",
                        "",
                        EXPECTED_SCHEMA,
                    )
                )
        for system in ("modular_dropout", "modular_no_dropout"):
            for region, summary in systems[system]["full_input"]["mean_region_dice"].items():
                writer.writerow(
                    ("C", system, region, _mean(summary), "", "", EXPECTED_SCHEMA)
                )
        comparisons = cast(dict[str, Any], aggregate["primary_and_secondary_comparisons"])
        for name, comparison in comparisons.items():
            interval = comparison["hierarchical_subject_bootstrap_95ci"]
            writer.writerow(
                (
                    "D",
                    name,
                    "incomplete_raw_accuracy",
                    comparison["mean"],
                    interval[0],
                    interval[1],
                    EXPECTED_SCHEMA,
                )
            )
        bins = aggregate["modular_dropout_subject_profile_bins"]
        for name, count in bins["counts"].items():
            writer.writerow(
                ("metadata", "modular_dropout", name, count, "", "", EXPECTED_SCHEMA)
            )


def main() -> None:
    args = build_parser().parse_args()
    payload = load_heldout_summary(args.summary.resolve())
    render(payload, output_prefix=args.output_prefix.resolve())


if __name__ == "__main__":
    main()
