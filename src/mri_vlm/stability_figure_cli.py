"""Render the development-only QA target-validity figure from an audited summary."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, cast


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render QA resolution-stability figure")
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--screen-resolution", type=int, default=32)
    return parser


def load_summary(path: Path, *, screen_resolution: int) -> dict[str, Any]:
    summary = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    if summary.get("test_cases_read") != 0:
        raise ValueError("target-validity figure requires a development-only audit")
    resolutions = summary.get("resolutions")
    if not isinstance(resolutions, list) or screen_resolution not in resolutions:
        raise ValueError("screen resolution is absent from the audit")
    if str(screen_resolution) not in summary.get("train_derived_ambiguity_screen", {}):
        raise ValueError("audit has no train-derived ambiguity screen")
    return summary


def render(summary: dict[str, Any], output_prefix: Path, *, screen_resolution: int) -> None:
    try:
        matplotlib: Any = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        plt: Any = importlib.import_module("matplotlib.pyplot")
    except ImportError as error:
        raise RuntimeError("install the project with the 'figures' extra") from error

    plt.rcParams.update({"font.size": 7.2, "axes.titlesize": 8.3, "axes.labelsize": 7.6})
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    resolutions = [int(value) for value in summary["resolutions"]]
    by_resolution = summary["by_resolution"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), constrained_layout=True)

    series = (
        ("All categorical", "all_categorical_stable", colors[0]),
        ("Enhancing present", "presence", colors[1]),
        ("Laterality", "laterality", colors[2]),
        ("Edema vs core", "relative_volume", colors[3]),
    )
    for label, key, color in series:
        values = []
        for resolution in resolutions:
            aggregate = by_resolution[str(resolution)]["all_development"]
            value = (
                aggregate[key]
                if key == "all_categorical_stable"
                else aggregate["categorical_stability"][key]
            )
            values.append(100.0 * float(value))
        axes[0].plot(resolutions, values, marker="o", label=label, color=color, linewidth=1.5)
    axes[0].set_xticks(resolutions)
    axes[0].set_ylim(82, 101)
    axes[0].set_xlabel("Isotropic label grid")
    axes[0].set_ylabel("Answers unchanged (%)")
    axes[0].set_title("A  Resolution sensitivity", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=6.1, loc="lower right")
    axes[0].spines[["top", "right"]].set_visible(False)

    mae = [
        100.0 * float(by_resolution[str(size)]["all_development"]["enhancing_fraction_mae"])
        for size in resolutions
    ]
    p95 = [
        100.0
        * float(
            by_resolution[str(size)]["all_development"][
                "enhancing_fraction_p95_absolute_error"
            ]
        )
        for size in resolutions
    ]
    axes[1].plot(resolutions, mae, marker="o", color=colors[0], label="MAE")
    axes[1].plot(resolutions, p95, marker="s", color=colors[1], label="95th percentile")
    axes[1].set_xticks(resolutions)
    axes[1].set_xlabel("Isotropic label grid")
    axes[1].set_ylabel("Enhancing-fraction error (points)")
    axes[1].set_title("B  Continuous-target error", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, fontsize=6.3)
    axes[1].spines[["top", "right"]].set_visible(False)

    screen = summary["train_derived_ambiguity_screen"][str(screen_resolution)]
    questions = ("presence", "laterality", "relative_volume")
    labels = ("Enhancing\npresent", "Laterality", "Edema vs\ncore")
    retained = [int(screen[question]["validation_retained_cases"]) for question in questions]
    total_validation = int(
        by_resolution[str(screen_resolution)]["by_split"]["validation"]["cases"]
    )
    bars = axes[2].bar(
        labels,
        [100.0 * value / total_validation for value in retained],
        color=colors[1:],
    )
    for bar, question, count in zip(bars, questions, retained, strict=True):
        answers = screen[question]["validation_reference_answer_counts"]
        distribution = {
            "presence": f"yes {answers.get('yes', 0)}\nno {answers.get('no', 0)}",
            "laterality": f"L {answers.get('left', 0)}\nR {answers.get('right', 0)}",
            "relative_volume": (
                f"edema {answers.get('edema', 0)}\ncore {answers.get('tumor core', 0)}"
            ),
        }[question]
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() - 3.0,
            f"{count}/{total_validation}\n{distribution}",
            ha="center",
            va="top",
            fontsize=5.8,
            color="white",
            fontweight="bold",
        )
    axes[2].set_ylim(0, 108)
    axes[2].set_ylabel("Validation cases retained (%)")
    axes[2].set_title(
        f"C  Stable targets after screening ({screen_resolution}³)",
        loc="left",
        fontweight="bold",
    )
    axes[2].spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Mask-derived QA validity on development subjects (n={summary['cases']}; test unread)",
        fontsize=9.8,
        fontweight="bold",
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    summary_path = args.summary.resolve()
    summary = load_summary(summary_path, screen_resolution=args.screen_resolution)
    prefix = args.output_prefix.resolve()
    render(summary, prefix, screen_resolution=args.screen_resolution)
    source_digest = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    metadata = {
        "status": "development-only-target-validity-figure",
        "source": str(summary_path),
        "source_sha256": source_digest,
        "screen_resolution": args.screen_resolution,
        "test_cases_read": 0,
    }
    prefix.with_name(f"{prefix.name}_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "pass", "output_prefix": str(prefix)}))


if __name__ == "__main__":
    main()
