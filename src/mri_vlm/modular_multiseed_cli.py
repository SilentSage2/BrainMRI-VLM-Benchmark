"""Run the modular 3D MR baseline with dropout ablations over frozen training seeds."""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

from mri_vlm.conditions import condition_id, modality_conditions
from mri_vlm.multiseed_cli import hierarchical_bootstrap_ci
from mri_vlm.schema import Modality


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run modular MR multi-seed evaluation")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/processed/v3-48"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260914, 20260915, 20260916])
    parser.add_argument("--split-seed", type=int, default=20260914)
    parser.add_argument("--train-cases", type=int, default=64)
    parser.add_argument("--validation-cases", type=int, default=16)
    parser.add_argument("--spatial-size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    results_dir = output_dir / "seeds"
    checkpoints_dir = output_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "status": "running",
        "test_cases_read": 0,
        "split_seed": args.split_seed,
        "seeds": args.seeds,
        "completed": [],
        "config": {
            "train_cases": args.train_cases,
            "validation_cases": args.validation_cases,
            "spatial_size": args.spatial_size,
            "epochs": args.epochs,
            "width": args.width,
            "learning_rate": args.learning_rate,
        },
    }
    progress_path = output_dir / "progress.json"
    _write(progress_path, manifest)
    payloads: dict[str, list[dict[str, Any]]] = {"dropout": [], "no_dropout": []}
    for seed in args.seeds:
        for mode in payloads:
            result_path = results_dir / f"seed-{seed}-{mode}.json"
            if not result_path.is_file():
                command = [
                    sys.executable,
                    "-m",
                    "mri_vlm.baseline_cli",
                    str(args.dataset_root.resolve()),
                    "--output",
                    str(result_path),
                    "--checkpoint",
                    str(checkpoints_dir / f"seed-{seed}-{mode}.pt"),
                    "--cache-dir",
                    str(args.cache_dir.resolve()),
                    "--train-cases",
                    str(args.train_cases),
                    "--validation-cases",
                    str(args.validation_cases),
                    "--spatial-size",
                    str(args.spatial_size),
                    "--epochs",
                    str(args.epochs),
                    "--width",
                    str(args.width),
                    "--learning-rate",
                    str(args.learning_rate),
                    "--seed",
                    str(seed),
                    "--split-seed",
                    str(args.split_seed),
                ]
                if mode == "no_dropout":
                    command.append("--no-modality-dropout")
                subprocess.run(command, check=True)
            payload = cast(dict[str, Any], json.loads(result_path.read_text(encoding="utf-8")))
            validate_result(payload, seed=seed, split_seed=args.split_seed, mode=mode)
            payloads[mode].append(payload)
            manifest["completed"].append({"seed": seed, "mode": mode})
            _write(progress_path, manifest)
    aggregate = aggregate_results(
        payloads, bootstrap_seed=args.split_seed, bootstrap_samples=args.bootstrap_samples
    )
    manifest["status"] = "complete"
    manifest["aggregate"] = aggregate
    _write(progress_path, manifest)
    _write(output_dir / "summary.json", manifest)
    print(json.dumps({"status": "complete", "output": str(output_dir / "summary.json")}))


def validate_result(
    payload: dict[str, Any], *, seed: int, split_seed: int, mode: str
) -> None:
    if payload.get("test_cases_read") != 0:
        raise ValueError("modular aggregation refuses a result that read test cases")
    if payload.get("seed") != seed or payload["config"].get("split_seed") != split_seed:
        raise ValueError("seed or split seed mismatch")
    expected = mode == "dropout"
    if payload["config"].get("modality_dropout") is not expected:
        raise ValueError("modality-dropout ablation mismatch")


def aggregate_results(
    payloads: dict[str, list[dict[str, Any]]],
    *,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    conditions = [condition_id(item) for item in modality_conditions()]
    full_id = condition_id(frozenset(Modality))
    missing = [item for item in conditions if item != full_id]
    reference_ids = payloads["dropout"][0]["validation_case_ids"]
    for items in payloads.values():
        if any(item["validation_case_ids"] != reference_ids for item in items):
            raise ValueError("all modular runs must use identical validation subjects")
    summaries = {
        mode: {
            "full_input": _mean_condition(items, full_id),
            "missing_contrast_mean": _mean_conditions(items, missing),
        }
        for mode, items in payloads.items()
    }
    paired_by_seed = []
    for dropout, no_dropout in zip(
        payloads["dropout"], payloads["no_dropout"], strict=True
    ):
        dropout_scores = _subject_missing_scores(dropout, missing)
        no_dropout_scores = _subject_missing_scores(no_dropout, missing)
        paired_by_seed.append(
            [a - b for a, b in zip(dropout_scores, no_dropout_scores, strict=True)]
        )
    seed_means = [statistics.mean(values) for values in paired_by_seed]
    return {
        "test_cases_read": 0,
        "summaries": summaries,
        "dropout_minus_no_dropout_missing_symbolic_accuracy": {
            "mean": statistics.mean(seed_means),
            "seed_means": seed_means,
            "hierarchical_subject_bootstrap_95ci": hierarchical_bootstrap_ci(
                paired_by_seed, seed=bootstrap_seed, samples=bootstrap_samples
            ),
        },
    }


def _mean_condition(items: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    results = [item["evaluation"][condition] for item in items]
    return _mean_metrics(results)


def _mean_conditions(items: list[dict[str, Any]], conditions: list[str]) -> dict[str, Any]:
    results = [item["evaluation"][condition] for item in items for condition in conditions]
    return _mean_metrics(results)


def _mean_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    region_names = ("whole_tumor", "tumor_core", "enhancing_tumor")
    return {
        "symbolic_balanced_accuracy": statistics.mean(
            item["symbolic_balanced_accuracy"] for item in results
        ),
        "symbolic_answer_accuracy": statistics.mean(
            item["symbolic_answer_accuracy"] for item in results
        ),
        "mean_region_dice": {
            name: statistics.mean(item["mean_region_dice"][name] for item in results)
            for name in region_names
        },
        "segmentation_confidence_proxy_ece_5bin": statistics.mean(
            item["segmentation_confidence_proxy_ece_5bin"] for item in results
        ),
        "selective_coverage": statistics.mean(item["selective_coverage"] for item in results),
    }


def _subject_missing_scores(payload: dict[str, Any], conditions: list[str]) -> list[float]:
    by_condition = [payload["evaluation"][condition] for condition in conditions]
    reference_ids = by_condition[0]["subject_ids"]
    if any(item["subject_ids"] != reference_ids for item in by_condition):
        raise ValueError("subject order changed between contrast conditions")
    return [
        statistics.mean(item["subject_symbolic_scores"][index] for item in by_condition)
        for index in range(len(reference_ids))
    ]


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
