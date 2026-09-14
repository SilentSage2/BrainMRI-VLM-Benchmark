"""Run and aggregate a frozen-cohort, multi-seed matched MRI-VLM experiment."""

import argparse
import json
import random
import statistics
import subprocess
import sys
from pathlib import Path
from typing import cast


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run matched MRI-VLM models over fixed seeds")
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
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be unique")
    output_dir = args.output_dir.resolve()
    results_dir = output_dir / "seeds"
    checkpoints_dir = output_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "running",
        "test_cases_read": 0,
        "split_seed": args.split_seed,
        "seeds": args.seeds,
        "config": {
            "train_cases": args.train_cases,
            "validation_cases": args.validation_cases,
            "spatial_size": args.spatial_size,
            "epochs": args.epochs,
            "width": args.width,
            "learning_rate": args.learning_rate,
            "cache_dir": str(args.cache_dir.resolve()),
        },
        "completed_seeds": [],
    }
    manifest_path = output_dir / "progress.json"
    _write_json(manifest_path, manifest)
    payloads: list[dict[str, object]] = []
    for seed in args.seeds:
        result_path = results_dir / f"seed-{seed}.json"
        if not result_path.is_file():
            command = [
                sys.executable,
                "-m",
                "mri_vlm.matched_cli",
                str(args.dataset_root.resolve()),
                "--output",
                str(result_path),
                "--checkpoint-dir",
                str(checkpoints_dir / f"seed-{seed}"),
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
            subprocess.run(command, check=True)
        payload = cast(dict[str, object], json.loads(result_path.read_text(encoding="utf-8")))
        validate_seed_result(payload, seed=seed, split_seed=args.split_seed)
        payloads.append(payload)
        manifest["completed_seeds"] = [
            cast(int, item["seed"]) for item in payloads
        ]
        _write_json(manifest_path, manifest)
    aggregate = aggregate_results(
        payloads, bootstrap_seed=args.split_seed, bootstrap_samples=args.bootstrap_samples
    )
    manifest["status"] = "complete"
    manifest["aggregate"] = aggregate
    _write_json(manifest_path, manifest)
    _write_json(output_dir / "summary.json", manifest)
    print(json.dumps({"status": "complete", "output": str(output_dir / "summary.json")}))


def validate_seed_result(payload: dict[str, object], *, seed: int, split_seed: int) -> None:
    if payload.get("test_cases_read") != 0:
        raise ValueError("multi-seed aggregation refuses a result that read test cases")
    if payload.get("seed") != seed:
        raise ValueError("seed result does not match requested training seed")
    config = cast(dict[str, object], payload.get("config"))
    if config.get("split_seed") != split_seed:
        raise ValueError("seed result does not use the frozen split seed")


def aggregate_results(
    payloads: list[dict[str, object]], *, bootstrap_seed: int, bootstrap_samples: int
) -> dict[str, object]:
    if len(payloads) < 2:
        raise ValueError("multi-seed aggregation requires at least two completed seeds")
    train_ids = payloads[0]["train_case_ids"]
    validation_ids = payloads[0]["validation_case_ids"]
    if any(
        payload["train_case_ids"] != train_ids
        or payload["validation_case_ids"] != validation_ids
        for payload in payloads[1:]
    ):
        raise ValueError("all seeds must use identical frozen train/validation subjects")
    effects: dict[str, object] = {}
    for comparator in ("answer_only", "unconditional_auxiliary"):
        per_seed_differences = [
            _subject_differences(payload, comparator=comparator) for payload in payloads
        ]
        seed_means = [statistics.mean(values) for values in per_seed_differences]
        effects[f"grounded_minus_{comparator}_answer_accuracy"] = {
            "mean": statistics.mean(seed_means),
            "seed_means": seed_means,
            "hierarchical_subject_bootstrap_95ci": hierarchical_bootstrap_ci(
                per_seed_differences, seed=bootstrap_seed, samples=bootstrap_samples
            ),
        }
    return {
        "seeds": [payload["seed"] for payload in payloads],
        "subjects_per_seed": len(cast(list[object], validation_ids)),
        "test_cases_read": 0,
        "full_input_paired_effects": effects,
    }


def hierarchical_bootstrap_ci(
    values_by_seed: list[list[float]], *, seed: int, samples: int
) -> tuple[float, float]:
    if len(values_by_seed) < 2 or any(not values for values in values_by_seed) or samples < 2:
        raise ValueError("hierarchical bootstrap requires two seeds, subjects, and two samples")
    generator = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        sampled_seeds = generator.choices(values_by_seed, k=len(values_by_seed))
        seed_means = [
            statistics.mean(generator.choices(values, k=len(values)))
            for values in sampled_seeds
        ]
        estimates.append(statistics.mean(seed_means))
    estimates.sort()
    return estimates[int(0.025 * (samples - 1))], estimates[int(0.975 * (samples - 1))]


def _subject_differences(payload: dict[str, object], *, comparator: str) -> list[float]:
    evaluation = cast(dict[str, object], payload["evaluation"])
    grounded = cast(dict[str, object], evaluation["question_grounded"])
    reference = cast(dict[str, object], evaluation[comparator])
    full_id = "flair+t1+t1gd+t2"
    grounded_scores = cast(
        list[float], cast(dict[str, object], grounded[full_id])["subject_answer_scores"]
    )
    reference_scores = cast(
        list[float], cast(dict[str, object], reference[full_id])["subject_answer_scores"]
    )
    return [a - b for a, b in zip(grounded_scores, reference_scores, strict=True)]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
