"""One-shot, authorization-gated held-out evaluator. Never call during development."""

import argparse
import hashlib
import json
import tomllib
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch

from mri_vlm.baseline_cli import evaluate as evaluate_modular
from mri_vlm.baseline_cli import load_cases
from mri_vlm.conditions import condition_id, modality_conditions
from mri_vlm.matched_cli import MatchedCase, evaluate_role
from mri_vlm.modeling import MRIVLM3D
from mri_vlm.multiseed_cli import hierarchical_bootstrap_ci
from mri_vlm.preprocess import PreprocessSpec
from mri_vlm.qa_v1 import (
    ANSWER_VOCAB_V1,
    FROZEN_FRACTION_THRESHOLDS,
    build_word_vocabulary,
    materialize_targets,
)
from mri_vlm.real_qa import SplitQAExample, generate_real_examples
from mri_vlm.schema import Modality, Split
from mri_vlm.segmentation import ResidualUNet3D
from mri_vlm.split import assign_subject

AUTHORIZATION_TEXT = "AUTHORIZE MRI-VLM HELDOUT V1 ONCE"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one-shot MRI-VLM held-out evaluation")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--authorization-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = args.config.resolve()
    repo = config_path.parent.parent
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    manifest_path = args.manifest.resolve()
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
    validate_frozen_inputs(config, manifest, manifest_path=manifest_path, repo=repo)
    validate_authorization(args.authorization_file)
    lock_path = initialize_lock(args.output_dir.resolve(), config, manifest_path)

    subjects = cast(list[str], manifest["subjects"])
    invalid_subject = any(
        assign_subject(subject, seed=config["split_seed"]) is not Split.TEST
        for subject in subjects
    )
    if invalid_subject:
        raise ValueError("held-out manifest contains a non-test subject")
    root = args.dataset_root.resolve()
    examples = generate_real_examples(
        root,
        seed=config["split_seed"],
        include_splits=frozenset({Split.TEST}),
        include_case_ids=frozenset(subjects),
    )
    grouped: dict[str, list[SplitQAExample]] = defaultdict(list)
    for item in examples:
        grouped[item.example.case_id].append(item)
    spec = PreprocessSpec(
        spatial_size=config["spatial_size"],
        version=config["preprocessing"],
        preserve_background_zero=True,
        crop_to_foreground=True,
    )
    cases = load_cases(root, args.output_dir.resolve() / "cache", tuple(subjects), spec)
    vocabulary = build_word_vocabulary()
    prepared = tuple(
        MatchedCase(
            case,
            materialize_targets(
                case,
                tuple(grouped[case.case_id]),
                FROZEN_FRACTION_THRESHOLDS,
                vocabulary,
            ),
        )
        for case in cases
    )
    targets_by_case = {item.case.case_id: item.targets for item in prepared}
    conditions = modality_conditions()
    results_dir = args.output_dir.resolve() / "systems"
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = cast(list[dict[str, Any]], config["checkpoints"])
    evaluations: dict[str, list[dict[str, Any]]] = {
        "question_grounded": [],
        "modular_dropout": [],
        "modular_no_dropout": [],
    }
    for checkpoint in checkpoints:
        system = cast(str, checkpoint["system"])
        seed = cast(int, checkpoint["seed"])
        result_path = results_dir / f"{system}-seed-{seed}.json"
        if result_path.is_file():
            payload = cast(dict[str, Any], json.loads(result_path.read_text(encoding="utf-8")))
            evaluations[system].append(payload["evaluation"])
            continue
        checkpoint_path = repo / cast(str, checkpoint["path"])
        saved = cast(
            dict[str, Any], torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        )
        if system == "question_grounded":
            model = MRIVLM3D(
                vocab_size=len(vocabulary) + 1,
                answer_classes=len(ANSWER_VOCAB_V1),
                width=4,
                question_conditioned_evidence=True,
            )
            model.load_state_dict(saved["state_dict"])
            evaluation = evaluate_role(
                model,
                prepared,
                conditions=conditions,
                role="question_grounded",
                bootstrap_seed=seed,
            )
        else:
            modular = ResidualUNet3D(width=cast(int, saved["width"]))
            modular.load_state_dict(saved["state_dict"])
            evaluation = evaluate_modular(
                modular,
                tuple(item.case for item in prepared),
                targets_by_case,
                conditions=conditions,
                bootstrap_seed=seed,
            )
        payload = {
            "system": system,
            "seed": seed,
            "test_cases_read": len(subjects),
            "preprocessing": asdict(spec),
            "checkpoint_sha256": checkpoint["sha256"],
            "evaluation": evaluation,
        }
        _write(result_path, payload)
        evaluations[system].append(evaluation)

    aggregate = aggregate_heldout(
        evaluations,
        bootstrap_seed=config["bootstrap_seed_primary"],
        bootstrap_samples=config["bootstrap_samples"],
    )
    if aggregate["test_cases_read"] != len(subjects):
        raise ValueError("held-out aggregate subject count mismatch")
    output = {
        "status": "complete-one-shot-heldout-v1",
        "protocol": config["protocol"],
        "result_schema": config["result_schema"],
        "test_cases_read": len(subjects),
        "subject_count": len(subjects),
        "aggregate": aggregate,
    }
    _write(args.output_dir.resolve() / "summary.json", output)
    _write(
        lock_path,
        {
            "status": "complete",
            "protocol": config["protocol"],
            "manifest_sha256": _sha256(manifest_path),
            "test_cases_read": len(subjects),
        },
    )
    print(json.dumps({"status": output["status"], "output": str(args.output_dir / "summary.json")}))


def validate_authorization(path: Path) -> None:
    if not path.is_file() or path.read_text(encoding="utf-8").strip() != AUTHORIZATION_TEXT:
        raise PermissionError("exact one-shot held-out authorization file is required")


def validate_frozen_inputs(
    config: dict[str, Any], manifest: dict[str, Any], *, manifest_path: Path, repo: Path
) -> None:
    if config.get("status") != "locked-awaiting-explicit-user-authorization":
        raise ValueError("held-out config is not locked")
    if config.get("result_schema") != "heldout-v1-complete-aggregate-20260914":
        raise ValueError("held-out result schema changed")
    subjects = manifest.get("subjects")
    if not isinstance(subjects, list) or len(subjects) != config.get("test_subject_count"):
        raise ValueError("held-out subject manifest count mismatch")
    if _sha256(manifest_path) != config.get("test_manifest_sha256"):
        raise ValueError("held-out subject manifest hash mismatch")
    for checkpoint in cast(list[dict[str, Any]], config["checkpoints"]):
        path = repo / cast(str, checkpoint["path"])
        if not path.is_file() or _sha256(path) != checkpoint["sha256"]:
            raise ValueError(f"checkpoint missing or changed: {path}")


def initialize_lock(output_dir: Path, config: dict[str, Any], manifest_path: Path) -> Path:
    if output_dir.name != "heldout_v1":
        raise ValueError("frozen output directory must be named heldout_v1")
    lock_path = output_dir / "ONE_SHOT_LOCK.json"
    if output_dir.exists():
        if not lock_path.is_file():
            raise FileExistsError("held-out output exists without a valid lock")
        state = cast(dict[str, Any], json.loads(lock_path.read_text(encoding="utf-8")))
        if state.get("status") != "started" or state.get("protocol") != config["protocol"]:
            raise FileExistsError("held-out run is already complete or belongs to another protocol")
        return lock_path
    output_dir.mkdir(parents=True)
    _write(
        lock_path,
        {
            "status": "started",
            "protocol": config["protocol"],
            "manifest_sha256": _sha256(manifest_path),
            "test_cases_read": "begins-after-this-lock",
        },
    )
    return lock_path


def aggregate_heldout(
    evaluations: dict[str, list[dict[str, Any]]],
    *,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    full_id = condition_id(frozenset(Modality))
    conditions = [condition_id(item) for item in modality_conditions()]
    missing = [item for item in conditions if item != full_id]
    _validate_subject_alignment(evaluations, conditions)
    comparisons: dict[str, Any] = {}
    grounded = evaluations["question_grounded"]
    for system in ("modular_dropout", "modular_no_dropout"):
        comparisons[f"{system}_minus_question_grounded_missing_accuracy"] = _paired_effect(
            evaluations[system],
            grounded,
            conditions=missing,
            left_key="subject_symbolic_scores",
            right_key="subject_answer_scores",
            bootstrap_seed=bootstrap_seed,
            bootstrap_samples=bootstrap_samples,
        )
    comparisons["modular_dropout_minus_no_dropout_missing_accuracy"] = _paired_effect(
        evaluations["modular_dropout"],
        evaluations["modular_no_dropout"],
        conditions=missing,
        left_key="subject_symbolic_scores",
        right_key="subject_symbolic_scores",
        bootstrap_seed=bootstrap_seed,
        bootstrap_samples=bootstrap_samples,
    )
    systems = {
        "question_grounded": _system_summary(
            grounded,
            conditions,
            missing,
            scalar_keys=(
                "answer_accuracy",
                "balanced_answer_accuracy",
                "ece_5bin",
                "grounded_answer_accuracy",
            ),
            mapping_key="mean_evidence_dice_by_type",
        ),
        **{
            system: _system_summary(
                evaluations[system],
                conditions,
                missing,
                scalar_keys=(
                    "symbolic_answer_accuracy",
                    "symbolic_balanced_accuracy",
                    "enhancing_fraction_mae",
                    "segmentation_confidence_proxy_ece_5bin",
                    "selective_coverage",
                    "selective_answer_accuracy",
                ),
                mapping_key="mean_region_dice",
            )
            for system in ("modular_dropout", "modular_no_dropout")
        },
    }
    dropout_profiles = _mean_profiles(
        evaluations["modular_dropout"], missing, key="subject_symbolic_scores"
    )
    bins = {
        "success_ge_0.80": sum(value >= 0.80 for value in dropout_profiles),
        "boundary_gt_0.40_lt_0.80": sum(0.40 < value < 0.80 for value in dropout_profiles),
        "failure_le_0.40": sum(value <= 0.40 for value in dropout_profiles),
    }
    return {
        "test_cases_read": len(dropout_profiles),
        "conditions": conditions,
        "incomplete_conditions": missing,
        "systems": systems,
        "primary_and_secondary_comparisons": comparisons,
        "modular_dropout_subject_profile_bins": {
            "denominator": len(dropout_profiles),
            "counts": bins,
            "subject_identifiers_published": False,
        },
    }


def _paired_effect(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    conditions: list[str],
    left_key: str,
    right_key: str,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    paired_by_seed: list[list[float]] = []
    for left_result, right_result in zip(left, right, strict=True):
        left_scores = _mean_subject_scores(left_result, conditions, key=left_key)
        right_scores = _mean_subject_scores(right_result, conditions, key=right_key)
        paired_by_seed.append(
            [a - b for a, b in zip(left_scores, right_scores, strict=True)]
        )
    seed_means = [sum(values) / len(values) for values in paired_by_seed]
    return {
        "mean": sum(seed_means) / len(seed_means),
        "seed_means": seed_means,
        "hierarchical_subject_bootstrap_95ci": hierarchical_bootstrap_ci(
            paired_by_seed, seed=bootstrap_seed, samples=bootstrap_samples
        ),
    }


def _system_summary(
    results: list[dict[str, Any]],
    conditions: list[str],
    missing: list[str],
    *,
    scalar_keys: tuple[str, ...],
    mapping_key: str,
) -> dict[str, Any]:
    by_condition: dict[str, Any] = {}
    for condition in conditions:
        by_condition[condition] = {
            **{
                key: _metric_summary(results, (condition,), key=key)
                for key in scalar_keys
            },
            mapping_key: _mapping_summary(results, (condition,), key=mapping_key),
        }
    return {
        "by_condition": by_condition,
        "full_input": by_condition[condition_id(frozenset(Modality))],
        "mean_across_incomplete_conditions": {
            **{key: _metric_summary(results, tuple(missing), key=key) for key in scalar_keys},
            mapping_key: _mapping_summary(results, tuple(missing), key=mapping_key),
        },
    }


def _metric_summary(
    results: list[dict[str, Any]], conditions: tuple[str, ...], *, key: str
) -> dict[str, Any]:
    raw = [result[condition].get(key) for result in results for condition in conditions]
    values = [float(value) for value in raw if _is_number(value)]
    return {
        "mean": sum(values) / len(values) if values else None,
        "values": raw,
        "defined_value_count": len(values),
    }


def _mapping_summary(
    results: list[dict[str, Any]], conditions: tuple[str, ...], *, key: str
) -> dict[str, Any]:
    names = sorted(
        {
            name
            for result in results
            for condition in conditions
            for name in cast(dict[str, Any], result[condition].get(key, {}))
        }
    )
    output: dict[str, Any] = {}
    for name in names:
        raw = [
            cast(dict[str, Any], result[condition].get(key, {})).get(name)
            for result in results
            for condition in conditions
        ]
        values = [float(cast(float, value)) for value in raw if _is_number(value)]
        output[name] = {
            "mean": sum(values) / len(values) if values else None,
            "values": raw,
            "defined_value_count": len(values),
        }
    return output


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _validate_subject_alignment(
    evaluations: dict[str, list[dict[str, Any]]], conditions: list[str]
) -> None:
    if not evaluations["modular_dropout"]:
        raise ValueError("held-out aggregate requires completed seed evaluations")
    reference = evaluations["modular_dropout"][0][conditions[0]].get("subject_ids")
    if not isinstance(reference, list) or not reference:
        raise ValueError("held-out evaluation is missing ordered subject identifiers")
    for system, results in evaluations.items():
        if len(results) != len(evaluations["modular_dropout"]):
            raise ValueError(f"held-out seed count mismatch for {system}")
        for result in results:
            for condition in conditions:
                if result[condition].get("subject_ids") != reference:
                    raise ValueError(f"held-out subject order mismatch for {system}/{condition}")


def _mean_profiles(
    results: list[dict[str, Any]], conditions: list[str], *, key: str
) -> list[float]:
    by_seed = [_mean_subject_scores(result, conditions, key=key) for result in results]
    return [
        sum(values) / len(values)
        for values in zip(*by_seed, strict=True)
    ]


def _mean_subject_scores(
    evaluation: dict[str, Any], conditions: list[str], *, key: str
) -> list[float]:
    values = [evaluation[condition][key] for condition in conditions]
    return [
        sum(scores[index] for scores in values) / len(values)
        for index in range(len(values[0]))
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
