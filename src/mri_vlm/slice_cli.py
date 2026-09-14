"""Train a fixed-policy axial 2D VLM on the frozen MRI QA V1 task."""

import argparse
import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch

from mri_vlm.baseline_cli import load_cases, select_case_ids
from mri_vlm.conditions import modality_conditions
from mri_vlm.matched_cli import (
    MatchedCase,
    _fingerprint,
    _set_seed,
    _sha256,
    answer_class_weights,
    evaluate_role,
    target_counts,
    train_role,
)
from mri_vlm.modeling import SliceVLM2D
from mri_vlm.preprocess import PreprocessSpec
from mri_vlm.qa_v1 import (
    ANSWER_VOCAB_V1,
    FROZEN_FRACTION_THRESHOLDS,
    build_word_vocabulary,
    materialize_targets,
)
from mri_vlm.real_qa import SplitQAExample, generate_real_examples
from mri_vlm.schema import Split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train fixed-policy axial slice VLM")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/processed/baseline-v2"))
    parser.add_argument("--train-cases", type=int, default=64)
    parser.add_argument("--validation-cases", type=int, default=16)
    parser.add_argument("--spatial-size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--split-seed", type=int, default=20260914)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _set_seed(args.seed)
    started = time.monotonic()
    root = args.dataset_root.resolve()
    ids = select_case_ids(
        root,
        seed=args.split_seed,
        train_count=args.train_cases,
        validation_count=args.validation_cases,
    )
    examples = generate_real_examples(
        root,
        seed=args.split_seed,
        include_splits=frozenset({Split.TRAIN, Split.VALIDATION}),
        include_case_ids=frozenset(ids[Split.TRAIN] + ids[Split.VALIDATION]),
    )
    grouped: dict[str, list[SplitQAExample]] = defaultdict(list)
    for item in examples:
        grouped[item.example.case_id].append(item)
    spec = PreprocessSpec(
        spatial_size=args.spatial_size,
        version="brain-crop-zero-background-zscore-v2",
        preserve_background_zero=True,
        crop_to_foreground=True,
    )
    vocabulary = build_word_vocabulary()
    prepared: dict[Split, tuple[MatchedCase, ...]] = {}
    for split in (Split.TRAIN, Split.VALIDATION):
        cases = load_cases(root, args.cache_dir.resolve(), ids[split], spec)
        prepared[split] = tuple(
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
    weights = answer_class_weights(prepared[Split.TRAIN], len(ANSWER_VOCAB_V1))
    model = SliceVLM2D(
        vocab_size=len(vocabulary) + 1,
        answer_classes=len(ANSWER_VOCAB_V1),
        width=args.width,
    )
    fingerprint = _fingerprint(
        {
            "model": "SliceVLM2D-fixed-axial-quartiles-v1",
            "seed": args.seed,
            "split_seed": args.split_seed,
            "train_case_ids": ids[Split.TRAIN],
            "validation_case_ids": ids[Split.VALIDATION],
            "epochs": args.epochs,
            "spatial_size": args.spatial_size,
            "width": args.width,
            "learning_rate": args.learning_rate,
            "preprocessing": asdict(spec),
        }
    )
    history, best_epoch, best_state = train_role(
        model,
        prepared[Split.TRAIN],
        prepared[Split.VALIDATION],
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        class_weights=weights,
        evidence_mode="none",
        seed=args.seed,
    )
    model.load_state_dict(best_state)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "complete": True,
            "run_fingerprint": fingerprint,
            "role": "slice_vlm_2d",
            "state_dict": best_state,
            "best_epoch": best_epoch,
            "history": history,
            "training_seconds": time.monotonic() - started,
        },
        args.checkpoint,
    )
    output = {
        "status": "single-seed-development-slice-baseline",
        "scope": "fixed-policy axial 2D VLM on QA V1; no test cases read",
        "test_cases_read": 0,
        "seed": args.seed,
        "run_fingerprint": fingerprint,
        "config": {
            "split_seed": args.split_seed,
            "train_cases": len(prepared[Split.TRAIN]),
            "validation_cases": len(prepared[Split.VALIDATION]),
            "spatial_size": args.spatial_size,
            "epochs": args.epochs,
            "width": args.width,
            "learning_rate": args.learning_rate,
            "slice_policy": "axial indices at 25%, 50%, and 75% of preprocessed depth",
            "preprocessing": asdict(spec),
        },
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_case_ids": ids[Split.TRAIN],
        "validation_case_ids": ids[Split.VALIDATION],
        "target_counts": {
            split.value: target_counts(prepared[split])
            for split in (Split.TRAIN, Split.VALIDATION)
        },
        "best_epoch": best_epoch,
        "history": history,
        "evaluation": evaluate_role(
            model,
            prepared[Split.VALIDATION],
            conditions=modality_conditions(),
            role="answer_only",
            bootstrap_seed=args.seed,
        ),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "wall_seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
