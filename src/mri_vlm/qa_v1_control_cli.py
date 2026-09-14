"""Question-only prior for the frozen balanced QA V1 target space."""

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from mri_vlm.baseline_cli import load_cases, select_case_ids
from mri_vlm.controls import bootstrap_mean_ci
from mri_vlm.matched_cli import MatchedCase, target_counts
from mri_vlm.preprocess import PreprocessSpec
from mri_vlm.qa_v1 import (
    ANSWER_VOCAB_V1,
    FROZEN_FRACTION_THRESHOLDS,
    build_word_vocabulary,
    materialize_targets,
)
from mri_vlm.real_qa import SplitQAExample, generate_real_examples
from mri_vlm.schema import QuestionType, Split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the QA V1 question-only prior")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/processed/baseline-v2"))
    parser.add_argument("--train-cases", type=int, default=64)
    parser.add_argument("--validation-cases", type=int, default=16)
    parser.add_argument("--spatial-size", type=int, default=48)
    parser.add_argument("--split-seed", type=int, default=20260914)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
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
    predictions = fit_question_only_v1(prepared[Split.TRAIN])
    result = evaluate_question_only_v1(
        prepared[Split.VALIDATION],
        predictions,
        bootstrap_seed=args.split_seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    output = {
        "status": "validation-only-qa-v1-control",
        "scope": "question-family prior on frozen QA V1; no images and no test cases read",
        "test_cases_read": 0,
        "split_seed": args.split_seed,
        "config": {
            "train_cases": len(prepared[Split.TRAIN]),
            "validation_cases": len(prepared[Split.VALIDATION]),
            "spatial_size": args.spatial_size,
            "preprocessing": asdict(spec),
            "fraction_thresholds": asdict(FROZEN_FRACTION_THRESHOLDS),
        },
        "train_case_ids": ids[Split.TRAIN],
        "validation_case_ids": ids[Split.VALIDATION],
        "target_counts": {
            split.value: target_counts(prepared[split])
            for split in (Split.TRAIN, Split.VALIDATION)
        },
        "predictions": {key.value: ANSWER_VOCAB_V1[value] for key, value in predictions.items()},
        "evaluation": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "output": str(args.output)}))


def fit_question_only_v1(cases: tuple[MatchedCase, ...]) -> dict[QuestionType, int]:
    grouped: dict[QuestionType, Counter[int]] = defaultdict(Counter)
    for case in cases:
        for target in case.targets:
            grouped[target.question_type][target.answer_index] += 1
    return {
        question_type: min(counts, key=lambda value: (-counts[value], value))
        for question_type, counts in grouped.items()
    }


def evaluate_question_only_v1(
    cases: tuple[MatchedCase, ...],
    predictions: dict[QuestionType, int],
    *,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, object]:
    hits_by_type_answer: dict[str, dict[str, list[bool]]] = defaultdict(
        lambda: defaultdict(list)
    )
    subject_scores: list[float] = []
    all_hits: list[bool] = []
    for case in cases:
        hits: list[bool] = []
        for target in case.targets:
            hit = predictions[target.question_type] == target.answer_index
            hits.append(hit)
            all_hits.append(hit)
            hits_by_type_answer[target.question_type.value][target.answer].append(hit)
        subject_scores.append(sum(hits) / len(hits))
    balanced_by_type = {
        question_type: sum(sum(hits) / len(hits) for hits in by_answer.values())
        / len(by_answer)
        for question_type, by_answer in hits_by_type_answer.items()
    }
    return {
        "subjects": len(cases),
        "questions": len(all_hits),
        "answer_accuracy": sum(all_hits) / len(all_hits),
        "balanced_answer_accuracy": sum(balanced_by_type.values()) / len(balanced_by_type),
        "balanced_accuracy_by_type": balanced_by_type,
        "answer_subject_bootstrap_95ci": bootstrap_mean_ci(
            subject_scores, seed=bootstrap_seed, samples=bootstrap_samples
        ),
        "subject_answer_scores": subject_scores,
        "grounded_answer_accuracy": None,
    }


if __name__ == "__main__":
    main()
