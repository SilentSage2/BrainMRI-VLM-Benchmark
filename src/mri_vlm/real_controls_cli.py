"""Run real-data question-only and symbolic-oracle controls without touching test labels."""

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from mri_vlm.controls import evaluate_question_only, fit_question_only
from mri_vlm.data.msd import discover_training_cases
from mri_vlm.real_qa import generate_real_examples
from mri_vlm.schema import Split
from mri_vlm.split import assign_subject


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run real MSD QA controls")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.dataset_root.resolve()
    examples = generate_real_examples(
        root,
        seed=args.seed,
        include_splits=frozenset({Split.TRAIN, Split.VALIDATION}),
    )
    train = tuple(item for item in examples if item.split is Split.TRAIN)
    validation = tuple(item for item in examples if item.split is Split.VALIDATION)
    subject_counts = Counter(
        assign_subject(case.case_id, seed=args.seed).value
        for case in discover_training_cases(root)
    )
    learned = fit_question_only(train)
    question_only = evaluate_question_only(
        validation,
        learned,
        bootstrap_seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    output = {
        "status": "validation-only",
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "example_counts": {
            "train": len(train),
            "validation": len(validation),
        },
        "subject_counts": {
            **dict(sorted(subject_counts.items())),
            "test_status": "no predictions; aggregate labels previously audited",
        },
        "validation_question_counts": dict(
            sorted(Counter(item.example.question_type.value for item in validation).items())
        ),
        "question_only_predictions": {
            key.value: value
            for key, value in sorted(learned.items(), key=lambda pair: pair[0].value)
        },
        "question_only": asdict(question_only),
        "reference_mask_symbolic_oracle": {
            "answer_accuracy": 1.0,
            "grounded_answer_accuracy": 1.0,
            "unanswerable_hallucination_rate": 0.0,
            "purpose": "materialization consistency upper bound; not a learned baseline",
        },
        "limitations": [
            "fixed templates expose the control to question-type leakage",
            "symbolic oracle reads reference labels and is not a deployable model",
            "this command reads no test labels, but aggregate test labels were previously audited",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = args.output.with_name(f"{args.output.stem}_manifest.jsonl")
    with manifest.open("w", encoding="utf-8") as stream:
        for item in examples:
            stream.write(json.dumps(item.to_dict(), sort_keys=True) + "\n")
    print(json.dumps({"status": "pass", "output": str(args.output), **output["example_counts"]}))


if __name__ == "__main__":
    main()
