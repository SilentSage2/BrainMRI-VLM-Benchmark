"""Run the metadata-only V0 MRI-VLM audit."""

import argparse
import json

from mri_vlm.audit import audit_examples, audit_partitions
from mri_vlm.conditions import modality_conditions
from mri_vlm.metrics import Prediction, evaluate_grounded_qa
from mri_vlm.split import split_cases
from mri_vlm.synthetic import fixture_cases, fixture_evidence, fixture_examples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the MRI-VLM V0 fixture")
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--count", type=int, default=60)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cases = fixture_cases(args.count)
    examples = fixture_examples(cases)
    partitions = split_cases(cases, seed=args.seed)
    audit_partitions(partitions)
    audit_examples(cases, examples)
    evidence = fixture_evidence(examples)
    oracle_metrics = evaluate_grounded_qa(
        examples,
        {
            example.example_id: Prediction(
                answer=example.answer,
                evidence_voxels=evidence.get(example.example_id, frozenset()),
            )
            for example in examples
        },
        evidence,
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "seed": args.seed,
                "cases": len(cases),
                "qa_examples": len(examples),
                "counterfactual_pairs": sum(
                    example.counterfactual_group is not None for example in examples
                )
                // 2,
                "modality_conditions": len(modality_conditions()),
                "oracle_protocol_metrics": {
                    "answer_accuracy": oracle_metrics.answer_accuracy,
                    "mean_evidence_dice": oracle_metrics.mean_evidence_dice,
                    "grounded_answer_accuracy": oracle_metrics.grounded_answer_accuracy,
                    "unanswerable_hallucination_rate": (
                        oracle_metrics.unanswerable_hallucination_rate
                    ),
                    "counterfactual_consistency": oracle_metrics.counterfactual_consistency,
                },
                "split_counts": {
                    split.value: len(items) for split, items in partitions.items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
