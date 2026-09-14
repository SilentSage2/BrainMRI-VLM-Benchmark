"""Run the metadata-only V0 MRI-VLM audit."""

import argparse
import json

from mri_vlm.audit import audit_examples, audit_partitions
from mri_vlm.split import split_cases
from mri_vlm.synthetic import fixture_cases, fixture_examples


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
    print(
        json.dumps(
            {
                "status": "pass",
                "seed": args.seed,
                "cases": len(cases),
                "qa_examples": len(examples),
                "split_counts": {
                    split.value: len(items) for split, items in partitions.items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
