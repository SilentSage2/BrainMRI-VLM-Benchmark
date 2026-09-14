"""Command-line audit for the deterministic V0 fixture."""

import argparse
import json

from scivlm.audit import audit_partitions
from scivlm.split import split_records
from scivlm.synthetic import fixture_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the SciVLM V0 synthetic fixture")
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--count", type=int, default=60)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    partitions = split_records(
        fixture_records(args.count),
        seed=args.seed,
        held_out_families=frozenset({"correlation"}),
    )
    audit_partitions(partitions)
    print(
        json.dumps(
            {
                "status": "pass",
                "seed": args.seed,
                "records": args.count,
                "split_counts": {
                    split.value: len(records) for split, records in partitions.items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
