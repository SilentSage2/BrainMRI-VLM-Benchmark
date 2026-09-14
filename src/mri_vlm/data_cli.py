"""Audit a locally obtained MSD Task01 dataset without writing derivatives."""

import argparse
import json
from pathlib import Path

from mri_vlm.audit import audit_partitions
from mri_vlm.data.msd import NibabelMetadataReader, build_case_records
from mri_vlm.fingerprint import sha256_json
from mri_vlm.split import split_cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit local MSD Task01_BrainTumour data")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--seed", type=int, default=20260914)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cases = build_case_records(args.dataset_root.resolve(), NibabelMetadataReader())
    partitions = split_cases(cases, seed=args.seed)
    audit_partitions(partitions)
    manifest_fingerprint = sha256_json(
        [
            {
                "case_id": case.case_id,
                "label_sha256": case.label_sha256,
                "volumes": [
                    {"modality": volume.modality.value, "sha256": volume.sha256}
                    for volume in case.volumes
                ],
            }
            for case in cases
        ]
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "cases": len(cases),
                "dataset_version": cases[0].dataset_version,
                "manifest_sha256": manifest_fingerprint,
                "split_counts": {
                    split.value: len(items) for split, items in partitions.items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
