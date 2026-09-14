"""MRI and derived-QA integrity checks."""

from collections.abc import Iterable, Mapping

from mri_vlm.schema import CaseRecord, GroundedQAExample, Split
from mri_vlm.split import validate_subject_isolation


def audit_partitions(partitions: Mapping[Split, Iterable[CaseRecord]]) -> None:
    materialized = {split: tuple(cases) for split, cases in partitions.items()}
    validate_subject_isolation(materialized)
    case_ids: set[str] = set()
    subject_ids: set[str] = set()
    fingerprint_owners: dict[str, Split] = {}
    for split, cases in materialized.items():
        for case in cases:
            if case.case_id in case_ids or case.subject_id in subject_ids:
                raise ValueError("duplicate case or subject ID")
            case_ids.add(case.case_id)
            subject_ids.add(case.subject_id)
            _claim(case.label_sha256, split, fingerprint_owners)
            for volume in case.volumes:
                _claim(volume.sha256, split, fingerprint_owners)


def audit_examples(cases: Iterable[CaseRecord], examples: Iterable[GroundedQAExample]) -> None:
    case_by_id = {case.case_id: case for case in cases}
    example_ids: set[str] = set()
    for example in examples:
        if example.example_id in example_ids:
            raise ValueError(f"duplicate example ID: {example.example_id}")
        example_ids.add(example.example_id)
        case = case_by_id.get(example.case_id)
        if case is None:
            raise ValueError(f"unknown case ID: {example.case_id}")
        if case.subject_id != example.subject_id:
            raise ValueError("QA subject does not match its MRI case")


def _claim(digest: str, split: Split, owners: dict[str, Split]) -> None:
    previous = owners.setdefault(digest, split)
    if previous is not split:
        raise ValueError(f"fingerprint occurs in both {previous} and {split}")
