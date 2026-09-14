"""Deterministic subject-level split assignment."""

import hashlib
from collections.abc import Iterable, Mapping

from mri_vlm.schema import CaseRecord, Split


def assign_subject(
    subject_id: str,
    *,
    seed: int,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> Split:
    if not subject_id:
        raise ValueError("subject_id must be non-empty")
    if not 0.0 < train_fraction < 1.0 or not 0.0 <= validation_fraction < 1.0:
        raise ValueError("invalid split fractions")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("fractions must leave a test interval")
    digest = hashlib.sha256(f"{seed}:{subject_id}".encode()).digest()
    position = int.from_bytes(digest[:8], "big") / 2**64
    if position < train_fraction:
        return Split.TRAIN
    if position < train_fraction + validation_fraction:
        return Split.VALIDATION
    return Split.TEST


def split_cases(
    cases: Iterable[CaseRecord], *, seed: int
) -> dict[Split, tuple[CaseRecord, ...]]:
    buckets: dict[Split, list[CaseRecord]] = {split: [] for split in Split}
    for case in cases:
        buckets[assign_subject(case.subject_id, seed=seed)].append(case)
    return {split: tuple(items) for split, items in buckets.items()}


def validate_subject_isolation(partitions: Mapping[Split, Iterable[CaseRecord]]) -> None:
    owners: dict[str, Split] = {}
    for split, cases in partitions.items():
        for case in cases:
            previous = owners.setdefault(case.subject_id, split)
            if previous is not split:
                raise ValueError(f"subject {case.subject_id!r} occurs in {previous} and {split}")
