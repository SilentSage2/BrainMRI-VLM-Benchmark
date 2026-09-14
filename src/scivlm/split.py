"""Deterministic source-group splitting and leakage validation."""

import hashlib
from collections.abc import Iterable, Mapping

from scivlm.schema import FigureTextRecord, Split


def assign_group(
    source_group: str,
    *,
    seed: int,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> Split:
    """Assign a complete source group using a stable hash."""
    if not source_group:
        raise ValueError("source_group must be non-empty")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train and validation fractions must leave a test split")

    digest = hashlib.sha256(f"{seed}:{source_group}".encode()).digest()
    position = int.from_bytes(digest[:8], "big") / 2**64
    if position < train_fraction:
        return Split.TRAIN
    if position < train_fraction + validation_fraction:
        return Split.VALIDATION
    return Split.TEST


def split_records(
    records: Iterable[FigureTextRecord],
    *,
    seed: int,
    held_out_families: frozenset[str] = frozenset(),
) -> dict[Split, tuple[FigureTextRecord, ...]]:
    """Split source groups, reserving declared generator families for test."""
    buckets: dict[Split, list[FigureTextRecord]] = {split: [] for split in Split}
    materialized = tuple(records)
    group_families: dict[str, str] = {}
    for record in materialized:
        previous = group_families.setdefault(record.source_group, record.generator_family)
        if previous != record.generator_family:
            raise ValueError(
                f"source group {record.source_group!r} has multiple generator families"
            )
    for record in materialized:
        split = (
            Split.TEST
            if record.generator_family in held_out_families
            else assign_group(record.source_group, seed=seed)
        )
        buckets[split].append(record)
    return {split: tuple(items) for split, items in buckets.items()}


def validate_group_isolation(partitions: Mapping[Split, Iterable[FigureTextRecord]]) -> None:
    """Reject source groups assigned to more than one split."""
    ownership: dict[str, Split] = {}
    for split, records in partitions.items():
        for record in records:
            previous = ownership.setdefault(record.source_group, split)
            if previous is not split:
                raise ValueError(
                    f"source group {record.source_group!r} occurs in {previous} and {split}"
                )
