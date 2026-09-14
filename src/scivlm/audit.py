"""Dataset integrity and leakage checks."""

from collections.abc import Iterable, Mapping

from scivlm.schema import FigureTextRecord, Split
from scivlm.split import validate_group_isolation


def audit_partitions(partitions: Mapping[Split, Iterable[FigureTextRecord]]) -> None:
    """Validate identifiers, provenance, groups, and exact cross-split duplicates."""
    materialized = {split: tuple(records) for split, records in partitions.items()}
    validate_group_isolation(materialized)

    figure_identity: dict[str, tuple[str, str]] = {}
    caption_ids: set[str] = set()
    spec_owner: dict[str, Split] = {}
    image_owner: dict[str, Split] = {}

    for split, records in materialized.items():
        for record in records:
            identity = (record.source_group, record.image_sha256)
            previous_identity = figure_identity.setdefault(record.figure_id, identity)
            if previous_identity != identity:
                raise ValueError(f"inconsistent figure ID: {record.figure_id}")
            if record.caption_id in caption_ids:
                raise ValueError(f"duplicate caption ID: {record.caption_id}")
            caption_ids.add(record.caption_id)
            _claim_fingerprint("spec", record.spec_sha256, split, spec_owner)
            _claim_fingerprint("image", record.image_sha256, split, image_owner)


def _claim_fingerprint(
    kind: str, digest: str, split: Split, ownership: dict[str, Split]
) -> None:
    previous = ownership.setdefault(digest, split)
    if previous is not split:
        raise ValueError(f"{kind} fingerprint occurs in both {previous} and {split}")
