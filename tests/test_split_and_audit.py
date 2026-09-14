import pytest

from scivlm.audit import audit_partitions
from scivlm.schema import FigureTextRecord, Split
from scivlm.split import assign_group, split_records, validate_group_isolation
from scivlm.synthetic import fixture_records


def test_group_assignment_is_deterministic() -> None:
    assert assign_group("study-12", seed=7) == assign_group("study-12", seed=7)


def test_fixture_split_passes_full_audit() -> None:
    partitions = split_records(
        fixture_records(),
        seed=20260914,
        held_out_families=frozenset({"correlation"}),
    )
    assert sum(map(len, partitions.values())) == 60
    assert all(
        record.generator_family != "correlation"
        for split in (Split.TRAIN, Split.VALIDATION)
        for record in partitions[split]
    )
    audit_partitions(partitions)


def test_group_leakage_is_rejected() -> None:
    record = fixture_records(3)[0]
    with pytest.raises(ValueError, match="source group"):
        validate_group_isolation({Split.TRAIN: (record,), Split.TEST: (record,)})


def test_cross_split_spec_duplicate_is_rejected() -> None:
    first, second, *_ = fixture_records(3)
    duplicate_spec = FigureTextRecord(
        figure_id=second.figure_id,
        caption_id=second.caption_id,
        caption=second.caption,
        source_group=second.source_group,
        generator_family=second.generator_family,
        generator_version=second.generator_version,
        chart_type=second.chart_type,
        relation_type=second.relation_type,
        visual_style=second.visual_style,
        spec_sha256=first.spec_sha256,
        image_sha256=second.image_sha256,
    )
    with pytest.raises(ValueError, match="spec fingerprint"):
        audit_partitions({Split.TRAIN: (first,), Split.TEST: (duplicate_spec,)})


def test_multiple_captions_for_one_figure_are_allowed_within_split() -> None:
    first = fixture_records(3)[0]
    alternate_caption = FigureTextRecord(
        figure_id=first.figure_id,
        caption_id="alternate-caption",
        caption="An alternate description of the same figure.",
        source_group=first.source_group,
        generator_family=first.generator_family,
        generator_version=first.generator_version,
        chart_type=first.chart_type,
        relation_type=first.relation_type,
        visual_style=first.visual_style,
        spec_sha256=first.spec_sha256,
        image_sha256=first.image_sha256,
    )
    audit_partitions({Split.TRAIN: (first, alternate_caption)})
