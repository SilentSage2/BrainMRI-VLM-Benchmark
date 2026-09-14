import pytest

from scivlm.schema import FigureTextRecord


def make_record(**overrides: str) -> FigureTextRecord:
    values = {
        "figure_id": "figure-1",
        "caption_id": "caption-1",
        "caption": "Series A increases over time.",
        "source_group": "group-1",
        "generator_family": "trend",
        "generator_version": "v0",
        "chart_type": "line",
        "relation_type": "increasing",
        "visual_style": "minimal",
        "spec_sha256": "a" * 64,
        "image_sha256": "b" * 64,
    }
    values.update(overrides)
    return FigureTextRecord(**values)


def test_valid_record() -> None:
    assert make_record().source_group == "group-1"


@pytest.mark.parametrize("field", ["spec_sha256", "image_sha256"])
def test_invalid_digest_is_rejected(field: str) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        make_record(**{field: "not-a-digest"})
