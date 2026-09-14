"""Immutable data contracts used by preparation and evaluation."""

from dataclasses import dataclass
from enum import StrEnum


class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class FigureTextRecord:
    figure_id: str
    caption_id: str
    caption: str
    source_group: str
    generator_family: str
    generator_version: str
    chart_type: str
    relation_type: str
    visual_style: str
    spec_sha256: str
    image_sha256: str

    def __post_init__(self) -> None:
        text_fields = (
            self.figure_id,
            self.caption_id,
            self.caption,
            self.source_group,
            self.generator_family,
            self.generator_version,
            self.chart_type,
            self.relation_type,
            self.visual_style,
        )
        if any(not value.strip() for value in text_fields):
            raise ValueError("record text fields must be non-empty")
        for label, digest in (
            ("spec_sha256", self.spec_sha256),
            ("image_sha256", self.image_sha256),
        ):
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")
