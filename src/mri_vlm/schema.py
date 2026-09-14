"""Immutable MRI, language, answer, and evidence contracts."""

from dataclasses import dataclass
from enum import StrEnum


class Modality(StrEnum):
    FLAIR = "flair"
    T1 = "t1"
    T1GD = "t1gd"
    T2 = "t2"


class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class AnswerKind(StrEnum):
    CATEGORICAL = "categorical"
    NUMERIC = "numeric"


class QuestionType(StrEnum):
    PRESENCE = "presence"
    LATERALITY = "laterality"
    RELATIVE_VOLUME = "relative_volume"
    ENHANCING_FRACTION = "enhancing_fraction"
    CROSS_REGION_COMPARISON = "cross_region_comparison"
    UNANSWERABLE = "unanswerable"


ALL_MODALITIES = frozenset(Modality)


def validate_digest(label: str, digest: str) -> None:
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class VolumeRecord:
    modality: Modality
    sha256: str
    shape: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]

    def __post_init__(self) -> None:
        validate_digest("volume sha256", self.sha256)
        if any(dimension <= 0 for dimension in self.shape):
            raise ValueError("volume dimensions must be positive")
        if any(spacing <= 0.0 for spacing in self.spacing_mm):
            raise ValueError("voxel spacing must be positive")


@dataclass(frozen=True, slots=True)
class CaseRecord:
    case_id: str
    subject_id: str
    dataset_id: str
    dataset_version: str
    volumes: tuple[VolumeRecord, ...]
    label_sha256: str
    label_values: tuple[int, ...]

    def __post_init__(self) -> None:
        if any(
            not value.strip()
            for value in (self.case_id, self.subject_id, self.dataset_id, self.dataset_version)
        ):
            raise ValueError("case identity and provenance must be non-empty")
        validate_digest("label sha256", self.label_sha256)
        modalities = [volume.modality for volume in self.volumes]
        if len(modalities) != len(set(modalities)):
            raise ValueError("case contains duplicate modalities")
        if frozenset(modalities) != ALL_MODALITIES:
            raise ValueError("source case must contain all four MRI modalities")
        if len({volume.shape for volume in self.volumes}) != 1:
            raise ValueError("registered modality shapes must agree")
        if len({volume.spacing_mm for volume in self.volumes}) != 1:
            raise ValueError("registered modality spacing must agree")
        if tuple(sorted(set(self.label_values))) != self.label_values or not self.label_values:
            raise ValueError("label values must be sorted and unique")
        if self.label_values[0] != 0:
            raise ValueError("label values must include background 0")


@dataclass(frozen=True, slots=True)
class GroundedQAExample:
    example_id: str
    case_id: str
    subject_id: str
    question: str
    question_type: QuestionType
    answer_kind: AnswerKind
    answer: str | None
    evidence_sha256: str | None
    counterfactual_group: str | None = None

    def __post_init__(self) -> None:
        if any(
            not value.strip()
            for value in (self.example_id, self.case_id, self.subject_id, self.question)
        ):
            raise ValueError("QA identity and question must be non-empty")
        if bool(self.answer is None) != bool(self.evidence_sha256 is None):
            raise ValueError("answer and evidence must both be present or both be absent")
        if self.answer is not None and not self.answer.strip():
            raise ValueError("answer must be non-empty when present")
        if self.evidence_sha256 is not None:
            validate_digest("evidence sha256", self.evidence_sha256)
        if self.counterfactual_group is not None and not self.counterfactual_group.strip():
            raise ValueError("counterfactual group must be non-empty when present")
        if (self.question_type is QuestionType.UNANSWERABLE) != (self.answer is None):
            raise ValueError("unanswerable question type and null answer must agree")

    @property
    def is_answerable(self) -> bool:
        return self.answer is not None
