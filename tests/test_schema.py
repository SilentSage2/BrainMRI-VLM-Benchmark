import pytest

from mri_vlm.schema import AnswerKind, GroundedQAExample, Modality, VolumeRecord


def test_volume_rejects_non_positive_shape() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        VolumeRecord(Modality.FLAIR, "a" * 64, (32, 0, 24), (1.0, 1.0, 1.5))


def test_answer_and_evidence_must_agree() -> None:
    with pytest.raises(ValueError, match="both"):
        GroundedQAExample(
            example_id="example-1",
            case_id="case-1",
            subject_id="subject-1",
            question="Is the region present?",
            question_type="presence",
            answer_kind=AnswerKind.CATEGORICAL,
            answer="yes",
            evidence_sha256=None,
        )


def test_unanswerable_example_has_no_answer_or_evidence() -> None:
    example = GroundedQAExample(
        example_id="example-1",
        case_id="case-1",
        subject_id="subject-1",
        question="What is absent?",
        question_type="unanswerable",
        answer_kind=AnswerKind.CATEGORICAL,
        answer=None,
        evidence_sha256=None,
    )
    assert not example.is_answerable
