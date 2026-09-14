import pytest

from mri_vlm.audit import audit_examples, audit_partitions
from mri_vlm.schema import GroundedQAExample, Split
from mri_vlm.split import assign_subject, split_cases, validate_subject_isolation
from mri_vlm.synthetic import fixture_cases, fixture_examples


def test_subject_assignment_is_deterministic() -> None:
    assert assign_subject("subject-9", seed=7) == assign_subject("subject-9", seed=7)


def test_fixture_passes_case_and_qa_audits() -> None:
    cases = fixture_cases()
    partitions = split_cases(cases, seed=20260914)
    assert sum(map(len, partitions.values())) == len(cases)
    audit_partitions(partitions)
    audit_examples(cases, fixture_examples(cases))


def test_subject_leakage_is_rejected() -> None:
    case = fixture_cases(3)[0]
    with pytest.raises(ValueError, match="subject"):
        validate_subject_isolation({Split.TRAIN: (case,), Split.TEST: (case,)})


def test_qa_subject_must_match_case() -> None:
    cases = fixture_cases(3)
    original = fixture_examples(cases)[0]
    mismatched = GroundedQAExample(
        example_id=original.example_id,
        case_id=original.case_id,
        subject_id="different-subject",
        question=original.question,
        question_type=original.question_type,
        answer_kind=original.answer_kind,
        answer=original.answer,
        evidence_sha256=original.evidence_sha256,
    )
    with pytest.raises(ValueError, match="does not match"):
        audit_examples(cases, (mismatched,))
