import pytest

from mri_vlm.metrics import Prediction, answer_correct, dice_score, evaluate_grounded_qa
from mri_vlm.schema import AnswerKind, GroundedQAExample


def make_example(
    example_id: str, answer: str | None, kind: AnswerKind = AnswerKind.CATEGORICAL
) -> GroundedQAExample:
    return GroundedQAExample(
        example_id=example_id,
        case_id=f"case-{example_id}",
        subject_id=f"subject-{example_id}",
        question="Fixture question?",
        question_type="fixture",
        answer_kind=kind,
        answer=answer,
        evidence_sha256="a" * 64 if answer is not None else None,
    )


def test_dice_empty_policy_and_overlap() -> None:
    assert dice_score(set(), set()) == 1.0
    assert dice_score({1, 2}, {2, 3}) == 0.5


def test_numeric_answers_use_tolerance() -> None:
    assert answer_correct("10", "10.1", AnswerKind.NUMERIC)
    assert not answer_correct("10", "11", AnswerKind.NUMERIC)


def test_grounded_metrics_match_hand_computation() -> None:
    examples = (
        make_example("categorical", "yes"),
        make_example("numeric", "10", AnswerKind.NUMERIC),
        make_example("unanswerable", None),
    )
    metrics = evaluate_grounded_qa(
        examples,
        {
            "categorical": Prediction("Yes!", frozenset({1, 2})),
            "numeric": Prediction("10.1", frozenset({3})),
            "unanswerable": Prediction("present", frozenset()),
        },
        {
            "categorical": frozenset({1, 2}),
            "numeric": frozenset({3, 4}),
        },
    )
    assert metrics.answer_accuracy == 1.0
    assert metrics.mean_evidence_dice == pytest.approx((1.0 + 2 / 3) / 2)
    assert metrics.grounded_answer_accuracy == 1.0
    assert metrics.unanswerable_hallucination_rate == 1.0
