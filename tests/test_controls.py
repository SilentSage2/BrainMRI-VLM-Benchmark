import pytest

from mri_vlm.controls import bootstrap_mean_ci, evaluate_question_only, fit_question_only
from mri_vlm.real_qa import SplitQAExample, _laterality
from mri_vlm.schema import AnswerKind, GroundedQAExample, QuestionType, Split


def _item(subject: str, question_type: QuestionType, answer: str | None) -> SplitQAExample:
    return SplitQAExample(
        Split.TRAIN,
        GroundedQAExample(
            example_id=f"{subject}:{question_type.value}",
            case_id=subject,
            subject_id=subject,
            question="question",
            question_type=question_type,
            answer_kind=(
                AnswerKind.NUMERIC
                if question_type is QuestionType.ENHANCING_FRACTION
                else AnswerKind.CATEGORICAL
            ),
            answer=answer,
            evidence_sha256="a" * 64 if answer is not None else None,
        ),
    )


def test_question_only_fits_train_and_reports_zero_grounding() -> None:
    train = (
        _item("a", QuestionType.PRESENCE, "yes"),
        _item("b", QuestionType.PRESENCE, "yes"),
        _item("c", QuestionType.PRESENCE, "no"),
        _item("a", QuestionType.ENHANCING_FRACTION, "0.1"),
        _item("b", QuestionType.ENHANCING_FRACTION, "0.3"),
        _item("a", QuestionType.UNANSWERABLE, None),
    )
    predictions = fit_question_only(train)
    result = evaluate_question_only(
        train, predictions, bootstrap_seed=1, bootstrap_samples=100
    )

    assert predictions[QuestionType.PRESENCE] == "yes"
    assert predictions[QuestionType.ENHANCING_FRACTION] == "0.200000"
    assert result.answer_accuracy == pytest.approx(0.4)
    assert result.grounded_answer_accuracy == 0.0
    assert result.unanswerable_hallucination_rate == 0.0


def test_bootstrap_is_deterministic() -> None:
    first = bootstrap_mean_ci([0.0, 0.5, 1.0], seed=3, samples=100)
    second = bootstrap_mean_ci([0.0, 0.5, 1.0], seed=3, samples=100)
    assert first == second


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [(10.0, 1.0, "left"), (1.0, 10.0, "right"), (10.0, 10.4, "midline")],
)
def test_laterality_rule(left: float, right: float, expected: str) -> None:
    assert _laterality(left, right) == expected
