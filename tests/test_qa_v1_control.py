import torch

from mri_vlm.matched_cli import MatchedCase
from mri_vlm.preprocess import PreprocessedCase
from mri_vlm.qa_v1 import QATargetV1
from mri_vlm.qa_v1_control_cli import evaluate_question_only_v1, fit_question_only_v1
from mri_vlm.schema import QuestionType


def _case(case_id: str, answers: tuple[tuple[QuestionType, int, str], ...]) -> MatchedCase:
    case = PreprocessedCase(
        case_id=case_id,
        volumes=torch.zeros(4, 2, 2, 2),
        label=torch.zeros(2, 2, 2, dtype=torch.long),
        cache_key="a" * 64,
        volume_sha256="b" * 64,
        label_sha256="c" * 64,
        cache_hit=False,
    )
    targets = tuple(
        QATargetV1(question_type, answer, index, (1,), torch.zeros(2, 2, 2))
        for question_type, index, answer in answers
    )
    return MatchedCase(case, targets)


def test_question_only_v1_uses_training_majority() -> None:
    cases = (
        _case("a", ((QuestionType.LATERALITY, 0, "left"),)),
        _case("b", ((QuestionType.LATERALITY, 0, "left"),)),
        _case("c", ((QuestionType.LATERALITY, 1, "right"),)),
    )
    predictions = fit_question_only_v1(cases)
    result = evaluate_question_only_v1(
        cases, predictions, bootstrap_seed=2, bootstrap_samples=100
    )
    assert predictions[QuestionType.LATERALITY] == 0
    assert result["answer_accuracy"] == 2 / 3
    assert result["balanced_answer_accuracy"] == 0.5
