import torch

from mri_vlm.preprocess import PreprocessedCase
from mri_vlm.qa_v1 import (
    FractionThresholds,
    batch_targets,
    build_word_vocabulary,
    fraction_bin,
    materialize_targets,
)
from mri_vlm.real_qa import SplitQAExample
from mri_vlm.schema import AnswerKind, GroundedQAExample, QuestionType, Split


def _example(question_type: QuestionType, answer: str) -> SplitQAExample:
    return SplitQAExample(
        split=Split.TRAIN,
        example=GroundedQAExample(
            example_id=f"case:{question_type.value}",
            case_id="case",
            subject_id="case",
            question="question",
            question_type=question_type,
            answer_kind=(
                AnswerKind.NUMERIC
                if question_type is QuestionType.ENHANCING_FRACTION
                else AnswerKind.CATEGORICAL
            ),
            answer=answer,
            evidence_sha256="0" * 64,
        ),
    )


def test_fraction_bins_include_ordered_boundaries() -> None:
    thresholds = FractionThresholds(0.1, 0.2, 0.3)
    assert fraction_bin(0.1, thresholds) == "fraction q1"
    assert fraction_bin(0.11, thresholds) == "fraction q2"
    assert fraction_bin(0.3, thresholds) == "fraction q3"
    assert fraction_bin(0.9, thresholds) == "fraction q4"


def test_materialized_evidence_differs_by_question() -> None:
    label = torch.zeros(4, 4, 4, dtype=torch.long)
    label[0, 0, 0] = 1
    label[0, 0, 1] = 2
    label[0, 0, 2] = 2
    label[0, 0, 3] = 3
    case = PreprocessedCase("case", torch.zeros(4, 4, 4, 4), label, "k", "v", "l", False)
    examples = (
        _example(QuestionType.LATERALITY, "midline"),
        _example(QuestionType.RELATIVE_VOLUME, "tumor core"),
        _example(QuestionType.ENHANCING_FRACTION, "0.5"),
    )
    targets = materialize_targets(
        case, examples, FractionThresholds(0.1, 0.2, 0.3), build_word_vocabulary()
    )
    assert [int(target.evidence.sum()) for target in targets] == [4, 3, 1]
    tokens, answers, evidence = batch_targets(targets)
    assert tokens.shape[0] == answers.shape[0] == evidence.shape[0] == 3


def test_unstable_categorical_target_is_excluded() -> None:
    label = torch.zeros(4, 4, 4, dtype=torch.long)
    label[0, 0, 0] = 1
    case = PreprocessedCase("case", torch.zeros(4, 4, 4, 4), label, "k", "v", "l", False)
    targets = materialize_targets(
        case,
        (_example(QuestionType.LATERALITY, "right"),),
        FractionThresholds(0.1, 0.2, 0.3),
        build_word_vocabulary(),
    )
    assert targets == ()
