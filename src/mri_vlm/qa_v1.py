"""Balanced, resolution-checked QA targets for matched development experiments."""

import re
from dataclasses import dataclass

import torch
from torch import Tensor

from mri_vlm.cohort import percentile
from mri_vlm.pilot_cli import _symbolic_answer
from mri_vlm.preprocess import PreprocessedCase
from mri_vlm.real_qa import SplitQAExample
from mri_vlm.schema import QuestionType

ANSWER_VOCAB_V1 = (
    "left",
    "right",
    "midline",
    "edema",
    "tumor core",
    "fraction q1",
    "fraction q2",
    "fraction q3",
    "fraction q4",
)
ANSWER_TO_INDEX_V1 = {answer: index for index, answer in enumerate(ANSWER_VOCAB_V1)}

QUESTION_TEXT = {
    QuestionType.LATERALITY: "Which hemisphere contains more whole tumor volume?",
    QuestionType.RELATIVE_VOLUME: "Which is larger, edema or tumor core?",
    QuestionType.ENHANCING_FRACTION: "What fraction of whole tumor volume is enhancing?",
}


@dataclass(frozen=True, slots=True)
class FractionThresholds:
    q1: float
    median: float
    q3: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.q1 <= self.median <= self.q3 <= 1.0:
            raise ValueError("fraction thresholds must be ordered in [0, 1]")


FROZEN_FRACTION_THRESHOLDS = FractionThresholds(0.07244455, 0.16695966, 0.27325105)


@dataclass(frozen=True, slots=True)
class QATargetV1:
    question_type: QuestionType
    answer: str
    answer_index: int
    question_tokens: tuple[int, ...]
    evidence: Tensor


def fit_fraction_thresholds(examples: tuple[SplitQAExample, ...]) -> FractionThresholds:
    values = [
        float(item.example.answer or "nan")
        for item in examples
        if item.example.question_type is QuestionType.ENHANCING_FRACTION
    ]
    if not values:
        raise ValueError("training examples contain no enhancing fractions")
    return FractionThresholds(
        q1=percentile(values, 0.25),
        median=percentile(values, 0.5),
        q3=percentile(values, 0.75),
    )


def build_word_vocabulary() -> dict[str, int]:
    words = sorted({word for text in QUESTION_TEXT.values() for word in _words(text)})
    return {word: index + 1 for index, word in enumerate(words)}


def materialize_targets(
    case: PreprocessedCase,
    examples: tuple[SplitQAExample, ...],
    thresholds: FractionThresholds,
    vocabulary: dict[str, int],
) -> tuple[QATargetV1, ...]:
    targets: list[QATargetV1] = []
    for item in examples:
        question_type = item.example.question_type
        if question_type not in QUESTION_TEXT:
            continue
        if question_type is QuestionType.ENHANCING_FRACTION:
            answer = fraction_bin(float(item.example.answer or "nan"), thresholds)
            evidence = case.label == 3
        else:
            answer = item.example.answer or ""
            if _symbolic_answer(case.label, question_type) != answer:
                continue
            if question_type is QuestionType.LATERALITY:
                evidence = case.label > 0
            else:
                evidence = (
                    case.label == 1
                    if answer == "edema"
                    else ((case.label == 2) | (case.label == 3))
                )
        tokens = tuple(vocabulary[word] for word in _words(QUESTION_TEXT[question_type]))
        targets.append(
            QATargetV1(
                question_type=question_type,
                answer=answer,
                answer_index=ANSWER_TO_INDEX_V1[answer],
                question_tokens=tokens,
                evidence=evidence,
            )
        )
    return tuple(targets)


def fraction_bin(value: float, thresholds: FractionThresholds) -> str:
    if not 0.0 <= value <= 1.0:
        raise ValueError("enhancing fraction must be in [0, 1]")
    if value <= thresholds.q1:
        return "fraction q1"
    if value <= thresholds.median:
        return "fraction q2"
    if value <= thresholds.q3:
        return "fraction q3"
    return "fraction q4"


def batch_targets(targets: tuple[QATargetV1, ...]) -> tuple[Tensor, Tensor, Tensor]:
    if not targets:
        raise ValueError("at least one QA target is required")
    length = max(len(target.question_tokens) for target in targets)
    tokens = torch.zeros(len(targets), length, dtype=torch.long)
    for index, target in enumerate(targets):
        tokens[index, : len(target.question_tokens)] = torch.tensor(target.question_tokens)
    return (
        tokens,
        torch.tensor([target.answer_index for target in targets], dtype=torch.long),
        torch.stack([target.evidence for target in targets]).float(),
    )


def _words(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z]+", text.lower()))
