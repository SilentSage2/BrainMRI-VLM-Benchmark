"""Leakage controls and deterministic symbolic upper bound for real QA."""

import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from mri_vlm.metrics import answer_correct
from mri_vlm.real_qa import SplitQAExample
from mri_vlm.schema import AnswerKind, QuestionType


@dataclass(frozen=True, slots=True)
class ControlResult:
    answer_accuracy: float
    grounded_answer_accuracy: float
    unanswerable_hallucination_rate: float
    subject_bootstrap_95ci: tuple[float, float]
    answerable_count: int
    unanswerable_count: int
    by_question_type: Mapping[str, float]


def fit_question_only(
    examples: Sequence[SplitQAExample],
) -> dict[QuestionType, str | None]:
    grouped: dict[QuestionType, list[str]] = defaultdict(list)
    for item in examples:
        if item.example.answer is not None:
            grouped[item.example.question_type].append(item.example.answer)
    predictions: dict[QuestionType, str | None] = {QuestionType.UNANSWERABLE: None}
    for question_type, answers in grouped.items():
        kind = next(
            item.example.answer_kind
            for item in examples
            if item.example.question_type is question_type
        )
        if kind is AnswerKind.NUMERIC:
            predictions[question_type] = f"{statistics.median(map(float, answers)):.6f}"
        else:
            counts = Counter(answers)
            predictions[question_type] = min(counts, key=lambda value: (-counts[value], value))
    return predictions


def evaluate_question_only(
    examples: Sequence[SplitQAExample],
    predictions: Mapping[QuestionType, str | None],
    *,
    bootstrap_seed: int,
    bootstrap_samples: int = 2000,
) -> ControlResult:
    hits_by_subject: dict[str, list[bool]] = defaultdict(list)
    hits_by_type: dict[str, list[bool]] = defaultdict(list)
    unanswerable = 0
    hallucinations = 0
    for item in examples:
        example = item.example
        prediction = predictions[example.question_type]
        if example.answer is None:
            unanswerable += 1
            hallucinations += prediction is not None
            continue
        hit = prediction is not None and answer_correct(
            example.answer, prediction, example.answer_kind
        )
        hits_by_subject[example.subject_id].append(hit)
        hits_by_type[example.question_type.value].append(hit)
    subject_scores = [sum(hits) / len(hits) for hits in hits_by_subject.values()]
    answerable = sum(len(hits) for hits in hits_by_subject.values())
    total_hits = sum(sum(hits) for hits in hits_by_subject.values())
    return ControlResult(
        answer_accuracy=total_hits / answerable,
        grounded_answer_accuracy=0.0,
        unanswerable_hallucination_rate=hallucinations / unanswerable,
        subject_bootstrap_95ci=bootstrap_mean_ci(
            subject_scores, seed=bootstrap_seed, samples=bootstrap_samples
        ),
        answerable_count=answerable,
        unanswerable_count=unanswerable,
        by_question_type={
            key: sum(hits) / len(hits) for key, hits in sorted(hits_by_type.items())
        },
    )


def bootstrap_mean_ci(
    values: Sequence[float], *, seed: int, samples: int
) -> tuple[float, float]:
    if not values or samples < 2:
        raise ValueError("bootstrap requires values and at least two samples")
    generator = random.Random(seed)
    estimates = sorted(
        statistics.mean(generator.choices(values, k=len(values))) for _ in range(samples)
    )
    lower = estimates[int(0.025 * (samples - 1))]
    upper = estimates[int(0.975 * (samples - 1))]
    return lower, upper
