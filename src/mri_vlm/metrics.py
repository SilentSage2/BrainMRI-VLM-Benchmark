"""Answer, voxel-evidence, and hallucination metrics for MRI-VLM evaluation."""

import math
import re
from collections.abc import Mapping, Set
from dataclasses import dataclass

from mri_vlm.schema import AnswerKind, GroundedQAExample


def normalize_answer(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9.+-]", " ", value.lower()).split())


def answer_correct(
    reference: str,
    prediction: str,
    kind: AnswerKind,
    *,
    absolute_tolerance: float = 0.05,
    relative_tolerance: float = 0.02,
) -> bool:
    if kind is AnswerKind.CATEGORICAL:
        return normalize_answer(reference) == normalize_answer(prediction)
    try:
        reference_number = float(reference)
        prediction_number = float(prediction)
    except ValueError:
        return False
    return math.isclose(
        reference_number,
        prediction_number,
        abs_tol=absolute_tolerance,
        rel_tol=relative_tolerance,
    )


def dice_score(predicted: Set[int], target: Set[int]) -> float:
    if not predicted and not target:
        return 1.0
    return 2.0 * len(predicted & target) / (len(predicted) + len(target))


@dataclass(frozen=True, slots=True)
class Prediction:
    answer: str | None
    evidence_voxels: frozenset[int]


@dataclass(frozen=True, slots=True)
class GroundedQAMetrics:
    answer_accuracy: float
    mean_evidence_dice: float
    grounded_answer_accuracy: float
    unanswerable_hallucination_rate: float
    unanswerable_evidence_rate: float
    counterfactual_consistency: float | None
    answerable_count: int
    unanswerable_count: int
    counterfactual_pair_count: int


def evaluate_grounded_qa(
    examples: tuple[GroundedQAExample, ...],
    predictions: Mapping[str, Prediction],
    reference_evidence: Mapping[str, frozenset[int]],
    *,
    evidence_threshold: float = 0.5,
    absolute_tolerance: float = 0.05,
    relative_tolerance: float = 0.02,
) -> GroundedQAMetrics:
    if not examples:
        raise ValueError("at least one example is required")
    example_ids = {example.example_id for example in examples}
    if set(predictions) != example_ids:
        raise ValueError("predictions must cover every example exactly")

    answerable = [example for example in examples if example.is_answerable]
    unanswerable = [example for example in examples if not example.is_answerable]
    if set(reference_evidence) != {example.example_id for example in answerable}:
        raise ValueError("reference evidence must cover answerable examples exactly")

    answer_hits = 0
    grounded_hits = 0
    grounded_by_id: dict[str, bool] = {}
    evidence_scores: list[float] = []
    for example in answerable:
        prediction = predictions[example.example_id]
        answer_hit = prediction.answer is not None and answer_correct(
            example.answer or "",
            prediction.answer,
            example.answer_kind,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        evidence = dice_score(
            prediction.evidence_voxels, reference_evidence[example.example_id]
        )
        answer_hits += answer_hit
        grounded_hits += answer_hit and evidence >= evidence_threshold
        grounded_by_id[example.example_id] = answer_hit and evidence >= evidence_threshold
        evidence_scores.append(evidence)

    hallucinations = sum(
        predictions[example.example_id].answer not in (None, "") for example in unanswerable
    )
    unsupported_evidence = sum(
        bool(predictions[example.example_id].evidence_voxels) for example in unanswerable
    )
    counterfactual_groups: dict[str, list[GroundedQAExample]] = {}
    for example in examples:
        if example.counterfactual_group is not None:
            counterfactual_groups.setdefault(example.counterfactual_group, []).append(example)
    for group, members in counterfactual_groups.items():
        if len(members) != 2:
            raise ValueError(f"counterfactual group {group!r} must contain exactly two examples")
        if len({member.subject_id for member in members}) != 1:
            raise ValueError(f"counterfactual group {group!r} crosses subjects")
        if any(not member.is_answerable for member in members):
            raise ValueError(f"counterfactual group {group!r} must be answerable")
    consistent_pairs = sum(
        all(grounded_by_id[member.example_id] for member in members)
        for members in counterfactual_groups.values()
    )
    answerable_count = len(answerable)
    unanswerable_count = len(unanswerable)
    return GroundedQAMetrics(
        answer_accuracy=answer_hits / answerable_count if answerable_count else 0.0,
        mean_evidence_dice=(
            sum(evidence_scores) / answerable_count if answerable_count else 0.0
        ),
        grounded_answer_accuracy=(
            grounded_hits / answerable_count if answerable_count else 0.0
        ),
        unanswerable_hallucination_rate=(
            hallucinations / unanswerable_count if unanswerable_count else 0.0
        ),
        unanswerable_evidence_rate=(
            unsupported_evidence / unanswerable_count if unanswerable_count else 0.0
        ),
        counterfactual_consistency=(
            consistent_pairs / len(counterfactual_groups) if counterfactual_groups else None
        ),
        answerable_count=answerable_count,
        unanswerable_count=unanswerable_count,
        counterfactual_pair_count=len(counterfactual_groups),
    )
