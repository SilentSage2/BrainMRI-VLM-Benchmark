"""Deterministic mask-verifiable questions from real MSD labels."""

import hashlib
import importlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mri_vlm.data.msd import discover_training_cases
from mri_vlm.schema import AnswerKind, GroundedQAExample, QuestionType, Split
from mri_vlm.split import assign_subject

ENHANCING_PRESENCE_ML = 0.1
MIDLINE_RELATIVE_MARGIN = 0.05


@dataclass(frozen=True, slots=True)
class SplitQAExample:
    split: Split
    example: GroundedQAExample

    def to_dict(self) -> dict[str, object]:
        return {"split": self.split.value, **asdict(self.example)}


def generate_real_examples(
    root: Path,
    *,
    seed: int,
    include_splits: frozenset[Split] = frozenset(Split),
) -> tuple[SplitQAExample, ...]:
    try:
        nib: Any = importlib.import_module("nibabel")
        np: Any = importlib.import_module("numpy")
    except ImportError as error:
        raise RuntimeError("install the project with the 'data' extra") from error

    generated: list[SplitQAExample] = []
    for case in discover_training_cases(root):
        split = assign_subject(case.case_id, seed=seed)
        if split not in include_splits:
            continue
        image: Any = nib.load(str(case.label))
        label = np.asanyarray(image.dataobj)
        spacing = image.header.get_zooms()[:3]
        voxel_ml = float(spacing[0] * spacing[1] * spacing[2]) / 1000.0
        whole = label > 0
        edema = label == 1
        core = (label == 2) | (label == 3)
        enhancing = label == 3
        volumes = {
            "whole": int(np.count_nonzero(whole)) * voxel_ml,
            "edema": int(np.count_nonzero(edema)) * voxel_ml,
            "core": int(np.count_nonzero(core)) * voxel_ml,
            "enhancing": int(np.count_nonzero(enhancing)) * voxel_ml,
        }
        x_midpoint = label.shape[0] // 2
        left_ml = int(np.count_nonzero(whole[:x_midpoint])) * voxel_ml
        right_ml = int(np.count_nonzero(whole[x_midpoint:])) * voxel_ml
        laterality = _laterality(left_ml, right_ml)
        definitions = (
            (
                QuestionType.PRESENCE,
                AnswerKind.CATEGORICAL,
                "Is an enhancing tumor component present?",
                "yes" if volumes["enhancing"] >= ENHANCING_PRESENCE_ML else "no",
                enhancing,
            ),
            (
                QuestionType.LATERALITY,
                AnswerKind.CATEGORICAL,
                "Which hemisphere contains more whole-tumor volume?",
                laterality,
                whole,
            ),
            (
                QuestionType.RELATIVE_VOLUME,
                AnswerKind.CATEGORICAL,
                "Which is larger, edema or tumor core?",
                "edema" if volumes["edema"] >= volumes["core"] else "tumor core",
                whole,
            ),
            (
                QuestionType.ENHANCING_FRACTION,
                AnswerKind.NUMERIC,
                "What fraction of whole-tumor volume is enhancing?",
                f"{volumes['enhancing'] / volumes['whole']:.6f}",
                whole,
            ),
            (
                QuestionType.CROSS_REGION_COMPARISON,
                AnswerKind.CATEGORICAL,
                "Is tumor-core volume greater than half of whole-tumor volume?",
                "yes" if volumes["core"] > 0.5 * volumes["whole"] else "no",
                whole,
            ),
        )
        for question_type, answer_kind, question, answer, evidence in definitions:
            generated.append(
                SplitQAExample(
                    split=split,
                    example=GroundedQAExample(
                        example_id=f"{case.case_id}:{question_type.value}",
                        case_id=case.case_id,
                        subject_id=case.case_id,
                        question=question,
                        question_type=question_type,
                        answer_kind=answer_kind,
                        answer=answer,
                        evidence_sha256=_mask_digest(np, evidence),
                    ),
                )
            )
        generated.append(
            SplitQAExample(
                split=split,
                example=GroundedQAExample(
                    example_id=f"{case.case_id}:unanswerable",
                    case_id=case.case_id,
                    subject_id=case.case_id,
                    question="How much did tumor volume change from the prior scan?",
                    question_type=QuestionType.UNANSWERABLE,
                    answer_kind=AnswerKind.NUMERIC,
                    answer=None,
                    evidence_sha256=None,
                ),
            )
        )
    return tuple(generated)


def _laterality(left_ml: float, right_ml: float) -> str:
    total = left_ml + right_ml
    if total <= 0.0:
        raise ValueError("laterality requires a non-empty tumor")
    if abs(left_ml - right_ml) / total <= MIDLINE_RELATIVE_MARGIN:
        return "midline"
    return "left" if left_ml > right_ml else "right"


def _mask_digest(np: Any, mask: Any) -> str:
    contiguous = np.ascontiguousarray(mask, dtype=np.uint8)
    payload = str(tuple(mask.shape)).encode() + b":" + contiguous.tobytes()
    return hashlib.sha256(payload).hexdigest()
