"""Deterministic metadata and QA fixtures; no medical images."""

import hashlib

from mri_vlm.schema import (
    AnswerKind,
    CaseRecord,
    GroundedQAExample,
    Modality,
    VolumeRecord,
)


def fixture_cases(count: int = 60) -> tuple[CaseRecord, ...]:
    if count < 3:
        raise ValueError("fixture requires at least three cases")
    cases: list[CaseRecord] = []
    for index in range(count):
        subject_id = f"subject-{index:04d}"
        cases.append(
            CaseRecord(
                case_id=f"case-{index:04d}",
                subject_id=subject_id,
                dataset_id="msd-task01-brain-tumour",
                dataset_version="v0-fixture-1",
                volumes=tuple(
                    VolumeRecord(
                        modality=modality,
                        sha256=_digest(f"{subject_id}:{modality.value}"),
                        shape=(32, 32, 24),
                        spacing_mm=(1.0, 1.0, 1.5),
                    )
                    for modality in Modality
                ),
                label_sha256=_digest(f"{subject_id}:label"),
                label_values=(0, 1, 2, 3),
            )
        )
    return tuple(cases)


def fixture_examples(cases: tuple[CaseRecord, ...]) -> tuple[GroundedQAExample, ...]:
    examples: list[GroundedQAExample] = []
    for index, case in enumerate(cases):
        answerable = index % 5 != 0
        examples.append(
            GroundedQAExample(
                example_id=f"example-{index:04d}",
                case_id=case.case_id,
                subject_id=case.subject_id,
                question=(
                    "Is an enhancing component represented in the reference mask?"
                    if answerable
                    else "What is the volume of the absent comparison scan?"
                ),
                question_type="presence" if answerable else "unanswerable",
                answer_kind=AnswerKind.CATEGORICAL,
                answer="yes" if answerable else None,
                evidence_sha256=(
                    _digest(f"{case.subject_id}:evidence") if answerable else None
                ),
            )
        )
    return tuple(examples)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()
