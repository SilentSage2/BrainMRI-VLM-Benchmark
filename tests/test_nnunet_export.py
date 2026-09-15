import json
from pathlib import Path

import pytest

from mri_vlm.nnunet_export import (
    build_phase2_plan,
    nnunet_dataset_json,
    nnunet_split_json,
)
from mri_vlm.schema import Split
from mri_vlm.split import assign_subject


def _make_dataset(root: Path, *, subjects: list[str]) -> None:
    (root / "imagesTr").mkdir()
    (root / "labelsTr").mkdir()
    for subject in subjects:
        (root / "imagesTr" / f"{subject}.nii.gz").write_bytes(b"image")
        (root / "labelsTr" / f"{subject}.nii.gz").write_bytes(b"label")
    (root / "dataset.json").write_text(
        json.dumps({"modality": {"0": "FLAIR", "1": "T1w", "2": "T1gd", "3": "T2w"}}),
        encoding="utf-8",
    )


def _write_manifest(path: Path, *, subjects: list[str], seed: int) -> None:
    path.write_text(
        json.dumps(
            {
                "seed": seed,
                "split": "test",
                "subject_count": len(subjects),
                "subjects": subjects,
            }
        ),
        encoding="utf-8",
    )


def test_phase2_plan_excludes_retired_subjects_and_preserves_split(tmp_path: Path) -> None:
    seed = 17
    subjects = [f"case_{index:03d}" for index in range(40)]
    retired = [
        subject for subject in subjects if assign_subject(subject, seed=seed) is Split.TEST
    ]
    _make_dataset(tmp_path, subjects=subjects)
    manifest = tmp_path / "retired.json"
    _write_manifest(manifest, subjects=retired, seed=seed)

    plan = build_phase2_plan(tmp_path, manifest, split_seed=seed)

    assert not set(plan.development_subjects) & set(retired)
    assert set(plan.retired_subjects) == set(retired)
    assert len(plan.cases) == len(plan.development_subjects)
    assert nnunet_dataset_json(plan)["numTraining"] == len(plan.development_subjects)
    split = nnunet_split_json(plan)[0]
    assert split["train"] == list(plan.training_subjects)
    assert split["val"] == list(plan.validation_subjects)


def test_phase2_plan_rejects_incomplete_retired_manifest(tmp_path: Path) -> None:
    seed = 23
    subjects = [f"case_{index:03d}" for index in range(30)]
    retired = [
        subject for subject in subjects if assign_subject(subject, seed=seed) is Split.TEST
    ]
    _make_dataset(tmp_path, subjects=subjects)
    manifest = tmp_path / "retired.json"
    _write_manifest(manifest, subjects=retired[:-1], seed=seed)

    with pytest.raises(ValueError, match="exactly match"):
        build_phase2_plan(tmp_path, manifest, split_seed=seed)


def test_region_order_matches_hierarchical_brats_labels(tmp_path: Path) -> None:
    seed = 31
    subjects = [f"case_{index:03d}" for index in range(30)]
    retired = [
        subject for subject in subjects if assign_subject(subject, seed=seed) is Split.TEST
    ]
    _make_dataset(tmp_path, subjects=subjects)
    manifest = tmp_path / "retired.json"
    _write_manifest(manifest, subjects=retired, seed=seed)
    payload = nnunet_dataset_json(build_phase2_plan(tmp_path, manifest, split_seed=seed))

    assert payload["labels"] == {
        "background": 0,
        "whole_tumor": [1, 2, 3],
        "tumor_core": [2, 3],
        "enhancing_tumor": [3],
    }
    assert payload["regions_class_order"] == [1, 2, 3]
