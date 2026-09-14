import json
from pathlib import Path

import pytest

from mri_vlm.data.msd import (
    NiftiMetadata,
    build_case_records,
    discover_training_cases,
    load_modality_mapping,
)
from mri_vlm.schema import Modality


class FakeReader:
    def __init__(self, *, label_affine: str = "a" * 64) -> None:
        self.label_affine = label_affine

    def read(self, path: Path, *, read_labels: bool) -> NiftiMetadata:
        if read_labels:
            return NiftiMetadata(
                shape=(8, 9, 10),
                spacing_mm=(1.0, 1.0, 1.5),
                affine_sha256=self.label_affine,
                finite=True,
                label_values=(0, 1, 2, 3),
            )
        return NiftiMetadata(
            shape=(8, 9, 10, 4),
            spacing_mm=(1.0, 1.0, 1.5, 1.0),
            affine_sha256="a" * 64,
            finite=True,
            label_values=None,
        )


def make_dataset(root: Path) -> None:
    (root / "imagesTr").mkdir()
    (root / "labelsTr").mkdir()
    (root / "imagesTr" / "BRATS_001.nii.gz").write_bytes(b"image fixture")
    (root / "labelsTr" / "BRATS_001.nii.gz").write_bytes(b"label fixture")
    (root / "dataset.json").write_text(
        json.dumps({"modality": {"0": "FLAIR", "1": "T1w", "2": "T1gd", "3": "T2w"}}),
        encoding="utf-8",
    )


def test_build_case_record_from_4d_msd_layout(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    records = build_case_records(tmp_path, FakeReader())
    assert len(records) == 1
    record = records[0]
    assert record.subject_id == "BRATS_001"
    assert tuple(volume.modality for volume in record.volumes) == tuple(Modality)
    assert all(volume.shape == (8, 9, 10) for volume in record.volumes)
    assert len({volume.sha256 for volume in record.volumes}) == 4


def test_discovery_rejects_unpaired_image(tmp_path: Path) -> None:
    (tmp_path / "imagesTr").mkdir()
    (tmp_path / "labelsTr").mkdir()
    (tmp_path / "imagesTr" / "BRATS_001.nii.gz").write_bytes(b"image")
    with pytest.raises(ValueError, match="missing labels"):
        discover_training_cases(tmp_path)


def test_modality_mapping_must_have_four_known_channels(tmp_path: Path) -> None:
    (tmp_path / "dataset.json").write_text(
        json.dumps({"modality": {"0": "FLAIR", "1": "T1w"}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="exactly four"):
        load_modality_mapping(tmp_path)


def test_image_and_label_affines_must_match(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    with pytest.raises(ValueError, match="affine"):
        build_case_records(tmp_path, FakeReader(label_affine="b" * 64))
