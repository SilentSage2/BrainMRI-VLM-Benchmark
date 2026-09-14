"""Audited adapter for Medical Segmentation Decathlon Task01_BrainTumour."""

import hashlib
import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from mri_vlm.schema import CaseRecord, Modality, VolumeRecord


@dataclass(frozen=True, slots=True)
class NiftiMetadata:
    shape: tuple[int, ...]
    spacing_mm: tuple[float, ...]
    affine_sha256: str
    finite: bool
    label_values: tuple[int, ...] | None


class MetadataReader(Protocol):
    def read(self, path: Path, *, read_labels: bool) -> NiftiMetadata: ...


@dataclass(frozen=True, slots=True)
class RawCasePaths:
    case_id: str
    image: Path
    label: Path


class NibabelMetadataReader:
    """Load full arrays for an integrity audit; requires the `data` extra."""

    def read(self, path: Path, *, read_labels: bool) -> NiftiMetadata:
        try:
            nib = importlib.import_module("nibabel")
            np = importlib.import_module("numpy")
        except ImportError as error:
            raise RuntimeError("install the project with the 'data' extra") from error

        image: Any = nib.load(str(path))
        array: Any = np.asanyarray(image.dataobj)
        finite = bool(np.isfinite(array).all())
        affine = np.asarray(image.affine, dtype="<f8")
        affine_sha256 = hashlib.sha256(affine.tobytes()).hexdigest()
        label_values: tuple[int, ...] | None = None
        if read_labels:
            unique = np.unique(array)
            if not bool(np.equal(unique, np.floor(unique)).all()):
                raise ValueError(f"label volume contains non-integer values: {path}")
            label_values = tuple(int(value) for value in unique.tolist())
        return NiftiMetadata(
            shape=tuple(int(value) for value in image.shape),
            spacing_mm=tuple(float(value) for value in image.header.get_zooms()),
            affine_sha256=affine_sha256,
            finite=finite,
            label_values=label_values,
        )


def discover_training_cases(root: Path) -> tuple[RawCasePaths, ...]:
    images = root / "imagesTr"
    labels = root / "labelsTr"
    if not images.is_dir() or not labels.is_dir():
        raise ValueError("dataset root must contain imagesTr and labelsTr directories")

    cases: list[RawCasePaths] = []
    label_names = {path.name for path in labels.glob("*.nii.gz")}
    image_names = {path.name for path in images.glob("*.nii.gz")}
    if image_names != label_names:
        missing_labels = sorted(image_names - label_names)
        missing_images = sorted(label_names - image_names)
        raise ValueError(
            "image/label mismatch; "
            f"missing labels={missing_labels}, missing images={missing_images}"
        )
    for name in sorted(image_names):
        cases.append(
            RawCasePaths(
                case_id=_strip_nifti_suffix(name),
                image=images / name,
                label=labels / name,
            )
        )
    if not cases:
        raise ValueError("no training NIfTI cases found")
    return tuple(cases)


def load_modality_mapping(root: Path) -> tuple[Modality, ...]:
    description_path = root / "dataset.json"
    try:
        raw = cast(dict[str, Any], json.loads(description_path.read_text(encoding="utf-8")))
    except FileNotFoundError as error:
        raise ValueError("dataset.json is required") from error
    mapping_raw = raw.get("modality") or raw.get("channel_names")
    if not isinstance(mapping_raw, Mapping):
        raise ValueError("dataset.json must define modality or channel_names")
    mapping = cast(Mapping[str, object], mapping_raw)
    if set(mapping) != {"0", "1", "2", "3"}:
        raise ValueError("dataset must define exactly four indexed MRI channels")
    return tuple(_parse_modality(mapping[str(index)]) for index in range(4))


def build_case_records(root: Path, reader: MetadataReader) -> tuple[CaseRecord, ...]:
    modalities = load_modality_mapping(root)
    records: list[CaseRecord] = []
    for raw_case in discover_training_cases(root):
        image = reader.read(raw_case.image, read_labels=False)
        label = reader.read(raw_case.label, read_labels=True)
        _validate_pair(raw_case, image, label)
        image_sha256 = sha256_file(raw_case.image)
        records.append(
            CaseRecord(
                case_id=raw_case.case_id,
                subject_id=raw_case.case_id,
                dataset_id="msd-task01-brain-tumour",
                dataset_version=_dataset_version(root),
                volumes=tuple(
                    VolumeRecord(
                        modality=modality,
                        sha256=hashlib.sha256(
                            f"{image_sha256}:channel:{index}".encode()
                        ).hexdigest(),
                        shape=cast(tuple[int, int, int], image.shape[:3]),
                        spacing_mm=cast(tuple[float, float, float], image.spacing_mm[:3]),
                    )
                    for index, modality in enumerate(modalities)
                ),
                label_sha256=sha256_file(raw_case.label),
                label_values=label.label_values or (),
            )
        )
    return tuple(records)


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_pair(
    raw_case: RawCasePaths, image: NiftiMetadata, label: NiftiMetadata
) -> None:
    if not image.finite or not label.finite:
        raise ValueError(f"case {raw_case.case_id} contains non-finite values")
    if len(image.shape) != 4 or image.shape[3] != 4:
        raise ValueError(f"case {raw_case.case_id} image must have four channels")
    if len(label.shape) != 3 or label.shape != image.shape[:3]:
        raise ValueError(f"case {raw_case.case_id} image/label shapes do not agree")
    if image.spacing_mm[:3] != label.spacing_mm[:3]:
        raise ValueError(f"case {raw_case.case_id} image/label spacing does not agree")
    if image.affine_sha256 != label.affine_sha256:
        raise ValueError(f"case {raw_case.case_id} image/label affine does not agree")
    if label.label_values is None:
        raise ValueError(f"case {raw_case.case_id} label values were not read")
    if not set(label.label_values) <= {0, 1, 2, 3}:
        raise ValueError(f"case {raw_case.case_id} contains an unexpected label value")


def _parse_modality(value: object) -> Modality:
    normalized = str(value).lower().replace("-", "").replace("_", "")
    aliases = {
        "flair": Modality.FLAIR,
        "t1": Modality.T1,
        "t1w": Modality.T1,
        "t1gd": Modality.T1GD,
        "t1ce": Modality.T1GD,
        "t1weightedgd": Modality.T1GD,
        "t2": Modality.T2,
        "t2w": Modality.T2,
    }
    try:
        return aliases[normalized]
    except KeyError as error:
        raise ValueError(f"unsupported MRI modality: {value!r}") from error


def _dataset_version(root: Path) -> str:
    return sha256_file(root / "dataset.json")[:16]


def _strip_nifti_suffix(name: str) -> str:
    if not name.endswith(".nii.gz"):
        raise ValueError(f"not a compressed NIfTI filename: {name}")
    return name[: -len(".nii.gz")]
