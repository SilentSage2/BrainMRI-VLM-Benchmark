"""Versioned real-MRI preprocessing cache with tensor-level verification."""

import hashlib
import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as functional
from torch import Tensor


@dataclass(frozen=True, slots=True)
class PreprocessSpec:
    spatial_size: int = 24
    clip_standard_deviations: float = 5.0
    version: str = "whole-volume-zscore-resample-v1"
    preserve_background_zero: bool = False
    crop_to_foreground: bool = False

    def __post_init__(self) -> None:
        if self.spatial_size <= 0 or self.spatial_size % 2:
            raise ValueError("spatial_size must be a positive even integer")
        if self.clip_standard_deviations <= 0.0 or not self.version.strip():
            raise ValueError("clip and version must be positive/non-empty")


@dataclass(frozen=True, slots=True)
class PreprocessedCase:
    case_id: str
    volumes: Tensor
    label: Tensor
    cache_key: str
    volume_sha256: str
    label_sha256: str
    cache_hit: bool


def load_preprocessed_case(
    dataset_root: Path,
    cache_root: Path,
    case_id: str,
    spec: PreprocessSpec,
) -> PreprocessedCase:
    image_path = dataset_root / "imagesTr" / f"{case_id}.nii.gz"
    label_path = dataset_root / "labelsTr" / f"{case_id}.nii.gz"
    cache_key = _cache_key(image_path, label_path, spec)
    cache_path = cache_root / spec.version / f"{case_id}-{cache_key[:16]}.pt"
    if cache_path.is_file():
        loaded_payload = cast(
            dict[str, Any],
            torch.load(cache_path, map_location="cpu", weights_only=True),
        )
        return _validate_payload(
            loaded_payload, case_id=case_id, cache_key=cache_key, cache_hit=True
        )

    volumes, label = preprocess_nifti(image_path, label_path, spec)
    payload: dict[str, object] = {
        "case_id": case_id,
        "cache_key": cache_key,
        "spec": asdict(spec),
        "volumes": volumes,
        "label": label,
        "volume_sha256": tensor_sha256(volumes),
        "label_sha256": tensor_sha256(label),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(cache_path)
    return _validate_payload(payload, case_id=case_id, cache_key=cache_key, cache_hit=False)


def preprocess_nifti(
    image_path: Path, label_path: Path, spec: PreprocessSpec
) -> tuple[Tensor, Tensor]:
    try:
        nib: Any = importlib.import_module("nibabel")
        np: Any = importlib.import_module("numpy")
    except ImportError as error:
        raise RuntimeError("install the project with the 'data' extra") from error

    image = np.asanyarray(nib.load(str(image_path)).dataobj, dtype=np.float32)
    label = np.asanyarray(nib.load(str(label_path)).dataobj, dtype=np.int64)
    channels = []
    for index in range(4):
        channel = image[..., index]
        foreground_mask = channel != 0
        foreground = channel[foreground_mask]
        if foreground.size == 0:
            raise ValueError(f"empty MRI contrast in {image_path}")
        mean = float(foreground.mean())
        deviation = max(float(foreground.std()), 1e-6)
        normalized = np.clip(
            (channel - mean) / deviation,
            -spec.clip_standard_deviations,
            spec.clip_standard_deviations,
        )
        if spec.preserve_background_zero:
            normalized = np.where(foreground_mask, normalized, 0.0)
        channels.append(normalized)
    volumes = torch.from_numpy(np.stack(channels)).permute(0, 3, 2, 1).unsqueeze(0)
    labels = torch.from_numpy(label).permute(2, 1, 0)[None, None].float()
    if spec.crop_to_foreground:
        volumes, labels = crop_and_pad_foreground(volumes.squeeze(0), labels.squeeze(0))
        volumes = volumes.unsqueeze(0)
        labels = labels.unsqueeze(0)
    volumes = functional.interpolate(
        volumes,
        size=(spec.spatial_size,) * 3,
        mode="trilinear",
        align_corners=False,
    ).squeeze(0)
    labels = functional.interpolate(
        labels,
        size=(spec.spatial_size,) * 3,
        mode="nearest",
    ).squeeze(0).squeeze(0).long()
    return volumes.contiguous(), labels.contiguous()


def crop_and_pad_foreground(volumes: Tensor, label: Tensor) -> tuple[Tensor, Tensor]:
    """Crop by MRI support only, then symmetrically pad to a cube."""
    if volumes.ndim != 4 or label.ndim != 4 or label.shape[0] != 1:
        raise ValueError("expected volumes [C,D,H,W] and label [1,D,H,W]")
    if label.shape[1:] != volumes.shape[1:]:
        raise ValueError("volume and label shapes must agree")
    foreground = volumes.ne(0).any(dim=0)
    coordinates = torch.nonzero(foreground, as_tuple=False)
    if coordinates.numel() == 0:
        raise ValueError("MRI foreground is empty")
    lower = coordinates.min(dim=0).values
    upper = coordinates.max(dim=0).values + 1
    slices = tuple(slice(int(lower[i]), int(upper[i])) for i in range(3))
    cropped_volumes = volumes[(slice(None), *slices)]
    cropped_label = label[(slice(None), *slices)]
    target = max(cropped_volumes.shape[1:])
    pads: list[int] = []
    for dimension in reversed(cropped_volumes.shape[1:]):
        difference = target - dimension
        pads.extend((difference // 2, difference - difference // 2))
    return functional.pad(cropped_volumes, pads), functional.pad(cropped_label, pads)


def tensor_sha256(tensor: Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    header = f"{contiguous.dtype}:{tuple(contiguous.shape)}:".encode()
    return hashlib.sha256(header + contiguous.numpy().tobytes()).hexdigest()


def _cache_key(image_path: Path, label_path: Path, spec: PreprocessSpec) -> str:
    sources = []
    for path in (image_path, label_path):
        stat = path.stat()
        sources.append({"name": path.name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    payload = json.dumps(
        {"spec": asdict(spec), "sources": sources}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _validate_payload(
    payload: dict[str, Any], *, case_id: str, cache_key: str, cache_hit: bool
) -> PreprocessedCase:
    if payload.get("case_id") != case_id or payload.get("cache_key") != cache_key:
        raise ValueError("cache identity mismatch")
    volumes = payload.get("volumes")
    label = payload.get("label")
    if not isinstance(volumes, Tensor) or not isinstance(label, Tensor):
        raise ValueError("cache tensors are missing")
    if volumes.ndim != 4 or volumes.shape[0] != 4 or label.shape != volumes.shape[1:]:
        raise ValueError("cache tensor shapes are invalid")
    volume_digest = tensor_sha256(volumes)
    label_digest = tensor_sha256(label)
    if volume_digest != payload.get("volume_sha256") or label_digest != payload.get("label_sha256"):
        raise ValueError("cache tensor fingerprint mismatch")
    return PreprocessedCase(
        case_id=case_id,
        volumes=volumes,
        label=label,
        cache_key=cache_key,
        volume_sha256=volume_digest,
        label_sha256=label_digest,
        cache_hit=cache_hit,
    )
