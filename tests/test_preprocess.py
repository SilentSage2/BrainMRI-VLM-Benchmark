from typing import Any

import pytest
import torch

from mri_vlm.preprocess import (
    PreprocessSpec,
    _validate_payload,
    crop_and_pad_foreground,
    tensor_sha256,
)


def test_tensor_digest_covers_shape_dtype_and_values() -> None:
    tensor = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)
    assert tensor_sha256(tensor) == tensor_sha256(tensor.clone())
    assert tensor_sha256(tensor) != tensor_sha256(tensor + 1)
    assert tensor_sha256(tensor) != tensor_sha256(tensor.double())


def test_preprocess_spec_rejects_odd_spatial_size() -> None:
    with pytest.raises(ValueError, match="even"):
        PreprocessSpec(spatial_size=15)


def test_cache_validation_detects_tensor_change() -> None:
    volumes = torch.zeros(4, 8, 8, 8)
    label = torch.zeros(8, 8, 8, dtype=torch.long)
    payload: dict[str, Any] = {
        "case_id": "case",
        "cache_key": "key",
        "volumes": volumes,
        "label": label,
        "volume_sha256": tensor_sha256(volumes),
        "label_sha256": tensor_sha256(label),
    }
    payload["volumes"][0, 0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="fingerprint"):
        _validate_payload(payload, case_id="case", cache_key="key", cache_hit=True)


def test_cache_payload_round_trip_identity() -> None:
    volumes = torch.randn(4, 8, 8, 8)
    label = torch.randint(0, 4, (8, 8, 8))
    payload: dict[str, Any] = {
        "case_id": "case",
        "cache_key": "key",
        "volumes": volumes,
        "label": label,
        "volume_sha256": tensor_sha256(volumes),
        "label_sha256": tensor_sha256(label),
    }
    result = _validate_payload(payload, case_id="case", cache_key="key", cache_hit=True)
    assert result.cache_hit
    assert result.volume_sha256 == tensor_sha256(volumes)
    assert result.label_sha256 == tensor_sha256(label)


def test_foreground_crop_uses_image_and_pads_to_cube() -> None:
    volumes = torch.zeros(4, 8, 10, 12)
    volumes[:, 2:6, 3:8, 4:10] = 1.0
    label = torch.zeros(1, 8, 10, 12)
    label[:, 3:5, 4:7, 5:9] = 2
    cropped, cropped_label = crop_and_pad_foreground(volumes, label)
    assert cropped.shape == (4, 6, 6, 6)
    assert cropped_label.shape == (1, 6, 6, 6)
    assert int((cropped_label == 2).sum()) == 24


def test_foreground_crop_rejects_label_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="agree"):
        crop_and_pad_foreground(torch.zeros(4, 8, 8, 8), torch.zeros(1, 7, 8, 8))
