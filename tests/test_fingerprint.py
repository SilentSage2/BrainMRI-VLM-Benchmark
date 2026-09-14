import pytest

from mri_vlm.fingerprint import canonical_json, sha256_json


def test_fingerprint_ignores_mapping_order() -> None:
    left = {"modalities": ["flair", "t1"], "subject": "fixture"}
    right = {"subject": "fixture", "modalities": ["flair", "t1"]}
    assert canonical_json(left) == canonical_json(right)
    assert sha256_json(left) == sha256_json(right)


def test_non_finite_number_is_rejected() -> None:
    with pytest.raises(ValueError):
        sha256_json({"spacing": float("nan")})
