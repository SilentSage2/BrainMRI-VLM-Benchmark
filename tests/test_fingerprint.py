import pytest

from scivlm.fingerprint import canonical_json, sha256_json


def test_fingerprint_is_independent_of_mapping_order() -> None:
    left = {"series": [1, 2, 3], "label": "control"}
    right = {"label": "control", "series": [1, 2, 3]}
    assert canonical_json(left) == canonical_json(right)
    assert sha256_json(left) == sha256_json(right)


def test_non_finite_numbers_are_rejected() -> None:
    with pytest.raises(ValueError):
        sha256_json({"value": float("nan")})
