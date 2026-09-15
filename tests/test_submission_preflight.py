import pytest

from mri_vlm.submission_preflight_cli import validate_manifest


def test_test_manifest_contract() -> None:
    valid = {"subjects": ["a", "b"], "subject_count": 2, "split": "test", "seed": 20260914}
    validate_manifest(valid)
    with pytest.raises(ValueError, match="count or uniqueness"):
        invalid = {
            "subjects": ["a", "a"],
            "subject_count": 2,
            "split": "test",
            "seed": 20260914,
        }
        validate_manifest(invalid)
