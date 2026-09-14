import pytest

from mri_vlm.modular_multiseed_cli import validate_result


def test_modular_result_requires_sealed_test() -> None:
    payload = {
        "test_cases_read": 1,
        "seed": 3,
        "config": {"split_seed": 2, "modality_dropout": True},
    }
    with pytest.raises(ValueError, match="test cases"):
        validate_result(payload, seed=3, split_seed=2, mode="dropout")
