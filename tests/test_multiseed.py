import pytest

from mri_vlm.multiseed_cli import hierarchical_bootstrap_ci, validate_seed_result


def test_hierarchical_bootstrap_is_deterministic() -> None:
    values = [[0.0, 0.5, 1.0], [-0.5, 0.0, 0.5], [0.25, 0.25, 0.25]]
    first = hierarchical_bootstrap_ci(values, seed=7, samples=200)
    second = hierarchical_bootstrap_ci(values, seed=7, samples=200)
    assert first == second
    assert first[0] <= 0.25 <= first[1]


def test_seed_result_refuses_test_read() -> None:
    payload: dict[str, object] = {
        "test_cases_read": 1,
        "seed": 3,
        "config": {"split_seed": 2},
    }
    with pytest.raises(ValueError, match="test cases"):
        validate_seed_result(payload, seed=3, split_seed=2)
