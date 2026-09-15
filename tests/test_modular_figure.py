import pytest

from mri_vlm.modular_figure_cli import select_failure_profiles


def test_failure_selection_is_extreme_and_deterministic() -> None:
    profiles = {f"case-{index}": float(index) for index in range(8)}
    assert select_failure_profiles(profiles, count=2) == (
        "case-0",
        "case-1",
        "case-6",
        "case-7",
    )
    with pytest.raises(ValueError, match="not enough"):
        select_failure_profiles({"a": 0.0}, count=1)
