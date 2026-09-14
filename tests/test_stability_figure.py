import json

import pytest

from mri_vlm.stability_figure_cli import load_summary


def test_figure_summary_rejects_test_read(tmp_path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "test_cases_read": 1,
                "resolutions": [32],
                "train_derived_ambiguity_screen": {"32": {}},
            }
        )
    )
    with pytest.raises(ValueError, match="development-only"):
        load_summary(path, screen_resolution=32)


def test_figure_summary_requires_screen_resolution(tmp_path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "test_cases_read": 0,
                "resolutions": [24],
                "train_derived_ambiguity_screen": {"24": {}},
            }
        )
    )
    with pytest.raises(ValueError, match="absent"):
        load_summary(path, screen_resolution=32)
