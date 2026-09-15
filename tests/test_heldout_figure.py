import json
from pathlib import Path

import pytest

from mri_vlm.heldout_figure_cli import EXPECTED_SCHEMA, load_heldout_summary


def test_heldout_figure_requires_complete_frozen_schema(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"status": "development"}), encoding="utf-8")
    with pytest.raises(ValueError, match="completed one-shot"):
        load_heldout_summary(path)


def test_heldout_figure_accepts_complete_66_subject_summary(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    payload = {
        "status": "complete-one-shot-heldout-v1",
        "result_schema": EXPECTED_SCHEMA,
        "test_cases_read": 66,
        "subject_count": 66,
        "aggregate": {"test_cases_read": 66, "conditions": [f"c{index}" for index in range(15)]},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_heldout_summary(path) == payload
