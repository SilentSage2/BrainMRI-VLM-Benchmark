import json
from pathlib import Path

import pytest

from mri_vlm.matched_figure_cli import load_test_sealed


def test_matched_figure_refuses_test_read(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"test_cases_read": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="not test-sealed"):
        load_test_sealed(path)
