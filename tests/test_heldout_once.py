from pathlib import Path

import pytest

from mri_vlm.heldout_once_cli import AUTHORIZATION_TEXT, initialize_lock, validate_authorization


def test_heldout_requires_exact_authorization(tmp_path: Path) -> None:
    path = tmp_path / "authorization.txt"
    path.write_text("no", encoding="utf-8")
    with pytest.raises(PermissionError, match="exact"):
        validate_authorization(path)
    path.write_text(AUTHORIZATION_TEXT, encoding="utf-8")
    validate_authorization(path)


def test_one_shot_lock_rejects_completed_run(tmp_path: Path) -> None:
    output = tmp_path / "heldout_v1"
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    config = {"protocol": "test"}
    lock = initialize_lock(output, config, manifest)
    lock.write_text('{"status":"complete","protocol":"test"}', encoding="utf-8")
    with pytest.raises(FileExistsError, match="already complete"):
        initialize_lock(output, config, manifest)
