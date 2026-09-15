"""Audit the frozen submission package without opening held-out images or labels."""

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any, cast

FORBIDDEN_SUFFIXES = (".nii", ".nii.gz", ".pt", ".pth", ".ckpt")
SECRET_PATTERNS = (
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ghp_[A-Za-z0-9]{30,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MRI-VLM submission preflight")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--results-root", type=Path, default=Path("artifacts/results"))
    parser.add_argument("--manifest", type=Path, default=Path("configs/test_subject_manifest.json"))
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo = args.repo_root.resolve()
    manifest = cast(dict[str, Any], json.loads((repo / args.manifest).read_text(encoding="utf-8")))
    validate_manifest(manifest)
    result_files = sorted((repo / args.results_root).rglob("*.json"))
    observed: list[dict[str, object]] = []
    for path in result_files:
        payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        if "test_cases_read" not in payload:
            continue
        reads = payload["test_cases_read"]
        if reads != 0:
            raise ValueError(f"nonzero test access recorded before unsealing: {path}")
        observed.append({"path": str(path.relative_to(repo)), "test_cases_read": reads})
    tracked = tracked_files(repo)
    forbidden = [
        path
        for path in tracked
        if path.startswith(("data/", "artifacts/", "weights/"))
        or path.endswith(FORBIDDEN_SUFFIXES)
    ]
    if forbidden:
        raise ValueError(f"forbidden data/weight artifacts are tracked: {forbidden}")
    secret_hits = scan_secrets(repo, tracked)
    if secret_hits:
        raise ValueError(f"possible secrets in tracked files: {secret_hits}")
    heldout_path = repo / "artifacts/results/heldout_v1"
    if heldout_path.exists():
        raise ValueError("heldout_v1 output already exists; test may have been unsealed")
    report = {
        "status": "pass",
        "scope": "pre-unseal submission audit; filenames and development artifacts only",
        "test_reads_observed": 0,
        "test_result_manifests_checked": observed,
        "frozen_test_subject_count": manifest["subject_count"],
        "tracked_file_count": len(tracked),
        "forbidden_tracked_files": [],
        "secret_pattern_hits": [],
        "heldout_output_absent": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "output": str(args.output)}))


def validate_manifest(manifest: dict[str, Any]) -> None:
    subjects = manifest.get("subjects")
    if not isinstance(subjects, list) or not all(isinstance(item, str) for item in subjects):
        raise ValueError("test manifest subjects must be a list of identifiers")
    if manifest.get("subject_count") != len(subjects) or len(set(subjects)) != len(subjects):
        raise ValueError("test manifest count or uniqueness mismatch")
    if manifest.get("split") != "test" or manifest.get("seed") != 20260914:
        raise ValueError("test manifest split contract changed")


def tracked_files(repo: Path) -> list[str]:
    completed = subprocess.run(
        ("git", "ls-files", "-z"),
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return [item for item in completed.stdout.decode().split("\0") if item]


def scan_secrets(repo: Path, tracked: list[str]) -> list[str]:
    hits: list[str] = []
    for relative in tracked:
        path = repo / relative
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            hits.append(relative)
    return hits


if __name__ == "__main__":
    main()
