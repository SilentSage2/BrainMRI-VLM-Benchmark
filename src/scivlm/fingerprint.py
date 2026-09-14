"""Canonical content fingerprints for declarative figure specifications."""

import hashlib
import json


def canonical_json(value: object) -> bytes:
    """Serialize JSON-compatible content deterministically."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    """Return the SHA-256 fingerprint of canonical JSON content."""
    return hashlib.sha256(canonical_json(value)).hexdigest()
