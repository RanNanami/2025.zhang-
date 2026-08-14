"""Deterministic hashing helpers with no model or protocol knowledge."""

from __future__ import annotations

import hashlib
from pathlib import Path


_READ_CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file read in 1 MiB chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_READ_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()
