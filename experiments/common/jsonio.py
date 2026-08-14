"""JSON writers that preserve the strict runner's existing byte format."""

from __future__ import annotations

import json
import os
from pathlib import Path


def write_json(path: Path, payload: object) -> None:
    """Write sorted, indented UTF-8 JSON without a trailing newline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_json_atomic(path: Path, payload: object) -> None:
    """Write the same JSON bytes through a sibling temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
