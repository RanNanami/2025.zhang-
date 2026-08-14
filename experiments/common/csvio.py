"""Generic CSV row output with explicit legacy-compatible empty handling."""

from __future__ import annotations

import csv
from pathlib import Path


def write_csv_rows(
    path: Path,
    rows: list[dict[str, object]],
    *,
    skip_empty: bool = False,
) -> None:
    """Write union-ordered dictionary fields using Python's CSV defaults.

    ``skip_empty`` preserves the two empty-input behaviors used by the
    migrated callers. It changes no row, field, quoting, or newline semantics.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if skip_empty and not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
