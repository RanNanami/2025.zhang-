"""CSV output primitives for experiment diagnostics.

This module owns only row serialization and file-handle lifetime. It does not
inspect model state, choose candidates, mutate learning state, or interpret
diagnostic rows.
"""

from __future__ import annotations

import atexit
import csv
import gzip
from pathlib import Path


def write_diagnostic_csv(
    path: Path,
    rows: list[dict[str, object]],
    *,
    compress: bool = False,
) -> Path | None:
    """Write a diagnostic CSV or CSV.GZ without changing row contents."""

    if not rows:
        return None
    actual_path = (
        path.with_suffix(path.suffix + ".gz") if compress else path
    )
    actual_path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        handle_context = gzip.open(
            actual_path,
            "wt",
            encoding="utf-8",
            newline="",
        )
    else:
        handle_context = actual_path.open(
            "w",
            encoding="utf-8",
            newline="",
        )
    with handle_context as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return actual_path


class _PendingCsvAppendSink:
    """Keep compressed trace rows durable without streaming zlib calls."""

    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        existed = path.exists() and path.stat().st_size > 0
        self._pending = path.with_suffix(path.suffix + ".pending")
        self._handle = self._pending.open(
            "a",
            encoding="utf-8",
            newline="",
            buffering=1,
        )
        self._writer = csv.DictWriter(self._handle, fieldnames=fieldnames)
        if not existed and self._pending.stat().st_size == 0:
            self._writer.writeheader()
        self.flush()

    def write(self, rows: list[dict[str, object]]) -> None:
        self._writer.writerows(rows)
        self.flush()

    def flush(self) -> None:
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def commit(self, path: Path) -> None:
        """Compress pending rows once after the model call path ends."""

        self.close()
        if not self._pending.exists() or self._pending.stat().st_size == 0:
            return
        final_exists = path.exists() and path.stat().st_size > 0
        mode = "at" if final_exists else "wt"
        with self._pending.open("r", encoding="utf-8", newline="") as source:
            with gzip.open(path, mode, encoding="utf-8", newline="") as target:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)
        self._pending.unlink()


_GZIP_CSV_APPEND_SINKS: dict[Path, _PendingCsvAppendSink] = {}


def close_diagnostic_writers() -> None:
    """Commit process-local gzip sinks before summaries or process exit."""

    items = list(_GZIP_CSV_APPEND_SINKS.items())
    _GZIP_CSV_APPEND_SINKS.clear()
    for path, sink in items:
        sink.commit(path)


atexit.register(close_diagnostic_writers)


def append_diagnostic_csv(
    path: Path,
    rows: list[dict[str, object]],
    *,
    compress: bool = False,
) -> Path | None:
    """Append a homogeneous trace batch and flush it out of process memory."""

    if not rows:
        return None
    actual_path = path.with_suffix(path.suffix + ".gz") if compress else path
    if compress:
        actual_path.parent.mkdir(parents=True, exist_ok=True)
        key = actual_path.resolve()
        sink = _GZIP_CSV_APPEND_SINKS.get(key)
        if sink is None:
            sink = _PendingCsvAppendSink(key, list(rows[0]))
            _GZIP_CSV_APPEND_SINKS[key] = sink
        sink.write(rows)
        return actual_path
    actual_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not actual_path.exists() or actual_path.stat().st_size == 0
    with actual_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()
    return actual_path
