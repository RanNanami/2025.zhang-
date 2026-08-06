"""Prepare an isolated, trace-consistent resume directory.

This utility never imports the experiment module and never unpickles a model.
It copies the durable checkpoint, trims append-only CSV traces at the recorded
checkpoint boundary, and leaves the failed source directory untouched.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
from pathlib import Path


INDEX_FIELDS = ("actual_record_index", "record_index", "input_index")
KEY_FIELDS = (
    "actual_record_index",
    "record_index",
    "input_index",
    "horizon_step",
    "column",
    "target_column",
    "neuron_index",
    "segment_id",
    "segment_identity",
    "event_id",
    "source_cell_id",
)
SKIP_NAMES = {
    "native_crash_faulthandler.log",
    "native_environment.json",
    "progress.jsonl",
    "result.json",
    "failure.json",
    "run.log",
    "stdout.log",
    "stderr.log",
    "combined.log",
    "command.json",
    "checkpoint.metadata.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_metadata_path(path: Path) -> Path:
    return path.with_name(path.stem + ".metadata.json")


def read_metadata(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("atomic_complete") is not True:
        raise ValueError("checkpoint metadata is not atomically complete")
    checkpoint = Path(str(payload["checkpoint_path"]))
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    expected = str(payload["checkpoint_sha256"])
    actual = sha256(checkpoint)
    if actual != expected:
        raise ValueError(
            f"checkpoint hash mismatch: expected {expected}, got {actual}"
        )
    return payload


def open_text(path: Path, mode: str):
    if path.name.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def row_index(row: dict[str, str]) -> int | None:
    for field in INDEX_FIELDS:
        value = row.get(field, "")
        if value not in {"", None}:
            try:
                index = int(float(value))
            except ValueError:
                continue
            # input_index is one-based in Fig.9 predictions.
            return index - 1 if field == "input_index" else index
    return None


def semantic_key(row: dict[str, str]) -> tuple[tuple[str, str], ...] | None:
    values = tuple(
        (field, row.get(field, ""))
        for field in KEY_FIELDS
        if row.get(field, "") not in {"", None}
    )
    return values or None


def trim_csv(path: Path, boundary: int) -> dict[str, int]:
    kept: list[dict[str, str]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    rows_read = 0
    with open_text(path, "rt") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return {"rows_read": 0, "rows_kept": 0, "duplicates_removed": 0}
        fieldnames = list(reader.fieldnames)
        for row in reader:
            rows_read += 1
            index = row_index(row)
            if index is not None and index >= boundary:
                continue
            key = semantic_key(row)
            if key is not None and key in seen:
                continue
            if key is not None:
                seen.add(key)
            kept.append(row)
    temporary = path.with_name(
        path.stem + ".resume.tmp" + (".gz" if path.name.endswith(".gz") else "")
    )
    with open_text(temporary, "wt") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    temporary.replace(path)
    # Opening the complete rewritten stream verifies gzip's end marker.
    with open_text(path, "rt") as handle:
        for _ in handle:
            pass
    return {
        "rows_read": rows_read,
        "rows_kept": len(kept),
        "duplicates_removed": rows_read - len(kept),
    }


def prepare(source: Path, target: Path, metadata_path: Path) -> dict[str, object]:
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"target must be new or empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    metadata = read_metadata(metadata_path)
    boundary = int(metadata["next_index"])
    source_checkpoint = Path(str(metadata["checkpoint_path"]))
    checkpoint_name = source_checkpoint.name
    copied: list[str] = []
    trimmed: list[dict[str, object]] = []

    for source_path in source.rglob("*"):
        if not source_path.is_file():
            continue
        relative = source_path.relative_to(source)
        if relative.parts and relative.parts[0] in {"analysis", "dumps"}:
            continue
        if source_path.name in SKIP_NAMES or source_path.name.endswith(".tmp"):
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        copied.append(str(relative))

    checkpoint_destination = target / checkpoint_name
    if source_checkpoint.resolve() != checkpoint_destination.resolve():
        shutil.copy2(source_checkpoint, checkpoint_destination)
    sidecar = source_checkpoint.with_name(
        source_checkpoint.name + ".branch_provenance.json"
    )
    if sidecar.exists():
        shutil.copy2(sidecar, target / sidecar.name)
    for trace in sorted(target.rglob("*.csv")) + sorted(target.rglob("*.csv.gz")):
        stats = trim_csv(trace, boundary)
        stats["path"] = str(trace.relative_to(target))
        trimmed.append(stats)

    resume_metadata = dict(metadata)
    resume_metadata["checkpoint_path"] = str(checkpoint_destination.resolve())
    resume_metadata["checkpoint_sha256"] = sha256(checkpoint_destination)
    resume_metadata["source_run_directory"] = str(source.resolve())
    resume_metadata["resume_boundary"] = boundary
    (target / "checkpoint.metadata.json").write_text(
        json.dumps(resume_metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    (target / "progress.jsonl").write_text(
        json.dumps(
            {
                "current_index": boundary,
                "checkpoint_boundary": True,
                "source": "checkpoint.metadata.json",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "source_run_directory": str(source.resolve()),
        "target_run_directory": str(target.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "checkpoint_path": str(checkpoint_destination.resolve()),
        "checkpoint_sha256": sha256(checkpoint_destination),
        "next_index": boundary,
        "copied_files": copied,
        "trimmed_traces": trimmed,
        "original_preserved": True,
    }
    (target / "resume_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--target-run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-metadata", type=Path)
    args = parser.parse_args()
    source = args.source_run_dir.resolve()
    metadata = (
        args.checkpoint_metadata.resolve()
        if args.checkpoint_metadata
        else checkpoint_metadata_path(source / "checkpoint.pkl")
    )
    manifest = prepare(source, args.target_run_dir.resolve(), metadata)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
