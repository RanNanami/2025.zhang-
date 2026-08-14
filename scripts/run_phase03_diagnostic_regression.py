"""Capture Phase 03 diagnostic ON/OFF and output compatibility evidence."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from capture_phase01_baselines import load_strict_checkpoint, model_fingerprints


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = {
    "segment_reinforcement": {
        "flags": [
            "--segment-reinforcement-diagnostic",
            "--segment-reinforcement-level",
            "segment",
        ],
        "prefixes": ("segment_reinforcement_",),
        "classification": "READ_ONLY_TRACE",
    },
    "actual_branch_provenance": {
        "flags": [
            "--actual-branch-provenance-diagnostic",
            "--actual-branch-provenance-level",
            "segment",
        ],
        "prefixes": ("actual_",),
        "classification": "READ_ONLY_TRACE_PROVENANCE",
    },
    "match_overlap": {
        "flags": [
            "--match-overlap-diagnostic",
            "--match-overlap-level",
            "segment",
            "--match-overlap-compress",
        ],
        "prefixes": ("match_overlap_",),
        "classification": "READ_ONLY_TRACE_COMPLEX",
    },
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scientific_ledger_rows(path: Path) -> list[dict[str, str]]:
    """Exclude only measured wall-time and resident-memory columns."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.pop("runtime_so_far", None)
        row.pop("memory_RSS", None)
    return rows


def without_protocol_provenance(value: Any) -> Any:
    """Remove commit-derived hashes while preserving diagnostic semantics."""

    if isinstance(value, dict):
        return {
            key: without_protocol_provenance(item)
            for key, item in value.items()
            if key != "strict_protocol_sha256"
        }
    if isinstance(value, list):
        return [without_protocol_provenance(item) for item in value]
    return value


def run_case(
    repo: Path,
    output: Path,
    flags: list[str],
) -> dict[str, Any]:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    checkpoint = output / "checkpoint.pkl"
    command = [
        sys.executable,
        "experiments/fig9_strict_reproduction.py",
        "--data",
        "data/paper_nyc_taxi.csv",
        "--perturbed-data",
        "data/paper_nyc_taxi_perturb.csv",
        "--limit",
        "20",
        "--warmup",
        "6",
        "--prediction-horizon",
        "5",
        "--tie-break-seed",
        "0",
        "--l-match",
        "4",
        "--continuous-impl",
        "reference",
        "--streams",
        "original",
        "--output-dir",
        str(output),
        "--checkpoint-path",
        str(checkpoint),
        "--checkpoint-every",
        "1",
        "--long-sequence-ledger",
        *flags,
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(repo / "src"), str(repo)]
    )
    completed = subprocess.run(
        command,
        cwd=repo,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    (output / "runner.stdout.log").write_text(
        completed.stdout,
        encoding="utf-8",
    )
    (output / "runner.stderr.log").write_text(
        completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"runner exited {completed.returncode}: {output}"
        )
    payload = load_strict_checkpoint(checkpoint)
    model = payload["model"]
    summary = json.loads(
        (output / "original_summary.json").read_text(encoding="utf-8")
    )
    return {
        "output": output,
        "checkpoint_next_index": payload["next_index"],
        "prediction_sha256": sha256_file(output / "original_predictions.csv"),
        "scenario_ledger": scientific_ledger_rows(
            output / "long_sequence_activity_trace.csv"
        ),
        "summary_science": {
            key: summary[key]
            for key in (
                "mape",
                "coverage",
                "predictions",
                "mean_raw_column_count",
                "peak_raw_column_count",
                "final_model_fingerprint",
                "final_rng_fingerprint",
                "final_segment_count",
                "final_synapse_count",
            )
        },
        "model_fingerprints": model_fingerprints(model),
    }


def selected_artifacts(
    output: Path,
    prefixes: tuple[str, ...],
) -> dict[str, Path]:
    return {
        path.name: path
        for path in output.iterdir()
        if path.is_file()
        and path.name.startswith(prefixes)
        and not path.name.endswith(".pending")
    }


def compare_artifacts(
    fixture: str,
    before: Path,
    after: Path,
    prefixes: tuple[str, ...],
) -> list[dict[str, object]]:
    old_files = selected_artifacts(before, prefixes)
    new_files = selected_artifacts(after, prefixes)
    rows: list[dict[str, object]] = []
    for name in sorted(set(old_files) | set(new_files)):
        old_path = old_files.get(name)
        new_path = new_files.get(name)
        present_both = old_path is not None and new_path is not None
        raw_equal = (
            present_both and old_path.read_bytes() == new_path.read_bytes()
        )
        content_equal = raw_equal
        reason = ""
        old_sha = sha256_file(old_path) if old_path is not None else ""
        new_sha = sha256_file(new_path) if new_path is not None else ""
        old_content_sha = old_sha
        new_content_sha = new_sha
        if present_both and name.endswith(".gz"):
            old_payload = gzip.decompress(old_path.read_bytes())
            new_payload = gzip.decompress(new_path.read_bytes())
            old_content_sha = sha256_bytes(old_payload)
            new_content_sha = sha256_bytes(new_payload)
            content_equal = old_payload == new_payload
            if content_equal and not raw_equal:
                reason = "gzip_header_metadata_only"
        elif present_both and name.endswith(".json") and not raw_equal:
            old_value = without_protocol_provenance(
                json.loads(old_path.read_text(encoding="utf-8"))
            )
            new_value = without_protocol_provenance(
                json.loads(new_path.read_text(encoding="utf-8"))
            )
            old_payload = json.dumps(
                old_value,
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
            new_payload = json.dumps(
                new_value,
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
            old_content_sha = sha256_bytes(old_payload)
            new_content_sha = sha256_bytes(new_payload)
            content_equal = old_payload == new_payload
            if content_equal:
                reason = "strict_protocol_sha256_commit_provenance_only"
        rows.append(
            {
                "fixture": fixture,
                "artifact": name,
                "present_before": old_path is not None,
                "present_after": new_path is not None,
                "raw_byte_equal": raw_equal,
                "scientific_content_equal": content_equal,
                "before_sha256": old_sha,
                "after_sha256": new_sha,
                "before_content_sha256": old_content_sha,
                "after_content_sha256": new_content_sha,
                "difference_reason": reason,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-repo", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()

    before_repo = args.before_repo.resolve()
    work_dir = args.work_dir.resolve()
    report_dir = args.report_dir.resolve()
    parity_rows: list[dict[str, object]] = []
    compatibility_rows: list[dict[str, object]] = []
    for fixture, settings in FIXTURES.items():
        shared = work_dir / "shared" / fixture
        before_snapshot = work_dir / "before" / fixture
        before = run_case(before_repo, shared, settings["flags"])
        if before_snapshot.exists():
            shutil.rmtree(before_snapshot)
        shutil.copytree(shared, before_snapshot)

        off = run_case(ROOT, work_dir / "off" / fixture, [])
        on = run_case(ROOT, shared, settings["flags"])
        science_keys = (
            "prediction_sha256",
            "scenario_ledger",
            "checkpoint_next_index",
            "summary_science",
            "model_fingerprints",
        )
        differences = [key for key in science_keys if off[key] != on[key]]
        parity_rows.append(
            {
                "fixture": fixture,
                "classification": settings["classification"],
                "records": 20,
                "scientific_exact": not differences,
                "different_fields": ";".join(differences),
                "prediction_sha_equal": (
                    off["prediction_sha256"] == on["prediction_sha256"]
                ),
                "science_state_equal": (
                    off["model_fingerprints"] == on["model_fingerprints"]
                ),
                "rng_equal": (
                    off["model_fingerprints"]["RNG_SHA256"]
                    == on["model_fingerprints"]["RNG_SHA256"]
                ),
                "scenario_ledger_equal": (
                    off["scenario_ledger"] == on["scenario_ledger"]
                ),
                "segment_synapse_equal": (
                    off["summary_science"]["final_segment_count"]
                    == on["summary_science"]["final_segment_count"]
                    and off["summary_science"]["final_synapse_count"]
                    == on["summary_science"]["final_synapse_count"]
                ),
            }
        )
        compatibility_rows.extend(
            compare_artifacts(
                fixture,
                before_snapshot,
                shared,
                settings["prefixes"],
            )
        )
        print(f"captured diagnostic fixture {fixture}", flush=True)

    write_csv(report_dir / "DIAGNOSTIC_ON_OFF_PARITY.csv", parity_rows)
    write_csv(
        report_dir / "DIAGNOSTIC_OUTPUT_COMPATIBILITY.csv",
        compatibility_rows,
    )
    result = {
        "all_science_parity_exact": all(
            bool(row["scientific_exact"]) for row in parity_rows
        ),
        "all_artifact_content_compatible": all(
            bool(row["scientific_content_equal"])
            for row in compatibility_rows
        ),
        "parity": parity_rows,
        "compatibility": compatibility_rows,
    }
    (report_dir / "diagnostic_regression.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not (
        result["all_science_parity_exact"]
        and result["all_artifact_content_compatible"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
