"""Compare two Phase 01 manifests on frozen behavioral fields only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BEHAVIOR_FIELDS = (
    "prediction_sha256",
    "metric",
    "scenario_summary",
    "SCIENCE_MODEL_SHA256",
    "FULL_STATE_SHA256",
    "RNG_SHA256",
    "LEARNING_RNG_SHA256",
    "DECODE_RNG_SHA256",
    "PREVIOUS_ACTIVE_SHA256",
    "PREVIOUS_WINNER_SHA256",
    "CANDIDATE_SHA256",
    "structure",
)


def _load_runs(path: Path) -> dict[str, dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {str(run["name"]): run for run in manifest["runs"]}


def compare_manifests(reference: Path, candidate: Path) -> dict[str, Any]:
    """Return an exact field-by-field comparison suitable for audit output."""

    expected = _load_runs(reference)
    actual = _load_runs(candidate)
    fixture_names_match = set(expected) == set(actual)
    rows: list[dict[str, Any]] = []
    for name in sorted(set(expected) | set(actual)):
        if name not in expected or name not in actual:
            rows.append(
                {
                    "name": name,
                    "exact_match": False,
                    "missing_from": "reference" if name not in expected else "candidate",
                    "different_fields": [],
                }
            )
            continue
        different_fields = [
            field
            for field in BEHAVIOR_FIELDS
            if expected[name].get(field) != actual[name].get(field)
        ]
        rows.append(
            {
                "name": name,
                "exact_match": not different_fields,
                "different_fields": different_fields,
            }
        )
    return {
        "reference_manifest": str(reference),
        "candidate_manifest": str(candidate),
        "behavior_fields": list(BEHAVIOR_FIELDS),
        "fixture_names_match": fixture_names_match,
        "all_six_exact_match": (
            fixture_names_match
            and len(rows) == 6
            and all(row["exact_match"] for row in rows)
        ),
        "fixtures": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = compare_manifests(args.reference.resolve(), args.candidate.resolve())
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["all_six_exact_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
