"""Run each frozen refactor fixture in an isolated child process.

The repository has historical Windows native-crash evidence in long-lived
single Python processes. This runner does not hide such failures: it records
each child exit code and output. Isolation lets architecture migrations still
compare all six frozen scientific fixtures without treating isolated success
as evidence that the native crash is fixed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from compare_phase01_baselines import compare_manifests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = (
    ROOT
    / "reports/refactor/phase01_20260814_233129/baselines/pre_refactor"
    / "baseline_manifest.json"
)
FIXTURES = tuple(
    (family, size)
    for family in ("fig8", "fig9")
    for size in (10, 20, 50)
)


def run_fixture(output_root: Path, family: str, size: int) -> dict[str, object]:
    """Capture one fixture in a fresh interpreter and return its manifest row."""

    shard = output_root / "shards" / f"{family}_{size}"
    if shard.exists():
        shutil.rmtree(shard)
    command = [
        sys.executable,
        "scripts/capture_phase01_baselines.py",
        "--output",
        str(shard),
        "--families",
        family,
        "--sizes",
        str(size),
        "--repeat",
        "1",
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT)]
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    (shard.parent / f"{family}_{size}.stdout.log").write_text(
        completed.stdout,
        encoding="utf-8",
    )
    (shard.parent / f"{family}_{size}.stderr.log").write_text(
        completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{family}_{size} child exited {completed.returncode}; "
            f"see {shard.parent}"
        )
    manifest = json.loads(
        (shard / "baseline_manifest.json").read_text(encoding="utf-8")
    )
    runs = manifest.get("runs", [])
    if len(runs) != 1:
        raise RuntimeError(f"{family}_{size} produced {len(runs)} runs")
    print(f"captured isolated {family}_{size}", flush=True)
    return runs[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs = [run_fixture(output, family, size) for family, size in FIXTURES]
    manifest_path = output / "baseline_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "classification": "PHASE03_ISOLATED_GOLDEN_REGRESSION",
                "runs": runs,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    comparison = compare_manifests(args.reference.resolve(), manifest_path)
    comparison_path = output / "comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison, indent=2, ensure_ascii=False))
    if not comparison["all_six_exact_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
