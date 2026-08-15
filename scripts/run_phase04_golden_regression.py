"""Run Phase 04 golden fixtures with audited native-crash retries.

Each fixture still runs in a fresh interpreter. Ordinary Python failures stop
immediately. Only known Windows native termination codes are retried, and every
attempt remains recorded so a successful comparison cannot be mistaken for a
native-runtime fix.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from compare_phase01_baselines import compare_manifests
from run_phase03_golden_regression import DEFAULT_REFERENCE, FIXTURES, ROOT


NATIVE_EXIT_MARKERS = (
    "3221225477",  # 0xC0000005 STATUS_ACCESS_VIOLATION
    "3221226505",  # 0xC0000409 STATUS_STACK_BUFFER_OVERRUN
    "0xC0000005",
    "0xC0000409",
)
NATIVE_RETURN_CODES = {
    3221225477,
    3221226505,
    -1073741819,
    -1073740791,
}


def _is_native_failure(completed: subprocess.CompletedProcess[str]) -> bool:
    text = completed.stdout + "\n" + completed.stderr
    return completed.returncode in NATIVE_RETURN_CODES or any(
        marker in text for marker in NATIVE_EXIT_MARKERS
    )


def capture_fixture(
    output_root: Path,
    family: str,
    size: int,
    max_native_attempts: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Return one successful fixture and the complete child-attempt ledger."""

    fixture_root = output_root / "shards" / f"{family}_{size}"
    fixture_root.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, object]] = []
    for attempt in range(1, max_native_attempts + 1):
        attempt_dir = fixture_root / f"attempt_{attempt}"
        command = [
            sys.executable,
            "scripts/capture_phase01_baselines.py",
            "--output",
            str(attempt_dir),
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
        stdout_path = fixture_root / f"attempt_{attempt}.stdout.log"
        stderr_path = fixture_root / f"attempt_{attempt}.stderr.log"
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        native_failure = _is_native_failure(completed)
        attempts.append(
            {
                "family": family,
                "size": size,
                "attempt": attempt,
                "exit_code": completed.returncode,
                "native_failure": native_failure,
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
            }
        )
        if completed.returncode == 0:
            manifest = json.loads(
                (attempt_dir / "baseline_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            runs = manifest.get("runs", [])
            if len(runs) != 1:
                raise RuntimeError(
                    f"{family}_{size} attempt {attempt} produced {len(runs)} runs"
                )
            print(
                f"captured isolated {family}_{size} on attempt {attempt}",
                flush=True,
            )
            return runs[0], attempts
        if not native_failure:
            raise RuntimeError(
                f"{family}_{size} ordinary child failure on attempt {attempt}; "
                f"see {stderr_path}"
            )
        print(
            f"native failure for {family}_{size} on attempt {attempt}; retrying",
            flush=True,
        )
    raise RuntimeError(
        f"{family}_{size} did not complete after {max_native_attempts} "
        "audited native attempts"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--max-native-attempts", type=int, default=5)
    args = parser.parse_args()
    if args.max_native_attempts < 1:
        raise ValueError("max native attempts must be positive")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, object]] = []
    attempt_ledger: list[dict[str, object]] = []
    for family, size in FIXTURES:
        run, attempts = capture_fixture(
            output,
            family,
            size,
            args.max_native_attempts,
        )
        runs.append(run)
        attempt_ledger.extend(attempts)

    manifest_path = output / "baseline_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "classification": "PHASE04_ISOLATED_GOLDEN_REGRESSION",
                "runs": runs,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "attempt_ledger.json").write_text(
        json.dumps(attempt_ledger, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    comparison = compare_manifests(args.reference.resolve(), manifest_path)
    comparison["attempt_ledger"] = attempt_ledger
    comparison["native_crash_fixed"] = False
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
