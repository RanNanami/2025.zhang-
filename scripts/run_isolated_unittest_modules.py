"""Run every unittest module in a fresh Python process.

This runner records assertion failures separately from Windows native process
termination. It is a reporting fallback and does not claim that an isolated
pass is equivalent to single-process discovery.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NATIVE_ACCESS_VIOLATION = 0xC0000005


def normalized_exit_code(code: int) -> int:
    return code if code >= 0 else code + 2**32


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT)])
    environment["PYTHONFAULTHANDLER"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    rows: list[dict[str, object]] = []
    for test_path in sorted((ROOT / "tests").glob("test_*.py")):
        module = f"tests.{test_path.stem}"
        command = [sys.executable, "-m", "unittest", module, "-v"]
        started = time.perf_counter()
        timed_out = False
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=args.timeout,
            )
            exit_code = normalized_exit_code(completed.returncode)
            stdout = completed.stdout
            stderr = completed.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = -1
            stdout = (exc.stdout or b"").decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = (exc.stderr or b"").decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        elapsed = time.perf_counter() - started
        combined = stdout + stderr
        (output / f"{test_path.stem}.stdout.log").write_text(stdout, encoding="utf-8")
        (output / f"{test_path.stem}.stderr.log").write_text(stderr, encoding="utf-8")
        match = re.search(r"Ran (\d+) tests?", combined)
        test_count = int(match.group(1)) if match else None
        status = (
            "TIMEOUT"
            if timed_out
            else "PASS"
            if exit_code == 0
            else "NATIVE_CRASH"
            if exit_code == NATIVE_ACCESS_VIOLATION
            else "FAIL"
        )
        row = {
            "module": module,
            "status": status,
            "exit_code": exit_code,
            "test_count": test_count,
            "elapsed_seconds": elapsed,
            "stdout_log": f"{test_path.stem}.stdout.log",
            "stderr_log": f"{test_path.stem}.stderr.log",
            "tail": "\n".join(combined.splitlines()[-20:]),
        }
        rows.append(row)
        print(f"{status:12s} {module} ({test_count if test_count is not None else '?'} tests)", flush=True)

    summary = {
        "captured_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "python": sys.version,
        "module_count": len(rows),
        "passed_module_count": sum(row["status"] == "PASS" for row in rows),
        "failed_module_count": sum(row["status"] != "PASS" for row in rows),
        "native_crash_count": sum(row["status"] == "NATIVE_CRASH" for row in rows),
        "total_reported_tests": sum(int(row["test_count"] or 0) for row in rows),
        "single_process_equivalent": False,
        "modules": rows,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    raise SystemExit(0 if summary["failed_module_count"] == 0 else 1)


if __name__ == "__main__":
    main()
