from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class Fig9ProcessLoggingTests(unittest.TestCase):
    def test_full_traceback_capture_regression(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stdout_log = root / "stdout.log"
            stderr_log = root / "stderr.log"
            combined_log = root / "combined.log"
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_logged_process.py",
                    "--stdout-log", str(stdout_log),
                    "--stderr-log", str(stderr_log),
                    "--combined-log", str(combined_log),
                    "--result-json", str(root / "result.json"),
                    "--failure-json", str(root / "failure.json"),
                    "--checkpoint", str(root / "checkpoint.pkl"),
                    "--",
                    sys.executable,
                    "-c",
                    "raise RuntimeError('FULL_TRACEBACK_CAPTURE_TEST')",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            stderr = stderr_log.read_text(encoding="utf-8")
            self.assertIn("Traceback (most recent call last)", stderr)
            self.assertIn("RuntimeError", stderr)
            self.assertIn("FULL_TRACEBACK_CAPTURE_TEST", stderr)
            self.assertIn("FULL_TRACEBACK_CAPTURE_TEST", completed.stdout)

    def test_stderr_traceback_and_native_exit_code_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = root / "run.log"
            result = root / "process_result.json"
            failure = root / "failure.json"
            checkpoint = root / "checkpoint.pkl"
            child = (
                "import sys; print('stdout-line', flush=True); "
                "print('stderr-line', file=sys.stderr, flush=True); "
                "raise RuntimeError('deliberate failure')"
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_logged_process.py",
                    "--log",
                    str(log),
                    "--result-json",
                    str(result),
                    "--failure-json",
                    str(failure),
                    "--checkpoint",
                    str(checkpoint),
                    "--",
                    sys.executable,
                    "-c",
                    child,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 1)
            content = log.read_text(encoding="utf-8")
            self.assertIn("stdout-line", content)
            self.assertIn("stderr-line", content)
            self.assertIn("Traceback", content)
            payload = json.loads(failure.read_text(encoding="utf-8"))
            self.assertNotEqual(payload["exit_code"], 0)
            self.assertIn("RuntimeError: deliberate failure", payload["exception"])
            self.assertIn("last_completed_index", payload)

    def test_success_writes_result_without_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_logged_process.py",
                    "--log",
                    str(root / "run.log"),
                    "--result-json",
                    str(root / "result.json"),
                    "--failure-json",
                    str(root / "failure.json"),
                    "--checkpoint",
                    str(root / "checkpoint.pkl"),
                    "--",
                    sys.executable,
                    "-c",
                    "print('ok', flush=True)",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0)
            self.assertEqual(
                json.loads((root / "result.json").read_text(encoding="utf-8"))[
                    "exit_code"
                ],
                0,
            )
            self.assertFalse((root / "failure.json").exists())


if __name__ == "__main__":
    unittest.main()
