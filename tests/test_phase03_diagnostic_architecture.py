from __future__ import annotations

import ast
import csv
import gzip
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.common.diagnostic_io import (
    append_diagnostic_csv,
    close_diagnostic_writers,
    write_diagnostic_csv,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/refactor/phase03"


class Phase03ImportArchitectureTests(unittest.TestCase):
    def test_seqmem_does_not_import_experiments(self) -> None:
        violations = []
        for path in (ROOT / "src/seqmem").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                if any(name == "experiments" or name.startswith("experiments.") for name in names):
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
        self.assertEqual([], violations)

    def test_strict_import_keeps_oracle_and_teacher_modules_unloaded(self) -> None:
        code = """
import sys
import experiments.fig9_strict_reproduction
blocked = sorted(
    name for name in sys.modules
    if name in {
        'experiments.diagnostics.fig9_oracle_candidate',
        'experiments.diagnostics.fig9_candidate_context_oracle',
        'experiments.diagnostics.fig9_teacher_forced_identity',
    }
)
print(';'.join(blocked))
raise SystemExit(bool(blocked))
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertEqual("", completed.stdout.strip())

    def test_loader_imports_only_requested_family(self) -> None:
        code = """
import sys
from experiments.fig9.diagnostic_loader import load_readout_dynamics
load_readout_dynamics()
loaded = sorted(
    name for name in sys.modules
    if name.startswith('experiments.diagnostics.fig9_')
)
print(';'.join(loaded))
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        self.assertEqual(
            "experiments.diagnostics.fig9_readout_dynamics",
            completed.stdout.strip(),
        )

    def test_only_scientific_competition_remains_a_direct_import(self) -> None:
        path = ROOT / "experiments/fig9_strict_reproduction.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules = sorted(
            node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("experiments.diagnostics")
        )
        self.assertEqual(
            ["experiments.diagnostics.fig9_competitive_inhibition"],
            modules,
        )

    def test_diagnostic_cli_defaults_remain_off(self) -> None:
        from experiments import fig9_strict_reproduction as strict

        with patch.object(sys, "argv", ["fig9_strict_reproduction.py"]):
            args = strict.parse_args()
        flags = (
            "oracle_candidate_diagnostic",
            "candidate_separability_trace",
            "branch_provenance_diagnostic",
            "preselection_segment_diagnostic",
            "teacher_forced_winner_diagnostic",
            "observe_scenario_diagnostic",
            "reference_neuron_selection_diagnostic",
            "intracolumn_selection_diagnostic",
            "readout_dynamics_trace",
            "match_overlap_diagnostic",
            "segment_context_composition_diagnostic",
            "context_trajectory_diagnostic",
            "temporal_context_diagnostic",
            "autonomous_context_provenance_diagnostic",
            "segment_reinforcement_diagnostic",
            "ambiguity_diagnostic",
            "independent_reference_diagnostic",
            "actual_branch_provenance_diagnostic",
        )
        self.assertTrue(all(getattr(args, name) is False for name in flags))
        self.assertEqual("off", args.competition_mode)


class Phase03DiagnosticIoTests(unittest.TestCase):
    def tearDown(self) -> None:
        close_diagnostic_writers()

    def test_one_shot_plain_and_gzip_content_are_exact(self) -> None:
        rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        expected = b"a,b\r\n1,x\r\n2,y\r\n"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plain = write_diagnostic_csv(root / "trace.csv", rows)
            compressed = write_diagnostic_csv(
                root / "trace_gzip.csv",
                rows,
                compress=True,
            )
            self.assertEqual(expected, plain.read_bytes())
            self.assertEqual(expected, gzip.decompress(compressed.read_bytes()))

    def test_streamed_compressed_batches_preserve_rows_and_close(self) -> None:
        rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        expected = b"a,b\r\n1,x\r\n2,y\r\n"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "stream.csv"
            actual = append_diagnostic_csv(path, rows[:1], compress=True)
            append_diagnostic_csv(path, rows[1:], compress=True)
            close_diagnostic_writers()
            self.assertEqual(expected, gzip.decompress(actual.read_bytes()))
            self.assertFalse((root / "stream.csv.gz.pending").exists())

    def test_three_diagnostic_on_off_fixtures_are_scientifically_exact(self) -> None:
        path = REPORT_DIR / "DIAGNOSTIC_ON_OFF_PARITY.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(3, len(rows))
        self.assertTrue(all(row["scientific_exact"] == "True" for row in rows))
        self.assertTrue(all(not row["different_fields"] for row in rows))

    def test_migrated_diagnostic_artifacts_keep_scientific_content(self) -> None:
        path = REPORT_DIR / "DIAGNOSTIC_OUTPUT_COMPATIBILITY.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertGreaterEqual(len(rows), 3)
        self.assertTrue(
            all(row["scientific_content_equal"] == "True" for row in rows)
        )


if __name__ == "__main__":
    unittest.main()
