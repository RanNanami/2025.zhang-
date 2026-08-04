import csv
import tempfile
import unittest
from pathlib import Path

from experiments.diagnostics.assemble_fig9_resumed_diagnostics import (
    merge_trace,
    scenario_events_from_context,
    scenario_events_from_observe,
)


class Fig9ResumeDiagnosticAssemblyTests(unittest.TestCase):
    @staticmethod
    def _write(path: Path, rows: list[dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_merge_uses_checkpoint_boundary_and_formal_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before = root / "before.csv"
            after = root / "after.csv"
            output = root / "output.csv"
            self._write(before, [{"actual_record_index": i} for i in (199, 200, 219, 220)])
            self._write(after, [{"actual_record_index": i} for i in (219, 220, 244, 245)])
            count = merge_trace(
                before,
                after,
                output,
                index_field="actual_record_index",
                resume_index=220,
                minimum_index=200,
                maximum_index_exclusive=245,
            )
            with output.open(encoding="utf-8", newline="") as handle:
                indices = [int(row["actual_record_index"]) for row in csv.DictReader(handle)]
        self.assertEqual(count, 4)
        self.assertEqual(indices, [200, 219, 220, 244])

    def test_observe_trace_restores_event_missing_from_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = root / "context.csv"
            observe = root / "observe.csv"
            self._write(
                context,
                [{
                    "actual_record_index": 220,
                    "observed_field": "passenger",
                    "observed_column": 10,
                    "observe_scenario": "scenario2",
                }],
            )
            self._write(
                observe,
                [
                    {
                        "actual_record_index": 220,
                        "field": "passenger",
                        "encoded_column": 10,
                        "observe_scenario": "scenario2",
                    },
                    {
                        "actual_record_index": 220,
                        "field": "passenger",
                        "encoded_column": 11,
                        "observe_scenario": "scenario3",
                    },
                ],
            )
            context_events = scenario_events_from_context(context)
            observe_events = scenario_events_from_observe(observe)
        self.assertEqual(len(context_events), 1)
        self.assertEqual(len(observe_events), 2)
        self.assertEqual(observe_events[(220, "passenger", 11)], "scenario3")


if __name__ == "__main__":
    unittest.main()
