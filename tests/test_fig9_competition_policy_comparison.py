from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.diagnostics.compare_fig9_competition_policies import (
    align_traces,
    compare_policy_rows,
    write_comparison,
)


class Fig9CompetitionPolicyComparisonTests(unittest.TestCase):
    def _row(
        self,
        index: int,
        step: int,
        *,
        targets: str,
        false: str,
    ) -> dict[str, str]:
        return {
            "prediction_input_index": str(index),
            "horizon_step": str(step),
            "target_timestamp": f"2015-01-01 0{step}:00:00",
            "prediction_available": "True",
            "absolute_error": "10",
            "target_passenger": "100",
            "raw_candidate_column_count": "10",
            "emitted_column_count": "5",
            "raw_target_total_hits": "3",
            "raw_target_passenger_hits": "1",
            "emitted_target_total_hits": str(len(targets.split())),
            "emitted_target_passenger_hits": "1",
            "target_total_column_count": "3",
            "target_passenger_column_count": "1",
            "target_suppression_ratio": "0.5",
            "emitted_false_column_ratio": "0.4",
            "classification": "TARGET_PRESENT_BUT_SUPPRESSED",
            "emitted_target_total_columns": targets,
            "emitted_false_columns": false,
            "raw_false_column_count": "7",
            "target_suppressed_only_due_to_same_batch_order": "1",
            "target_suppressed_by_earlier_batch": "2",
            "target_best_candidate_batch_index": "3",
            "false_best_candidate_batch_index": "2",
        }

    def test_strict_alignment_accepts_reordered_rows(self) -> None:
        sequential = [
            self._row(2, 1, targets="1", false="8"),
            self._row(1, 1, targets="1", false="8"),
        ]
        batched = list(reversed(sequential))

        aligned = align_traces(sequential, batched)

        self.assertEqual(len(aligned), 2)
        self.assertEqual(aligned[0][0]["prediction_input_index"], "1")

    def test_strict_alignment_rejects_missing_row(self) -> None:
        sequential = [self._row(1, 1, targets="1", false="8")]

        with self.assertRaisesRegex(ValueError, "alignment mismatch"):
            align_traces(sequential, [])

    def test_strict_alignment_rejects_duplicate_key(self) -> None:
        row = self._row(1, 1, targets="1", false="8")

        with self.assertRaisesRegex(ValueError, "duplicate sequential"):
            align_traces([row, dict(row)], [row])

    def test_comparison_counts_target_and_false_rescue(self) -> None:
        sequential = [self._row(1, 1, targets="1", false="8")]
        batched = [self._row(1, 1, targets="1 2", false="8 9")]

        rows, summary = compare_policy_rows(sequential, batched)

        self.assertEqual(rows[0]["target_rescued_by_batching"], 1)
        self.assertEqual(rows[0]["false_rescued_by_batching"], 1)
        self.assertEqual(summary["target_rescued_by_batching_count"], 1)
        self.assertEqual(summary["false_rescued_by_batching_count"], 1)

    def test_comparison_outputs_are_explicitly_diagnostic(self) -> None:
        sequential = [self._row(1, 1, targets="1", false="8")]
        batched = [self._row(1, 1, targets="1 2", false="8")]
        rows, summary = compare_policy_rows(sequential, batched)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_comparison(output, rows, summary)
            csv_text = (
                output / "competition_policy_comparison.csv"
            ).read_text(encoding="utf-8")
            json_text = (
                output / "competition_policy_comparison.json"
            ).read_text(encoding="utf-8")

        self.assertIn("diagnostic_only", csv_text)
        self.assertIn('"diagnostic_only": true', json_text)

    def test_historical_trace_without_appended_columns_is_readable(self) -> None:
        sequential = [self._row(1, 1, targets="1", false="8")]
        batched = [self._row(1, 1, targets="1", false="8")]
        for row in (sequential[0], batched[0]):
            row.pop("emitted_target_total_columns")
            row.pop("emitted_false_columns")

        rows, summary = compare_policy_rows(sequential, batched)

        self.assertEqual(rows[0]["target_rescued_by_batching"], 0)
        self.assertEqual(rows[0]["false_rescued_by_batching"], 0)
        self.assertEqual(summary["aligned_rows"], 1)


if __name__ == "__main__":
    unittest.main()
