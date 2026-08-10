from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from experiments.diagnostics.analyze_fig9_l2_l4_long_sequence import (
    RANGES,
    expected_predictions,
    overall_stepwise_error,
    process_attempt_summary,
    record_range,
)
from experiments.fig9_strict_reproduction import (
    write_long_sequence_activity_trace,
)


class Fig9LongSequenceTests(unittest.TestCase):
    def test_expected_count_and_requested_ranges(self) -> None:
        protocol = {
            "data_file_rows_used": 500,
            "warmup_length": 200,
            "prediction_horizon": 5,
        }
        self.assertEqual(expected_predictions(protocol), 295)
        self.assertEqual(record_range(200), "200-249")
        self.assertEqual(record_range(494), "450-494")
        self.assertEqual(record_range(495), "outside_requested_ranges")
        self.assertEqual(len(RANGES), 6)

    def test_ledger_writer_preserves_one_row_per_observation(self) -> None:
        rows = [
            {
                "record_index": 200,
                "L_match": 4,
                "live_segments": 12,
                "raw_predicted_columns_step5": 8,
            },
            {
                "record_index": 201,
                "L_match": 4,
                "live_segments": 13,
                "raw_predicted_columns_step5": 7,
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "long_sequence_activity_trace.csv"
            write_long_sequence_activity_trace(path, rows)
            with path.open("r", encoding="utf-8", newline="") as handle:
                saved = list(csv.DictReader(handle))
            self.assertEqual(len(saved), 2)
            self.assertEqual(saved[1]["record_index"], "201")
            self.assertEqual(saved[0]["raw_predicted_columns_step5"], "8")

    def test_overall_stepwise_error_uses_all_requested_records(self) -> None:
        density = []
        for index in (200, 250, 300, 350, 400, 450):
            density.append(
                {
                    "record_index": str(index),
                    "horizon_step": "1",
                    "actual_future_passenger": "10",
                    "absolute_error": "2",
                }
            )
        row = overall_stepwise_error(
            {"L_match": 2, "density": density}
        )
        self.assertEqual(row["prediction_rows_step1"], 6)
        self.assertAlmostEqual(row["MAPE_step1"], 0.2)
        self.assertEqual(row["prediction_rows_step5"], 0)

    def test_process_attempt_summary_counts_recovery_and_native_crashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "process_a.json").write_text(
                json.dumps({"elapsed_seconds": 1.5, "exit_code": 0}),
                encoding="utf-8",
            )
            (directory / "process_b.json").write_text(
                json.dumps({"elapsed_seconds": 2.5, "exit_code": 3221225477}),
                encoding="utf-8",
            )
            self.assertEqual(
                process_attempt_summary(directory),
                {
                    "process_attempt_count": 2,
                    "process_runtime_seconds_total": 4.0,
                    "native_crash_count": 1,
                },
            )


if __name__ == "__main__":
    unittest.main()
