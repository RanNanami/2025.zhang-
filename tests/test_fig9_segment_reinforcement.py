from __future__ import annotations

import csv
import gzip
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from experiments.diagnostics.analyze_fig9_wrong_segment_reinforcement import (
    OUTPUT_NAMES,
    analyze,
    reinforcement_class,
)
from experiments.fig9 import TaxiRecord
from experiments.fig9_strict_reproduction import (
    Fig9StrictConfig,
    load_strict_checkpoint,
    model_long_term_fingerprint,
    model_rng_fingerprint,
    run_strict_stream,
)


class SegmentReinforcementDiagnosticTests(unittest.TestCase):
    def test_reference_unavailable_is_never_wrong(self) -> None:
        for status in (
            "REFERENCE_UNAVAILABLE",
            "POST_OBSERVATION_REFERENCE_INVALID",
        ):
            self.assertEqual(
                reinforcement_class(
                    {
                        "reference_status": status,
                        "ambiguous_segment": True,
                        "selected_segment_reference_compatible": False,
                    }
                ),
                "REFERENCE_UNAVAILABLE",
            )

    def test_primary_labels_are_mutually_exclusive(self) -> None:
        cases = (
            ({"reference_status": "REFERENCE_COMPATIBLE", "selected_segment_reference_compatible": True, "selected_neuron_reference_compatible": True}, "REFERENCE_COMPATIBLE"),
            ({"reference_status": "REFERENCE_COMPATIBLE", "selected_segment_reference_compatible": True, "selected_neuron_reference_compatible": True, "ambiguous_neuron": True}, "AMBIGUOUS_BUT_CORRECT"),
            ({"reference_status": "REFERENCE_INCOMPATIBLE"}, "WRONG_REINFORCEMENT_EVENT"),
            ({"reference_status": "REFERENCE_INCOMPATIBLE", "ambiguous_segment": True}, "AMBIGUOUS_AND_WRONG"),
            ({"reference_status": "REFERENCE_AMBIGUOUS"}, "REFERENCE_AMBIGUOUS"),
        )
        for row, expected in cases:
            self.assertEqual(reinforcement_class(row), expected)

    def test_runner_trace_is_noninterfering_and_checkpoint_safe(self) -> None:
        records = [
            TaxiRecord(
                datetime(2014, 7, 1, hour=(index // 2) % 24, minute=30 * (index % 2)),
                100.0 + index,
            )
            for index in range(10)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "tiny.csv"
            data.write_text("timestamp,passenger_count\n", encoding="utf-8")
            common = {
                "records": records,
                "data_path": data,
                "stream_label": "original",
                "config": Fig9StrictConfig(warmup=4, horizon=1),
                "limit": 10,
                "print_fingerprint": False,
            }
            plain = run_strict_stream(output_dir=root / "plain", **common)
            traced = run_strict_stream(
                output_dir=root / "traced",
                checkpoint_path=root / "traced.pkl",
                checkpoint_at_index=9,
                segment_reinforcement_diagnostic=True,
                segment_reinforcement_level="segment",
                **common,
            )
            self.assertEqual(
                (root / "plain" / "original_predictions.csv").read_bytes(),
                (root / "traced" / "original_predictions.csv").read_bytes(),
            )
            for field in (
                "mape",
                "coverage",
                "final_model_fingerprint",
                "final_rng_fingerprint",
                "final_segment_count",
            ):
                self.assertEqual(plain[field], traced[field])
            checkpoint = load_strict_checkpoint(root / "traced.pkl")
            self.assertEqual(
                model_long_term_fingerprint(checkpoint["model"]),
                traced["final_model_fingerprint"],
            )
            self.assertEqual(
                model_rng_fingerprint(checkpoint["model"]),
                traced["final_rng_fingerprint"],
            )
            with gzip.open(
                root / "traced" / "segment_reinforcement_event_trace.csv.gz",
                "rt",
                encoding="utf-8",
                newline="",
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            self.assertTrue(all(row["reinforced"] == "True" for row in rows))

    def test_analyzer_writes_complete_output_set(self) -> None:
        base = {
            "diagnostic_only": True,
            "ground_truth_does_not_affect_model": True,
            "reinforcement_trace_does_not_affect_learning": True,
            "actual_record_index": 10,
            "field": "passenger",
            "observe_scenario": "scenario2",
            "selected_overlap": 2,
            "l2_only_match": True,
            "ambiguous_segment": True,
            "ambiguous_neuron": True,
            "reference_applicable": False,
            "reference_status": "POST_OBSERVATION_REFERENCE_INVALID",
            "reinforced": True,
            "reinforcement_delta": 0.2,
            "future_reinforcement_count_5": 1,
            "reinforced_again_within_5": True,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, overlap in (("l2", 2), ("l4", 4)):
                run = root / name
                run.mkdir()
                row = {**base, "L_match": overlap, "selected_overlap": overlap}
                path = run / "segment_reinforcement_event_trace.csv.gz"
                with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerow(row)
            output = root / "analysis"
            result = analyze(
                l2_run=root / "l2",
                l4_run=root / "l4",
                output_dir=output,
                bootstrap_samples=20,
            )
            self.assertEqual(result["mechanism_conclusion"], "REFERENCE_COVERAGE_INSUFFICIENT")
            for name in OUTPUT_NAMES:
                self.assertTrue((output / name).exists(), name)
            self.assertTrue((output / "reinforcement_consistency_checks.json").exists())
            self.assertTrue((output / "FIG9_WRONG_SEGMENT_REINFORCEMENT_REPORT.md").exists())


if __name__ == "__main__":
    unittest.main()
