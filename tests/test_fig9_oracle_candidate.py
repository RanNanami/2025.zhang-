from __future__ import annotations

import csv
import inspect
import json
import pickle
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from experiments.diagnostics.fig9_competitive_inhibition import (
    CompetitionCandidate,
    CompetitionSettings,
    compete_prediction_candidates,
    emitted_prediction_code,
)
from experiments.diagnostics.fig9_oracle_candidate import (
    CLASSIFICATION_THRESHOLDS,
    FieldColumnRanges,
    analyze_oracle_step,
    classify_step,
    split_field_columns,
    summarize_oracle_rows,
)
from experiments.fig9 import TaxiRecord
from experiments.fig9_strict_reproduction import (
    Fig9StrictConfig,
    build_fig9_encoder,
    model_long_term_fingerprint,
    model_rng_fingerprint,
    run_strict_stream,
)
from seqmem.encoding import SpikeEvent, SymbolCode
from seqmem.model import PredictionCandidate, Segment


class Fig9OracleCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name)
        cls.data_path = cls.root / "tiny.csv"
        cls.data_path.write_text(
            "timestamp,passenger_count\n", encoding="utf-8"
        )
        start = datetime(2014, 7, 1)
        cls.records = [
            TaxiRecord(
                start + timedelta(minutes=30 * index),
                1000.0 + 25.0 * index,
            )
            for index in range(18)
        ]
        cls.settings = CompetitionSettings(
            mode="competitive_raw",
            inhibition_strength=0.1,
            inhibition_tau=0.02,
        )
        common = {
            "records": cls.records,
            "data_path": cls.data_path,
            "stream_label": "original",
            "config": Fig9StrictConfig(warmup=6),
            "limit": 0,
            "print_fingerprint": False,
            "competition_settings": cls.settings,
            "checkpoint_at_index": 10,
        }
        cls.off_checkpoint = cls.root / "off.pkl"
        cls.on_checkpoint = cls.root / "on.pkl"
        cls.off_summary = run_strict_stream(
            output_dir=cls.root / "off",
            checkpoint_path=cls.off_checkpoint,
            **common,
        )
        cls.on_summary = run_strict_stream(
            output_dir=cls.root / "on",
            checkpoint_path=cls.on_checkpoint,
            oracle_candidate_diagnostic=True,
            **common,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def _candidate(
        self,
        column: int,
        score: float,
        order: int,
        *,
        time: float = 0.0,
    ) -> CompetitionCandidate:
        return CompetitionCandidate(
            column_index=column,
            candidate=PredictionCandidate(
                neuron_index=order,
                score=score,
                time=time,
                segment=Segment(),
            ),
            original_order=order,
        )

    def _analysis_row(self) -> dict[str, object]:
        candidates = [
            self._candidate(0, 1.2, 0),
            self._candidate(2, 1.05, 1),
            self._candidate(4, 1.3, 2),
            self._candidate(5, 1.4, 3),
        ]
        result = compete_prediction_candidates(
            candidates,
            threshold=1.0,
            inhibition_strength=0.1,
            inhibition_tau=0.02,
            simultaneous_tolerance=0.0,
        )
        raw = SymbolCode(
            tuple(
                SpikeEvent(
                    column=item.column_index,
                    time=item.candidate.time,
                )
                for item in candidates
            )
        )
        target = SymbolCode(
            (
                SpikeEvent(0, 0.0),
                SpikeEvent(2, 0.0),
                SpikeEvent(4, 0.0),
            )
        )
        return analyze_oracle_step(
            prediction_input_index=7,
            input_timestamp="2014-07-01 03:00:00",
            horizon_step=1,
            target_timestamp="2014-07-01 03:30:00",
            target_passenger=1000.0,
            decoded_prediction=900.0,
            absolute_error=100.0,
            raw_prediction=raw,
            emitted_prediction=emitted_prediction_code(raw, result),
            competition_result=result,
            target_code=target,
            ranges=FieldColumnRanges.from_sizes(2, 2, 2),
        )

    def test_oracle_off_and_on_predictions_csv_are_identical(self) -> None:
        off = (self.root / "off" / "original_predictions.csv").read_bytes()
        on = (self.root / "on" / "original_predictions.csv").read_bytes()
        self.assertEqual(on, off)

    def test_oracle_on_preserves_model_and_rng_fingerprints(self) -> None:
        self.assertEqual(
            self.on_summary["final_model_fingerprint"],
            self.off_summary["final_model_fingerprint"],
        )
        self.assertEqual(
            self.on_summary["final_rng_fingerprint"],
            self.off_summary["final_rng_fingerprint"],
        )

    def test_oracle_on_preserves_competition_trace(self) -> None:
        runtime_fields = {
            "prediction_runtime_seconds",
            "decode_runtime_seconds",
            "competition_runtime_seconds",
            "observe_runtime",
        }

        def stable_rows(path: Path) -> list[dict[str, str]]:
            with path.open("r", encoding="utf-8", newline="") as handle:
                return [
                    {
                        key: value
                        for key, value in row.items()
                        if key not in runtime_fields
                    }
                    for row in csv.DictReader(handle)
                ]

        self.assertEqual(
            stable_rows(self.root / "on" / "competition_trace.csv"),
            stable_rows(self.root / "off" / "competition_trace.csv"),
        )

    def test_oracle_on_preserves_checkpoint_v1(self) -> None:
        with self.off_checkpoint.open("rb") as handle:
            off = pickle.load(handle)
        with self.on_checkpoint.open("rb") as handle:
            on = pickle.load(handle)

        self.assertEqual(set(on), set(off))
        self.assertEqual(on["checkpoint_format"], "fig9-strict-v1")
        self.assertNotIn("oracle", on)
        self.assertEqual(
            model_long_term_fingerprint(on["model"]),
            model_long_term_fingerprint(off["model"]),
        )
        self.assertEqual(
            model_rng_fingerprint(on["model"]),
            model_rng_fingerprint(off["model"]),
        )

    def test_oracle_outputs_are_explicitly_analysis_only(self) -> None:
        oracle_summary = json.loads(
            (self.root / "on" / "oracle_candidate_summary.json").read_text(
                encoding="utf-8"
            )
        )
        run_summary = json.loads(
            (self.root / "on" / "original_summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(oracle_summary["diagnostic_only"])
        self.assertTrue(oracle_summary["uses_ground_truth_for_analysis_only"])
        self.assertTrue(
            oracle_summary["ground_truth_does_not_affect_prediction"]
        )
        self.assertTrue(run_summary["uses_ground_truth_for_analysis_only"])

    def test_strict_protocol_contains_no_oracle_marker(self) -> None:
        protocol = json.loads(
            (self.root / "on" / "original_protocol.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(protocol["propagation_mode"], "raw")
        self.assertNotIn("oracle", protocol)
        self.assertNotIn("uses_ground_truth_for_analysis_only", protocol)

    def test_ground_truth_is_absent_from_competition_interfaces(self) -> None:
        for function in (
            compete_prediction_candidates,
            emitted_prediction_code,
        ):
            parameters = inspect.signature(function).parameters
            for forbidden in (
                "target",
                "expected",
                "ground_truth",
                "future_record",
            ):
                self.assertNotIn(forbidden, parameters)

    def test_target_columns_match_composite_encoder_output(self) -> None:
        config = Fig9StrictConfig()
        encoder = build_fig9_encoder(config)
        code = encoder.encode((2.0, 17.0, 12345.0))
        ranges = FieldColumnRanges.from_sizes(
            config.weekday_columns,
            config.time_columns,
            config.passenger_columns,
        )
        fields = split_field_columns(code, ranges)

        self.assertEqual(fields["all"], set(code.columns))
        self.assertEqual(len(fields["weekday"]), config.k)
        self.assertEqual(len(fields["time"]), config.k)
        self.assertEqual(len(fields["passenger"]), config.k)

    def test_three_field_ranges_are_disjoint(self) -> None:
        code = SymbolCode(
            (
                SpikeEvent(0, 0.0),
                SpikeEvent(1, 0.1),
                SpikeEvent(2, 0.2),
                SpikeEvent(3, 0.3),
                SpikeEvent(4, 0.4),
                SpikeEvent(5, 0.5),
            )
        )
        fields = split_field_columns(
            code, FieldColumnRanges.from_sizes(2, 2, 2)
        )

        self.assertEqual(fields["weekday"], {0, 1})
        self.assertEqual(fields["time"], {2, 3})
        self.assertEqual(fields["passenger"], {4, 5})

    def test_raw_and_emitted_target_recall_are_correct(self) -> None:
        row = self._analysis_row()
        self.assertEqual(row["raw_target_total_hits"], 3)
        self.assertEqual(row["raw_target_total_recall"], 1.0)
        self.assertEqual(row["emitted_target_total_hits"], 2)
        self.assertEqual(row["emitted_target_total_recall"], 2 / 3)

    def test_suppressed_target_columns_are_reported(self) -> None:
        row = self._analysis_row()
        self.assertEqual(row["suppressed_target_time_columns"], "2")
        self.assertEqual(row["target_survival_ratio"], 2 / 3)
        self.assertAlmostEqual(row["target_suppression_ratio"], 1 / 3)

    def test_target_scores_and_ranks_use_saved_decisions(self) -> None:
        row = self._analysis_row()
        self.assertEqual(row["target_candidate_count"], 3)
        self.assertEqual(row["target_candidate_emitted_count"], 2)
        self.assertEqual(row["target_candidate_suppressed_count"], 1)
        self.assertEqual(row["target_candidate_best_rank"], 2)
        self.assertEqual(row["target_candidate_median_rank"], 3)

    def test_missing_provenance_is_not_invented(self) -> None:
        row = self._analysis_row()
        self.assertFalse(row["provenance_available"])
        self.assertEqual(row["target_branch_coherence"], "")
        self.assertEqual(row["emitted_branch_coherence"], "")

    def test_empty_predictions_and_duplicate_columns_are_safe(self) -> None:
        target = SymbolCode(
            (
                SpikeEvent(0, 0.0),
                SpikeEvent(0, 0.2),
                SpikeEvent(2, 0.1),
            )
        )
        row = analyze_oracle_step(
            prediction_input_index=1,
            input_timestamp="",
            horizon_step=1,
            target_timestamp="",
            target_passenger=1.0,
            decoded_prediction="",
            absolute_error="",
            raw_prediction=None,
            emitted_prediction=None,
            competition_result=None,
            target_code=target,
            ranges=FieldColumnRanges.from_sizes(2, 2, 2),
        )

        self.assertEqual(row["target_total_column_count"], 2)
        self.assertEqual(row["raw_target_total_recall"], 0.0)
        self.assertEqual(row["classification"], "TARGET_ABSENT_FROM_RAW")

    def test_classification_thresholds_are_explicit(self) -> None:
        self.assertEqual(
            CLASSIFICATION_THRESHOLDS[
                "mostly_suppressed_survival_ratio_below"
            ],
            0.5,
        )
        classification = classify_step(
            raw_target_recall=1.0,
            emitted_target_recall=0.8,
            passenger_recall=0.8,
            target_survival_ratio=0.8,
            emitted_false_ratio=0.1,
            relative_error=0.5,
            provenance_available=False,
            target_branch_coherence="",
            target_score_max=1.2,
            false_score_max=1.1,
        )
        self.assertEqual(
            classification, "TARGET_SURVIVES_BUT_DECODE_WRONG"
        )

    def test_summary_reports_first_target_loss_step(self) -> None:
        row = self._analysis_row()
        preserved = dict(row)
        preserved["prediction_input_index"] = 1
        preserved["horizon_step"] = 1
        preserved["emitted_target_total_recall"] = 1.0
        lost = dict(preserved)
        lost["horizon_step"] = 2
        lost["emitted_target_total_recall"] = 0.2
        summary = summarize_oracle_rows(
            [preserved, lost],
            horizon=2,
        )

        self.assertEqual(
            summary["first_clear_target_loss_step_distribution"],
            {"2": 1},
        )


if __name__ == "__main__":
    unittest.main()
