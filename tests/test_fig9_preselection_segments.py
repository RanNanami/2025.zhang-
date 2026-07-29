from __future__ import annotations

import csv
import gzip
import hashlib
import inspect
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from experiments.diagnostics.analyze_fig9_preselection_segments import analyze
from experiments.diagnostics.fig9_branch_provenance import (
    BranchProvenanceRegistry,
)
from experiments.diagnostics.fig9_competitive_inhibition import (
    CompetitionSettings,
)
from experiments.diagnostics.fig9_preselection_segments import (
    _replacement_rows,
    build_preselection_rows,
    stable_selection_group_id,
)
from experiments.fig9 import TaxiRecord
from experiments.fig9_strict_reproduction import (
    Fig9StrictConfig,
    branch_checkpoint_sidecar_path,
    load_strict_checkpoint,
    model_long_term_fingerprint,
    model_rng_fingerprint,
    run_strict_stream,
)
from seqmem.model import (
    PreselectionReplacementTrace,
    PreselectionSegmentTrace,
    PreselectionTrace,
    Segment,
    SequentialMemory,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_csv(path: Path) -> list[dict[str, str]]:
    runtime_fields = {
        "prediction_runtime_seconds",
        "decode_runtime_seconds",
        "competition_runtime_seconds",
        "observe_runtime",
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            {
                key: value
                for key, value in row.items()
                if key not in runtime_fields
            }
            for row in csv.DictReader(handle)
        ]


class Fig9PreselectionUnitTests(unittest.TestCase):
    def test_stable_group_id_uses_every_fingerprint_input(self) -> None:
        arguments = {
            "input_index": 4,
            "horizon_step": 2,
            "group_type": "column_time_event",
            "group_column": 8,
            "group_neuron": None,
            "stable_time_key": 1.25,
        }
        original = stable_selection_group_id(**arguments)
        self.assertEqual(original, stable_selection_group_id(**arguments))
        for key, replacement in (
            ("input_index", 5),
            ("horizon_step", 3),
            ("group_type", "column"),
            ("group_column", 9),
            ("group_neuron", 1),
            ("stable_time_key", 1.30),
        ):
            changed = dict(arguments)
            changed[key] = replacement
            self.assertNotEqual(
                original,
                stable_selection_group_id(**changed),
            )

    def test_offline_mapping_does_not_predict_observe_or_learn(self) -> None:
        source = inspect.getsource(build_preselection_rows)
        for forbidden in (
            "predict_code(",
            ".observe(",
            ".observe_code(",
            "learn_actual_code(",
        ):
            self.assertNotIn(forbidden, source)

    def test_ground_truth_is_absent_from_prediction_signature(self) -> None:
        parameters = inspect.signature(SequentialMemory.predict_code).parameters
        for forbidden in ("target", "expected", "ground_truth", "oracle"):
            self.assertNotIn(forbidden, parameters)

    def test_internal_replacement_cannot_change_column_label(self) -> None:
        first_segment = Segment()
        second_segment = Segment()
        first = PreselectionSegmentTrace(
            target_column=3,
            target_neuron=0,
            segment_index=0,
            inspection_order=0,
            segment=first_segment,
            segment_identity=id(first_segment),
            active_source_count=1,
            active_matched_synapse_count=1,
            sum_active_weights=1.0,
            dendritic_threshold=1.0,
        )
        second = PreselectionSegmentTrace(
            target_column=3,
            target_neuron=1,
            segment_index=1,
            inspection_order=1,
            segment=second_segment,
            segment_identity=id(second_segment),
            active_source_count=1,
            active_matched_synapse_count=1,
            sum_active_weights=1.0,
            dendritic_threshold=1.0,
        )
        replacements = [
            PreselectionReplacementTrace(
                "column_time_event",
                3,
                None,
                1.1,
                first_segment,
                second_segment,
                id(first_segment),
                id(second_segment),
                "score",
                1.0,
                1.1,
                (0, 0),
                (0, 1),
                "EVENT_SCORE_SELECTION",
            ),
        ]
        rows = _replacement_rows(
            replacements=replacements,
            trace=PreselectionTrace(
                segments=[first, second],
                replacements=replacements,
            ),
            registry=BranchProvenanceRegistry(),
            input_index=1,
            horizon_step=1,
            target_columns={3},
        )
        self.assertTrue(rows[0]["previous_is_target"])
        self.assertTrue(rows[0]["replacement_is_target"])

    def test_empty_analysis_is_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = analyze(
                run_dir=root / "missing",
                output_dir=root / "analysis",
                policy_label="batched",
                bootstrap_samples=2,
                bootstrap_seed=0,
            )
            self.assertTrue(summary["empty_trace"])
            self.assertEqual(summary["inspected_segments"], 0)


class Fig9PreselectionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.data_path = cls.root / "tiny.csv"
        cls.data_path.write_text(
            "timestamp,passenger_count\n",
            encoding="utf-8",
        )
        start = datetime(2014, 7, 1)
        cls.records = [
            TaxiRecord(
                start + timedelta(minutes=30 * index),
                1000.0 + index * 20.0,
            )
            for index in range(18)
        ]
        settings = CompetitionSettings(
            mode="competitive_raw",
            inhibition_strength=0.1,
            inhibition_tau=0.02,
            simultaneous_policy="batched",
            simultaneous_bin_width=0.005,
        )
        cls.results = {}
        for label, enabled in (("off", False), ("on", True)):
            directory = cls.root / label
            checkpoint = directory / "checkpoint.pkl"
            cls.results[label] = run_strict_stream(
                records=cls.records,
                data_path=cls.data_path,
                stream_label="original",
                output_dir=directory,
                config=Fig9StrictConfig(warmup=6),
                limit=0,
                print_fingerprint=False,
                competition_settings=settings,
                oracle_candidate_diagnostic=True,
                branch_provenance_diagnostic=True,
                branch_provenance_level="candidate",
                preselection_segment_diagnostic=enabled,
                preselection_segment_level="crossing",
                checkpoint_path=checkpoint,
                checkpoint_at_index=10,
            )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_on_off_preserves_all_outputs_and_fingerprints(self) -> None:
        off = self.root / "off"
        on = self.root / "on"
        self.assertEqual(
            _sha256(off / "original_predictions.csv"),
            _sha256(on / "original_predictions.csv"),
        )
        self.assertEqual(
            _stable_csv(off / "competition_trace.csv"),
            _stable_csv(on / "competition_trace.csv"),
        )
        self.assertEqual(
            (off / "oracle_candidate_trace.csv").read_bytes(),
            (on / "oracle_candidate_trace.csv").read_bytes(),
        )
        self.assertEqual(
            (off / "branch_candidate_trace.csv").read_bytes(),
            (on / "branch_candidate_trace.csv").read_bytes(),
        )
        self.assertEqual(
            (off / "branch_segment_trace.csv").read_bytes(),
            (on / "branch_segment_trace.csv").read_bytes(),
        )
        self.assertEqual(
            self.results["off"]["final_model_fingerprint"],
            self.results["on"]["final_model_fingerprint"],
        )
        self.assertEqual(
            self.results["off"]["final_rng_fingerprint"],
            self.results["on"]["final_rng_fingerprint"],
        )
        off_protocol = json.loads(
            (off / "original_protocol.json").read_text(encoding="utf-8")
        )
        on_protocol = json.loads(
            (on / "original_protocol.json").read_text(encoding="utf-8")
        )
        self.assertEqual(off_protocol, on_protocol)
        self.assertFalse(
            any("preselection" in key for key in on_protocol)
        )

    def test_funnel_counts_are_monotonic_and_accounted(self) -> None:
        path = self.root / "on" / "preselection_funnel_trace.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(rows)
        for row in rows:
            inspected = int(row["inspected_segment_count"])
            computed = int(row["response_computed_count"])
            positive = int(row["positive_response_count"])
            crossed = int(row["threshold_crossing_count"])
            event_winners = int(row["event_winner_count"])
            candidates = int(row["saved_candidate_count"])
            self.assertGreaterEqual(inspected, computed)
            self.assertGreaterEqual(computed, positive)
            self.assertGreaterEqual(positive, crossed)
            self.assertGreaterEqual(crossed, event_winners)
            self.assertGreaterEqual(event_winners, candidates)
            self.assertEqual(int(row["unknown_elimination_count"]), 0)

    def test_crossing_schema_and_saved_candidate_mapping(self) -> None:
        directory = self.root / "on"
        with (directory / "preselection_segment_trace.csv").open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            segments = list(csv.DictReader(handle))
        self.assertTrue(segments)
        self.assertTrue(all(row["crossed_threshold"] == "True" for row in segments))
        candidates = [
            row
            for row in segments
            if row["became_prediction_candidate"] == "True"
        ]
        self.assertTrue(candidates)
        self.assertTrue(
            all(row["segment_provenance_id"] for row in candidates)
        )
        self.assertTrue(
            all(row["furthest_stage_reached"] == "PREDICTION_CANDIDATE"
                for row in candidates)
        )
        self.assertTrue(
            all(row["is_target_neuron_when_definable"] == ""
                for row in segments)
        )
        with (directory / "preselection_group_trace.csv").open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            groups = list(csv.DictReader(handle))
        self.assertTrue(groups)
        self.assertTrue(
            all(row["mixed_target_false_group"] == "False" for row in groups)
        )

    def test_checkpoint_v1_and_stable_sidecar_ids(self) -> None:
        off_path = self.root / "off" / "checkpoint.pkl"
        on_path = self.root / "on" / "checkpoint.pkl"
        off = load_strict_checkpoint(off_path)
        on = load_strict_checkpoint(on_path)
        self.assertEqual(off["checkpoint_format"], "fig9-strict-v1")
        self.assertEqual(on["checkpoint_format"], "fig9-strict-v1")
        self.assertEqual(
            model_long_term_fingerprint(off["model"]),  # type: ignore[arg-type]
            model_long_term_fingerprint(on["model"]),  # type: ignore[arg-type]
        )
        self.assertEqual(
            model_rng_fingerprint(off["model"]),  # type: ignore[arg-type]
            model_rng_fingerprint(on["model"]),  # type: ignore[arg-type]
        )
        left = json.loads(
            branch_checkpoint_sidecar_path(off_path).read_text(
                encoding="utf-8"
            )
        )
        right = json.loads(
            branch_checkpoint_sidecar_path(on_path).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(left, right)

    def test_protocol_and_analysis_outputs(self) -> None:
        directory = self.root / "on"
        protocol = json.loads(
            (directory / "preselection_protocol.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(protocol["diagnostic_only"])
        self.assertFalse(protocol["trace_truncated"])
        self.assertIn("no independent per-neuron", protocol["nonexistent_stage"])
        output = self.root / "analysis"
        summary = analyze(
            run_dir=directory,
            output_dir=output,
            policy_label="batched",
            bootstrap_samples=10,
            bootstrap_seed=0,
        )
        self.assertFalse(summary["empty_trace"])
        for name in (
            "preselection_stage_summary.csv",
            "preselection_field_summary.csv",
            "preselection_horizon_summary.csv",
            "target_segment_survival.csv",
            "target_loss_stage_summary.csv",
            "internal_group_competition.csv",
            "segment_score_separability.csv",
            "segment_context_separability.csv",
            "winner_vs_discarded_summary.csv",
            "preselection_bootstrap_ci.csv",
            "preselection_analysis_summary.json",
            "FIG9_PRESELECTION_SEGMENT_REPORT.md",
        ):
            self.assertTrue((output / name).exists(), name)

    def test_summary_full_and_gzip_levels(self) -> None:
        settings = CompetitionSettings(
            mode="competitive_raw",
            inhibition_strength=0.1,
            inhibition_tau=0.02,
            simultaneous_policy="batched",
            simultaneous_bin_width=0.005,
        )
        summary_dir = self.root / "summary_level"
        run_strict_stream(
            records=self.records[:14],
            data_path=self.data_path,
            stream_label="original",
            output_dir=summary_dir,
            config=Fig9StrictConfig(warmup=6),
            limit=0,
            print_fingerprint=False,
            competition_settings=settings,
            oracle_candidate_diagnostic=True,
            preselection_segment_diagnostic=True,
            preselection_segment_level="summary",
        )
        self.assertTrue(
            (summary_dir / "preselection_funnel_trace.csv").exists()
        )
        self.assertFalse(
            (summary_dir / "preselection_segment_trace.csv").exists()
        )
        full_dir = self.root / "full_level"
        run_strict_stream(
            records=self.records[:14],
            data_path=self.data_path,
            stream_label="original",
            output_dir=full_dir,
            config=Fig9StrictConfig(warmup=6),
            limit=0,
            print_fingerprint=False,
            competition_settings=settings,
            oracle_candidate_diagnostic=True,
            preselection_segment_diagnostic=True,
            preselection_segment_level="full",
            preselection_segment_compress=True,
        )
        compressed = full_dir / "preselection_segment_trace.csv.gz"
        self.assertTrue(compressed.exists())
        with gzip.open(
            compressed,
            "rt",
            encoding="utf-8",
            newline="",
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(rows)
        self.assertTrue(any(row["crossed_threshold"] == "False" for row in rows))
        truncated_dir = self.root / "truncated_level"
        run_strict_stream(
            records=self.records[:14],
            data_path=self.data_path,
            stream_label="original",
            output_dir=truncated_dir,
            config=Fig9StrictConfig(warmup=6),
            limit=0,
            print_fingerprint=False,
            competition_settings=settings,
            oracle_candidate_diagnostic=True,
            preselection_segment_diagnostic=True,
            preselection_segment_level="full",
            preselection_max_rows=5,
        )
        protocol = json.loads(
            (truncated_dir / "preselection_protocol.json").read_text(
                encoding="utf-8"
            )
        )
        summary = json.loads(
            (truncated_dir / "preselection_summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(protocol["trace_truncated"])
        self.assertEqual(protocol["requested_max_rows"], 5)
        self.assertEqual(protocol["actual_rows"]["segments"], 5)
        self.assertGreater(
            summary["total_segment_rows_before_truncation"],
            summary["actual_segment_rows"],
        )


if __name__ == "__main__":
    unittest.main()
