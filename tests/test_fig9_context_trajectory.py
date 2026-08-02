from __future__ import annotations

import csv
import gzip
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from experiments.diagnostics.analyze_fig9_context_trajectory_decomposition import (
    analyze,
)
from experiments.diagnostics.fig9_context_trajectory import (
    ColumnTransition,
    ContextTrajectoryTracker,
    LOSS_REASONS,
    _stage_loss,
)
from experiments.diagnostics.fig9_oracle_candidate import FieldColumnRanges
from experiments.fig9.data import TaxiRecord
from experiments.fig9_strict_reproduction import (
    Fig9StrictConfig,
    load_strict_checkpoint,
    model_long_term_fingerprint,
    model_rng_fingerprint,
    run_strict_stream,
)
from seqmem.encoding import SpikeEvent, SymbolCode
from seqmem.model import PreselectionSegmentTrace, PreselectionTrace, Segment


def transition(**overrides: object) -> ColumnTransition:
    values: dict[str, object] = {
        "trajectory_kind": "autonomous_rollout",
        "actual_record_index": 1,
        "effective_record_index": 2,
        "horizon_step": 1,
        "column": 5,
        "segment_inspected": True,
        "overlap_positive": True,
        "response_computed": True,
        "threshold_crossed": True,
        "firing_time_valid": True,
        "saved_candidate": True,
        "intracolumn_selected": True,
        "intercolumn_survived": True,
        "emitted_winner": True,
        "entered_previous_winners": True,
        "present_in_next_context": True,
    }
    values.update(overrides)
    return ColumnTransition(**values)  # type: ignore[arg-type]


class ContextTrajectoryUnitTests(unittest.TestCase):
    def test_loss_stages_are_mutually_exclusive(self) -> None:
        cases = (
            ({"segment_inspected": False}, "NO_INSPECTED_SEGMENT"),
            ({"overlap_positive": False}, "INSPECTED_ZERO_OVERLAP"),
            ({"response_computed": False}, "RESPONSE_BELOW_THRESHOLD"),
            ({"threshold_crossed": False}, "RESPONSE_BELOW_THRESHOLD"),
            ({"firing_time_valid": False}, "CROSSING_INVALID_TIME"),
            ({"saved_candidate": False}, "CANDIDATE_NOT_SAVED"),
            ({"intracolumn_selected": False}, "LOST_INTRACOLUMN"),
            ({"intercolumn_survived": False}, "LOST_INTERCOLUMN"),
            ({"emitted_winner": False}, "LOST_INTERCOLUMN"),
            (
                {"entered_previous_winners": False},
                "EMITTED_NOT_PROPAGATED",
            ),
        )
        for overrides, expected in cases:
            with self.subTest(expected=expected):
                reason = _stage_loss(transition(**overrides))
                self.assertEqual(reason, expected)
                self.assertIn(reason, LOSS_REASONS)

    def test_actual_and_autonomous_histories_remain_separate(self) -> None:
        tracker = ContextTrajectoryTracker()
        model = SimpleNamespace(
            columns=[
                SimpleNamespace(neurons=[object(), object()])
                for _ in range(10)
            ]
        )
        segment = Segment()
        item = PreselectionSegmentTrace(
            target_column=5,
            target_neuron=0,
            segment_index=0,
            inspection_order=0,
            segment=segment,
            segment_identity=id(segment),
            active_source_count=1,
            active_matched_synapse_count=1,
            sum_active_weights=1.0,
            dendritic_threshold=1.0,
            response_computed=True,
            crossed_threshold=True,
            valid_firing_time=True,
            became_prediction_candidate=True,
        )
        raw = SymbolCode(events=(SpikeEvent(column=5, time=0.1),))
        for kind, active in (
            ("actual_observation", {}),
            ("autonomous_rollout", {10: 0.1}),
        ):
            tracker.record_transition(
                model=model,  # type: ignore[arg-type]
                trajectory_kind=kind,
                actual_record_index=1,
                horizon_step=1 if kind == "autonomous_rollout" else 0,
                preselection_trace=PreselectionTrace(segments=[item]),
                raw_code=raw,
                competition_result=None,
                next_active_cells=active,
                next_winners=active,
            )
        source = {
            "actual_record_index": 3,
            "timestamp": "2014-07-01 01:30:00",
            "field": "passenger",
            "encoded_column": 100,
            "observe_scenario": "scenario3",
            "segment_provenance_id": "segment-1",
            "segment_creation_transition_index": 0,
            "source_field": "weekday",
            "source_column": 5,
            "source_neuron": 0,
            "source_cell_stable_id": 10,
            "source_loss_reason_primary": "SOURCE_COLUMN_NOT_ACTIVE",
        }
        rows = tracker.dependency_rows(
            source_rows=[source],
            ranges=FieldColumnRanges.from_sizes(30, 58, 482),
            level="column",
        )
        self.assertEqual(len(rows), 2)
        by_kind = {row["trajectory_kind"]: row for row in rows}
        self.assertEqual(
            by_kind["actual_observation"]["most_recent_loss_stage"],
            "EMITTED_NOT_PROPAGATED",
        )
        self.assertTrue(
            by_kind["autonomous_rollout"]["present_in_next_context"]
        )
        self.assertEqual(
            by_kind["actual_observation"]["source_loss_reason_joined"],
            "SOURCE_COLUMN_NOT_ACTIVE",
        )

    def test_only_inactive_source_rows_are_joined(self) -> None:
        tracker = ContextTrajectoryTracker()
        rows = tracker.dependency_rows(
            source_rows=[
                {
                    "source_loss_reason_primary": "SOURCE_EXACT_CELL_MATCHED",
                }
            ],
            ranges=FieldColumnRanges.from_sizes(30, 58, 482),
            level="column",
        )
        self.assertEqual(rows, [])


class ContextTrajectoryAnalyzerTests(unittest.TestCase):
    def test_analyzer_writes_required_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            output_dir = root / "analysis"
            run_dir.mkdir()
            rows = []
            for kind, reason, pattern in (
                (
                    "actual_observation",
                    "EMITTED_NOT_PROPAGATED",
                    "ACTIVE_THEN_LOST",
                ),
                (
                    "autonomous_rollout",
                    "LOST_INTERCOLUMN",
                    "ACTIVE_THEN_LOST",
                ),
            ):
                row = {
                    "actual_record_index": 220,
                    "observed_column": 100,
                    "segment_id": "segment-1",
                    "source_column": 5,
                    "source_neuron": "",
                    "source_field": "weekday",
                    "trajectory_kind": kind,
                    "history_available": True,
                    "first_loss_stage": reason,
                    "most_recent_loss_stage": reason,
                    "latest_loss_stage": reason,
                    "most_recent_loss_horizon_step": 1,
                    "trajectory_pattern": pattern,
                    "trajectory_loss_primary_reason": reason,
                    "recovery_count": 0,
                    **{stage: stage not in {"intercolumn_survived", "present_in_next_context"} for stage in (
                        "segment_inspected",
                        "overlap_positive",
                        "response_computed",
                        "threshold_crossed",
                        "firing_time_valid",
                        "saved_candidate",
                        "intracolumn_selected",
                        "intercolumn_survived",
                        "emitted_winner",
                        "entered_previous_winners",
                        "present_in_next_context",
                    )},
                }
                rows.append(row)
            path = run_dir / "context_trajectory_column_trace.csv.gz"
            with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            summary = analyze(
                run_dir=run_dir,
                output_dir=output_dir,
                bootstrap_samples=20,
                bootstrap_seed=0,
            )
            self.assertEqual(
                summary["recommended_next_step"],
                "INTERCOLUMN_RETENTION_FIX_REQUIRED",
            )
            for name in (
                "context_trajectory_stage_funnel.csv",
                "context_trajectory_loss_reason_summary.csv",
                "context_trajectory_pattern_summary.csv",
                "context_trajectory_latest_loss_summary.csv",
                "context_trajectory_source_field_summary.csv",
                "context_trajectory_horizon_summary.csv",
                "context_trajectory_recovery_summary.csv",
                "context_trajectory_join_coverage.json",
                "context_trajectory_bootstrap_ci.csv",
                "FIG9_CONTEXT_TRAJECTORY_DECOMPOSITION_REPORT.md",
            ):
                self.assertTrue((output_dir / name).exists(), name)


class ContextTrajectoryRunnerTests(unittest.TestCase):
    def test_runner_diagnostic_is_noninterfering(self) -> None:
        records = [
            TaxiRecord(
                datetime(
                    2014,
                    7,
                    1,
                    hour=(index // 2) % 24,
                    minute=30 * (index % 2),
                ),
                100.0 + index,
            )
            for index in range(10)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_path = root / "tiny.csv"
            data_path.write_text(
                "timestamp,passenger_count\n",
                encoding="utf-8",
            )
            config = Fig9StrictConfig(warmup=4, horizon=1)
            common = {
                "records": records,
                "data_path": data_path,
                "stream_label": "original",
                "config": config,
                "limit": 10,
                "print_fingerprint": False,
                "match_overlap_diagnostic": True,
                "match_overlap_level": "source",
                "match_overlap_compress": True,
            }
            plain = run_strict_stream(
                output_dir=root / "plain",
                checkpoint_path=root / "plain.pkl",
                checkpoint_at_index=9,
                **common,
            )
            traced = run_strict_stream(
                output_dir=root / "traced",
                checkpoint_path=root / "traced.pkl",
                checkpoint_at_index=9,
                context_trajectory_diagnostic=True,
                context_trajectory_level="column",
                **common,
            )
            streamed = run_strict_stream(
                output_dir=root / "streamed",
                checkpoint_path=root / "streamed.pkl",
                checkpoint_at_index=9,
                context_trajectory_diagnostic=True,
                context_trajectory_level="column",
                stream_diagnostic_traces=True,
                **common,
            )
            uncompressed = run_strict_stream(
                output_dir=root / "uncompressed",
                checkpoint_path=root / "uncompressed.pkl",
                checkpoint_at_index=9,
                context_trajectory_diagnostic=True,
                context_trajectory_level="column",
                context_trajectory_compress=False,
                **common,
            )
            self.assertEqual(
                (root / "plain" / "original_predictions.csv").read_bytes(),
                (root / "traced" / "original_predictions.csv").read_bytes(),
            )
            self.assertEqual(
                (
                    root / "plain" / "match_overlap_column_trace.csv"
                ).read_bytes(),
                (
                    root / "traced" / "match_overlap_column_trace.csv"
                ).read_bytes(),
            )
            for field in (
                "mape",
                "coverage",
                "final_model_fingerprint",
                "final_rng_fingerprint",
                "final_segment_count",
            ):
                self.assertEqual(plain.get(field), traced.get(field))
                self.assertEqual(traced.get(field), streamed.get(field))
                self.assertEqual(traced.get(field), uncompressed.get(field))
            self.assertEqual(
                (root / "traced" / "original_predictions.csv").read_bytes(),
                (root / "streamed" / "original_predictions.csv").read_bytes(),
            )
            with gzip.open(
                root / "traced" / "context_trajectory_column_trace.csv.gz",
                "rt",
                encoding="utf-8",
            ) as left, gzip.open(
                root / "streamed" / "context_trajectory_column_trace.csv.gz",
                "rt",
                encoding="utf-8",
            ) as right:
                self.assertEqual(left.read(), right.read())
            with gzip.open(
                root / "traced" / "context_trajectory_column_trace.csv.gz",
                "rt",
                encoding="utf-8",
            ) as left:
                self.assertEqual(
                    left.read(),
                    (
                        root
                        / "uncompressed"
                        / "context_trajectory_column_trace.csv"
                    ).read_text(encoding="utf-8"),
                )
            for path, summary in (
                (root / "plain.pkl", plain),
                (root / "traced.pkl", traced),
            ):
                checkpoint = load_strict_checkpoint(path)
                self.assertEqual(
                    model_long_term_fingerprint(checkpoint["model"]),
                    summary["final_model_fingerprint"],
                )
                self.assertEqual(
                    model_rng_fingerprint(checkpoint["model"]),
                    summary["final_rng_fingerprint"],
                )
            self.assertTrue(
                (
                    root
                    / "traced"
                    / "context_trajectory_protocol.json"
                ).exists()
            )


if __name__ == "__main__":
    unittest.main()
