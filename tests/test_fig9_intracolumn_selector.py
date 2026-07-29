from __future__ import annotations

import inspect
import csv
import json
import math
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from experiments.diagnostics.fig9_intracolumn_selector import (
    mark_competition_outcomes,
    selection_trace_rows,
    summarize_selection_rows,
)
from experiments.diagnostics.analyze_fig9_intracolumn_selector import analyze
from experiments.diagnostics.fig9_oracle_candidate import FieldColumnRanges
from experiments.diagnostics.fig9_competitive_inhibition import (
    CompetitionSettings,
)
from experiments.fig9 import TaxiRecord
from experiments.fig9_strict_reproduction import (
    Fig9StrictConfig,
    run_strict_stream,
)
from seqmem.encoding import SSTDDiscreteEncoder
from seqmem.model import (
    IntracolumnSelectionTrace,
    Segment,
    SequentialMemory,
    Synapse,
    SynapsePSPContribution,
)


class IntracolumnSelectorUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = SequentialMemory(
            SSTDDiscreteEncoder(num_columns=4, k=1, seed=3),
            num_neurons_per_column=4,
        )
        self.source_a = self.model._cell_id(3, 0)
        self.source_b = self.model._cell_id(3, 1)
        self.active = {self.source_a: 0.0, self.source_b: 0.0}
        self.segments = [
            Segment(
                synapses={
                    self.source_a: Synapse(self.source_a, weight=0.6),
                }
            ),
            Segment(
                synapses={
                    self.source_a: Synapse(self.source_a, weight=0.6),
                    self.source_b: Synapse(self.source_b, weight=0.6),
                }
            ),
            Segment(
                synapses={
                    self.source_b: Synapse(self.source_b, weight=0.6),
                }
            ),
        ]
        for neuron, segment in enumerate(self.segments):
            self.model.columns[0].neurons[neuron].segments.append(segment)

    def _metadata(self) -> dict[int, dict[str, object]]:
        counts = (1, 3, 2)
        peaks = (1.1, 1.4, 1.7)
        return {
            id(segment): {
                "peak_dendritic_potential": peaks[index],
                "crossing_synapse_contributions": tuple(
                    SynapsePSPContribution(
                        source_cell_id=self.source_a,
                        source_time=0.0,
                        arrival_time=0.5,
                        weight=0.6,
                        psp_contribution=0.5,
                    )
                    for _ in range(counts[index])
                ),
            }
            for index, segment in enumerate(self.segments)
        }

    def _pool(self):
        return {
            (0, 1.3): (0, 1.1, 1.30, self.segments[0]),
            (0, 1.2): (1, 1.5, 1.20, self.segments[1]),
            (0, 1.1): (2, 1.2, 1.10, self.segments[2]),
        }

    def _select(self, policy: str):
        trace = IntracolumnSelectionTrace()
        selected = self.model._select_intracolumn_event_winners(
            best_by_event=self._pool(),
            policy=policy,
            prediction_metadata=self._metadata(),
            active_sources=self.active,
            selection_trace=trace,
        )
        return selected[0][1][0], trace

    def test_each_policy_uses_declared_key(self) -> None:
        self.assertEqual(self._select("existing")[0], 2)
        self.assertEqual(self._select("max_candidate_score")[0], 1)
        self.assertEqual(self._select("max_response_peak")[0], 2)
        self.assertEqual(self._select("max_contributor_count")[0], 1)
        self.assertEqual(self._select("context_then_score")[0], 1)

    def test_single_candidate_is_identical_for_every_policy(self) -> None:
        pool = {(0, 1.2): (1, 1.5, 1.2, self.segments[1])}
        for policy in (
            "existing",
            "max_candidate_score",
            "max_response_peak",
            "max_contributor_count",
            "context_then_score",
        ):
            selected = self.model._select_intracolumn_event_winners(
                best_by_event=pool,
                policy=policy,
                prediction_metadata=self._metadata(),
                active_sources=self.active,
                selection_trace=None,
            )
            self.assertEqual(selected, [(0, pool[(0, 1.2)])])

    def test_empty_pool_is_safe(self) -> None:
        selected = self.model._select_intracolumn_event_winners(
            best_by_event={},
            policy="max_candidate_score",
            prediction_metadata={},
            active_sources=self.active,
            selection_trace=IntracolumnSelectionTrace(),
        )
        self.assertEqual(selected, [])

    def test_equal_metrics_keep_original_stable_order(self) -> None:
        pool = {
            (0, 1.2): (0, 1.5, 1.2, self.segments[0]),
            (0, 1.3): (1, 1.5, 1.2, self.segments[1]),
        }
        metadata = {
            id(segment): {
                "peak_dendritic_potential": 1.5,
                "crossing_synapse_contributions": (),
            }
            for segment in self.segments[:2]
        }
        for policy in (
            "max_candidate_score",
            "max_response_peak",
            "max_contributor_count",
        ):
            trace = IntracolumnSelectionTrace()
            selected = self.model._select_intracolumn_event_winners(
                best_by_event=pool,
                policy=policy,
                prediction_metadata=metadata,
                active_sources={},
                selection_trace=trace,
            )
            self.assertEqual(selected[0][1][0], 0)
            self.assertTrue(trace.groups[0].tie_break_used)

    def test_nonfinite_metric_sorts_after_finite(self) -> None:
        metadata = self._metadata()
        metadata[id(self.segments[1])]["peak_dendritic_potential"] = math.nan
        selected = self.model._select_intracolumn_event_winners(
            best_by_event=self._pool(),
            policy="max_response_peak",
            prediction_metadata=metadata,
            active_sources=self.active,
            selection_trace=IntracolumnSelectionTrace(),
        )
        self.assertEqual(selected[0][1][0], 2)

    def test_trace_projection_has_no_posthoc_target_label(self) -> None:
        _neuron, trace = self._select("max_candidate_score")
        rows = selection_trace_rows(
            trace=trace,
            policy="max_candidate_score",
            input_index=7,
            input_timestamp="2014-01-01 00:00:00",
            horizon_step=1,
            ranges=FieldColumnRanges.from_sizes(1, 1, 2),
        )
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["post_hoc_labels_present"])
        self.assertNotIn("target_column", rows[0])
        mark_competition_outcomes(rows, emitted_columns={0})
        self.assertTrue(rows[0]["emitted_after_competition"])
        self.assertFalse(rows[0]["suppressed_intercolumn"])
        summary = summarize_selection_rows(
            rows, policy="max_candidate_score"
        )
        self.assertTrue(summary["raw_reachable_column_set_parity"])
        self.assertFalse(summary["uses_ground_truth"])

    def test_selector_source_does_not_read_ground_truth_or_rng(self) -> None:
        source = inspect.getsource(
            SequentialMemory._select_intracolumn_event_winners
        )
        self.assertNotIn("ground_truth", source)
        self.assertNotIn("teacher", source)
        self.assertNotIn("_learning_rng", source)
        self.assertNotIn("_decode_rng", source)


class IntracolumnSelectorIntegrationTests(unittest.TestCase):
    @staticmethod
    def _records(count: int = 18) -> list[TaxiRecord]:
        start = datetime(2014, 7, 1)
        return [
            TaxiRecord(start, 1000.0)
            for index in range(count)
        ]

    def _run(self, directory: Path, *, policy: str, diagnostic: bool):
        config = Fig9StrictConfig(warmup=4)
        data_path = directory.parent / "tiny.csv"
        if not data_path.exists():
            data_path.write_text(
                "timestamp,passenger_count\n", encoding="utf-8"
            )
        return run_strict_stream(
            records=self._records(),
            data_path=data_path,
            stream_label="original",
            output_dir=directory,
            config=config,
            limit=18,
            competition_settings=CompetitionSettings(
                mode="competitive_raw",
                simultaneous_policy="batched",
            ),
            oracle_candidate_diagnostic=True,
            intracolumn_selection_policy=policy,
            intracolumn_selection_diagnostic=diagnostic,
            print_fingerprint=False,
        )

    @staticmethod
    def _stable_prediction_rows(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            for key in tuple(row):
                if "runtime" in key:
                    row.pop(key)
        return rows

    def test_existing_diagnostic_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            off = root / "off"
            on = root / "on"
            off.mkdir()
            on.mkdir()
            off_summary = self._run(off, policy="existing", diagnostic=False)
            on_summary = self._run(on, policy="existing", diagnostic=True)

            self.assertEqual(off_summary["mape"], on_summary["mape"])
            self.assertEqual(off_summary["coverage"], on_summary["coverage"])
            self.assertEqual(
                (off / "original_predictions.csv").read_bytes(),
                (on / "original_predictions.csv").read_bytes(),
            )
            self.assertEqual(
                off_summary["final_model_fingerprint"],
                on_summary["final_model_fingerprint"],
            )
            self.assertEqual(
                off_summary["final_rng_fingerprint"],
                on_summary["final_rng_fingerprint"],
            )
            self.assertEqual(
                self._stable_prediction_rows(off / "competition_trace.csv"),
                self._stable_prediction_rows(on / "competition_trace.csv"),
            )
            self.assertEqual(
                self._stable_prediction_rows(
                    off / "oracle_candidate_trace.csv"
                ),
                self._stable_prediction_rows(
                    on / "oracle_candidate_trace.csv"
                ),
            )
            self.assertNotIn(
                "intracolumn_selection_policy",
                (on / "original_protocol.json").read_text(encoding="utf-8"),
            )
            protocol = json.loads(
                (on / "intracolumn_selection_protocol.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(protocol["diagnostic_only"])
            self.assertEqual(protocol["policy"], "existing")

    def test_alternative_policy_preserves_training_and_rng(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = root / "left"
            right = root / "right"
            left.mkdir()
            right.mkdir()
            self._run(
                left, policy="max_candidate_score", diagnostic=True
            )
            self._run(
                right, policy="max_candidate_score", diagnostic=True
            )
            self.assertEqual(
                self._stable_prediction_rows(
                    left / "original_predictions.csv"
                ),
                self._stable_prediction_rows(
                    right / "original_predictions.csv"
                ),
            )
            self.assertEqual(
                json.loads(
                    (
                        left / "intracolumn_selection_summary.json"
                    ).read_text(encoding="utf-8")
                ),
                json.loads(
                    (
                        right / "intracolumn_selection_summary.json"
                    ).read_text(encoding="utf-8")
                ),
            )
            left_summary = json.loads(
                (left / "original_summary.json").read_text(encoding="utf-8")
            )
            right_summary = json.loads(
                (right / "original_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                left_summary["final_model_fingerprint"],
                right_summary["final_model_fingerprint"],
            )
            self.assertEqual(
                left_summary["final_rng_fingerprint"],
                right_summary["final_rng_fingerprint"],
            )

    def test_offline_analyzer_writes_required_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            baseline_dir = root / "baseline"
            output_dir = root / "analysis"
            run_dir.mkdir()
            baseline_dir.mkdir()
            selector_row = {
                "input_index": 21,
                "horizon_step": 1,
                "column": 100,
                "field": "passenger",
                "group_candidate_count": 2,
                "selected_differs_from_existing": True,
                "tie_break_used": False,
                "candidate_pool_fingerprint": "same-pool",
            }
            for directory in (run_dir, baseline_dir):
                with (
                    directory / "intracolumn_selection_trace.csv"
                ).open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle, fieldnames=list(selector_row)
                    )
                    writer.writeheader()
                    writer.writerow(selector_row)
                with (directory / "competition_trace.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    density = {
                        "horizon_step": 1,
                        "raw_predicted_column_count": 2,
                        "emitted_column_count": 1,
                        "absolute_error": 10,
                        "target": 100,
                        "prediction_missing": False,
                    }
                    writer = csv.DictWriter(handle, fieldnames=list(density))
                    writer.writeheader()
                    writer.writerow(density)
                (directory / "original_summary.json").write_text(
                    json.dumps(
                        {
                            "mape": 0.1,
                            "coverage": 1.0,
                            "final_rolling_mape": 0.2,
                            "mean_raw_column_count": 2.0,
                            "peak_raw_column_count": 2,
                        }
                    ),
                    encoding="utf-8",
                )

            result = analyze(
                run_dir=run_dir,
                baseline_dir=baseline_dir,
                output_dir=output_dir,
                policy_label="max_candidate_score",
                bootstrap_samples=10,
                bootstrap_seed=0,
            )
            self.assertTrue(result["step1_candidate_pool_parity"])
            for name in (
                "intracolumn_policy_summary.csv",
                "intracolumn_horizon_summary.csv",
                "intracolumn_field_summary.csv",
                "intracolumn_identity_summary.csv",
                "intracolumn_density_summary.csv",
                "intracolumn_policy_comparison.csv",
                "intracolumn_bootstrap_ci.csv",
                "INTRACOLUMN_SELECTOR_REPORT.md",
            ):
                self.assertTrue((output_dir / name).exists(), name)
