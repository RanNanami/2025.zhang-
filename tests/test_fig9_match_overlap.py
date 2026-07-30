from __future__ import annotations

import csv
import gzip
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from experiments.diagnostics.analyze_fig9_match_overlap_decomposition import (
    analyze,
    classify_record_range,
)
from experiments.diagnostics.analyze_fig9_timing_eligibility_decomposition import (
    analyze as analyze_timing_eligibility,
)
from experiments.diagnostics.fig9_branch_provenance import (
    BranchProvenanceRegistry,
)
from experiments.diagnostics.fig9_match_overlap import (
    DIAGNOSTIC_MARKERS,
    MATCH_OVERLAP_LEVELS,
    SourceTraceFilter,
    capture_match_overlap,
    classify_source_loss,
    classify_timing_eligibility_loss,
    filter_source_trace_rows,
    finalize_match_overlap,
)
from experiments.diagnostics.fig9_oracle_candidate import FieldColumnRanges
from experiments.fig9 import TaxiRecord
from experiments.fig9_strict_reproduction import (
    Fig9StrictConfig,
    load_strict_checkpoint,
    model_long_term_fingerprint,
    model_rng_fingerprint,
    run_strict_stream,
)
from seqmem.encoding import SSTDDiscreteEncoder, SpikeEvent, SymbolCode
from seqmem.model import (
    ObservationEventTrace,
    ObservationTrace,
    Segment,
    SequentialMemory,
    Synapse,
)


class MatchOverlapCaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = SequentialMemory(
            SSTDDiscreteEncoder(num_columns=6, k=1, seed=4),
            num_neurons_per_column=2,
        )
        self.ranges = FieldColumnRanges.from_sizes(2, 2, 2)
        self.record = TaxiRecord(datetime(2014, 7, 1, 0, 0), 100.0)
        self.code = SymbolCode((SpikeEvent(4, 0.0),))
        self.source_weekday = self.model._cell_id(0, 0)
        self.source_time = self.model._cell_id(2, 0)
        self.source_passenger = self.model._cell_id(5, 0)
        self.wrong_passenger = self.model._cell_id(5, 1)
        self.segment = Segment(
            synapses={
                self.source_weekday: Synapse(
                    self.source_weekday,
                    delay=0.5,
                    weight=0.5,
                ),
                self.source_time: Synapse(
                    self.source_time,
                    delay=0.5,
                    weight=0.6,
                ),
                self.source_passenger: Synapse(
                    self.source_passenger,
                    delay=0.5,
                    weight=0.7,
                ),
            },
            target_time=1.0,
        )
        self.model.columns[4].neurons[1].segments.append(self.segment)
        self.model.previous_active_cells = {
            self.source_weekday: 0.0,
            self.source_time: 0.0,
            self.wrong_passenger: 0.0,
        }
        self.model.previous_winners = {
            self.source_weekday: 0.0,
            self.wrong_passenger: 0.0,
        }
        self.registry = BranchProvenanceRegistry()
        self.registry.capture_new_segments(
            self.model,
            previous_segment_ids=set(),
            creation_transition_index=0,
            creation_sources={
                self.source_weekday: 0.0,
                self.source_time: 0.0,
                self.source_passenger: 0.0,
            },
        )
        self.registry.current_predicted_sources = {self.source_weekday}
        self.registry.current_burst_sources = {
            self.source_time,
            self.wrong_passenger,
        }

    def capture(self, level: str = "source"):
        return capture_match_overlap(
            model=self.model,
            code=self.code,
            registry=self.registry,
            stream_label="original",
            actual_record_index=3,
            actual_record=self.record,
            ranges=self.ranges,
            level=level,
        )

    def test_levels_are_explicit(self) -> None:
        self.assertEqual(MATCH_OVERLAP_LEVELS, {"summary", "segment", "source"})

    def test_unknown_level_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.capture("bad")

    def test_actual_overlap_matches_segment_formula(self) -> None:
        capture = self.capture()
        row = capture.segment_rows[0]
        expected = self.segment.timed_overlap(
            self.model.previous_active_cells,
            0.5,
            self.model.params.timing_tolerance,
        )
        self.assertEqual(row["actual_overlap"], expected)
        self.assertEqual(expected, 2)

    def test_lmatch_flags_are_fixed_state_only(self) -> None:
        row = self.capture().segment_rows[0]
        self.assertTrue(row["passes_L1"])
        self.assertTrue(row["passes_L2"])
        self.assertFalse(row["passes_L3"])
        self.assertFalse(row["passes_L4"])
        self.assertFalse(row["meets_L_match"])
        self.assertEqual(self.model.params.l_match, 3)

    def test_summary_level_does_not_allocate_segment_or_source_rows(self) -> None:
        capture = self.capture("summary")
        self.assertEqual(len(capture.column_rows), 1)
        self.assertEqual(capture.segment_rows, ())
        self.assertEqual(capture.source_rows, ())

    def test_segment_level_omits_source_rows(self) -> None:
        capture = self.capture("segment")
        self.assertEqual(len(capture.segment_rows), 1)
        self.assertEqual(capture.source_rows, ())

    def test_source_level_maps_fields(self) -> None:
        fields = {row["source_field"] for row in self.capture().source_rows}
        self.assertEqual(fields, {"weekday", "time", "passenger"})

    def test_source_level_schema_contains_auditable_identity_fields(self) -> None:
        row = self.capture().source_rows[0]
        required = {
            "segment_target_neuron",
            "source_cell_stable_id",
            "synaptic_weight",
            "synaptic_delay",
            "source_in_previous_winners",
            "source_in_current_active_cells",
            "source_in_predicted_cells",
            "source_in_burst_cells",
            "source_column_currently_active",
            "exact_source_neuron_currently_active",
            "same_column_other_neuron_active",
            "active_neuron_count_in_source_column",
            "contributes_to_actual_overlap",
            "timing_or_eligibility_allows_match",
            "source_loss_reason",
            "source_loss_reason_available",
        }
        self.assertTrue(required <= set(row))

    def test_all_cell_context_is_real_matching_context(self) -> None:
        row = self.capture().segment_rows[0]
        self.assertEqual(row["overlap_all_cell_current"], 2)

    def test_winner_only_context_is_separate(self) -> None:
        row = self.capture().segment_rows[0]
        self.assertEqual(row["overlap_winner_only"], 1)

    def test_predicted_only_context_is_separate(self) -> None:
        row = self.capture().segment_rows[0]
        self.assertEqual(row["overlap_predicted_only"], 1)

    def test_burst_only_context_is_separate(self) -> None:
        row = self.capture().segment_rows[0]
        self.assertEqual(row["overlap_burst_only"], 1)

    def test_without_burst_only_context(self) -> None:
        row = self.capture().segment_rows[0]
        self.assertEqual(row["overlap_without_burst_only"], 1)

    def test_same_and_cross_field_partition(self) -> None:
        row = self.capture().segment_rows[0]
        self.assertEqual(row["overlap_same_field_only"], 0)
        self.assertEqual(row["overlap_cross_field_only"], 2)

    def test_source_retention_uses_exact_cell_identity(self) -> None:
        row = self.capture().segment_rows[0]
        self.assertAlmostEqual(row["source_retention_ratio"], 2 / 3)

    def test_wrong_neuron_is_distinguished(self) -> None:
        passenger = next(
            row
            for row in self.capture().source_rows
            if row["source_column"] == 5
        )
        self.assertEqual(
            passenger["source_missing_reason"],
            "SOURCE_COLUMN_ACTIVE_WRONG_NEURON",
        )

    def test_inactive_column_is_distinguished(self) -> None:
        self.model.previous_active_cells.pop(self.source_time)
        time_source = next(
            row
            for row in self.capture().source_rows
            if row["source_column"] == 2
        )
        self.assertEqual(
            time_source["source_missing_reason"],
            "SOURCE_COLUMN_NOT_ACTIVE",
        )

    def test_timing_failure_is_distinguished(self) -> None:
        self.model.previous_active_cells[self.source_weekday] = 0.2
        weekday = next(
            row
            for row in self.capture().source_rows
            if row["source_column"] == 0
        )
        self.assertEqual(
            weekday["source_missing_reason"],
            "SOURCE_TIMING_OR_ELIGIBILITY_EXCLUDED",
        )

    def test_source_loss_primary_reasons_are_mutually_exclusive(self) -> None:
        base = {
            "contributes": False,
            "source_cell_available": True,
            "source_column_currently_active": False,
            "exact_source_neuron_currently_active": False,
            "same_column_other_neuron_active": False,
            "source_in_all_cell_context": False,
            "source_in_burst_cells": False,
            "source_in_predicted_cells": False,
            "source_in_previous_winners": False,
            "timing_or_eligibility_allows_match": False,
        }
        cases = {
            "SOURCE_EXACT_CELL_MATCHED": {"contributes": True},
            "SOURCE_COLUMN_ACTIVE_WRONG_NEURON": {
                "source_column_currently_active": True,
                "same_column_other_neuron_active": True,
            },
            "SOURCE_COLUMN_NOT_ACTIVE": {},
            "SOURCE_EXACT_NEURON_ACTIVE_NOT_IN_MATCH_CONTEXT": {
                "source_column_currently_active": True,
                "exact_source_neuron_currently_active": True,
            },
            "SOURCE_ONLY_AVAILABLE_THROUGH_BURST": {
                "source_column_currently_active": True,
                "exact_source_neuron_currently_active": True,
                "source_in_burst_cells": True,
            },
            "SOURCE_ONLY_AVAILABLE_THROUGH_PREDICTION": {
                "source_column_currently_active": True,
                "exact_source_neuron_currently_active": True,
                "source_in_predicted_cells": True,
            },
            "SOURCE_TIMING_OR_ELIGIBILITY_EXCLUDED": {
                "source_column_currently_active": True,
                "exact_source_neuron_currently_active": True,
                "source_in_all_cell_context": True,
            },
            "SOURCE_CELL_UNAVAILABLE": {
                "source_cell_available": False,
            },
        }
        for expected, overrides in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(
                    classify_source_loss(**(base | overrides)),
                    expected,
                )

    def test_timing_decomposition_uses_real_arrival_window(self) -> None:
        common = {
            "contributes": False,
            "source_cell_available": True,
            "source_in_matching_context": True,
            "source_in_active_cells": True,
            "source_in_predicted_cells": False,
            "source_in_burst_cells": False,
            "arrival_available": True,
            "arrival_in_matching_window": False,
            "matching_window_start": 0.47,
            "matching_window_end": 0.53,
            "segment_considered_by_matching": True,
            "synapse_considered_by_matching": True,
        }
        late = classify_timing_eligibility_loss(
            **common, computed_arrival_time=0.6
        )
        early = classify_timing_eligibility_loss(
            **common, computed_arrival_time=0.4
        )
        self.assertEqual(late[0], "SOURCE_IN_CONTEXT_DELAY_NOT_ARRIVED")
        self.assertEqual(
            early[0], "SOURCE_IN_CONTEXT_ARRIVAL_TOO_EARLY"
        )
        self.assertEqual(late[2], "Segment.timed_overlap")
        self.assertEqual(late[3], "src/seqmem/model.py")

    def test_context_decomposition_reasons_are_distinct(self) -> None:
        base = {
            "contributes": False,
            "source_cell_available": True,
            "source_in_matching_context": False,
            "source_in_active_cells": False,
            "source_in_predicted_cells": False,
            "source_in_burst_cells": False,
            "arrival_available": False,
            "arrival_in_matching_window": False,
            "computed_arrival_time": None,
            "matching_window_start": 0.47,
            "matching_window_end": 0.53,
            "segment_considered_by_matching": False,
            "synapse_considered_by_matching": False,
        }
        cases = (
            (
                {"source_in_active_cells": True},
                "SOURCE_ACTIVE_NOT_IN_MATCH_CONTEXT",
            ),
            (
                {"source_in_predicted_cells": True},
                "SOURCE_PREDICTED_NOT_IN_MATCH_CONTEXT",
            ),
            (
                {"source_in_burst_cells": True},
                "SOURCE_BURST_NOT_IN_MATCH_CONTEXT",
            ),
            ({}, "SEGMENT_NOT_ELIGIBLE"),
        )
        for overrides, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(
                    classify_timing_eligibility_loss(
                        **(base | overrides)
                    )[0],
                    expected,
                )

    def test_remaining_timing_eligibility_paths_are_explicit(self) -> None:
        base = {
            "contributes": False,
            "source_cell_available": True,
            "source_in_matching_context": False,
            "source_in_active_cells": False,
            "source_in_predicted_cells": False,
            "source_in_burst_cells": False,
            "arrival_available": False,
            "arrival_in_matching_window": False,
            "computed_arrival_time": None,
            "matching_window_start": 0.47,
            "matching_window_end": 0.53,
            "segment_considered_by_matching": True,
            "synapse_considered_by_matching": False,
        }
        cases = (
            ({}, "SYNAPSE_NOT_ELIGIBLE"),
            (
                {
                    "source_in_matching_context": True,
                    "source_in_active_cells": True,
                },
                "SOURCE_STATE_CAPTURE_UNAVAILABLE",
            ),
            (
                {
                    "source_in_matching_context": True,
                    "source_in_active_cells": True,
                    "arrival_available": True,
                    "computed_arrival_time": 0.55,
                    "synapse_considered_by_matching": True,
                },
                "SOURCE_IN_CONTEXT_DELAY_NOT_ARRIVED",
            ),
            (
                {
                    "source_in_matching_context": True,
                    "source_in_active_cells": True,
                    "arrival_available": True,
                    "computed_arrival_time": 0.5,
                    "synapse_considered_by_matching": True,
                },
                "SOURCE_OTHER_EXPLICIT_CODE_PATH",
            ),
            (
                {
                    "arrival_available": True,
                    "synapse_considered_by_matching": True,
                },
                "UNKNOWN_TIMING_ELIGIBILITY_REASON",
            ),
        )
        for overrides, expected in cases:
            with self.subTest(expected=expected):
                reason = classify_timing_eligibility_loss(
                    **(base | overrides)
                )[0]
                self.assertEqual(reason, expected)

        # Predicted firing time is trace metadata, not an input to the real
        # Segment.timed_overlap predicate. It must not invent a matching loss.
        predicted_time_invalid_reason = classify_timing_eligibility_loss(
            **base
        )[0]
        self.assertEqual(
            predicted_time_invalid_reason,
            "SYNAPSE_NOT_ELIGIBLE",
        )
        self.assertNotEqual(
            predicted_time_invalid_reason,
            "SOURCE_PREDICTED_TIME_INVALID",
        )

    def test_source_trace_filter_is_output_only_and_boundary_exact(self) -> None:
        rows = [
            {
                "actual_record_index": index,
                "field": field,
                "observe_scenario": scenario,
                "source_loss_reason_primary": reason,
                "source_still_present_in_segment": True,
            }
            for index, field, scenario, reason in (
                (199, "passenger", "scenario3", "SOURCE_COLUMN_NOT_ACTIVE"),
                (200, "passenger", "scenario3", "SOURCE_COLUMN_NOT_ACTIVE"),
                (244, "passenger", "scenario3", "SOURCE_COLUMN_NOT_ACTIVE"),
                (245, "passenger", "scenario3", "SOURCE_COLUMN_NOT_ACTIVE"),
                (220, "time", "scenario3", "SOURCE_COLUMN_NOT_ACTIVE"),
                (220, "passenger", "scenario2", "SOURCE_COLUMN_NOT_ACTIVE"),
                (220, "passenger", "scenario3", "SOURCE_EXACT_CELL_MATCHED"),
            )
        ]
        original = json.dumps(rows, sort_keys=True)
        filtered = filter_source_trace_rows(
            rows,
            SourceTraceFilter(
                field="passenger",
                scenario="scenario3",
                record_start=200,
                record_end=244,
                loss_only=True,
                existing_segments_only=True,
            ),
        )
        self.assertEqual(
            [row["actual_record_index"] for row in filtered],
            [200, 244],
        )
        self.assertEqual(json.dumps(rows, sort_keys=True), original)

    def test_empty_column_is_safe(self) -> None:
        empty = SymbolCode((SpikeEvent(3, 0.0),))
        capture = capture_match_overlap(
            model=self.model,
            code=empty,
            registry=self.registry,
            stream_label="original",
            actual_record_index=3,
            actual_record=self.record,
            ranges=self.ranges,
            level="segment",
        )
        self.assertEqual(capture.column_rows[0]["existing_segment_count"], 0)
        self.assertEqual(capture.column_rows[0]["best_matching_overlap"], 0)

    def test_capture_does_not_call_matching(self) -> None:
        with patch.object(
            self.model,
            "_best_matching_neuron",
            side_effect=AssertionError("matching rerun"),
        ):
            self.capture()

    def test_capture_does_not_change_rng_or_state(self) -> None:
        learning = self.model._learning_rng.getstate()
        decoding = self.model._decode_rng.getstate()
        active = self.model.previous_active_cells.copy()
        winners = self.model.previous_winners.copy()
        weights = {
            source: (synapse.weight, synapse.delay, synapse.age)
            for source, synapse in self.segment.synapses.items()
        }
        self.capture()
        self.assertEqual(self.model._learning_rng.getstate(), learning)
        self.assertEqual(self.model._decode_rng.getstate(), decoding)
        self.assertEqual(self.model.previous_active_cells, active)
        self.assertEqual(self.model.previous_winners, winners)
        self.assertEqual(
            {
                source: (synapse.weight, synapse.delay, synapse.age)
                for source, synapse in self.segment.synapses.items()
            },
            weights,
        )

    def test_finalize_uses_existing_observation_identity(self) -> None:
        trace = ObservationTrace(
            events=[
                ObservationEventTrace(
                    target_column=4,
                    target_time=0.0,
                    winner_neuron=1,
                    winner_cell_id=self.model._cell_id(4, 1),
                    scenario="scenario2",
                    was_predicted=False,
                    selected_segment=self.segment,
                    reinforced_segment=self.segment,
                    best_matching_segment=self.segment,
                    scenario_assignment_reason="MATCHING_SEGMENT_FOUND",
                )
            ]
        )
        final = finalize_match_overlap(self.capture(), trace, self.registry)
        self.assertEqual(final.column_rows[0]["observe_scenario"], "scenario2")
        self.assertTrue(final.segment_rows[0]["selected_as_best_matching"])
        self.assertTrue(final.segment_rows[0]["reinforced"])
        self.assertEqual(final.source_rows[0]["observe_scenario"], "scenario2")

    def test_markers_are_present_and_true(self) -> None:
        row = self.capture().column_rows[0]
        for key, value in DIAGNOSTIC_MARKERS.items():
            self.assertEqual(row[key], value)


class MatchOverlapAnalyzerTests(unittest.TestCase):
    def test_record_range_boundaries_and_invalid_values(self) -> None:
        cases = {
            0: "0-49",
            49: "0-49",
            50: "50-99",
            99: "50-99",
            100: "100-149",
            149: "100-149",
            150: "150-199",
            199: "150-199",
            200: "200-244",
            244: "200-244",
            None: "UNKNOWN_RECORD_RANGE",
            "": "UNKNOWN_RECORD_RANGE",
            -1: "UNKNOWN_RECORD_RANGE",
            245: "UNKNOWN_RECORD_RANGE",
            1.5: "UNKNOWN_RECORD_RANGE",
            "bad": "UNKNOWN_RECORD_RANGE",
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(classify_record_range(value), expected)

    def test_analyzer_writes_all_required_outputs(self) -> None:
        column_row = {
            **DIAGNOSTIC_MARKERS,
            "actual_record_index": 3,
            "field": "passenger",
            "observe_scenario": "scenario3",
            "existing_segment_count": 2,
            "best_matching_overlap": 1,
            "gap_to_L_match": 3,
            "best_segment_source_retention": 0.2,
            "best_segment_creation_current_jaccard": 0.1,
            "best_segment_current_context_jaccard": 0.1,
            "multiple_segments_meet_L_match": False,
            "multiple_neurons_meet_L_match": False,
        }
        segment_row = {
            **DIAGNOSTIC_MARKERS,
            "actual_record_index": 3,
            "timestamp": "2014-07-01 00:00:00",
            "field": "passenger",
            "encoded_column": 4,
            "segment_target_neuron": 1,
            "segment_synapse_count": 3,
            "source_weekday_count": 1,
            "source_time_count": 1,
            "source_passenger_count": 1,
            "actual_overlap": 1,
            "source_retention_ratio": 1 / 3,
            "creation_current_jaccard": 0.2,
            "creation_source_count": 3,
            "creation_source_currently_active": 1,
            "creation_source_currently_winner": 1,
            "selected_for_scenario": False,
            "reinforced": False,
            "overlap_all_cell_current": 1,
            "overlap_winner_only": 1,
            "overlap_predicted_only": 1,
            "overlap_burst_only": 0,
            "overlap_predicted_plus_winner": 1,
            "overlap_without_burst_only": 1,
            "overlap_same_field_only": 0,
            "overlap_cross_field_only": 1,
        }
        source_row = {
            **DIAGNOSTIC_MARKERS,
            "actual_record_index": 3,
            "field": "passenger",
            "source_missing_reason": "SOURCE_COLUMN_ACTIVE_WRONG_NEURON",
        }
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            output_dir = Path(temporary) / "analysis"
            run_dir.mkdir()
            self._write(run_dir / "match_overlap_column_trace.csv", [column_row])
            self._write_gzip(
                run_dir / "match_overlap_segment_trace.csv.gz",
                [segment_row],
            )
            self._write_gzip(
                run_dir / "match_overlap_source_trace.csv.gz",
                [source_row],
            )
            summary = analyze(
                run_dir=run_dir,
                output_dir=output_dir,
                bootstrap_samples=20,
                bootstrap_seed=0,
            )
            expected = {
                "match_overlap_field_summary.csv",
                "match_overlap_record_range_summary.csv",
                "passenger_overlap_distribution.csv",
                "segment_source_field_summary.csv",
                "source_retention_summary.csv",
                "source_loss_reason_summary.csv",
                "context_variant_overlap_summary.csv",
                "lmatch_counterfactual_summary.csv",
                "lmatch_ambiguity_summary.csv",
                "teacher_reference_overlap_summary.csv",
                "match_overlap_bootstrap_ci.csv",
                "match_overlap_summary.json",
                "FIG9_MATCH_OVERLAP_DECOMPOSITION_REPORT.md",
                "match_overlap_field_summary_fixed.csv",
                "match_overlap_record_range_summary_fixed.csv",
                "segment_source_field_summary_fixed.csv",
                "source_retention_summary_fixed.csv",
                "teacher_reference_overlap_summary_fixed.csv",
                "lmatch_counterfactual_summary_fixed.csv",
                "lmatch_ambiguity_summary_fixed.csv",
                "context_variant_overlap_summary_fixed.csv",
                "denominator_manifest.json",
                "analysis_consistency_checks.json",
                "FIG9_MATCH_OVERLAP_DECOMPOSITION_REPORT_FIXED.md",
                "source_field_summary.csv",
                "source_record_range_summary.csv",
                "source_context_membership_summary.csv",
                "source_identity_bootstrap_ci.csv",
                "source_trace_protocol.json",
                "SOURCE_LEVEL_MATCH_OVERLAP_REPORT.md",
            }
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                expected,
            )
            self.assertFalse(summary["passenger_200_244_available"])
            payload = json.loads(
                (output_dir / "match_overlap_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(payload["formal_250_not_run_by_analyzer"])
            self.assertTrue(payload["fixed_analysis_checks_passed"])

    def test_fixed_grouping_uses_actual_index_not_legacy_label(self) -> None:
        rows = [
            {
                **DIAGNOSTIC_MARKERS,
                "actual_record_index": index,
                "record_range": "0-49",
                "timestamp": f"t-{index}",
                "field": "passenger",
                "encoded_column": index,
                "observe_scenario": "scenario3",
                "best_matching_overlap": 0,
                "gap_to_L_match": 4,
            }
            for index in (0, 50, 100, 150, 200)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            output_dir = Path(temporary) / "analysis"
            run_dir.mkdir()
            self._write(run_dir / "match_overlap_column_trace.csv", rows)
            analyze(
                run_dir=run_dir,
                output_dir=output_dir,
                bootstrap_samples=10,
                bootstrap_seed=0,
            )
            with (
                output_dir / "match_overlap_record_range_summary_fixed.csv"
            ).open(encoding="utf-8", newline="") as handle:
                fixed = list(csv.DictReader(handle))
            self.assertEqual(
                {row["record_range"] for row in fixed},
                {"0-49", "50-99", "100-149", "150-199", "200-244"},
            )
            self.assertEqual(
                sum(int(row["observation_count"]) for row in fixed),
                len(rows),
            )
            manifest = json.loads(
                (output_dir / "denominator_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                manifest["record_range_source"],
                "actual_record_index",
            )

    def test_fixed_analysis_rejects_unknown_record_range(self) -> None:
        row = {
            **DIAGNOSTIC_MARKERS,
            "actual_record_index": "",
            "timestamp": "t",
            "field": "passenger",
            "encoded_column": 1,
            "observe_scenario": "scenario3",
            "best_matching_overlap": 0,
            "gap_to_L_match": 4,
        }
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            self._write(
                run_dir / "match_overlap_column_trace.csv",
                [row],
            )
            with self.assertRaisesRegex(
                AssertionError, "no_unknown_record_ranges"
            ):
                analyze(
                    run_dir=run_dir,
                    output_dir=Path(temporary) / "fixed",
                    bootstrap_samples=10,
                    bootstrap_seed=0,
                    fixed_only=True,
                )

    def test_timing_analyzer_writes_scoped_auditable_outputs(self) -> None:
        rows = [
            {
                "actual_record_index": 20,
                "timestamp": "2014-07-01 10:00:00",
                "field": "passenger",
                "encoded_column": 500,
                "observe_scenario": "scenario3",
                "segment_provenance_id": "s1",
                "segment_target_neuron": 2,
                "segment_synapse_count": 2,
                "source_loss_reason_primary": reason,
                "source_loss_reason_secondary": "explicit",
                "source_column_currently_active": True,
                "exact_source_neuron_currently_active": True,
                "source_in_previous_winners": False,
                "source_in_predicted_cells": False,
                "source_in_burst_cells": True,
                "source_in_matching_context": True,
                "source_in_active_cells": True,
                "arrival_in_matching_window": matched,
                "timing_used_by_actual_overlap": True,
                "source_predicted_time_available": False,
                "source_predicted_time_valid": False,
                "segment_considered_by_matching": True,
                "segment_eligible": False,
                "synapse_considered_by_matching": True,
                "synapse_eligible": matched,
                "contributes_to_actual_overlap": matched,
            }
            for reason, matched in (
                ("SOURCE_EXACT_CELL_MATCHED", True),
                ("SOURCE_IN_CONTEXT_DELAY_NOT_ARRIVED", False),
            )
        ]
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            output_dir = Path(temporary) / "analysis"
            run_dir.mkdir()
            self._write_gzip(
                run_dir / "match_overlap_source_trace.csv.gz",
                rows,
            )
            summary = analyze_timing_eligibility(
                run_dir=run_dir,
                output_dir=output_dir,
                bootstrap_samples=20,
                bootstrap_seed=0,
            )
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["unknown_reason_rate"], 0.0)
            self.assertTrue(
                (
                    output_dir
                    / "FIG9_TIMING_ELIGIBILITY_DECOMPOSITION_REPORT.md"
                ).exists()
            )
            with (
                output_dir / "timing_eligibility_reason_summary.csv"
            ).open(encoding="utf-8", newline="") as handle:
                summary_rows = list(csv.DictReader(handle))
            passenger_s3 = [
                row
                for row in summary_rows
                if row["scope"] == "PASSENGER_SCENARIO3"
            ]
            self.assertEqual(
                {int(row["source_row_denominator"]) for row in passenger_s3},
                {2},
            )

    def test_runner_trace_is_noninterfering(self) -> None:
        records = [
            TaxiRecord(
                datetime(2014, 7, 1, hour=(index // 2) % 24, minute=30 * (index % 2)),
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
            plain = run_strict_stream(
                records=records,
                data_path=data_path,
                stream_label="original",
                output_dir=root / "plain",
                config=config,
                limit=10,
                print_fingerprint=False,
            )
            traced = run_strict_stream(
                records=records,
                data_path=data_path,
                stream_label="original",
                output_dir=root / "traced",
                config=config,
                limit=10,
                print_fingerprint=False,
                match_overlap_diagnostic=True,
                match_overlap_level="segment",
                match_overlap_compress=True,
                checkpoint_path=root / "traced.pkl",
                checkpoint_at_index=9,
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
                self.assertEqual(plain.get(field), traced.get(field))
            self.assertTrue(
                (root / "traced" / "match_overlap_column_trace.csv").exists()
            )
            self.assertTrue(
                (
                    root
                    / "traced"
                    / "match_overlap_segment_trace.csv.gz"
                ).exists()
            )
            checkpoint = load_strict_checkpoint(root / "traced.pkl")
            self.assertEqual(
                model_long_term_fingerprint(checkpoint["model"]),
                traced["final_model_fingerprint"],
            )
            self.assertEqual(
                model_rng_fingerprint(checkpoint["model"]),
                traced["final_rng_fingerprint"],
            )

    @staticmethod
    def _write(path: Path, rows: list[dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_gzip(path: Path, rows: list[dict[str, object]]) -> None:
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
