from __future__ import annotations

import csv
import hashlib
import inspect
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from experiments.diagnostics.analyze_fig9_teacher_forced_identity import (
    CLASSIFICATIONS,
    align_identity_rows,
    analyze,
    bootstrap_rows,
    classify_identity,
    derive_reference_applicability,
    scoped_metric_rows,
)
from experiments.diagnostics.analyze_fig9_observe_scenario_assignment import (
    normalize_assignment_rows,
    summarize_assignments,
)
from experiments.diagnostics.analyze_fig9_reference_neuron_selection import (
    build_reference_replacement_trace,
    build_reference_selection_trace,
    build_reference_funnel,
    build_ranking_observations,
    classify_reference_loss,
)
from experiments.diagnostics.fig9_competitive_inhibition import (
    CompetitionSettings,
)
from experiments.diagnostics.fig9_teacher_forced_identity import (
    DIAGNOSTIC_MARKERS,
    build_teacher_forced_observation_rows,
    legacy_teacher_forced_rows,
    observe_scenario_rows,
    reference_applicability_from_trace,
    stable_observation_reference_id,
    winner_set,
)
from experiments.fig9 import TaxiRecord
from experiments.fig9_strict_reproduction import (
    Fig9StrictConfig,
    load_strict_checkpoint,
    run_strict_stream,
)
from seqmem.model import ObservationEventTrace, Segment


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_csv(path: Path) -> list[dict[str, str]]:
    runtime_fields = {
        "prediction_runtime_seconds",
        "decode_runtime_seconds",
        "competition_runtime_seconds",
        "observe_runtime",
        "runtime_seconds",
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


def _reference(
    *,
    neuron: str = "2",
    scenario: str = "scenario1",
    segment_id: str = "teacher-segment",
) -> dict[str, object]:
    return {
        "actual_record_index": 5,
        "actual_timestamp": "2014-07-01 02:30:00",
        "field": "passenger",
        "target_column": 100,
        "observed_winner_available": True,
        "observed_winner_neurons": neuron,
        "selected_segment_id": segment_id,
        "reinforced_segment_id": (
            segment_id if scenario in {"scenario1", "scenario2"} else ""
        ),
        "created_segment_id": (
            segment_id if scenario == "scenario3" else ""
        ),
        "observe_scenario": scenario,
        "scenario3": scenario == "scenario3",
        "creation_source_fingerprint": "teacher-source",
        "creation_source_cell_ids": "1 2",
        "current_source_fingerprint": "teacher-current",
        "segment_creation_transition_index": 4,
        "source_support_class": "predicted-supported",
    }


def _funnel(outcome: str = "TARGET_SEGMENT_EMITTED") -> dict[str, object]:
    return {
        "input_index": 5,
        "target_timestamp": "2014-07-01 02:30:00",
        "horizon_step": 1,
        "target_column_outcomes": json.dumps(
            [{"field": "passenger", "column": 100, "outcome": outcome}]
        ),
    }


def _segment(
    *,
    neuron: int = 2,
    candidate: bool = True,
    emitted: bool = True,
    segment_id: str = "teacher-segment",
) -> dict[str, object]:
    return {
        "input_index": 5,
        "target_timestamp": "2014-07-01 02:30:00",
        "horizon_step": 1,
        "target_column": 100,
        "target_neuron": neuron,
        "crossed_threshold": True,
        "became_prediction_candidate": candidate,
        "emitted_after_competition": emitted,
        "segment_provenance_id": segment_id,
        "creation_source_fingerprint": "teacher-source",
        "current_source_fingerprint": "teacher-current",
        "creation_transition_index": 4,
        "source_support_class": "predicted-supported",
    }


class TeacherForcedIdentityUnitTests(unittest.TestCase):
    def test_scenario1_reference_is_preexisting_predicted(self) -> None:
        segment = Segment()
        fields = reference_applicability_from_trace(
            ObservationEventTrace(
                1,
                0.1,
                2,
                12,
                "scenario1",
                True,
                selected_segment=segment,
                reinforced_segment=segment,
            )
        )
        self.assertEqual(
            fields["reference_applicability"],
            "PREEXISTING_PREDICTED_REFERENCE",
        )
        self.assertTrue(fields["neuron_reference_valid_before_observation"])
        self.assertTrue(fields["segment_reference_valid_before_observation"])

    def test_scenario2_reference_is_preexisting_matching_segment(self) -> None:
        segment = Segment()
        fields = reference_applicability_from_trace(
            ObservationEventTrace(
                1,
                0.1,
                2,
                12,
                "scenario2",
                False,
                selected_segment=segment,
                reinforced_segment=segment,
            )
        )
        self.assertEqual(
            fields["reference_applicability"],
            "PREEXISTING_MATCHING_SEGMENT_REFERENCE",
        )
        self.assertTrue(fields["exact_segment_match_applicable"])

    def test_scenario3_created_segment_is_operational_only(self) -> None:
        segment = Segment()
        fields = reference_applicability_from_trace(
            ObservationEventTrace(
                1,
                0.1,
                2,
                12,
                "scenario3",
                False,
                selected_segment=segment,
                created_segment=segment,
            )
        )
        self.assertEqual(
            fields["reference_applicability"],
            "POST_OBSERVATION_CREATED_SEGMENT",
        )
        self.assertFalse(fields["neuron_reference_valid_before_observation"])
        self.assertFalse(fields["exact_segment_match_applicable"])
        self.assertTrue(fields["operational_winner_only"])

    def test_old_trace_applicability_is_recovered_from_real_fields(self) -> None:
        reference = _reference(scenario="scenario1")
        reference["reinforced_segment_id"] = "teacher-segment"
        fields = derive_reference_applicability(
            reference,
            expected_target_index=5,
            maximum_observed_index=9,
        )
        self.assertEqual(
            fields["reference_applicability"],
            "PREEXISTING_PREDICTED_REFERENCE",
        )

    def test_boundary_applicability_uses_observed_index(self) -> None:
        fields = derive_reference_applicability(
            None,
            expected_target_index=10,
            maximum_observed_index=9,
        )
        self.assertEqual(
            fields["reference_applicability"],
            "REFERENCE_UNAVAILABLE_BOUNDARY",
        )

    def test_scoped_metrics_exclude_post_observation_references(self) -> None:
        rows = [
            {
                "policy": "batched",
                "reference_available": True,
                "neuron_reference_valid_before_observation": True,
                "segment_reference_valid_before_observation": True,
                "reference_applicability": "PREEXISTING_PREDICTED_REFERENCE",
                "reference_exclusion_reason": "",
                "target_column_raw_recall": True,
                "target_column_candidate_recall": True,
                "target_column_emitted_recall": True,
                "reference_neuron_threshold_crossing_recall": True,
                "reference_neuron_candidate_recall": True,
                "reference_neuron_emitted_recall": True,
                "correct_column_wrong_neuron": False,
                "exact_segment_match_applicable": True,
                "source_context_match_applicable": True,
                "exact_segment_match": True,
                "same_creation_transition": True,
                "creation_source_fingerprint_exact_match": True,
            },
            {
                "policy": "batched",
                "reference_available": True,
                "neuron_reference_valid_before_observation": False,
                "segment_reference_valid_before_observation": False,
                "reference_applicability": "POST_OBSERVATION_CREATED_SEGMENT",
                "reference_exclusion_reason": "post_observation_created_segment",
                "target_column_raw_recall": True,
                "target_column_candidate_recall": True,
                "target_column_emitted_recall": True,
                "reference_neuron_threshold_crossing_recall": False,
                "reference_neuron_candidate_recall": False,
                "reference_neuron_emitted_recall": False,
                "correct_column_wrong_neuron": True,
                "exact_segment_match_applicable": False,
                "source_context_match_applicable": False,
                "exact_segment_match": "",
                "same_creation_transition": "",
                "creation_source_fingerprint_exact_match": "",
            },
        ]
        metrics = scoped_metric_rows(rows, group_fields=("policy",))
        candidate = next(
            row
            for row in metrics
            if row["reference_scope"] == "PREEXISTING_NEURON_REFERENCES"
            and row["metric"] == "reference_neuron_candidate_recall"
        )
        self.assertEqual(candidate["numerator"], 1)
        self.assertEqual(candidate["denominator"], 1)
        self.assertEqual(candidate["excluded_count"], 1)
        exact = next(
            row
            for row in metrics
            if row["reference_scope"] == "PREEXISTING_SEGMENT_REFERENCES"
            and row["metric"] == "exact_segment_match"
        )
        self.assertEqual(exact["denominator"], 1)

    def test_crossing_to_candidate_loss_uses_real_stage_flags(self) -> None:
        self.assertEqual(
            classify_reference_loss(
                [
                    {
                        "predicted_time": "0.2",
                        "became_event_winner": True,
                        "became_prediction_candidate": False,
                        "emitted_after_competition": False,
                    }
                ]
            ),
            "REFERENCE_NEURON_EVENT_WINNER_BUT_LOST_COLUMN_SELECTION",
        )

    def test_reference_wrong_winner_pair_uses_same_column_neurons(self) -> None:
        identity = {
            **_reference(),
            "input_index": 5,
            "target_timestamp": "2014-07-01 02:30:00",
            "horizon_step": 1,
            "target_column": 100,
            "field": "passenger",
            "reference_winner_neurons": "2",
            "neuron_reference_valid_before_observation": True,
            "reference_applicability": "PREEXISTING_PREDICTED_REFERENCE",
            "teacher_forced_observe_scenario": "scenario1",
            "teacher_forced_segment_id": "reference-segment",
        }
        reference_segment = {
            **_segment(neuron=2, candidate=False, emitted=False),
            "segment_provenance_id": "reference-segment",
            "became_event_winner": True,
            "predicted_time": "0.3",
            "candidate_score_value": "1.1",
        }
        wrong = {
            **_segment(neuron=7, candidate=True, emitted=True),
            "segment_provenance_id": "wrong-segment",
            "became_event_winner": True,
            "predicted_time": "0.2",
            "candidate_score_value": "1.0",
        }
        funnels, pairs = build_reference_funnel(
            [identity],
            [reference_segment, wrong],
        )
        self.assertEqual(funnels[0]["actual_selected_autonomous_neuron"], 7)
        self.assertEqual(len(pairs), 1)
        self.assertTrue(pairs[0]["winner_was_earlier"])

    def test_ranking_observation_recognizes_same_neuron_different_segment(
        self,
    ) -> None:
        identity = {
            **_reference(),
            "input_index": 5,
            "target_timestamp": "2014-07-01 02:30:00",
            "horizon_step": 1,
            "target_column": 100,
            "field": "passenger",
            "reference_winner_neurons": "2",
            "neuron_reference_valid_before_observation": True,
            "reference_applicability": "PREEXISTING_PREDICTED_REFERENCE",
            "teacher_forced_observe_scenario": "scenario1",
            "teacher_forced_segment_id": "teacher-segment",
        }
        segment = {
            **_segment(neuron=2),
            "segment_provenance_id": "different-segment",
            "response_peak": "1.2",
            "first_crossing_time": "0.1",
            "current_context_jaccard": "0.5",
            "creation_current_source_jaccard": "0.5",
            "historical_winner_match": True,
            "segment_age": "0",
        }
        observations = build_ranking_observations([identity], [segment])
        self.assertTrue(observations)
        self.assertTrue(all(row["hit_at_1"] for row in observations))
        self.assertTrue(
            all(
                not row["reference_segment_exact_match"]
                for row in observations
            )
        )

    def test_observe_scenario_normalization_is_empty_safe(self) -> None:
        self.assertEqual(normalize_assignment_rows([]), [])
        self.assertEqual(summarize_assignments([]), [])

    def test_observe_level_projection_does_not_recompute_rows(self) -> None:
        row = {
            **DIAGNOSTIC_MARKERS,
            "actual_record_index": 1,
            "timestamp": "2014-07-01 00:30:00",
            "field": "passenger",
            "encoded_column": 100,
            "encoded_value": 1000.0,
            "observe_scenario": "scenario3",
            "scenario_assignment_reason": "NO_EXISTING_SEGMENT_IN_COLUMN",
            "reference_applicability": "POST_OBSERVATION_CREATED_SEGMENT",
            "neuron_reference_valid_before_observation": False,
            "segment_reference_valid_before_observation": False,
            "created_new_segment": True,
            "created_new_neuron_identity": True,
            "private_full_field": "kept-only-in-full",
        }
        summary = observe_scenario_rows([row], level="summary")
        full = observe_scenario_rows([row], level="full")
        self.assertNotIn("private_full_field", summary[0])
        self.assertEqual(full[0]["private_full_field"], "kept-only-in-full")
        self.assertEqual(
            full[0]["scenario_assignment_reason"],
            "NO_EXISTING_SEGMENT_IN_COLUMN",
        )

    def test_legacy_teacher_trace_keeps_old_reason_vocabulary(self) -> None:
        row = {
            "scenario_assignment_reason": (
                "CORRECTLY_PREDICTED_CELL_AVAILABLE"
            ),
            "predictive_neuron_ids": "2",
            "observe_scenario": "scenario1",
        }
        projected = legacy_teacher_forced_rows([row])[0]
        self.assertNotIn("predictive_neuron_ids", projected)
        self.assertEqual(
            projected["scenario_assignment_reason"],
            "PREDICTIVE_CELL_MATCHED",
        )

    def test_reference_selection_trace_keeps_real_stage_fields(self) -> None:
        identity = {
            **_reference(),
            "input_index": 5,
            "target_record_index": 6,
            "target_timestamp": "2014-07-01 02:30:00",
            "horizon_step": 1,
            "target_column": 100,
            "field": "passenger",
            "reference_winner_neurons": "2",
            "neuron_reference_valid_before_observation": True,
            "reference_applicability": "PREEXISTING_PREDICTED_REFERENCE",
            "teacher_forced_segment_id": "teacher-segment",
        }
        reference_segment = {
            **_segment(neuron=2, candidate=False, emitted=False),
            "policy": "batched",
            "response_available": True,
            "response_peak": 1.1,
            "predicted_time": 0.2,
            "became_event_winner": True,
            "elimination_stage": "COLUMN_EARLIEST_SELECTION",
            "elimination_reason": "later_predicted_time",
            "winner_segment_id": "wrong-segment",
            "rank_within_column_group": 2,
        }
        wrong_segment = {
            **_segment(
                neuron=3,
                candidate=True,
                emitted=True,
                segment_id="wrong-segment",
            ),
            "policy": "batched",
            "response_available": True,
            "response_peak": 1.2,
            "predicted_time": 0.1,
            "became_event_winner": True,
            "winner_segment_id": "wrong-segment",
            "rank_within_column_group": 1,
        }
        rows = build_reference_selection_trace(
            [identity],
            [reference_segment, wrong_segment],
            input_timestamps={5: "2014-07-01 02:00:00"},
        )
        reference_row = next(
            row for row in rows if row["is_reference_neuron"]
        )
        self.assertEqual(
            reference_row["comparison_lost_at_stage"],
            "COLUMN_EARLIEST_SELECTION",
        )
        self.assertEqual(reference_row["winning_neuron"], 3)
        self.assertFalse(reference_row["winner_is_reference_neuron"])

    def test_reference_replacement_trace_uses_captured_comparison(self) -> None:
        identity = {
            **_reference(),
            "input_index": 5,
            "target_record_index": 6,
            "target_timestamp": "2014-07-01 02:30:00",
            "horizon_step": 1,
            "target_column": 100,
            "field": "passenger",
            "reference_winner_neurons": "2",
            "neuron_reference_valid_before_observation": True,
            "reference_applicability": "PREEXISTING_PREDICTED_REFERENCE",
        }
        segments = [
            _segment(neuron=2, segment_id="reference"),
            _segment(neuron=3, segment_id="wrong"),
        ]
        replacements = [
            {
                "selection_group_id": "group",
                "previous_winner_id": "reference",
                "replacement_winner_id": "wrong",
                "comparison_field": "predicted_time",
                "previous_value": 0.2,
                "replacement_value": 0.1,
                "previous_tie_break_values": "[2]",
                "replacement_tie_break_values": "[3]",
                "replacement_stage": "COLUMN_EARLIEST_SELECTION",
            }
        ]
        rows = build_reference_replacement_trace(
            [identity],
            segments,
            replacements,
        )
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["previous_is_reference"])
        self.assertEqual(rows[0]["comparison_key"], "predicted_time")

    def test_reference_id_is_stable_and_uses_column(self) -> None:
        arguments = {
            "stream_label": "original",
            "actual_record_index": 8,
            "actual_timestamp": "2014-07-01 04:00:00",
            "target_column": 10,
        }
        reference_id = stable_observation_reference_id(**arguments)
        self.assertEqual(
            reference_id,
            stable_observation_reference_id(**arguments),
        )
        arguments["target_column"] = 11
        self.assertNotEqual(
            reference_id,
            stable_observation_reference_id(**arguments),
        )

    def test_winner_set_supports_single_multiple_and_missing(self) -> None:
        self.assertEqual(winner_set({"observed_winner_neurons": "3"}), {3})
        self.assertEqual(
            winner_set({"observed_winner_neurons": "3 7"}),
            {3, 7},
        )
        self.assertEqual(winner_set({}), set())

    def test_classification_covers_absent_below_and_no_firing(self) -> None:
        shared = {
            "reference_available": True,
            "target_column_candidate": False,
            "target_column_emitted": False,
            "reference_neuron_crossed": False,
            "reference_neuron_candidate": False,
            "reference_neuron_emitted": False,
        }
        self.assertEqual(
            classify_identity(
                funnel_outcome="TARGET_NO_SEGMENT_INSPECTED",
                **shared,
            ),
            CLASSIFICATIONS["A"],
        )
        self.assertEqual(
            classify_identity(
                funnel_outcome="TARGET_SEGMENT_BELOW_THRESHOLD",
                **shared,
            ),
            CLASSIFICATIONS["B"],
        )
        self.assertEqual(
            classify_identity(
                funnel_outcome=(
                    "TARGET_SEGMENT_CROSSED_BUT_NO_VALID_FIRING_TIME"
                ),
                **shared,
            ),
            CLASSIFICATIONS["C"],
        )

    def test_missing_reference_is_safe(self) -> None:
        rows = align_identity_rows(
            observation_rows=[],
            funnel_rows=[_funnel()],
            segment_rows=[_segment()],
            policy_label="batched",
        )
        self.assertEqual(rows[0]["classification"], CLASSIFICATIONS["I"])

    def test_multiple_winner_match_accepts_any_member(self) -> None:
        rows = align_identity_rows(
            observation_rows=[_reference(neuron="2 7")],
            funnel_rows=[_funnel()],
            segment_rows=[_segment(neuron=7)],
            policy_label="batched",
        )
        self.assertEqual(rows[0]["classification"], CLASSIFICATIONS["H"])
        self.assertEqual(rows[0]["winner_set_size"], 2)

    def test_correct_column_wrong_neuron_classification(self) -> None:
        rows = align_identity_rows(
            observation_rows=[_reference(neuron="2")],
            funnel_rows=[_funnel()],
            segment_rows=[_segment(neuron=7)],
            policy_label="batched",
        )
        self.assertEqual(rows[0]["classification"], CLASSIFICATIONS["G"])
        self.assertTrue(rows[0]["correct_column_wrong_neuron"])

    def test_reference_candidate_suppressed_classification(self) -> None:
        rows = align_identity_rows(
            observation_rows=[_reference()],
            funnel_rows=[
                _funnel(
                    "TARGET_SEGMENT_BECAME_CANDIDATE_BUT_SUPPRESSED_INTERCOLUMN"
                )
            ],
            segment_rows=[_segment(candidate=True, emitted=False)],
            policy_label="batched",
        )
        self.assertEqual(rows[0]["classification"], CLASSIFICATIONS["F"])

    def test_reference_present_but_not_candidate_classification(self) -> None:
        rows = align_identity_rows(
            observation_rows=[_reference()],
            funnel_rows=[
                _funnel("TARGET_SEGMENT_CROSSED_BUT_LOST_INTERNAL_SELECTION")
            ],
            segment_rows=[_segment(candidate=False, emitted=False)],
            policy_label="batched",
        )
        self.assertEqual(rows[0]["classification"], CLASSIFICATIONS["E"])

    def test_exact_segment_and_source_similarity(self) -> None:
        rows = align_identity_rows(
            observation_rows=[_reference()],
            funnel_rows=[_funnel()],
            segment_rows=[_segment()],
            policy_label="batched",
            creation_sources_by_segment={"teacher-segment": {1, 2}},
        )
        self.assertTrue(rows[0]["exact_segment_match"])
        self.assertEqual(
            rows[0]["creation_source_fingerprint_jaccard"],
            1.0,
        )
        self.assertTrue(rows[0]["same_creation_transition"])

    def test_created_after_observation_has_no_exact_match_requirement(self) -> None:
        rows = align_identity_rows(
            observation_rows=[_reference(scenario="scenario3")],
            funnel_rows=[_funnel()],
            segment_rows=[_segment(segment_id="older-segment")],
            policy_label="batched",
        )
        self.assertTrue(
            rows[0][
                "teacher_forced_segment_created_only_after_observation"
            ]
        )
        self.assertFalse(rows[0]["exact_segment_match_applicable"])
        self.assertEqual(rows[0]["exact_segment_match"], "")

    def test_timestamp_join_and_target_index_validation(self) -> None:
        rows = align_identity_rows(
            observation_rows=[_reference()],
            funnel_rows=[_funnel()],
            segment_rows=[_segment()],
            policy_label="batched",
        )
        self.assertTrue(rows[0]["target_index_mapping_valid"])
        changed = _reference()
        changed["actual_timestamp"] = "2014-07-01 03:00:00"
        missing = align_identity_rows(
            observation_rows=[changed],
            funnel_rows=[_funnel()],
            segment_rows=[_segment()],
            policy_label="batched",
        )
        self.assertEqual(missing[0]["classification"], CLASSIFICATIONS["I"])

    def test_same_reference_is_reused_by_multiple_rollouts(self) -> None:
        first = _funnel()
        second = _funnel()
        second["input_index"] = 4
        second["horizon_step"] = 2
        second_segment = _segment()
        second_segment["input_index"] = 4
        second_segment["horizon_step"] = 2
        rows = align_identity_rows(
            observation_rows=[_reference()],
            funnel_rows=[first, second],
            segment_rows=[_segment(), second_segment],
            policy_label="batched",
        )
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["reference_available"] for row in rows))

    def test_rollout_bootstrap_is_repeatable(self) -> None:
        rows = [
            {
                "input_index": index,
                "reference_available": True,
                "target_column_emitted_recall": index % 2 == 0,
                "reference_neuron_candidate_recall": True,
                "reference_neuron_emitted_recall": index % 2 == 0,
                "correct_column_wrong_neuron": index % 2 != 0,
            }
            for index in range(8)
        ]
        self.assertEqual(
            bootstrap_rows(rows, samples=20, seed=3),
            bootstrap_rows(rows, samples=20, seed=3),
        )

    def test_bootstrap_uses_scoped_and_conditional_denominators(self) -> None:
        rows = [
            {
                "input_index": 0,
                "field": "passenger",
                "horizon_step": 1,
                "reference_available": True,
                "neuron_reference_valid_before_observation": True,
                "reference_neuron_threshold_crossing_recall": True,
                "reference_neuron_candidate_recall": True,
                "reference_neuron_emitted_recall": False,
                "target_column_emitted_recall": True,
                "correct_column_wrong_neuron": True,
            },
            {
                "input_index": 0,
                "field": "passenger",
                "horizon_step": 1,
                "reference_available": True,
                "neuron_reference_valid_before_observation": False,
                "reference_neuron_threshold_crossing_recall": False,
                "reference_neuron_candidate_recall": False,
                "reference_neuron_emitted_recall": False,
                "target_column_emitted_recall": False,
                "correct_column_wrong_neuron": False,
            },
        ]
        output = bootstrap_rows(rows, samples=1, seed=0)
        indexed = {
            (
                row["reference_scope"],
                row["field"],
                row["horizon_step"],
                row["metric"],
            ): row
            for row in output
        }
        preexisting_crossing = indexed[
            (
                "PREEXISTING_NEURON_REFERENCES",
                "passenger",
                "1",
                "reference_neuron_crossing_recall",
            )
        ]
        operational_crossing = indexed[
            (
                "ALL_OPERATIONAL_REFERENCES",
                "passenger",
                "1",
                "reference_neuron_crossing_recall",
            )
        ]
        conditional_wrong = indexed[
            (
                "PREEXISTING_NEURON_REFERENCES",
                "passenger",
                "1",
                "correct_column_wrong_neuron_rate",
            )
        ]
        self.assertEqual(preexisting_crossing["mean"], 1.0)
        self.assertEqual(operational_crossing["mean"], 0.5)
        self.assertEqual(conditional_wrong["mean"], 1.0)

    def test_analysis_empty_trace_is_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = analyze(
                run_dir=root / "missing",
                output_dir=root / "analysis",
                policy_label="batched",
                bootstrap_samples=2,
                bootstrap_seed=0,
            )
            self.assertEqual(summary["target_columns"], 0)
            self.assertTrue(
                (root / "analysis" / "teacher_forced_identity_summary.json").exists()
            )

    def test_analysis_and_capture_do_not_call_prediction_or_observe(self) -> None:
        sources = (
            inspect.getsource(align_identity_rows),
            inspect.getsource(build_teacher_forced_observation_rows),
        )
        for source in sources:
            for forbidden in (
                "predict_code(",
                ".observe(",
                ".observe_code(",
                "learn_actual_code(",
            ):
                self.assertNotIn(forbidden, source)

    def test_all_required_safety_markers_are_true(self) -> None:
        self.assertTrue(all(DIAGNOSTIC_MARKERS.values()))


class TeacherForcedIdentityIntegrationTests(unittest.TestCase):
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
        for label, enabled, details in (
            ("off", False, False),
            ("on", True, False),
            ("details", True, True),
        ):
            directory = cls.root / label
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
                preselection_segment_diagnostic=True,
                preselection_segment_level="crossing",
                checkpoint_path=directory / "checkpoint.pkl",
                checkpoint_at_index=10,
                teacher_forced_winner_diagnostic=enabled,
                teacher_forced_winner_level="segment",
                observe_scenario_diagnostic=details,
                observe_scenario_level="full",
                reference_neuron_selection_diagnostic=details,
                reference_neuron_selection_level="crossing",
            )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_on_off_preserves_predictions_model_and_rng(self) -> None:
        off = self.root / "off"
        on = self.root / "on"
        self.assertEqual(
            _sha256(off / "original_predictions.csv"),
            _sha256(on / "original_predictions.csv"),
        )
        self.assertEqual(
            self.results["off"]["final_model_fingerprint"],
            self.results["on"]["final_model_fingerprint"],
        )
        self.assertEqual(
            self.results["off"]["final_rng_fingerprint"],
            self.results["on"]["final_rng_fingerprint"],
        )

    def test_on_off_preserves_existing_diagnostic_traces(self) -> None:
        off = self.root / "off"
        on = self.root / "on"
        self.assertEqual(
            _stable_csv(off / "competition_trace.csv"),
            _stable_csv(on / "competition_trace.csv"),
        )
        for name in (
            "oracle_candidate_trace.csv",
            "branch_candidate_trace.csv",
            "branch_segment_trace.csv",
            "preselection_segment_trace.csv",
            "preselection_group_trace.csv",
            "preselection_replacement_trace.csv",
        ):
            self.assertEqual(
                (off / name).read_bytes(),
                (on / name).read_bytes(),
                name,
            )
        self.assertEqual(
            _stable_csv(off / "preselection_funnel_trace.csv"),
            _stable_csv(on / "preselection_funnel_trace.csv"),
        )

    def test_detail_hooks_preserve_predictions_model_rng_and_traces(self) -> None:
        baseline = self.root / "on"
        details = self.root / "details"
        self.assertEqual(
            _sha256(baseline / "original_predictions.csv"),
            _sha256(details / "original_predictions.csv"),
        )
        self.assertEqual(
            self.results["on"]["final_model_fingerprint"],
            self.results["details"]["final_model_fingerprint"],
        )
        self.assertEqual(
            self.results["on"]["final_rng_fingerprint"],
            self.results["details"]["final_rng_fingerprint"],
        )
        for name in (
            "oracle_candidate_trace.csv",
            "branch_candidate_trace.csv",
            "branch_segment_trace.csv",
            "preselection_segment_trace.csv",
            "preselection_group_trace.csv",
            "preselection_replacement_trace.csv",
            "teacher_forced_observation_trace.csv",
        ):
            self.assertEqual(
                (baseline / name).read_bytes(),
                (details / name).read_bytes(),
                name,
            )

    def test_observe_detail_trace_has_exact_nonempty_reasons(self) -> None:
        path = self.root / "details" / "observe_scenario_trace.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(rows)
        self.assertTrue(all(row["scenario_assignment_reason"] for row in rows))
        allowed = {
            "CORRECTLY_PREDICTED_CELL_AVAILABLE",
            "MATCHING_SEGMENT_FOUND",
            "NO_EXISTING_SEGMENT_IN_COLUMN",
            "EXISTING_SEGMENTS_BELOW_L_MATCH",
            "MATCHING_SEGMENT_NOT_ELIGIBLE",
            "PREDICTED_TIME_INVALID",
        }
        self.assertTrue(
            {row["scenario_assignment_reason"] for row in rows} <= allowed
        )
        self.assertTrue(
            all(
                int(row["gap_to_L_match"])
                == 4 - int(row["best_matching_segment_overlap"])
                for row in rows
            )
        )

    def test_reference_rows_have_one_winner_and_real_scenarios(self) -> None:
        path = self.root / "on" / "teacher_forced_observation_trace.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(rows)
        self.assertTrue(
            all(int(row["observed_winner_count_in_column"]) == 1 for row in rows)
        )
        self.assertTrue(
            {row["observe_scenario"] for row in rows}
            <= {"scenario1", "scenario2", "scenario3"}
        )
        self.assertTrue(
            all(row["future_observation_does_not_affect_past_prediction"] == "True"
                for row in rows)
        )

    def test_scenario_segment_capture_and_created_semantics(self) -> None:
        path = self.root / "on" / "teacher_forced_observation_trace.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if row["observe_scenario"] in {"scenario1", "scenario2"}:
                self.assertEqual(
                    row["selected_segment_id"],
                    row["reinforced_segment_id"],
                )
            if row["observe_scenario"] == "scenario3" and row["created_segment_id"]:
                self.assertEqual(
                    row["selected_segment_id"],
                    row["created_segment_id"],
                )

    def test_checkpoint_format_and_strict_protocol_stay_clean(self) -> None:
        checkpoint = load_strict_checkpoint(
            self.root / "on" / "checkpoint.pkl"
        )
        self.assertEqual(checkpoint["checkpoint_format"], "fig9-strict-v1")
        protocol = json.loads(
            (self.root / "on" / "original_protocol.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(
            any("teacher" in key.lower() for key in protocol)
        )

    def test_off_has_no_teacher_trace_and_on_analysis_joins(self) -> None:
        self.assertFalse(
            (self.root / "off" / "teacher_forced_observation_trace.csv").exists()
        )
        summary = analyze(
            run_dir=self.root / "on",
            output_dir=self.root / "analysis",
            policy_label="batched",
            bootstrap_samples=10,
            bootstrap_seed=2,
        )
        self.assertGreater(summary["target_columns"], 0)
        self.assertEqual(summary["target_index_mapping_valid_rate"], 1.0)

    def test_reference_ids_survive_checkpoint_reload(self) -> None:
        path = self.root / "on" / "teacher_forced_observation_trace.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            row = next(csv.DictReader(handle))
        expected = stable_observation_reference_id(
            stream_label=row["stream_label"],
            actual_record_index=int(row["actual_record_index"]),
            actual_timestamp=row["actual_timestamp"],
            target_column=int(row["target_column"]),
        )
        self.assertEqual(row["observation_reference_id"], expected)
        self.assertEqual(
            load_strict_checkpoint(
                self.root / "on" / "checkpoint.pkl"
            )["checkpoint_format"],
            "fig9-strict-v1",
        )
