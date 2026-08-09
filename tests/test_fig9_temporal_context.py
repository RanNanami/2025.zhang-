import gzip
import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from experiments.diagnostics.fig9_temporal_context import (
    NA,
    TRAJECTORY_ACTUAL,
    TRAJECTORY_AUTONOMOUS,
    TemporalContextTracker,
    _age_metrics,
    protocol,
    source_id_fingerprint,
)


class _Registry:
    def provenance_for(self, segment):
        return SimpleNamespace(segment_provenance_id="segment-1")


def _tracker_row(tracker=None, *, trajectory=TRAJECTORY_AUTONOMOUS, record=10, step=1):
    tracker = tracker or TemporalContextTracker(
        actual_last_activation={1: 8, 2: 5},
        autonomous_last_activation={3: 0},
    )
    segment = SimpleNamespace(
        synapses={
            1: SimpleNamespace(age=2, delay=0.10),
            2: SimpleNamespace(age=3, delay=0.20),
            3: SimpleNamespace(age=4, delay=0.30),
        },
        creation_transition_index=2,
        scenario1_reinforcements=3,
        scenario2_reinforcements=1,
    )
    contributions = (
        SimpleNamespace(source_cell_id=1, source_time=0.10, arrival_time=0.20, psp_contribution=0.6),
        SimpleNamespace(source_cell_id=2, source_time=0.20, arrival_time=0.40, psp_contribution=0.5),
        SimpleNamespace(source_cell_id=3, source_time=0.30, arrival_time=0.60, psp_contribution=0.4),
    )
    return tracker._base_row(
        trajectory_kind=trajectory,
        actual_record_index=record,
        horizon_step=step,
        field="passenger",
        observed_encoded_column="",
        candidate_target_column=5,
        candidate_target_neuron=2,
        segment_id="segment-1",
        ordinal=0,
        candidate_score=1.2,
        source_ids=(1, 2, 3),
        contribution_items=contributions,
        segment=segment,
        candidate_time=0.5,
        target_columns={5},
        oracle_target_column=5,
        response_peak=1.1,
    )


class TemporalContextSummaryTests(unittest.TestCase):
    def test_01_protocol_is_diagnostic_only(self):
        self.assertTrue(protocol()["diagnostic_only"])

    def test_02_protocol_default_off(self):
        self.assertFalse(protocol()["default_enabled"])

    def test_03_protocol_is_causal(self):
        self.assertIn("records <", protocol()["causal_state"])

    def test_04_protocol_no_ground_truth_selection(self):
        self.assertFalse(protocol()["ground_truth_used_for_selection"])

    def test_05_protocol_no_future_selection(self):
        self.assertFalse(protocol()["future_information_used_for_selection"])

    def test_06_age_mean(self):
        self.assertEqual(_age_metrics([1, 3, 5])["mean_age"], 3.0)

    def test_07_age_median(self):
        self.assertEqual(_age_metrics([1, 3, 7, 9])["median_age"], 5.0)

    def test_08_age_min(self):
        self.assertEqual(_age_metrics([4, 1])["min_age"], 1)

    def test_09_age_max(self):
        self.assertEqual(_age_metrics([4, 1])["max_age"], 4)

    def test_10_age_std(self):
        self.assertAlmostEqual(_age_metrics([1, 3])["std_age"], 1.0)

    def test_11_age_span(self):
        self.assertEqual(_age_metrics([1, 3, 8])["age_span"], 7)

    def test_12_age_iqr(self):
        self.assertAlmostEqual(_age_metrics([1, 2, 3, 4])["iqr_age"], 1.5)

    def test_13_entropy_is_finite(self):
        self.assertTrue(_age_metrics([1, 2, 3])["age_entropy"] > 0)

    def test_14_recent_1(self):
        self.assertAlmostEqual(_age_metrics([0, 1, 3])["recent_1_fraction"], 1 / 3)

    def test_15_recent_3(self):
        self.assertAlmostEqual(_age_metrics([0, 1, 3])["recent_3_fraction"], 2 / 3)

    def test_16_recent_5(self):
        self.assertAlmostEqual(_age_metrics([0, 4, 5])["recent_5_fraction"], 2 / 3)

    def test_17_recent_10(self):
        self.assertAlmostEqual(_age_metrics([0, 9, 10])["recent_10_fraction"], 2 / 3)

    def test_18_recent_20(self):
        self.assertAlmostEqual(_age_metrics([0, 19, 20])["recent_20_fraction"], 2 / 3)

    def test_19_negative_age_is_removed(self):
        self.assertEqual(_age_metrics([-1, 2])["mean_age"], 2)

    def test_20_missing_age_is_na(self):
        self.assertEqual(_age_metrics([])["mean_age"], NA)

    def test_21_empty_recent_fraction_is_zero(self):
        self.assertEqual(_age_metrics([])["recent_5_fraction"], 0.0)

    def test_22_compactness_is_finite(self):
        self.assertTrue(_age_metrics([1, 4])["temporal_compactness"] > 0)

    def test_23_recentness_concentration_matches_recent5(self):
        metrics = _age_metrics([0, 4, 10])
        self.assertEqual(metrics["recentness_concentration"], metrics["recent_5_fraction"])

    def test_24_gap_mean(self):
        self.assertEqual(_age_metrics([1, 3, 8])["adjacent_age_gap_mean"], 3.5)

    def test_25_gap_max(self):
        self.assertEqual(_age_metrics([1, 3, 8])["adjacent_age_gap_max"], 5)

    def test_26_recent_cluster_size(self):
        self.assertEqual(_age_metrics([0, 2, 9])["recent_cluster_size"], 2)

    def test_27_source_fingerprint_is_order_invariant(self):
        self.assertEqual(source_id_fingerprint([3, 1, 2]), source_id_fingerprint([1, 2, 3]))

    def test_28_source_fingerprint_is_not_object_id(self):
        self.assertEqual(len(source_id_fingerprint([1, 2])), 64)

    def test_29_actual_and_autonomous_labels_are_distinct(self):
        self.assertNotEqual(TRAJECTORY_ACTUAL, TRAJECTORY_AUTONOMOUS)

    def test_30_actual_history_age(self):
        row = _tracker_row(trajectory=TRAJECTORY_ACTUAL)
        self.assertEqual(row["mean_any_age"], 3.5)

    def test_31_autonomous_history_is_separate(self):
        row = _tracker_row(trajectory=TRAJECTORY_AUTONOMOUS)
        self.assertEqual(row["predicted_history_source_count"], 1)
        self.assertEqual(row["actual_history_source_count"], 2)

    def test_32_unknown_history_is_not_zero(self):
        tracker = TemporalContextTracker()
        row = _tracker_row(tracker, trajectory=TRAJECTORY_ACTUAL)
        self.assertEqual(row["unknown_history_source_count"], 3)
        self.assertEqual(row["mean_any_age"], NA)

    def test_33_candidate_row_has_stable_id(self):
        row = _tracker_row()
        self.assertEqual(len(row["candidate_event_id"]), 64)

    def test_34_candidate_row_is_fixed_width(self):
        self.assertGreaterEqual(len(_tracker_row()), 60)

    def test_35_candidate_score_is_preserved(self):
        self.assertEqual(_tracker_row()["candidate_score"], 1.2)

    def test_36_response_peak_is_preserved(self):
        self.assertEqual(_tracker_row()["response_peak_time"], 1.1)

    def test_37_delay_mean_is_preserved(self):
        self.assertAlmostEqual(_tracker_row()["mean_contributor_delay"], 0.2)

    def test_38_delay_span_is_preserved(self):
        self.assertAlmostEqual(_tracker_row()["delay_span"], 0.2)

    def test_39_order_template_is_explicitly_unavailable(self):
        self.assertEqual(_tracker_row()["historical_order_agreement"], "TEMPORAL_TEMPLATE_UNAVAILABLE")

    def test_40_source_age_monotonicity_not_invented(self):
        self.assertEqual(_tracker_row()["source_age_monotonicity"], NA)

    def test_41_reinforcement_time_not_invented(self):
        self.assertEqual(_tracker_row()["time_since_last_segment_reinforcement"], NA)

    def test_42_reinforcement_count_preserved(self):
        self.assertEqual(_tracker_row()["segment_reinforcement_count"], 4)

    def test_43_recurrent_flag_is_step_sensitive(self):
        self.assertFalse(_tracker_row(step=1)["recurrent_trajectory_divergence"])
        self.assertTrue(_tracker_row(step=2)["recurrent_trajectory_divergence"])

    def test_44_checkpoint_round_trip(self):
        tracker = TemporalContextTracker(actual_last_activation={2: 4}, actual_transition_count=5)
        restored = TemporalContextTracker.from_checkpoint_payload(tracker.checkpoint_payload())
        self.assertEqual(restored.actual_last_activation, {2: 4})
        self.assertEqual(restored.actual_transition_count, 5)

    def test_45_begin_record_resets_autonomous_history(self):
        tracker = TemporalContextTracker(autonomous_last_activation={1: 1})
        tracker.begin_record(10)
        self.assertEqual(tracker.autonomous_last_activation, {})

    def test_46_record_actual_updates_only_actual_history(self):
        tracker = TemporalContextTracker(autonomous_last_activation={1: 1})
        tracker.record_actual_observation([2], 10)
        self.assertEqual(tracker.actual_last_activation[2], 10)
        self.assertEqual(tracker.autonomous_last_activation, {1: 1})

    def test_47_record_autonomous_updates_only_autonomous_history(self):
        tracker = TemporalContextTracker(actual_last_activation={1: 1})
        tracker.record_autonomous_state([2], 2)
        self.assertEqual(tracker.autonomous_last_activation[2], 2)
        self.assertEqual(tracker.actual_last_activation, {1: 1})

    def test_48_actual_source_summary_is_bounded(self):
        tracker = TemporalContextTracker(actual_last_activation={1: 1, 2: 2})
        tracker.record_actual_source_rows(
            source_rows=[
                {"contributes_to_actual_overlap": True, "segment_provenance_id": "s", "encoded_column": 5, "segment_target_neuron": 1, "field": "passenger", "source_cell_stable_id": 1, "source_synaptic_delay": 0.1, "actual_observation_time": 0.2},
                {"contributes_to_actual_overlap": True, "segment_provenance_id": "s", "encoded_column": 5, "segment_target_neuron": 1, "field": "passenger", "source_cell_stable_id": 2, "source_synaptic_delay": 0.1, "actual_observation_time": 0.2},
            ], actual_record_index=3, target_field_by_column=lambda _: "passenger", l_match=2,
        )
        self.assertEqual(tracker.actual_row_count, 1)

    def test_49_actual_source_false_rows_are_ignored(self):
        tracker = TemporalContextTracker()
        tracker.record_actual_source_rows(
            source_rows=[{"contributes_to_actual_overlap": False, "segment_provenance_id": "s"}],
            actual_record_index=3, target_field_by_column=lambda _: "passenger", l_match=2,
        )
        self.assertEqual(tracker.actual_row_count, 0)

    def test_50_streaming_trace_reaches_gzip_eof(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            tracker = TemporalContextTracker()
            tracker.enable_streaming(path)
            tracker._emit(_tracker_row(tracker))
            tracker.close()
            with gzip.open(path / "temporal_candidate_trace.csv.gz", "rt", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
