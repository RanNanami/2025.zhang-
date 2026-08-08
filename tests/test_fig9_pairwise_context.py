from __future__ import annotations

import inspect
import json
import gzip
import tempfile
import unittest

from experiments.diagnostics.analyze_fig9_pairwise_context import (
    ROW_UNIT_CANDIDATE,
    ROW_UNIT_PAIR,
    TRAJECTORY_ACTUAL,
    TRAJECTORY_AUTONOMOUS,
    _bootstrap_ci,
    _candidate_group_key,
    _candidate_pair_summary,
    _event_contexts,
    _parse_ids,
    _parse_metadata,
    _pearson,
    _ranking,
    _safe_ratio,
    _segment_source_sets,
    _source_metadata_from_trace,
    build_pair_trace,
    canonical_pair,
    pmi_like,
    shrunk_pmi,
    source_pairs,
    stable_pair_key,
    support_bucket,
)


class PairwiseContextTests(unittest.TestCase):
    def _base_candidate(self, **overrides):
        row = {
            "candidate_event_id": "c",
            "trajectory_kind": TRAJECTORY_AUTONOMOUS,
            "actual_record_index": "2",
            "horizon_step": "1",
            "field": "passenger",
            "candidate_target_column": "5",
            "candidate_target_neuron": "0",
            "stable_segment_provenance_id": "s",
            "overlap_source_ids": "[1,2]",
            "source_ids": "[1,2]",
            "is_target_column_candidate": "True",
            "original_candidate_score": "1.0",
            "selected_intracolumn": "True",
            "selected_after_intercolumn": "True",
            "L2_RELAXED_VS_L4": "True",
            "step_error": "0",
        }
        row.update(overrides)
        return row

    def test_pair_key_is_order_independent(self) -> None:
        self.assertEqual(canonical_pair(9, 2), (2, 9))
        self.assertEqual(stable_pair_key(9, 2), stable_pair_key(2, 9))
        self.assertEqual(source_pairs([3, 1, 2, 1]), [(1, 2), (1, 3), (2, 3)])

    def test_pair_key_does_not_use_object_identity_or_rng(self) -> None:
        source = inspect.getsource(stable_pair_key)
        self.assertNotIn("id(", source)
        self.assertNotIn("uuid", source.lower())
        self.assertNotIn("random", source.lower())

    def test_support_buckets_are_explicit(self) -> None:
        self.assertEqual([support_bucket(n) for n in (0, 1, 2, 4, 8)], ["0", "1", "2-3", "4-7", "8+"])

    def test_pmi_smoothing_and_shrinkage_are_finite_at_zero(self) -> None:
        value = pmi_like(0, 0, 0, 0)
        self.assertTrue(value == value)
        self.assertEqual(shrunk_pmi(value, 0), 0.0)

    def test_causal_pair_count_excludes_current_and_future_context(self) -> None:
        event_rows = [
            {"actual_record_index": 0, "context_source_ids": json.dumps([1, 2])},
            {"actual_record_index": 1, "context_source_ids": json.dumps([1, 2, 3])},
            {"actual_record_index": 2, "context_source_ids": json.dumps([2, 3])},
        ]
        source_rows = [
            {"actual_record_index": 2, "segment_provenance_id": "seg", "source_cell_stable_id": str(source), "source_field": "passenger", "source_column": str(source), "source_neuron": "0", "source_idf": "1.0", "source_live_segment_incidence": "2", "source_age": "1", "source_origin_type": "ACTUAL_WINNER"}
            for source in (1, 2)
        ]
        candidate_rows = [{
            "row_unit": "CANDIDATE_SEGMENT",
            "candidate_event_id": "candidate",
            "trajectory_kind": "actual_observation",
            "actual_record_index": "2",
            "horizon_step": "",
            "field": "passenger",
            "candidate_target_column": "5",
            "candidate_target_neuron": "0",
            "stable_segment_provenance_id": "seg",
            "overlap_source_ids": json.dumps([1, 2]),
            "source_ids": json.dumps([1, 2]),
            "is_target_column_candidate": "True",
            "original_candidate_score": "1.0",
            "L2_RELAXED_VS_L4": "True",
            "selected_intracolumn": "True",
            "step_error": "0",
        }]
        rows, summary = build_pair_trace(candidate_rows, event_rows=event_rows, source_rows=source_rows)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["row_unit"], ROW_UNIT_PAIR)
        self.assertEqual(rows[0]["pair_key"], "1:2")
        self.assertEqual(rows[0]["pair_actual_cooccurrence_count"], 2)
        self.assertEqual(rows[0]["pair_statistic_scope"], "ONLINE_CAUSAL_PREMATCH")
        self.assertEqual(summary["actual_context_scope"], "FULL_PREMATCH_CONTEXT")

    def test_actual_and_autonomous_trajectories_remain_separate(self) -> None:
        event_rows = [{"actual_record_index": 1, "context_source_ids": json.dumps([1, 2])}]
        source_rows = [
            {"actual_record_index": 1, "segment_provenance_id": "a", "source_cell_stable_id": str(source), "source_field": "time", "source_column": str(source), "source_neuron": "0", "source_idf": "1.0", "source_live_segment_incidence": "2", "source_age": "1", "source_origin_type": "ACTUAL_WINNER"}
            for source in (1, 2)
        ]
        base = {"candidate_event_id": "x", "actual_record_index": "1", "horizon_step": "1", "field": "time", "candidate_target_column": "5", "candidate_target_neuron": "0", "stable_segment_provenance_id": "a", "overlap_source_ids": "[1,2]", "source_ids": "[1,2]", "is_target_column_candidate": "False", "original_candidate_score": "1.0"}
        rows, _ = build_pair_trace([{**base, "trajectory_kind": "actual_observation"}, {**base, "candidate_event_id": "y", "trajectory_kind": "autonomous_rollout"}], event_rows=event_rows, source_rows=source_rows)
        self.assertEqual({row["trajectory_kind"] for row in rows}, {"actual_observation", "autonomous_rollout"})

    def test_canonical_pair_repeated_is_stable(self) -> None:
        self.assertEqual([canonical_pair(7, 3) for _ in range(10)], [(3, 7)] * 10)

    def test_source_pairs_remove_duplicates(self) -> None:
        self.assertEqual(source_pairs([4, 4, 4]), [])

    def test_source_pairs_empty_is_empty(self) -> None:
        self.assertEqual(source_pairs([]), [])

    def test_self_pair_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            canonical_pair(1, 1)

    def test_negative_support_bucket_is_zero(self) -> None:
        self.assertEqual(support_bucket(-4), "0")

    def test_pmi_is_finite_for_large_counts(self) -> None:
        self.assertTrue(abs(pmi_like(100, 120, 130, 200)) < 100)

    def test_pmi_alpha_changes_only_statistic(self) -> None:
        self.assertNotEqual(pmi_like(1, 2, 3, 10, alpha=1), pmi_like(1, 2, 3, 10, alpha=2))

    def test_shrunk_pmi_is_zero_without_support(self) -> None:
        self.assertEqual(shrunk_pmi(9.0, 0), 0.0)

    def test_shrunk_pmi_is_bounded_by_raw_value(self) -> None:
        self.assertLessEqual(shrunk_pmi(2.0, 1), 2.0)

    def test_parse_ids_is_sorted_and_unique(self) -> None:
        self.assertEqual(_parse_ids("[3, 1, 3, 2]"), (1, 2, 3))

    def test_parse_ids_rejects_non_list(self) -> None:
        self.assertEqual(_parse_ids('{"a": 1}'), ())

    def test_parse_ids_rejects_invalid_json(self) -> None:
        self.assertEqual(_parse_ids("not-json"), ())

    def test_parse_metadata_rejects_invalid_json(self) -> None:
        self.assertEqual(_parse_metadata("not-json"), {})

    def test_full_event_context_scope_is_preferred(self) -> None:
        contexts, scope = _event_contexts([{"actual_record_index": 0, "context_source_ids": "[1,2]"}], [])
        self.assertEqual(scope, "FULL_PREMATCH_CONTEXT")
        self.assertEqual(contexts[0], {1, 2})

    def test_empty_event_context_uses_explicit_fallback(self) -> None:
        contexts, scope = _event_contexts([], [{"actual_record_index": 0, "source_cell_stable_id": "9", "source_in_active_cells": "True"}])
        self.assertEqual(scope, "OBSERVED_ACTIVE_SEGMENT_SOURCE_SUBSET")
        self.assertEqual(contexts[0], {9})

    def test_source_metadata_is_indexed_by_record_and_segment(self) -> None:
        rows = [{"actual_record_index": "1", "segment_provenance_id": "s", "source_cell_stable_id": "2", "source_field": "time"}]
        result = _source_metadata_from_trace(rows)
        self.assertEqual(result[(1, "s")][2]["source_field"], "time")

    def test_segment_source_sets_keep_overlap_ids(self) -> None:
        row = self._base_candidate()
        result = _segment_source_sets([row], {})
        segment = result[(TRAJECTORY_AUTONOMOUS, 2, 1)]["s"]
        self.assertEqual(segment["overlap_source_ids"], (1, 2))

    def test_candidate_group_key_is_deterministic(self) -> None:
        row = self._base_candidate()
        self.assertEqual(_candidate_group_key(row), (2, 1, "passenger"))

    def test_overlap_source_ids_are_used_for_pairs(self) -> None:
        row = self._base_candidate(source_ids="[1,2,3]", overlap_source_ids="[1,2]")
        rows, _ = build_pair_trace([row], event_rows=[], source_rows=[])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["pair_key"], "1:2")

    def test_pair_segment_incidence_is_nonnegative(self) -> None:
        rows, _ = build_pair_trace([self._base_candidate()], event_rows=[], source_rows=[])
        self.assertGreaterEqual(rows[0]["pair_segment_cooccurrence_count"], 0)

    def test_pair_target_column_incidence_is_integer(self) -> None:
        rows, _ = build_pair_trace([self._base_candidate()], event_rows=[], source_rows=[])
        self.assertIsInstance(rows[0]["pair_target_column_incidence"], int)

    def test_same_field_pair_flag_is_boolean(self) -> None:
        rows, _ = build_pair_trace([self._base_candidate()], event_rows=[], source_rows=[])
        self.assertIsInstance(rows[0]["same_field_pair"], bool)

    def test_cross_field_pair_flag_is_boolean(self) -> None:
        rows, _ = build_pair_trace([self._base_candidate()], event_rows=[], source_rows=[])
        self.assertIsInstance(rows[0]["cross_field_pair"], bool)

    def test_origin_labels_are_posthoc_metadata(self) -> None:
        rows, _ = build_pair_trace([self._base_candidate()], event_rows=[], source_rows=[])
        self.assertIn("UNKNOWN", rows[0]["pair_origin_labels"])

    def test_candidate_summary_has_candidate_row_unit(self) -> None:
        pair_rows, _ = build_pair_trace([self._base_candidate()], event_rows=[], source_rows=[])
        summary = _candidate_pair_summary(pair_rows)
        self.assertEqual(summary[0]["row_unit"], ROW_UNIT_CANDIDATE)

    def test_candidate_summary_preserves_candidate_id(self) -> None:
        pair_rows, _ = build_pair_trace([self._base_candidate(candidate_event_id="keep-me")], event_rows=[], source_rows=[])
        self.assertEqual(_candidate_pair_summary(pair_rows)[0]["candidate_event_id"], "keep-me")

    def test_candidate_summary_pair_count_is_exact(self) -> None:
        pair_rows, _ = build_pair_trace([self._base_candidate(overlap_source_ids="[1,2,3]")], event_rows=[], source_rows=[])
        self.assertEqual(_candidate_pair_summary(pair_rows)[0]["pair_count"], 3)

    def test_paired_candidates_require_target_and_false(self) -> None:
        from experiments.diagnostics.analyze_fig9_pairwise_context import _paired_candidates
        rows = [self._base_candidate(candidate_event_id="target"), self._base_candidate(candidate_event_id="false", is_target_column_candidate="False")]
        pair_rows = []
        for row in rows:
            pair_rows.extend(build_pair_trace([row], event_rows=[], source_rows=[])[0])
        self.assertEqual(len(_paired_candidates(_candidate_pair_summary(pair_rows))), 1)

    def test_ranking_returns_zero_without_targets(self) -> None:
        self.assertEqual(_ranking([self._base_candidate(is_target_column_candidate="False")], "original_candidate_score")["paired_or_target_events"], 0)

    def test_ranking_top1_detects_target(self) -> None:
        rows = [self._base_candidate(candidate_event_id="target", original_candidate_score="2"), self._base_candidate(candidate_event_id="false", is_target_column_candidate="False", original_candidate_score="1")]
        self.assertEqual(_ranking(rows, "original_candidate_score")["target_top1_rate"], 1.0)

    def test_bootstrap_ci_is_deterministic(self) -> None:
        self.assertEqual(_bootstrap_ci([1.0, 2.0, 3.0], 7, samples=20), _bootstrap_ci([1.0, 2.0, 3.0], 7, samples=20))

    def test_bootstrap_ci_empty_is_none(self) -> None:
        self.assertEqual(_bootstrap_ci([], 7), (None, None))

    def test_pearson_is_one_for_identical_values(self) -> None:
        self.assertAlmostEqual(_pearson([1, 2, 3], [1, 2, 3]), 1.0)

    def test_pearson_is_none_for_constant_values(self) -> None:
        self.assertIsNone(_pearson([1, 1], [2, 3]))

    def test_safe_ratio_zero_denominator_is_zero(self) -> None:
        self.assertEqual(_safe_ratio(3, 0), 0.0)

    def test_callback_path_matches_list_path(self) -> None:
        row = self._base_candidate()
        direct, direct_summary = build_pair_trace([row], event_rows=[], source_rows=[])
        streamed = []
        _, streamed_summary = build_pair_trace([row], event_rows=[], source_rows=[], row_callback=streamed.append)
        self.assertEqual(direct, streamed)
        self.assertEqual(direct_summary["pair_rows"], streamed_summary["pair_rows"])

    def test_callback_path_does_not_change_trajectory(self) -> None:
        row = self._base_candidate(trajectory_kind=TRAJECTORY_ACTUAL)
        streamed = []
        build_pair_trace([row], event_rows=[], source_rows=[], row_callback=streamed.append)
        self.assertEqual(streamed[0]["trajectory_kind"], TRAJECTORY_ACTUAL)

    def test_current_context_is_not_counted_as_causal_history(self) -> None:
        row = self._base_candidate(actual_record_index="0")
        rows, _ = build_pair_trace([row], event_rows=[{"actual_record_index": 0, "context_source_ids": "[1,2]"}], source_rows=[])
        self.assertEqual(rows[0]["pair_actual_cooccurrence_count"], 0)

    def test_autonomous_rows_have_autonomous_trajectory(self) -> None:
        rows, _ = build_pair_trace([self._base_candidate()], event_rows=[], source_rows=[])
        self.assertEqual(rows[0]["trajectory_kind"], TRAJECTORY_AUTONOMOUS)

    def test_trace_rows_have_pair_row_unit(self) -> None:
        rows, _ = build_pair_trace([self._base_candidate()], event_rows=[], source_rows=[])
        self.assertEqual(rows[0]["row_unit"], ROW_UNIT_PAIR)

    def test_trace_does_not_mutate_candidate_input(self) -> None:
        row = self._base_candidate()
        before = dict(row)
        build_pair_trace([row], event_rows=[], source_rows=[])
        self.assertEqual(row, before)

    def test_analyzer_source_does_not_import_model(self) -> None:
        source = inspect.getsource(build_pair_trace)
        self.assertNotIn("SequentialMemory", source)

    def test_analyzer_source_has_no_observe_call(self) -> None:
        source = inspect.getsource(build_pair_trace)
        self.assertNotIn("observe(", source)

    def test_analyzer_source_has_no_learning_call(self) -> None:
        source = inspect.getsource(build_pair_trace)
        self.assertNotIn("learn(", source)

    def test_trace_id_is_not_object_id(self) -> None:
        self.assertNotIn("id(", inspect.getsource(build_pair_trace))

    def test_pair_key_has_no_random_dependency(self) -> None:
        self.assertNotIn("random", inspect.getsource(stable_pair_key).lower())

    def test_gzip_trace_can_be_read_to_eof(self) -> None:
        from experiments.diagnostics.analyze_fig9_pairwise_context import _GzipCsvSink
        with tempfile.TemporaryDirectory() as directory:
            path = __import__("pathlib").Path(directory) / "trace.csv.gz"
            sink = _GzipCsvSink(path)
            sink.write({"row_unit": ROW_UNIT_PAIR, "pair_key": "1:2"})
            sink.close()
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                self.assertIn("1:2", handle.read())


if __name__ == "__main__":
    unittest.main()
