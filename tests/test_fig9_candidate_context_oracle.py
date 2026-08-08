from __future__ import annotations

import csv
import gzip
import inspect
import tempfile
import unittest
from pathlib import Path

from experiments.diagnostics.fig9_candidate_context_oracle import (
    ROW_UNIT_CANDIDATE,
    classify_overlap_regime,
    project_actual_composition_rows,
    stable_candidate_event_id,
    validate_lossless_join,
    write_join_outputs,
)
from experiments.diagnostics.analyze_fig9_context_oracle import _stable_seed


class CandidateContextOracleTests(unittest.TestCase):
    def test_candidate_event_id_is_deterministic(self) -> None:
        kwargs = {
            "trajectory_kind": "autonomous_rollout",
            "actual_record_index": 201,
            "horizon_step": 2,
            "field": "passenger",
            "observed_encoded_column": "",
            "candidate_target_column": 320,
            "candidate_target_neuron": 4,
            "stable_segment_provenance_id": "segment-a",
            "candidate_generation_ordinal": 7,
            "observed_event_time": 0.42,
        }
        self.assertEqual(
            stable_candidate_event_id(**kwargs),
            stable_candidate_event_id(**kwargs),
        )

    def test_candidate_event_id_does_not_use_object_identity(self) -> None:
        source = inspect.getsource(stable_candidate_event_id)
        self.assertNotIn("id(segment", source)
        self.assertNotIn("uuid", source.lower())
        self.assertNotIn("rng", source.lower())

    def test_exact_match_regime_vocabulary_is_unambiguous(self) -> None:
        self.assertEqual(
            classify_overlap_regime(2),
            ("OVERLAP_EXACT_2", "L2_ONLY_STRICT"),
        )
        self.assertEqual(
            classify_overlap_regime(3),
            ("OVERLAP_EXACT_3", "L3_ONLY_VS_L4"),
        )
        self.assertEqual(classify_overlap_regime(4), ("OVERLAP_GE_4",))

    def test_projected_actual_rows_have_explicit_row_unit_and_posthoc_label(self) -> None:
        rows = project_actual_composition_rows(
            [
                {
                    "actual_record_index": 5,
                    "field": "passenger",
                    "encoded_column": 320,
                    "target_column": 320,
                    "target_neuron": 2,
                    "segment_provenance_id": "s1",
                    "actual_observation_time": 0.5,
                    "overlap_count": 2,
                    "passes_L2": True,
                    "passes_L3": False,
                    "passes_L4": False,
                    "selected": True,
                }
            ]
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["row_unit"], ROW_UNIT_CANDIDATE)
        self.assertTrue(rows[0]["L2_ONLY_STRICT"])
        self.assertTrue(rows[0]["is_target_column_candidate"])
        self.assertEqual(rows[0]["original_candidate_score"], "")

    def test_lossless_join_detects_duplicate_ids(self) -> None:
        row = {"candidate_event_id": "a"}
        result = validate_lossless_join([row], [dict(row)])
        self.assertTrue(result["lossless"])
        duplicate = validate_lossless_join([row, dict(row)], [dict(row)])
        self.assertFalse(duplicate["lossless"])
        self.assertGreater(duplicate["duplicate_candidate_event_ids"], 0)

    def test_join_outputs_are_gzip_readable(self) -> None:
        rows = [
            {
                "candidate_event_id": "a",
                "trajectory_kind": "autonomous_rollout",
                "row_unit": ROW_UNIT_CANDIDATE,
                "is_target_column_candidate": True,
            }
        ]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = write_join_outputs(output, rows=rows)
            self.assertTrue(result["lossless"])
            with gzip.open(
                output / "candidate_context_oracle_trace.csv.gz",
                "rt",
                encoding="utf-8",
                newline="",
            ) as handle:
                loaded = list(csv.DictReader(handle))
            self.assertEqual(loaded[0]["candidate_event_id"], "a")

    def test_oracle_label_is_not_in_model_selector_source(self) -> None:
        source = inspect.getsource(project_actual_composition_rows)
        self.assertNotIn("predict_code(", source)
        self.assertNotIn("observe_code(", source)
        self.assertNotIn("learn_actual_code(", source)

    def test_offline_bootstrap_seed_is_process_stable(self) -> None:
        self.assertEqual(
            _stable_seed("field", "passenger", "candidate_score"),
            _stable_seed("field", "passenger", "candidate_score"),
        )
        self.assertNotEqual(
            _stable_seed("field", "passenger", "candidate_score"),
            _stable_seed("field", "weekday", "candidate_score"),
        )


if __name__ == "__main__":
    unittest.main()
