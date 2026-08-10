from __future__ import annotations

import csv
import gzip
import hashlib
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from experiments.diagnostics.fig9_ambiguity import (
    build_actual_observation_rows,
    build_autonomous_rollout_rows,
    score_summary,
    stable_context_signature,
)
from experiments.diagnostics.fig9_competitive_inhibition import CompetitionSettings
from experiments.diagnostics.fig9_oracle_candidate import FieldColumnRanges
from experiments.fig9 import TaxiRecord
from experiments.fig9_strict_reproduction import Fig9StrictConfig, run_strict_stream
from seqmem.model import MatchingCandidateTrace, ObservationEventTrace, ObservationTrace, Segment, Synapse


class _Registry:
    def __init__(self, segments: dict[int, str]) -> None:
        self.segments = segments

    def provenance_for(self, segment: object):
        origin = self.segments.get(id(segment))
        return SimpleNamespace(segment_provenance_id=origin) if origin else None


class Fig9AmbiguityUnitTests(unittest.TestCase):
    def test_context_signature_is_stable_and_does_not_use_object_identity(self) -> None:
        sources = {11: 0.5, 4: 0.25}
        first = stable_context_signature(sources)
        second = stable_context_signature(dict(reversed(list(sources.items()))))
        self.assertEqual(first, second)
        self.assertNotIn(str(id(sources)), first)
        self.assertEqual(len(first), hashlib.sha256().digest_size * 2)

    def test_score_summary_is_deterministic(self) -> None:
        result = score_summary([1.0, 1.0, 0.5])
        self.assertEqual(result["top_score_tie_count"], 2)
        self.assertEqual(result["local_score_margin"], 0.0)
        self.assertEqual(result, score_summary([0.5, 1.0, 1.0]))

    def test_actual_rows_keep_l2_labels_and_candidate_identity(self) -> None:
        source = 99
        segment_a = Segment(synapses={source: Synapse(source, weight=0.5, age=7)})
        segment_b = Segment(synapses={source: Synapse(source, weight=0.5, age=9)})
        registry = _Registry({id(segment_a): "segment-a", id(segment_b): "segment-b"})
        trace = ObservationTrace(
            capture_scenario_details=True,
            pre_observe_active_sources={source: 0.25},
            events=[
                ObservationEventTrace(
                    target_column=12,
                    target_time=0.5,
                    winner_neuron=2,
                    winner_cell_id=122,
                    scenario="scenario1",
                    was_predicted=True,
                    selected_segment=segment_a,
                    matching_candidates=(
                        MatchingCandidateTrace(2, segment_a, 2, 1.2),
                        MatchingCandidateTrace(3, segment_b, 3, 1.1),
                    ),
                    best_matching_overlap=3,
                )
            ],
        )
        rows = build_actual_observation_rows(
            observation_trace=trace,
            registry=registry,
            active_sources=trace.pre_observe_active_sources,
            actual_record_index=222,
            input_timestamp="2014-07-10 00:00:00",
            ranges=FieldColumnRanges.from_sizes(2, 2, 20),
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["candidate_segment_id"] for row in rows}, {"segment-a", "segment-b"})
        self.assertEqual(rows[0]["l2_match_class"], "L2_ONLY_STRICT")
        self.assertEqual(rows[1]["l2_match_class"], "L3_ONLY_VS_L4")
        self.assertEqual({row["trajectory_kind"] for row in rows}, {"actual_observation"})

    def test_autonomous_rows_are_separate_and_have_no_ground_truth_label(self) -> None:
        rows = build_autonomous_rollout_rows(
            selection_rows=[
                {
                    "input_index": 8,
                    "horizon_step": 2,
                    "column": 9,
                    "field": "passenger",
                    "group_candidate_count": 3,
                    "selected_segment_id": "segment-x",
                    "selected_neuron": 2,
                    "selected_candidate_score": 1.2,
                    "local_score_margin": 0.1,
                    "within_column_candidate_segment_count": 3,
                    "context_signature": "context-hash",
                    "context_source_count": 4,
                }
            ],
            actual_record_index=7,
            input_timestamp="2014-07-10 00:00:00",
        )
        self.assertEqual(rows[0]["trajectory_kind"], "autonomous_rollout")
        self.assertEqual(rows[0]["candidate_segment_id"], "segment-x")
        self.assertEqual(rows[0]["l2_match_class"], "")
        self.assertEqual(rows[0]["context_signature"], "context-hash")
        self.assertEqual(rows[0]["context_source_count"], 4)
        self.assertNotIn("target_column", rows[0])


class Fig9AmbiguityIntegrationTests(unittest.TestCase):
    @staticmethod
    def _records(count: int = 18) -> list[TaxiRecord]:
        start = datetime(2014, 7, 1)
        return [TaxiRecord(start, 1000.0 + index) for index in range(count)]

    def _run(self, root: Path, enabled: bool):
        output = root / ("on" if enabled else "off")
        data_path = root / "tiny.csv"
        data_path.write_text("timestamp,passenger_count\n", encoding="utf-8")
        return run_strict_stream(
            records=self._records(),
            data_path=data_path,
            stream_label="original",
            output_dir=output,
            config=Fig9StrictConfig(warmup=4, l_match=2),
            limit=18,
            competition_settings=CompetitionSettings(
                mode="competitive_raw",
                simultaneous_policy="batched",
            ),
            lmatch_real_ablation=True,
            ambiguity_diagnostic=enabled,
            print_fingerprint=False,
        )

    def test_diagnostic_is_read_only_and_stream_is_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            off = self._run(root, False)
            on = self._run(root, True)
            for key in (
                "mape",
                "coverage",
                "final_model_fingerprint",
                "final_rng_fingerprint",
                "final_segment_count",
                "final_synapse_count",
            ):
                self.assertEqual(off[key], on[key], key)
            off_csv = (root / "off" / "original_predictions.csv").read_bytes()
            on_csv = (root / "on" / "original_predictions.csv").read_bytes()
            self.assertEqual(off_csv, on_csv)
            trace_path = root / "on" / "ambiguity_event_trace.csv.gz"
            self.assertTrue(trace_path.exists())
            with gzip.open(trace_path, "rt", encoding="utf-8", newline="") as handle:
                trace_rows = list(csv.DictReader(handle))
            self.assertTrue(trace_rows)
            self.assertEqual(
                {row["trajectory_kind"] for row in trace_rows},
                {"actual_observation", "autonomous_rollout"},
            )
            self.assertTrue(
                all(row["ground_truth_does_not_affect_model"] == "True" for row in trace_rows)
            )


if __name__ == "__main__":
    unittest.main()
