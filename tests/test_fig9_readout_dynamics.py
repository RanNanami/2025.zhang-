import csv
import gzip
import tempfile
import unittest
from pathlib import Path

from experiments.diagnostics.analyze_fig9_readout_dynamics import (
    contributor_audit,
    target_false_funnel,
)
from experiments.diagnostics.fig9_competitive_inhibition import (
    CompetitionCandidate,
    CompetitionDecision,
    CompetitionResult,
)
from experiments.diagnostics.fig9_readout_dynamics import (
    READOUT_TRACE_FIELDS,
    build_readout_column_rows,
    column_event_id,
    write_readout_trace,
)
from seqmem.encoding import SpikeEvent, SymbolCode
from seqmem.model import PredictionCandidate, Segment


class Fig9ReadoutDynamicsTests(unittest.TestCase):
    def selection(self, *, policy="existing"):
        return [{
            "policy": policy,
            "column": 2,
            "field": "Passenger",
            "group_candidate_count": 2,
            "within_column_candidate_neuron_count": 2,
            "local_top1_score": 1.2,
            "local_top2_score": 1.0,
            "local_score_margin": 0.2,
            "selected_neuron": 11 if policy == "max_candidate_score" else 10,
            "selected_segment_id": "2:11:1" if policy == "max_candidate_score" else "2:10:0",
            "selected_candidate_score": 1.2 if policy == "max_candidate_score" else 1.0,
            "selected_response_peak": 2.0 if policy == "max_candidate_score" else 1.5,
            "selected_predicted_time": 1.1 if policy == "max_candidate_score" else 1.0,
            "existing_policy_neuron": 10,
            "existing_policy_segment_id": "2:10:0",
            "existing_policy_candidate_index": 0,
            "existing_selected_neuron": 10,
            "existing_selected_segment": "2:10:0",
            "existing_selected_score": 1.0,
            "existing_selected_response_peak": 1.5,
            "existing_selected_predicted_time": 1.0,
            "maxscore_selected_neuron": 11,
            "maxscore_selected_segment": "2:11:1",
            "maxscore_selected_score": 1.2,
            "maxscore_selected_response_peak": 2.0,
            "maxscore_selected_predicted_time": 1.1,
        }]

    def test_column_event_id_is_deterministic_and_has_no_object_identity(self):
        first = column_event_id(7, 8, 2, "Passenger", 99)
        second = column_event_id(7, 8, 2, "Passenger", 99)
        self.assertEqual(first, second)
        self.assertNotIn("0x", first)
        self.assertNotIn("object", first)

    def test_missing_contributor_is_not_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.csv.gz"
            write_readout_trace(path, [{"contributor_count_if_available": None}])
            with gzip.open(path, "rt", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["contributor_count_if_available"], "NA")
            write_readout_trace(path, [{"contributor_count_if_available": 0}])
            with gzip.open(path, "rt", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["contributor_count_if_available"], "0")

    def test_posthoc_target_is_field_aware(self):
        rows = build_readout_column_rows(
            run_id="P0",
            record_index=10,
            anchor_index=11,
            horizon_step=1,
            field_for_column=lambda column: "Passenger" if column >= 2 else "Time",
            selection_rows=self.selection(),
            raw_code=SymbolCode((SpikeEvent(2, 1.0),)),
            competition_result=None,
            target_code=SymbolCode((SpikeEvent(2, 1.0), SpikeEvent(0, 0.5))),
        )
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["is_target_column"])
        self.assertEqual(rows[0]["target_label_source"], "posthoc_future_code")

    def test_preselector_and_maxscore_representative_are_persisted(self):
        rows = build_readout_column_rows(
            run_id="P3",
            record_index=10,
            anchor_index=11,
            horizon_step=1,
            field_for_column=lambda _column: "Passenger",
            selection_rows=self.selection(policy="max_candidate_score"),
            raw_code=SymbolCode((SpikeEvent(2, 1.0),)),
            competition_result=None,
            target_code=None,
        )
        row = rows[0]
        self.assertEqual(row["pre_selector_candidate_count"], 2)
        self.assertEqual(row["maxscore_selected_neuron"], 11)
        self.assertEqual(row["existing_selected_neuron"], 10)
        self.assertTrue(row["selector_changed_any"])
        self.assertAlmostEqual(row["selector_score_delta"], 0.2)
        self.assertAlmostEqual(row["selector_predicted_time_delta"], 0.1)

    def test_competition_outcomes_are_aggregate_per_column(self):
        candidate = PredictionCandidate(11, 1.2, 1.0, Segment())
        item = CompetitionCandidate(2, candidate, 0)
        decision = CompetitionDecision(
            candidate=item,
            original_score=1.2,
            accumulated_inhibition=0.3,
            effective_score=0.9,
            emitted=False,
            reason="effective_score_below_threshold",
            winning_predecessor_ids=("prior",),
            batch_index=4,
        )
        result = CompetitionResult(
            decisions=(decision,),
            emitted_candidates=(),
            inhibited_candidates=(item,),
            raw_candidate_count=1,
            raw_column_count=1,
            emitted_column_count=0,
        )
        rows = build_readout_column_rows(
            run_id="P2",
            record_index=10,
            anchor_index=11,
            horizon_step=2,
            field_for_column=lambda _column: "Passenger",
            selection_rows=self.selection(),
            raw_code=SymbolCode((SpikeEvent(2, 1.0),)),
            competition_result=result,
            target_code=None,
        )
        row = rows[0]
        self.assertEqual(row["competition_bucket"], 4)
        self.assertEqual(row["prior_competitor_count"], 1)
        self.assertAlmostEqual(row["inhibition_received"], 0.3)
        self.assertFalse(row["competition_survived"])
        self.assertTrue(row["competition_suppressed"])
        self.assertFalse(row["emitted"])

    def test_target_false_funnel_sums_are_explicit(self):
        rows = [
            {"run_id": "P2", "horizon_step": "1", "field": "Passenger", "is_target_column": "True", "pre_competition_present": "True", "emitted": "True", "competition_enabled": "True"},
            {"run_id": "P2", "horizon_step": "1", "field": "Passenger", "is_target_column": "False", "pre_competition_present": "True", "emitted": "False", "competition_enabled": "True"},
        ]
        funnel = target_false_funnel(rows, "P2")
        self.assertEqual(funnel[0]["target_candidate_availability"], 1)
        self.assertEqual(funnel[0]["false_candidate_count"], 1)
        self.assertEqual(funnel[0]["suppressed_false_count"], 1)
        self.assertEqual(funnel[0]["suppressed_target_count"], 0)

    def test_legacy_zero_contributor_ledger_is_marked_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "original_summary.json").write_text("{}", encoding="utf-8")
            (directory / "long_sequence_activity_trace.csv").write_text(
                "mean_contributors,peak_contributors,total_contributors_step1\n0.0,0.0,0\n",
                encoding="utf-8",
            )
            audit = contributor_audit("P0", directory)
        self.assertEqual(audit[0]["status"], "MISSING_LEGACY_ZERO_FALLBACK")
        self.assertEqual(audit[1]["status"], "MISSING_LEGACY_ZERO_FALLBACK")
        self.assertEqual(audit[2]["status"], "MISSING_LEGACY_ZERO_FALLBACK")
        self.assertTrue(all(row["status"] != "AVAILABLE" for row in audit))


if __name__ == "__main__":
    unittest.main()
