import csv
import gzip
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from experiments.diagnostics.fig9_autonomous_context_provenance import (
    AutonomousContextProvenanceTracker,
    PRE_ACTUAL,
    PRE_PREDICTED,
    _stable_rollout_id,
)
from seqmem.encoding import SpikeEvent, SymbolCode
from seqmem.model import (
    PredictionCandidate,
    Segment,
    Synapse,
    SynapsePSPContribution,
)


class _DummyModel:
    def __init__(self, candidate):
        self.last_prediction_candidates = {0: [candidate]}
        self.columns = [SimpleNamespace(neurons=[object(), object()])]
        self.params = SimpleNamespace(timing_tolerance=0.03)


def _candidate(source: int = 10, neuron: int = 1) -> PredictionCandidate:
    segment = Segment(synapses={source: Synapse(source=source)})
    contribution = SynapsePSPContribution(
        source_cell_id=source,
        source_time=0.1,
        arrival_time=0.1,
        weight=0.5,
        psp_contribution=0.25,
    )
    return PredictionCandidate(
        neuron_index=neuron,
        score=1.0,
        time=0.1,
        segment=segment,
        dendritic_crossing_time=0.09,
        crossing_synapse_contributions=(contribution,),
        peak_dendritic_potential=1.2,
        threshold_margin=0.2,
    )


class AutonomousContextProvenanceTests(unittest.TestCase):
    def test_rollout_id_is_deterministic_and_does_not_use_object_ids(self):
        first = _stable_rollout_id(
            stream_label="original",
            anchor_timestamp="2015-01-01 00:00:00",
            seed=0,
            trajectory_ordinal=1,
        )
        second = _stable_rollout_id(
            stream_label="original",
            anchor_timestamp="2015-01-01 00:00:00",
            seed=0,
            trajectory_ordinal=1,
        )
        self.assertEqual(first, second)
        self.assertNotIn(str(id(object())), first)

    def test_identity_and_current_activation_are_separate(self):
        tracker = AutonomousContextProvenanceTracker("original")
        tracker.begin_rollout(
            anchor_record_index=20,
            anchor_timestamp="2015-01-01 00:00:00",
            pre_rollout_active={10: 0.1, 11: 0.2},
            pre_rollout_predicted={11},
            pre_rollout_burst=set(),
        )
        self.assertTrue(tracker._states[10].identity_actual)
        self.assertEqual(tracker._states[10].activation_origin, PRE_ACTUAL)
        self.assertTrue(tracker._states[11].identity_actual)
        self.assertEqual(tracker._states[11].activation_origin, PRE_PREDICTED)

    def test_step_origin_and_depth_are_propagated(self):
        tracker = AutonomousContextProvenanceTracker("original")
        tracker.begin_rollout(
            anchor_record_index=20,
            anchor_timestamp="2015-01-01 00:00:00",
            pre_rollout_active={10: 0.1},
            pre_rollout_predicted=set(),
            pre_rollout_burst=set(),
        )
        model = _DummyModel(_candidate())
        code = SymbolCode((SpikeEvent(column=0, time=0.1),))
        tracker.record_step(
            model=model,
            raw=code,
            propagated=code,
            next_active_cells={1: 1.0},
            competition_result=None,
            actual_record_index=20,
            horizon_step=1,
            field_for_column=lambda _column: "passenger",
            target_columns=(0,),
            target_column_by_field={"passenger": 0},
            input_index=21,
            input_timestamp="2015-01-01 00:00:00",
            target_timestamp="2015-01-01 00:30:00",
        )
        self.assertEqual(len(tracker.step_rows), 1)
        self.assertEqual(tracker._states[1].generated_depth, 1)
        self.assertTrue(tracker._states[1].propagated_from_actual)
        model.last_prediction_candidates = {0: [_candidate(source=1, neuron=0)]}
        tracker.record_step(
            model=model,
            raw=code,
            propagated=code,
            next_active_cells={0: 1.0},
            competition_result=None,
            actual_record_index=20,
            horizon_step=2,
            field_for_column=lambda _column: "passenger",
            target_columns=(0,),
            target_column_by_field={"passenger": 0},
            input_index=21,
            input_timestamp="2015-01-01 00:00:00",
            target_timestamp="2015-01-01 00:30:00",
        )
        self.assertEqual(tracker.step_rows[1]["origin_step1_count"], 1)

    def test_candidate_trace_is_bounded_and_gzip_is_readable(self):
        tracker = AutonomousContextProvenanceTracker("original", level="candidate")
        tracker.begin_rollout(
            anchor_record_index=20,
            anchor_timestamp="2015-01-01 00:00:00",
            pre_rollout_active={10: 0.1},
            pre_rollout_predicted=set(),
            pre_rollout_burst=set(),
        )
        model = _DummyModel(_candidate())
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            tracker.enable_streaming(output)
            code = SymbolCode((SpikeEvent(column=0, time=0.1),))
            tracker.record_step(
                model=model,
                raw=code,
                propagated=code,
                next_active_cells={1: 1.0},
                competition_result=None,
                actual_record_index=20,
                horizon_step=1,
                field_for_column=lambda _column: "passenger",
                target_columns=(0,),
                target_column_by_field={"passenger": 0},
                input_index=21,
                input_timestamp="2015-01-01 00:00:00",
                target_timestamp="2015-01-01 00:30:00",
            )
            tracker.close()
            with gzip.open(
                output / "autonomous_candidate_provenance_trace.csv.gz",
                "rt",
                encoding="utf-8",
                newline="",
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["source_identity_actual_history_count"], "1")
            self.assertNotIn("segment_identity", rows[0])
            self.assertNotIn("object", rows[0]["candidate_event_id"])


if __name__ == "__main__":
    unittest.main()
