from __future__ import annotations

import unittest
from datetime import datetime

from experiments.diagnostics.fig9_branch_provenance import BranchProvenanceRegistry
from experiments.diagnostics.fig9_match_overlap import (
    capture_match_overlap,
    finalize_match_overlap,
)
from experiments.diagnostics.fig9_oracle_candidate import FieldColumnRanges
from experiments.diagnostics.fig9_segment_context_composition import (
    SegmentContextCompositionTracker,
)
from experiments.fig9 import TaxiRecord
from seqmem.encoding import SSTDDiscreteEncoder, SpikeEvent, SymbolCode
from seqmem.model import ObservationTrace, Segment, SequentialMemory, Synapse


class SegmentContextCompositionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = SequentialMemory(
            SSTDDiscreteEncoder(num_columns=6, k=1, seed=4),
            num_neurons_per_column=2,
        )
        self.ranges = FieldColumnRanges.from_sizes(2, 2, 2)
        self.record = TaxiRecord(datetime(2014, 7, 1, 0, 0), 100.0)
        source_a = self.model._cell_id(0, 0)
        source_b = self.model._cell_id(2, 0)
        self.segment = Segment(
            synapses={
                source_a: Synapse(source_a, delay=0.5),
                source_b: Synapse(source_b, delay=0.5),
            },
            target_time=1.0,
        )
        self.model.columns[4].neurons[1].segments.append(self.segment)
        self.model.previous_active_cells = {
            source_a: 0.0,
            source_b: 0.0,
        }
        self.model.previous_winners = {source_a: 0.0}
        self.registry = BranchProvenanceRegistry()
        self.registry.capture_new_segments(
            self.model,
            previous_segment_ids=set(),
            creation_transition_index=3,
            creation_sources=self.model.previous_winners,
        )
        self.registry.current_predicted_sources = {source_a}
        self.registry.current_burst_sources = {source_b}
        self.code = SymbolCode((SpikeEvent(4, 0.0),))

    def test_snapshot_is_read_only_and_keeps_multilabel_source_roles(self) -> None:
        before = (
            tuple(sorted(self.segment.synapses)),
            tuple(synapse.weight for synapse in self.segment.synapses.values()),
        )
        tracker = SegmentContextCompositionTracker(self.registry, level="source")
        tracker.capture_pre_observation(
            model=self.model,
            ranges=self.ranges,
            actual_record_index=4,
        )
        after = (
            tuple(sorted(self.segment.synapses)),
            tuple(synapse.weight for synapse in self.segment.synapses.values()),
        )
        self.assertEqual(before, after)
        roles = {
            int(source): str(row["source_origin_type"])
            for source, row in tracker._pending.source_stats.items()  # type: ignore[union-attr]
        }
        self.assertIn("ACTUAL_WINNER", roles[next(iter(roles))])
        self.assertIn("ACTUAL_BURST", roles[next(iter(roles))] if len(roles) == 1 else ";".join(roles.values()))

    def test_real_observation_produces_l2_composition_rows(self) -> None:
        tracker = SegmentContextCompositionTracker(self.registry, level="source")
        tracker.capture_pre_observation(
            model=self.model,
            ranges=self.ranges,
            actual_record_index=4,
        )
        capture = capture_match_overlap(
            model=self.model,
            code=self.code,
            registry=self.registry,
            stream_label="test",
            actual_record_index=4,
            actual_record=self.record,
            ranges=self.ranges,
            level="source",
        )
        observation = ObservationTrace(capture_scenario_details=True)
        self.model.observe_code(self.code, learn=False, observation_trace=observation)
        completed = finalize_match_overlap(capture, observation, self.registry)
        tracker.consume_transition(
            capture=capture,
            completed=completed,
            actual_record_index=4,
            stream_label="test",
        )
        self.assertEqual(len(tracker.match_rows), 1)
        self.assertEqual(tracker.match_rows[0]["overlap_count"], 2)
        self.assertTrue(tracker.match_rows[0]["passes_L2"])
        self.assertFalse(tracker.match_rows[0]["passes_L3"])
        self.assertEqual(len(tracker.source_rows), 2)
        self.assertEqual(tracker.match_rows[0]["trajectory_kind"], "actual_observation")

    def test_record_observation_only_updates_tracker_history(self) -> None:
        tracker = SegmentContextCompositionTracker(self.registry)
        before = tuple(
            (source, synapse.weight, synapse.age)
            for source, synapse in self.segment.synapses.items()
        )
        tracker.record_observation(model=self.model, actual_record_index=4)
        after = tuple(
            (source, synapse.weight, synapse.age)
            for source, synapse in self.segment.synapses.items()
        )
        self.assertEqual(before, after)
        self.assertGreaterEqual(len(tracker.source_history), 1)


if __name__ == "__main__":
    unittest.main()
