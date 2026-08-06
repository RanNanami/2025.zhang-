import unittest

from experiments.diagnostics.fig9_actual_branch_provenance import (
    ActualBranchProvenanceTracker,
    MARKERS,
    stable_id,
)


class ActualBranchProvenanceTests(unittest.TestCase):
    def test_repeated_actual_reinforcement_keeps_anchor_identity(self):
        tracker = ActualBranchProvenanceTracker()
        first = tracker.record_actual(
            segment_id="segment-a", index=10, field="passenger", column=4,
            neuron=2, source_cells={1: 0.5}, reinforced=True,
        )
        second = tracker.record_actual(
            segment_id="segment-a", index=20, field="passenger", column=4,
            neuron=2, source_cells={1: 0.5}, reinforced=True,
        )
        self.assertEqual(first["actual_anchor_ids_seen"], second["actual_anchor_ids_seen"])
        self.assertEqual(second["actual_anchor_count"], 1)

    def test_distinct_actual_source_is_mixed_not_silently_selected(self):
        tracker = ActualBranchProvenanceTracker()
        tracker.record_actual(
            segment_id="segment-a", index=10, field="passenger", column=4,
            neuron=2, source_cells={1: 0.5}, reinforced=False,
        )
        state = tracker.record_actual(
            segment_id="segment-a", index=20, field="passenger", column=4,
            neuron=2, source_cells={2: 0.5}, reinforced=False,
        )
        self.assertEqual(state["actual_anchor_count"], 2)
        self.assertEqual(state["actual_branch_count"], 2)
        self.assertTrue(state["mixed_actual_history"])
        self.assertEqual(state["mixed_history_class"], "ACTUAL_BRANCH_MIXED_HISTORY")
        self.assertEqual(state["mixture_trigger"], "REINFORCEMENT_ADDED_NEW_ACTUAL_ANCHOR")

    def test_anchor_identity_does_not_depend_on_first_seen_index(self):
        tracker_a = ActualBranchProvenanceTracker()
        tracker_b = ActualBranchProvenanceTracker()
        a = tracker_a.record_actual(
            segment_id="segment-a", index=10, field="time", column=104,
            neuron=3, source_cells={1, 2}, reinforced=False,
        )
        b = tracker_b.record_actual(
            segment_id="segment-a", index=999, field="time", column=104,
            neuron=3, source_cells={1, 2}, reinforced=False,
        )
        self.assertEqual(a["actual_anchor_ids_seen"], b["actual_anchor_ids_seen"])
        self.assertEqual(a["actual_branch_ids_seen"], b["actual_branch_ids_seen"])

    def test_checkpoint_round_trip_preserves_stable_ids(self):
        tracker = ActualBranchProvenanceTracker()
        tracker.record_actual(
            segment_id="segment-a", index=10, field="passenger", column=4,
            neuron=2, source_cells={1: 0.5}, reinforced=True,
        )
        restored = ActualBranchProvenanceTracker.from_checkpoint_payload(
            tracker.checkpoint_payload()
        )
        self.assertEqual(restored.segments, tracker.segments)
        self.assertEqual(restored.anchors, tracker.anchors)
        self.assertEqual(restored.branches, tracker.branches)

    def test_metadata_is_explicitly_non_interfering(self):
        self.assertTrue(MARKERS["branch_provenance_does_not_affect_model"])
        self.assertTrue(MARKERS["branch_provenance_does_not_affect_rng"])
        self.assertEqual(stable_id("x", 1), stable_id("x", 1))


if __name__ == "__main__":
    unittest.main()
