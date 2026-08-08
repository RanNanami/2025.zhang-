import csv
import gzip
import tempfile
import unittest
from pathlib import Path

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

    def test_registry_writer_preserves_anchor_fields(self):
        from experiments.diagnostics.fig9_actual_branch_provenance import write_registry

        tracker = ActualBranchProvenanceTracker()
        tracker._new_anchor(
            segment_id="segment",
            index=1,
            field="passenger",
            column=2,
            neuron=3,
            source_sig="source",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_registry(Path(temp_dir) / "registry.csv", tracker, compress=True)
            with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        anchor_rows = [row for row in rows if row.get("actual_anchor_id")]
        self.assertEqual(len(anchor_rows), 1)
        self.assertEqual(anchor_rows[0]["anchor_segment_provenance_id"], "segment")
        self.assertEqual(anchor_rows[0]["parent_actual_anchor_ids"], "")

    def test_one_lineage_can_contain_many_anchors(self):
        tracker = ActualBranchProvenanceTracker()
        first = tracker.record_actual(
            segment_id="segment-a", index=1, field="passenger", column=4,
            neuron=2, source_cells={1: 0.5}, reinforced=True,
        )
        first_anchor = first["actual_anchor_ids_seen"][0]
        second = tracker.record_actual(
            segment_id="segment-a", index=2, field="passenger", column=4,
            neuron=2, source_cells={2: 0.5}, reinforced=True,
            parent_anchor_ids=[first_anchor],
        )
        self.assertEqual(second["actual_lineage_count"], 1)
        self.assertEqual(second["actual_anchor_count"], 2)
        child_anchor = [value for value in second["actual_anchor_ids_seen"] if value != first_anchor][0]
        self.assertEqual(
            tracker.anchors[child_anchor]["actual_lineage_id"],
            tracker.anchors[first_anchor]["actual_lineage_id"],
        )
        self.assertEqual(tracker.anchors[child_anchor]["lineage_class"], "LINEAGE_CONTINUATION_UNIQUE")

    def test_multiple_parent_lineages_are_marked_as_merge(self):
        tracker = ActualBranchProvenanceTracker()
        first = tracker.record_actual(
            segment_id="segment-a", index=1, field="passenger", column=4,
            neuron=2, source_cells={1: 0.5}, reinforced=True,
        )
        second = tracker.record_actual(
            segment_id="segment-b", index=2, field="passenger", column=5,
            neuron=3, source_cells={2: 0.5}, reinforced=True,
        )
        parents = [first["actual_anchor_ids_seen"][0], second["actual_anchor_ids_seen"][0]]
        merged = tracker.record_actual(
            segment_id="segment-c", index=3, field="passenger", column=6,
            neuron=4, source_cells={3: 0.5}, reinforced=True,
            parent_anchor_ids=parents,
        )
        child = [value for value in merged["actual_anchor_ids_seen"] if value not in parents]
        self.assertEqual(len(child), 1)
        self.assertEqual(tracker.anchors[child[0]]["lineage_class"], "LINEAGE_MERGE")
        self.assertEqual(len(tracker.anchors[child[0]]["parent_actual_lineage_ids"].split("|")), 2)
        self.assertFalse(tracker.lineage_events[-1]["future_data_used"])


if __name__ == "__main__":
    unittest.main()
