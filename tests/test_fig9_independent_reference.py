import unittest

from experiments.diagnostics.fig9_independent_reference import (
    IndependentReferenceTracker,
    MARKERS,
    REFERENCE_STATUSES,
    _stable_branch_id,
)
from experiments.diagnostics.fig9_branch_provenance import SegmentOrigin


class IndependentReferenceTests(unittest.TestCase):
    def test_tracker_checkpoint_round_trip_uses_stable_fields(self):
        tracker = IndependentReferenceTracker()
        tracker.by_column[7] = [{
            "segment_provenance_id": "segment-7",
            "branch_provenance_id": "branch-7",
            "actual_record_index": 12,
            "target_column": 7,
        }]
        restored = IndependentReferenceTracker.from_checkpoint_payload(
            tracker.checkpoint_payload()
        )
        self.assertEqual(restored.by_column, tracker.by_column)
        self.assertNotIn("segment_object_id", restored.by_column[7][0])

    def test_branch_id_is_process_independent(self):
        origin = SegmentOrigin(
            segment_provenance_id="s",
            creation_transition_index=4,
            target_column=8,
            target_neuron=2,
            creation_target_time=0.5,
            creation_source_cell_ids=(1, 2),
            creation_source_fingerprint="f",
            stable_creation_ordinal=0,
        )
        self.assertEqual(_stable_branch_id(origin), _stable_branch_id(origin))

    def test_markers_and_statuses_are_explicit(self):
        self.assertTrue(MARKERS["reference_does_not_affect_selection"])
        self.assertIn("REFERENCE_NOT_FOUND", REFERENCE_STATUSES)
        self.assertIn("CIRCULAR_REFERENCE_INVALID", REFERENCE_STATUSES)


if __name__ == "__main__":
    unittest.main()
