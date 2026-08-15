from __future__ import annotations

import unittest

from seqmem._state_helpers import (
    build_transient_snapshot,
    shallow_copy_candidate_map,
    shallow_copy_dict,
    shallow_copy_list,
    shallow_copy_set,
)
from seqmem.model import PredictionCandidate, Segment, TransientStateSnapshot


class TransientStateHelperTests(unittest.TestCase):
    def test_container_helpers_return_shallow_copies(self) -> None:
        marker = object()
        mapping = {1: marker}
        sequence = [marker]
        members = {1}

        mapping_copy = shallow_copy_dict(mapping)
        sequence_copy = shallow_copy_list(sequence)
        members_copy = shallow_copy_set(members)

        self.assertEqual(mapping_copy, mapping)
        self.assertIsNot(mapping_copy, mapping)
        self.assertIs(mapping_copy[1], marker)
        self.assertEqual(sequence_copy, sequence)
        self.assertIsNot(sequence_copy, sequence)
        self.assertIs(sequence_copy[0], marker)
        self.assertEqual(members_copy, members)
        self.assertIsNot(members_copy, members)

    def test_candidate_map_copies_only_containers(self) -> None:
        candidate = object()
        candidates = {4: [candidate]}

        copied = shallow_copy_candidate_map(candidates)

        self.assertEqual(copied, candidates)
        self.assertIsNot(copied, candidates)
        self.assertIsNot(copied[4], candidates[4])
        self.assertIs(copied[4][0], candidate)

    def test_snapshot_construction_preserves_candidate_identity(self) -> None:
        import random

        segment = Segment()
        candidate = PredictionCandidate(2, 1.25, 0.4, segment)
        active = {1: 0.1}
        candidates = {3: [candidate]}
        decode_rng = random.Random(7)
        learning_rng = random.Random(11)

        snapshot = build_transient_snapshot(
            TransientStateSnapshot,
            previous_active_cells=active,
            previous_winners={1: 0.1},
            last_prediction_candidates=candidates,
            last_prediction_stats={"candidate_count": 1},
            last_observe_stats={"scenario1": 1},
            last_symbol_ranking=[(2, 1, "B")],
            previous_predicted_sources={1},
            previous_burst_only_sources={4},
            decode_rng=decode_rng,
            learning_rng=learning_rng,
        )

        self.assertEqual(snapshot.previous_active_cells, active)
        self.assertIsNot(snapshot.previous_active_cells, active)
        self.assertIsNot(snapshot.last_prediction_candidates, candidates)
        self.assertIsNot(snapshot.last_prediction_candidates[3], candidates[3])
        self.assertIs(snapshot.last_prediction_candidates[3][0], candidate)
        self.assertIs(snapshot.last_prediction_candidates[3][0].segment, segment)
        self.assertEqual(snapshot.decode_rng_state, decode_rng.getstate())
        self.assertEqual(snapshot.learning_rng_state, learning_rng.getstate())


if __name__ == "__main__":
    unittest.main()
