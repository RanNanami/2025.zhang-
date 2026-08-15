from __future__ import annotations

import unittest

from seqmem._state_helpers import (
    shallow_copy_candidate_map,
    shallow_copy_dict,
    shallow_copy_list,
    shallow_copy_set,
)


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


if __name__ == "__main__":
    unittest.main()
