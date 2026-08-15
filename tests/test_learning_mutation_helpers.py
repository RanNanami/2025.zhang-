from __future__ import annotations

import inspect
import random
import unittest

from seqmem._learning_helpers import (
    apply_failed_prediction_updates,
    apply_selected_segment_updates,
    depress_other_segment_updates,
)
from seqmem.model import Segment, Synapse


def synapse_state(segment: Segment) -> tuple[tuple[int, float, int, float], ...]:
    return tuple(
        (source, synapse.weight, synapse.age, synapse.delay)
        for source, synapse in segment.synapses.items()
    )


class LearningMutationHelperTests(unittest.TestCase):
    def test_selected_segment_updates_match_frozen_loop(self) -> None:
        actual = Segment(
            synapses={
                7: Synapse(7, weight=0.95, age=4),
                2: Synapse(2, weight=0.40, age=1),
                9: Synapse(9, weight=0.05, age=2),
            }
        )
        expected = Segment(
            synapses={
                source: Synapse(
                    source,
                    delay=synapse.delay,
                    weight=synapse.weight,
                    age=synapse.age,
                )
                for source, synapse in actual.synapses.items()
            }
        )
        contributed = {7, 2}
        for source, synapse in expected.synapses.items():
            if source in contributed:
                synapse.weight = min(1.0, synapse.weight + 0.1)
                synapse.age = 0
            else:
                synapse.weight = max(0.0, synapse.weight - 0.1)
                synapse.age += 1

        apply_selected_segment_updates(
            actual.synapses,
            contributed,
            delta_w=0.1,
            depress_noncontributing=True,
        )

        self.assertEqual(synapse_state(actual), synapse_state(expected))

    def test_other_segment_depression_preserves_objects_and_order(self) -> None:
        selected = Segment(synapses={1: Synapse(1, weight=0.5, age=0)})
        other = Segment(
            synapses={
                4: Synapse(4, weight=0.05, age=2),
                3: Synapse(3, weight=0.8, age=5),
            }
        )
        segments = [selected, other]
        identities = [id(segment) for segment in segments]
        source_order = list(other.synapses)

        depress_other_segment_updates(
            segments,
            selected,
            delta_w=0.1,
            enabled=True,
        )

        self.assertEqual([id(segment) for segment in segments], identities)
        self.assertEqual(list(other.synapses), source_order)
        self.assertEqual(synapse_state(selected), ((1, 0.5, 0, 1.0),))
        self.assertEqual(
            synapse_state(other),
            ((4, 0.0, 3, 1.0), (3, 0.7000000000000001, 6, 1.0)),
        )

    def test_failed_prediction_updates_match_frozen_loop(self) -> None:
        segment = Segment(
            synapses={
                5: Synapse(5, weight=0.20, age=7),
                1: Synapse(1, weight=0.04, age=2),
                8: Synapse(8, weight=0.90, age=0),
            }
        )
        apply_failed_prediction_updates(
            segment.synapses,
            {5, 1},
            delta_w_bad=0.05,
        )
        self.assertEqual(
            synapse_state(segment),
            ((5, 0.15000000000000002, 8, 1.0), (1, 0.0, 3, 1.0), (8, 0.9, 0, 1.0)),
        )

    def test_helpers_do_not_use_rng_or_replace_containers(self) -> None:
        rng = random.Random(19)
        before = rng.getstate()
        segment = Segment(synapses={1: Synapse(1)})
        synapses = segment.synapses
        apply_selected_segment_updates(
            synapses,
            {1},
            delta_w=0.1,
            depress_noncontributing=True,
        )
        apply_failed_prediction_updates(
            synapses,
            {1},
            delta_w_bad=0.05,
        )
        self.assertIs(segment.synapses, synapses)
        self.assertEqual(rng.getstate(), before)
        for helper in (
            apply_selected_segment_updates,
            depress_other_segment_updates,
            apply_failed_prediction_updates,
        ):
            self.assertNotIn("random", inspect.getsource(helper))


if __name__ == "__main__":
    unittest.main()
