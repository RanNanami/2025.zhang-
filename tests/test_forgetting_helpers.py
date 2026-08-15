from __future__ import annotations

import inspect
import random
import unittest

from seqmem._forgetting_helpers import forgetting_score, synapse_is_retained
from seqmem.model import MemoryParams, Neuron, Segment, Synapse


class ForgettingHelperTests(unittest.TestCase):
    def test_score_matches_frozen_expression_bit_for_bit(self) -> None:
        for weight in (0.0, 0.125, 0.5, 0.875, 1.0):
            for age in (0, 1, 7, 501):
                for l_weight, l_age in ((1.0, 1.0), (0.25, 2.5)):
                    expected = l_weight * (1.0 - weight) + l_age * age
                    actual = forgetting_score(weight, age, l_weight, l_age)
                    with self.subTest(
                        weight=weight,
                        age=age,
                        l_weight=l_weight,
                        l_age=l_age,
                    ):
                        self.assertEqual(actual.hex(), expected.hex())

    def test_retention_uses_strict_less_than(self) -> None:
        score = forgetting_score(0.5, 2, 1.0, 1.0)
        self.assertFalse(synapse_is_retained(0.5, 2, 1.0, 1.0, score))
        self.assertTrue(
            synapse_is_retained(0.5, 2, 1.0, 1.0, score + 0.001)
        )
        self.assertFalse(
            synapse_is_retained(0.5, 2, 1.0, 1.0, score - 0.001)
        )

    def test_synapse_method_remains_compatible(self) -> None:
        synapse = Synapse(source=3, weight=0.4, age=8)
        self.assertEqual(
            synapse.forgetting_score(0.25, 2.0).hex(),
            forgetting_score(0.4, 8, 0.25, 2.0).hex(),
        )

    def test_helpers_have_no_model_rng_or_mutation_access(self) -> None:
        rng = random.Random(23)
        before = rng.getstate()
        synapse_is_retained(0.5, 2, 1.0, 1.0, 25.0)
        self.assertEqual(rng.getstate(), before)
        source = inspect.getsource(forgetting_score) + inspect.getsource(
            synapse_is_retained
        )
        for forbidden in ("random", "SequentialMemory", ".weight =", ".age ="):
            self.assertNotIn(forbidden, source)

    def test_pruning_retains_order_and_object_identity(self) -> None:
        keep = Synapse(source=5, weight=0.9, age=0)
        remove = Synapse(source=2, weight=0.0, age=30)
        segment = Segment(synapses={5: keep, 2: remove})
        neuron = Neuron(segments=[segment])
        params = MemoryParams(
            l_weight=1.0,
            l_age=1.0,
            forgetting_threshold=25.0,
        )

        retained = {
            source: synapse
            for source, synapse in segment.synapses.items()
            if synapse_is_retained(
                synapse.weight,
                synapse.age,
                params.l_weight,
                params.l_age,
                params.forgetting_threshold,
            )
        }

        self.assertEqual(list(retained), [5])
        self.assertIs(retained[5], keep)
        self.assertIs(neuron.segments[0], segment)


if __name__ == "__main__":
    unittest.main()
