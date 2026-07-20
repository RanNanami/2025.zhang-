from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from fig8c_poem_memory import InterLayerSequenceMemory
from generate_fig8c_stress_dataset import generate_poems
from prepare_fig8c_poems import split_clauses
from seqmem.encoding import SSTDDiscreteEncoder


class Fig8CPoemTests(unittest.TestCase):
    def test_stress_dataset_is_unique_but_contextually_ambiguous(self) -> None:
        poems = generate_poems(100, shared_first_lines=10, seed=9)
        self.assertEqual(len({poem["title"] for poem in poems}), 100)
        self.assertEqual(len({"".join(poem["lines"]) for poem in poems}), 100)
        self.assertEqual(len({poem["lines"][0] for poem in poems}), 10)
        self.assertTrue(all(len(line) == 5 for poem in poems for line in poem["lines"]))

    def test_clause_split_recovers_four_five_character_lines(self) -> None:
        text = "abcde,fghij.klmno,pqrst."
        self.assertEqual(split_clauses([text]), ["abcde", "fghij", "klmno", "pqrst"])

    def test_goal_feedback_round_trip(self) -> None:
        titles = SSTDDiscreteEncoder(30, 3, seed=1)
        lines = SSTDDiscreteEncoder(30, 3, seed=2)
        memory = InterLayerSequenceMemory(titles, lines)
        expected = ("line-a", "line-b", "line-c", "line-d")
        memory.learn("title", expected)
        self.assertEqual(memory.retrieve("title", 4), list(expected))

    def test_source_neurons_separate_colliding_concepts(self) -> None:
        concepts = SSTDDiscreteEncoder(1, 1, seed=1)
        items = SSTDDiscreteEncoder(8, 2, seed=2)
        memory = InterLayerSequenceMemory(concepts, items, source_neurons=2)
        memory.learn("title-a", ("line-a",))
        memory.learn("title-b", ("line-b",))
        self.assertEqual(memory.retrieve("title-a", 1), ["line-a"])
        self.assertEqual(memory.retrieve("title-b", 1), ["line-b"])

    def test_unlearned_concept_has_no_feedback(self) -> None:
        concepts = SSTDDiscreteEncoder(10, 2, seed=1)
        items = SSTDDiscreteEncoder(10, 2, seed=2)
        memory = InterLayerSequenceMemory(concepts, items)
        memory.learn("known", ("line",))
        self.assertEqual(memory.retrieve("unknown", 1), [])


if __name__ == "__main__":
    unittest.main()
