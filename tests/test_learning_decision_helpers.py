from __future__ import annotations

import inspect
import random
import unittest

from seqmem._learning_helpers import (
    CANONICAL_PAPER_BRANCH_BY_INTERNAL,
    classify_observation_learning_branch,
)
from seqmem.encoding import SSTDDiscreteEncoder, SpikeEvent, SymbolCode
from seqmem.model import (
    MemoryParams,
    PredictionCandidate,
    Segment,
    SequentialMemory,
)


def legacy_branch_oracle(
    *,
    learning_enabled: bool,
    has_matching_segment: bool,
    was_predicted: bool,
    matching_segment_eligible: bool,
) -> str:
    """Frozen pre-Phase-06 branch conditions for differential testing."""

    if not has_matching_segment:
        return "scenario3"
    if not learning_enabled:
        return ""
    if was_predicted:
        return "scenario1"
    if matching_segment_eligible:
        return "scenario2"
    return "scenario3"


class LearningDecisionHelperTests(unittest.TestCase):
    def test_truth_table_matches_frozen_legacy_oracle(self) -> None:
        for learning_enabled in (False, True):
            for has_matching_segment in (False, True):
                for was_predicted in (False, True):
                    for matching_segment_eligible in (False, True):
                        values = {
                            "learning_enabled": learning_enabled,
                            "has_matching_segment": has_matching_segment,
                            "was_predicted": was_predicted,
                            "matching_segment_eligible": (
                                matching_segment_eligible
                            ),
                        }
                        with self.subTest(**values):
                            self.assertEqual(
                                classify_observation_learning_branch(**values),
                                legacy_branch_oracle(**values),
                            )

    def test_canonical_mapping_preserves_historical_labels(self) -> None:
        self.assertEqual(
            CANONICAL_PAPER_BRANCH_BY_INTERNAL["scenario1"], "PAPER_S1"
        )
        self.assertEqual(
            CANONICAL_PAPER_BRANCH_BY_INTERNAL["scenario2"], "PAPER_S2A"
        )
        self.assertEqual(
            CANONICAL_PAPER_BRANCH_BY_INTERNAL["scenario3"], "PAPER_S2B"
        )
        self.assertEqual(
            CANONICAL_PAPER_BRANCH_BY_INTERNAL[
                "wrong_prediction_punishment"
            ],
            "PAPER_S3",
        )
        with self.assertRaises(TypeError):
            CANONICAL_PAPER_BRANCH_BY_INTERNAL["scenario1"] = "changed"  # type: ignore[index]

    def test_helper_does_not_consume_learning_or_decode_rng(self) -> None:
        learning_rng = random.Random(7)
        decode_rng = random.Random(11)
        learning_before = learning_rng.getstate()
        decode_before = decode_rng.getstate()

        classify_observation_learning_branch(
            learning_enabled=True,
            has_matching_segment=True,
            was_predicted=False,
            matching_segment_eligible=True,
        )

        self.assertEqual(learning_rng.getstate(), learning_before)
        self.assertEqual(decode_rng.getstate(), decode_before)

    def test_helper_source_has_no_traversal_rng_or_float_recalculation(self) -> None:
        source = inspect.getsource(classify_observation_learning_branch)
        for forbidden in (
            "for ",
            "while ",
            "random",
            "timed_overlap",
            "sum(",
            "max(",
            "sort",
        ):
            self.assertNotIn(forbidden, source)

    def test_s1_mutation_precedes_punishment_and_publication(self) -> None:
        model = SequentialMemory(
            SSTDDiscreteEncoder(num_columns=4, k=1, seed=1),
            num_neurons_per_column=2,
            params=MemoryParams(continuous_dynamics=False),
            tie_break_seed=3,
        )
        segment = Segment(target_time=0.0)
        model.columns[0].neurons[1].segments.append(segment)
        candidate = PredictionCandidate(1, 1.0, 0.0, segment)
        model.last_prediction_candidates = {0: [candidate]}
        calls: list[str] = []

        model._reinforce_segment = lambda *args, **kwargs: calls.append("reinforce")  # type: ignore[method-assign]
        model._punish_wrong_predictions = lambda *args, **kwargs: calls.append("punish")  # type: ignore[method-assign]
        model.observe_code(
            SymbolCode(events=(SpikeEvent(column=0, time=0.0),)),
            learn=True,
        )

        self.assertEqual(calls, ["reinforce", "punish"])
        self.assertEqual(model.last_observe_stats["scenario1"], 1)
        source = inspect.getsource(SequentialMemory.observe_code)
        self.assertLess(
            source.index("self._punish_wrong_predictions"),
            source.index("self.previous_active_cells = active_cells"),
        )


if __name__ == "__main__":
    unittest.main()
