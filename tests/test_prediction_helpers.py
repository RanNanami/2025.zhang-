from __future__ import annotations

import inspect
import math
import random
import unittest

from seqmem._prediction_helpers import (
    build_emitted_spike_events,
    build_prediction_candidate,
    build_prediction_stats,
    prediction_event_key,
    prediction_time_is_outside_window,
    segment_upper_bound_is_below_threshold,
)
from seqmem.encoding import SpikeEvent
from seqmem.model import (
    PredictionCandidate,
    Segment,
    SynapsePSPContribution,
)


class PredictionHelperTests(unittest.TestCase):
    def test_upper_bound_predicate_matches_original_expression(self) -> None:
        for v_rest in (-0.25, 0.0, math.nan):
            for upper_bound in (0.0, 0.5, 1.0, math.nan):
                for threshold in (0.5, 1.0):
                    with self.subTest(
                        v_rest=v_rest,
                        upper_bound=upper_bound,
                        threshold=threshold,
                    ):
                        self.assertEqual(
                            segment_upper_bound_is_below_threshold(
                                v_rest,
                                upper_bound,
                                threshold,
                            ),
                            v_rest + upper_bound < threshold,
                        )

    def test_firing_window_predicate_preserves_boundaries(self) -> None:
        cycle = 1.0
        tolerance = 0.03
        upper = cycle + cycle / 2.0 + tolerance
        cases = (
            (cycle - 1e-12, True),
            (cycle, False),
            (upper, False),
            (upper + 1e-12, True),
        )
        for target_time, expected in cases:
            with self.subTest(target_time=target_time):
                self.assertEqual(
                    prediction_time_is_outside_window(
                        target_time,
                        cycle,
                        tolerance,
                    ),
                    expected,
                )

    def test_event_key_uses_exact_existing_rounding(self) -> None:
        for value in (1.0, 1.1234567890124, 1.1234567890126):
            self.assertEqual(
                prediction_event_key(7, value),
                (7, round(value, 12)),
            )

    def test_candidate_construction_preserves_objects_and_float_values(self) -> None:
        segment = Segment()
        kept = SynapsePSPContribution(1, 0.1, 0.2, 0.5, 0.25)
        dropped = object()
        metadata = {
            "first_threshold_crossing_time": 1.125,
            "crossing_synapse_contributions": (kept, dropped),
            "peak_dendritic_potential": 1.75,
            "predicted_soma_firing_time": 1.25,
        }

        candidate = build_prediction_candidate(
            candidate_factory=PredictionCandidate,
            contribution_type=SynapsePSPContribution,
            neuron_index=3,
            score=1.5,
            target_time=1.25,
            cycle_period=1.0,
            segment=segment,
            metadata=metadata,
            dendrite_threshold=1.0,
        )

        self.assertIs(candidate.segment, segment)
        self.assertIs(candidate.crossing_synapse_contributions[0], kept)
        self.assertEqual(candidate.time.hex(), (1.25 - 1.0).hex())
        self.assertEqual(candidate.threshold_margin.hex(), (1.75 - 1.0).hex())
        self.assertEqual(candidate.dendritic_crossing_time, 1.125)
        self.assertEqual(candidate.predicted_soma_firing_time, 1.25)

    def test_stats_and_events_preserve_key_and_event_order(self) -> None:
        stats = build_prediction_stats(
            active_source_count=1,
            candidate_segment_count=2,
            threshold_crossing_segment_count=3,
            accepted_candidate_count=4,
            raw_event_count=5,
            raw_predicted_column_count=6,
        )
        self.assertEqual(
            list(stats),
            [
                "active_source_count",
                "candidate_segment_count",
                "threshold_crossing_segment_count",
                "accepted_candidate_count",
                "raw_event_count",
                "raw_predicted_column_count",
            ],
        )
        first = Segment()
        second = Segment()
        selected = [
            (8, (1, 0.9, 1.25, first)),
            (2, (3, 0.8, 1.50, second)),
        ]
        events = build_emitted_spike_events(
            selected,
            cycle_period=1.0,
            event_factory=SpikeEvent,
        )
        self.assertEqual(
            events,
            (SpikeEvent(8, 0.25), SpikeEvent(2, 0.5)),
        )

    def test_helpers_do_not_access_rng_model_or_scientific_engines(self) -> None:
        rng = random.Random(29)
        before = rng.getstate()
        prediction_event_key(1, 1.0)
        self.assertEqual(rng.getstate(), before)
        helpers = (
            segment_upper_bound_is_below_threshold,
            prediction_time_is_outside_window,
            prediction_event_key,
            build_prediction_candidate,
            build_prediction_stats,
            build_emitted_spike_events,
        )
        source = "\n".join(inspect.getsource(helper) for helper in helpers)
        for forbidden in (
            "random",
            "SequentialMemory",
            "spike_response",
            "potential(",
            "sorted(",
            ".sort(",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
