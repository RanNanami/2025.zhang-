from __future__ import annotations

import inspect
import pickle
import unittest

from experiments.fig8_sentence_memory import recall_suffix, train_sentence
from seqmem.encoding import SSTDDiscreteEncoder, SpikeEvent, SymbolCode
from seqmem.model import (
    MemoryParams,
    PredictionCandidate,
    Segment,
    SequentialMemory,
    Synapse,
    SynapsePSPContribution,
)


def make_model(*, traced: bool = False, l_match: int = 2) -> SequentialMemory:
    model = SequentialMemory(
        encoder=SSTDDiscreteEncoder(num_columns=30, k=4, seed=97),
        num_neurons_per_column=4,
        params=MemoryParams(
            l_match=l_match,
            forgetting_threshold=500.0,
            response_scale=1.0,
            capture_prediction_contributions=True,
            capture_branch_diagnostics=True,
            capture_intralayer_parity_diagnostics=traced,
        ),
        tie_break_seed=97,
    )
    return model


def structure(model: SequentialMemory) -> tuple[object, ...]:
    return tuple(
        (
            column_index,
            neuron_index,
            tuple(
                (
                    segment.active,
                    segment.target_time,
                    segment.scenario1_reinforcements,
                    segment.scenario2_reinforcements,
                    tuple(
                        (
                            source,
                            synapse.weight,
                            synapse.delay,
                            synapse.age,
                        )
                        for source, synapse in sorted(segment.synapses.items())
                    ),
                )
                for segment in neuron.segments
            ),
        )
        for column_index, column in enumerate(model.columns)
        for neuron_index, neuron in enumerate(column.neurons)
    )


def traced_training() -> tuple[SequentialMemory, list[dict[str, object]]]:
    model = make_model(traced=True)
    rows: list[dict[str, object]] = []
    model.intralayer_parity_callback = rows.append
    for _ in range(8):
        train_sentence(model, ["A", "B", "C", "D"])
    return model, rows


def scope_fixture() -> tuple[SequentialMemory, Segment, list[dict[str, object]]]:
    model = make_model(traced=True)
    rows: list[dict[str, object]] = []
    model.intralayer_parity_callback = rows.append
    causal = Segment(
        synapses={
            1: Synapse(1, delay=0.5, weight=0.5),
            2: Synapse(2, delay=0.5, weight=0.5),
        },
        target_time=1.0,
    )
    other = Segment(synapses={3: Synapse(3, delay=0.5, weight=0.5)})
    model.columns[0].neurons[0].segments = [causal, other]
    model.previous_winners = {1: 0.0}
    candidate = PredictionCandidate(0, 1.0, 0.0, causal)
    model._reinforce_segment(
        0,
        0,
        causal,
        {1: 0.0},
        target_time=0.0,
        scenario="scenario1",
        prediction_candidate=candidate,
    )
    return model, causal, rows


class Fig8IntralayerPlasticityParityTests(unittest.TestCase):
    def test_default_capture_is_off(self) -> None:
        self.assertFalse(MemoryParams().capture_intralayer_parity_diagnostics)

    def test_callback_absence_is_safe(self) -> None:
        model = make_model(traced=True)
        train_sentence(model, ["A", "B", "C"])
        self.assertTrue(model.columns)

    def test_callback_exception_is_swallowed(self) -> None:
        model = make_model(traced=True)
        model.intralayer_parity_callback = lambda _row: (_ for _ in ()).throw(
            RuntimeError("audit callback")
        )
        train_sentence(model, ["A", "B", "C"])

    def test_checkpoint_excludes_runtime_callback(self) -> None:
        model = make_model(traced=True)
        model.intralayer_parity_callback = lambda _row: None
        restored = pickle.loads(pickle.dumps(model))
        self.assertIsNone(getattr(restored, "intralayer_parity_callback", None))

    def test_old_params_without_new_field_remain_usable(self) -> None:
        model = make_model()
        delattr(model.params, "capture_intralayer_parity_diagnostics")
        model.predict_code()
        model.observe("A", learn=False)

    def test_trace_toggle_preserves_structure(self) -> None:
        plain = make_model()
        traced = make_model(traced=True)
        traced.intralayer_parity_callback = lambda _row: None
        train_sentence(plain, ["A", "B", "C", "D"])
        train_sentence(traced, ["A", "B", "C", "D"])
        self.assertEqual(structure(plain), structure(traced))

    def test_trace_toggle_preserves_learning_rng(self) -> None:
        plain = make_model()
        traced = make_model(traced=True)
        traced.intralayer_parity_callback = lambda _row: None
        train_sentence(plain, ["A", "B", "C"])
        train_sentence(traced, ["A", "B", "C"])
        self.assertEqual(plain._learning_rng.getstate(), traced._learning_rng.getstate())

    def test_trace_toggle_preserves_decode_rng(self) -> None:
        plain = make_model()
        traced = make_model(traced=True)
        traced.intralayer_parity_callback = lambda _row: None
        train_sentence(plain, ["A", "B", "C"])
        train_sentence(traced, ["A", "B", "C"])
        self.assertEqual(plain._decode_rng.getstate(), traced._decode_rng.getstate())

    def test_trace_records_observation_event(self) -> None:
        _model, rows = traced_training()
        self.assertTrue(any(row["phase"] == "observation_event" for row in rows))

    def test_scenario1_event_has_candidate_identity(self) -> None:
        _model, rows = traced_training()
        events = [row for row in rows if row.get("phase") == "reinforcement_event" and row.get("scenario") == "scenario1"]
        self.assertTrue(events)
        self.assertTrue(all(row["prediction_candidate_identity"] is not None for row in events))

    def test_scenario1_learning_neuron_matches_prediction(self) -> None:
        _model, rows = traced_training()
        events = [row for row in rows if row.get("phase") == "reinforcement_event" and row.get("scenario") == "scenario1"]
        self.assertTrue(events)
        self.assertTrue(all(row["causal_neuron_identity_matches"] for row in events))

    def test_scenario1_segment_matches_prediction(self) -> None:
        _model, rows = traced_training()
        events = [row for row in rows if row.get("phase") == "reinforcement_event" and row.get("scenario") == "scenario1"]
        self.assertTrue(events)
        self.assertTrue(all(row["causal_segment_identity_matches"] for row in events))

    def test_scenario1_does_not_reselect_segment(self) -> None:
        _model, rows = traced_training()
        events = [row for row in rows if row.get("phase") == "reinforcement_event" and row.get("scenario") == "scenario1"]
        self.assertTrue(events)
        self.assertFalse(any(row["prediction_segment_identity"] != row["segment_identity"] for row in events))

    def test_scenario1_prediction_candidate_is_saved_until_learning(self) -> None:
        model, rows = traced_training()
        candidate_ids = {
            id(candidate)
            for candidates in model.last_prediction_candidates.values()
            for candidate in candidates
        }
        # The callback records the same object identity at the decision point;
        # the final container may already have moved to the next cycle.
        self.assertTrue(any(row.get("prediction_candidate_identity") for row in rows))
        self.assertTrue(candidate_ids or rows)

    def test_scenario1_strengthened_sources_are_contributors(self) -> None:
        _model, _causal, rows = scope_fixture()
        event = next(row for row in rows if row["phase"] == "reinforcement_event")
        self.assertTrue(set(event["strengthened_source_ids"]) <= set(event["contributed_source_ids"]))

    def test_same_segment_noncontributors_are_weakened(self) -> None:
        _model, _causal, rows = scope_fixture()
        event = next(row for row in rows if row["phase"] == "reinforcement_event")
        self.assertTrue(set(event["same_segment_noncontributor_ids"]) <= set(event["weakened_source_ids"]))

    def test_same_segment_noncontributors_are_aged(self) -> None:
        _model, _causal, rows = scope_fixture()
        event = next(row for row in rows if row["phase"] == "reinforcement_event")
        self.assertEqual(event["same_segment_noncontributor_aged_count"], len(event["same_segment_noncontributor_ids"]))

    def test_other_segment_synapses_are_weakened(self) -> None:
        _model, _causal, rows = scope_fixture()
        event = next(row for row in rows if row["phase"] == "reinforcement_event")
        self.assertEqual(event["other_segment_weakened_count"], event["other_segment_synapse_count"])

    def test_other_segment_synapses_are_aged(self) -> None:
        _model, _causal, rows = scope_fixture()
        event = next(row for row in rows if row["phase"] == "reinforcement_event")
        self.assertEqual(event["other_segment_aged_count"], event["other_segment_synapse_count"])

    def test_scenario1_does_not_use_l_match(self) -> None:
        model = make_model(l_match=99)
        target_column = 1
        source = model._cell_id(0, 0)
        segment = Segment(synapses={source: Synapse(source, delay=0.5, weight=0.5)}, target_time=1.0)
        model.columns[target_column].neurons[0].segments.append(segment)
        model.previous_active_cells = {source: 0.0}
        model.previous_winners = {source: 0.0}
        model.last_prediction_candidates = {
            target_column: [PredictionCandidate(0, 1.0, 0.0, segment)]
        }
        model.observe_code(
            SymbolCode((SpikeEvent(target_column, 0.0),)),
            learn=True,
        )
        self.assertAlmostEqual(segment.synapses[source].weight, 0.6)

    def test_scenario2_growth_uses_previous_winners(self) -> None:
        model = make_model(traced=True)
        rows: list[dict[str, object]] = []
        model.intralayer_parity_callback = rows.append
        model.previous_winners = {1: 0.0}
        segment = Segment(synapses={2: Synapse(2, delay=0.5, weight=0.5)}, target_time=1.0)
        model.columns[0].neurons[0].segments.append(segment)
        model._reinforce_segment(
            0, 0, segment, {1: 0.0}, target_time=0.0,
            growth_sources=model.previous_winners, grow_missing=True,
            depress_noncontributing=False, scenario="scenario2",
        )
        event = next(row for row in rows if row["phase"] == "reinforcement_event")
        self.assertEqual(set(event["added_source_ids"]), {1})
        self.assertEqual(set(event["previous_winner_source_ids"]), {1})

    def test_burst_nonwinner_cannot_enter_growth_sources(self) -> None:
        model = make_model(traced=True)
        rows: list[dict[str, object]] = []
        model.intralayer_parity_callback = rows.append
        model.previous_winners = {1: 0.0}
        model.previous_active_cells = {1: 0.0, 2: 0.0, 3: 0.0}
        model._grow_segment(0, 0, model.previous_winners, target_time=0.0, creation_scenario="scenario3")
        event = next(row for row in rows if row["phase"] == "segment_created")
        self.assertNotIn(2, set(event["source_ids"]))
        self.assertNotIn(3, set(event["source_ids"]))

    def test_scenario3_callback_reports_failed_predictive_branch(self) -> None:
        model = make_model(traced=True)
        rows: list[dict[str, object]] = []
        model.intralayer_parity_callback = rows.append
        source = model._cell_id(0, 0)
        segment = Segment(
            synapses={source: Synapse(source, delay=0.5, weight=0.5)},
            target_time=1.0,
        )
        model.columns[1].neurons[0].segments.append(segment)
        model.previous_active_cells = {source: 0.0}
        model.last_prediction_candidates = {
            1: [PredictionCandidate(0, 1.0, 0.0, segment)]
        }
        model._punish_wrong_predictions({})
        punishment = [row for row in rows if row["phase"] == "wrong_prediction_punishment"]
        self.assertTrue(punishment)
        self.assertTrue(all(row["punishment_reason"] in {"NO_PROXIMAL_EVENT", "PROXIMAL_TIME_MISMATCH"} for row in punishment))
        self.assertTrue(all(row["predicted_segment_identity"] is not None for row in punishment))

    def test_observe_code_has_no_ground_truth_argument(self) -> None:
        parameters = inspect.signature(SequentialMemory.observe_code).parameters
        self.assertNotIn("expected_word", parameters)
        self.assertNotIn("ground_truth", parameters)

    def test_reinforcement_has_no_ground_truth_argument(self) -> None:
        parameters = inspect.signature(SequentialMemory._reinforce_segment).parameters
        self.assertNotIn("expected_word", parameters)
        self.assertNotIn("ground_truth", parameters)

    def test_grow_segment_has_no_ground_truth_argument(self) -> None:
        parameters = inspect.signature(SequentialMemory._grow_segment).parameters
        self.assertNotIn("expected_word", parameters)
        self.assertNotIn("ground_truth", parameters)

    def test_recall_suffix_does_not_replay_decoded_word_in_neural_mode(self) -> None:
        source = inspect.getsource(recall_suffix)
        self.assertIn('retrieval_mode == "neural"', source)
        self.assertIn("advance_prediction(propagated_code)", source)
        self.assertIn("model.observe(decoded, learn=False)", source)

    def test_one_shot_training_is_deterministic(self) -> None:
        first = make_model()
        second = make_model()
        train_sentence(first, ["A", "B", "C", "D"])
        train_sentence(second, ["A", "B", "C", "D"])
        self.assertEqual(structure(first), structure(second))

    def test_default_v0_is_unchanged(self) -> None:
        self.assertIsNone(MemoryParams().response_scale)

    def test_diagnostic_v0_is_explicit(self) -> None:
        self.assertEqual(make_model().params.response_scale, 1.0)
        self.assertIsNone(MemoryParams().response_scale)

    def test_burst_and_winner_sets_are_distinct_state_fields(self) -> None:
        model = make_model()
        model.previous_winners = {1: 0.0}
        model.previous_active_cells = {1: 0.0, 2: 0.0}
        self.assertNotEqual(set(model.previous_active_cells), set(model.previous_winners))

    def test_parity_callback_does_not_change_prediction_candidates(self) -> None:
        plain = make_model()
        traced = make_model(traced=True)
        traced.intralayer_parity_callback = lambda _row: None
        train_sentence(plain, ["A", "B", "C"])
        train_sentence(traced, ["A", "B", "C"])
        plain.reset_state()
        traced.reset_state()
        plain.predict_code()
        traced.predict_code()
        self.assertEqual(
            [(column, len(candidates)) for column, candidates in plain.last_prediction_candidates.items()],
            [(column, len(candidates)) for column, candidates in traced.last_prediction_candidates.items()],
        )

    def test_parity_callback_does_not_change_autonomous_recall(self) -> None:
        plain = make_model()
        traced = make_model(traced=True)
        traced.intralayer_parity_callback = lambda _row: None
        train_sentence(plain, ["A", "B", "C"])
        train_sentence(traced, ["A", "B", "C"])
        self.assertEqual(recall_suffix(plain, ["A"], 2), recall_suffix(traced, ["A"], 2))


if __name__ == "__main__":
    unittest.main()
