from __future__ import annotations

import unittest

from seqmem.dynamics import (
    DSDynamicsParams,
    DSNeuronState,
    DelayedSpike,
    dendritic_potential,
    spike_response,
)
from seqmem.encoding import (
    SSTDCompositeEncoder,
    SSTDDiscreteEncoder,
    SSTDPeriodicEncoder,
    SSTDRealValueEncoder,
    SpikeEvent,
    SymbolCode,
)
from seqmem.model import (
    MemoryParams,
    PredictionCandidate,
    Segment,
    SequentialMemory,
    Synapse,
)

from experiments.fig7_sequence_prediction import (
    build_sequence_set,
    make_stream,
    paper_forgetting_threshold,
)
from experiments.fig9_paper_snn import (
    error_ratio,
    learn_actual_code,
    mape,
    project_sparse_prediction,
    rank_sparse_predictions,
    reference_rolling_mape,
    rollout,
)


class PaperEncodingTests(unittest.TestCase):
    def test_nearby_real_values_have_more_overlap(self) -> None:
        encoder = SSTDRealValueEncoder(
            num_columns=101,
            k=5,
            minimum=0.0,
            maximum=100.0,
        )
        reference = set(encoder.encode(50.0).columns)
        nearby = set(encoder.encode(50.4).columns)
        distant = set(encoder.encode(90.0).columns)

        self.assertGreater(len(reference & nearby), len(reference & distant))

    def test_real_value_likelihood_decode_round_trip(self) -> None:
        encoder = SSTDRealValueEncoder(
            num_columns=101,
            k=5,
            minimum=0.0,
            maximum=100.0,
        )
        value = 42.4
        decoded = encoder.decode_likelihood(encoder.encode(value))
        self.assertLessEqual(abs(decoded - value), encoder.spacing / 2.0)

    def test_real_value_likelihood_search_handles_range_boundaries(self) -> None:
        encoder = SSTDRealValueEncoder(482, 10, 0.0, 40000.0)
        low = encoder.decode_likelihood(encoder.encode(0.0))
        high = encoder.decode_likelihood(encoder.encode(40000.0))
        self.assertLessEqual(low, encoder.spacing / 2.0)
        self.assertAlmostEqual(high, 40000.0)

    def test_real_value_likelihood_search_is_global_for_noisy_codes(self) -> None:
        encoder = SSTDRealValueEncoder(21, 5, 0.0, 20.0)
        expected = encoder.encode(15.0)
        noisy = SymbolCode(
            events=(SpikeEvent(encoder.column_offset, 0.0),) + expected.events[1:]
        )
        decoded = encoder.decode_likelihood(noisy)
        self.assertGreater(decoded, 10.0)

    def test_periodic_boundary_wraps(self) -> None:
        encoder = SSTDPeriodicEncoder(num_columns=48, k=5, period=24.0)
        late = set(encoder.encode(23.9).columns)
        early = set(encoder.encode(0.1).columns)
        noon = set(encoder.encode(12.0).columns)

        self.assertGreater(len(late & early), len(late & noon))

    def test_composite_encoder_keeps_field_column_ranges_disjoint(self) -> None:
        day = SSTDPeriodicEncoder(
            num_columns=30, k=10, period=7.0, column_offset=0
        )
        time = SSTDPeriodicEncoder(
            num_columns=58, k=10, period=48.0, column_offset=30
        )
        value = SSTDRealValueEncoder(
            num_columns=482,
            k=10,
            minimum=0.0,
            maximum=50000.0,
            column_offset=88,
        )
        encoder = SSTDCompositeEncoder([day, time, value])
        code = encoder.encode([2.0, 23.0, 12000.0])

        self.assertEqual(encoder.num_columns, 570)
        self.assertEqual(len(code.events), 30)
        self.assertEqual(len([column for column in code.columns if column < 30]), 10)
        self.assertEqual(len([column for column in code.columns if 30 <= column < 88]), 10)
        self.assertEqual(len([column for column in code.columns if column >= 88]), 10)

    def test_symbol_decode_distinguishes_firing_order(self) -> None:
        encoder = SSTDDiscreteEncoder(num_columns=8, k=2, seed=1)
        encoder._codes["forward"] = SymbolCode(
            events=(SpikeEvent(2, 0.0), SpikeEvent(5, 0.5))
        )
        encoder._codes["reverse"] = SymbolCode(
            events=(SpikeEvent(5, 0.0), SpikeEvent(2, 0.5))
        )
        encoder._symbols_by_event = {
            (2, 0.0): {"forward"}, (5, 0.5): {"forward"},
            (5, 0.0): {"reverse"}, (2, 0.5): {"reverse"},
        }
        model = SequentialMemory(
            encoder=encoder,
            num_neurons_per_column=2,
            params=MemoryParams(timing_tolerance=0.01),
        )
        model.predict_code = lambda: encoder.encode("reverse")  # type: ignore[method-assign]

        self.assertEqual(model.predict_symbol(), "reverse")

    def test_multiple_symbol_predictions_are_ranked_and_bounded(self) -> None:
        encoder = SSTDDiscreteEncoder(num_columns=8, k=2, seed=1)
        encoder._codes["best"] = SymbolCode(
            events=(SpikeEvent(2, 0.0), SpikeEvent(5, 0.5))
        )
        encoder._codes["partial"] = SymbolCode(
            events=(SpikeEvent(2, 0.0), SpikeEvent(6, 0.5))
        )
        encoder._codes["wrong-order"] = SymbolCode(
            events=(SpikeEvent(5, 0.0), SpikeEvent(2, 0.5))
        )
        encoder._symbols_by_event = {
            (2, 0.0): {"best", "partial"},
            (5, 0.5): {"best"},
            (6, 0.5): {"partial"},
            (5, 0.0): {"wrong-order"},
            (2, 0.5): {"wrong-order"},
        }
        model = SequentialMemory(
            encoder=encoder,
            num_neurons_per_column=2,
            params=MemoryParams(timing_tolerance=0.01),
        )
        model.predict_code = lambda: encoder.encode("best")  # type: ignore[method-assign]

        self.assertEqual(
            model.predict_symbols(max_predictions=2, min_overlap=1),
            ["best", "partial"],
        )

    def test_multiple_predictions_keep_one_column_at_distinct_times(self) -> None:
        encoder = SSTDDiscreteEncoder(num_columns=8, k=2, seed=1)
        encoder._codes["forward"] = SymbolCode(
            events=(SpikeEvent(2, 0.0), SpikeEvent(5, 0.5))
        )
        encoder._codes["reverse"] = SymbolCode(
            events=(SpikeEvent(5, 0.0), SpikeEvent(2, 0.5))
        )
        segment = Segment()
        model = SequentialMemory(encoder=encoder, num_neurons_per_column=2)

        def predict_both_orders() -> SymbolCode:
            model.last_prediction_candidates = {
                2: [
                    PredictionCandidate(0, 1.0, 0.0, segment),
                    PredictionCandidate(1, 1.0, 0.5, segment),
                ],
                5: [
                    PredictionCandidate(0, 1.0, 0.0, segment),
                    PredictionCandidate(1, 1.0, 0.5, segment),
                ],
            }
            return SymbolCode(
                events=(
                    SpikeEvent(2, 0.0),
                    SpikeEvent(5, 0.0),
                    SpikeEvent(2, 0.5),
                    SpikeEvent(5, 0.5),
                )
            )

        model.predict_code = predict_both_orders  # type: ignore[method-assign]

        self.assertEqual(
            set(
                model.predict_symbols(
                    max_predictions=2,
                    min_overlap=2,
                    candidate_symbols={"forward", "reverse"},
                )
            ),
            {"forward", "reverse"},
        )

    def test_forgetting_function_matches_paper_equation(self) -> None:
        synapse = Synapse(source=1, weight=0.5, age=3)
        self.assertAlmostEqual(synapse.forgetting_score(10.0, 1.0), 8.0)


class DSDynamicsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.params = DSDynamicsParams()

    def test_spike_response_has_unit_peak_after_arrival(self) -> None:
        self.assertEqual(spike_response(-0.01, self.params), 0.0)
        self.assertAlmostEqual(
            spike_response(self.params.kernel_peak_time, self.params), 1.0
        )

    def test_delays_align_spikes_at_dendritic_threshold(self) -> None:
        spikes = [
            DelayedSpike(firing_time=0.0, delay=0.20, weight=0.5),
            DelayedSpike(firing_time=0.1, delay=0.10, weight=0.5),
        ]
        evaluation_time = 0.20 + self.params.kernel_peak_time
        potential = dendritic_potential(spikes, evaluation_time, self.params)
        self.assertAlmostEqual(potential, self.params.dendrite_threshold)

    def test_segment_response_matches_explicit_delayed_spikes(self) -> None:
        segment = Segment(
            synapses={
                2: Synapse(source=2, delay=0.20, weight=0.5),
                4: Synapse(source=4, delay=0.10, weight=0.5),
            }
        )
        sources = {2: 0.0, 4: 0.1}
        target = 0.2
        explicit = dendritic_potential(
            [
                DelayedSpike(0.0, 0.20, 0.5),
                DelayedSpike(0.1, 0.10, 0.5),
            ],
            target + self.params.kernel_peak_time,
            self.params,
        )
        self.assertAlmostEqual(
            segment.response_score(sources, target, self.params), explicit
        )

    def test_phase_precession_and_depolarization_enable_prediction(self) -> None:
        state = DSNeuronState()
        state.trigger_dendritic_spike(0.0)

        self.assertTrue(state.is_depolarized(0.25, self.params))
        self.assertTrue(state.try_fire(0.5, self.params))
        self.assertTrue(state.is_refractory(0.6, self.params))
        self.assertEqual(state.membrane_potential(0.6, self.params), float("-inf"))


class PlasticityTests(unittest.TestCase):
    def setUp(self) -> None:
        encoder = SSTDDiscreteEncoder(num_columns=4, k=1, seed=3)
        self.model = SequentialMemory(encoder=encoder, num_neurons_per_column=2)

    def test_reinforcement_ages_other_context_segments(self) -> None:
        active = Segment(
            synapses={
                1: Synapse(1, delay=0.5, weight=0.5),
                2: Synapse(2, delay=0.5, weight=0.5),
            }
        )
        other = Segment(synapses={3: Synapse(3, weight=0.5)})
        self.model.columns[0].neurons[0].segments = [active, other]

        self.model._reinforce_segment(0, 0, active, {1: 0.0}, target_time=0.0)

        self.assertAlmostEqual(active.synapses[1].weight, 0.6)
        self.assertEqual(active.synapses[1].age, 0)
        self.assertAlmostEqual(active.synapses[2].weight, 0.4)
        self.assertEqual(active.synapses[2].age, 1)
        self.assertAlmostEqual(other.synapses[3].weight, 0.4)
        self.assertEqual(other.synapses[3].age, 1)

    def test_new_delay_targets_dendrite_half_cycle_before_soma(self) -> None:
        self.model._grow_segment(0, 0, {1: 0.1}, target_time=0.4)
        segment = self.model.columns[0].neurons[0].segments[0]
        synapse = segment.synapses[1]

        self.assertAlmostEqual(synapse.delay, 0.8)
        self.assertAlmostEqual(0.1 + synapse.delay, segment.target_time - 0.5)

    def test_new_segment_leaves_existing_synapses_unchanged(self) -> None:
        old = Segment(synapses={2: Synapse(2, weight=0.5, age=4)})
        self.model.columns[0].neurons[0].segments = [old]

        self.model._grow_segment(0, 0, {1: 0.0}, target_time=0.0)

        self.assertEqual(old.synapses[2].age, 4)

    def test_pruning_lazily_compacts_the_incoming_index(self) -> None:
        segment = Segment(synapses={1: Synapse(1, weight=0.5, age=30)})
        neuron = self.model.columns[0].neurons[0]
        neuron.segments = [segment]
        self.model._incoming_index[1] = [(0, 0, segment)]

        self.model._prune_neuron(neuron)

        self.assertIn(1, self.model._dirty_incoming_sources)
        self.assertEqual(self.model._live_incoming(1), [])
        self.assertNotIn(1, self.model._dirty_incoming_sources)

    def test_wrong_prediction_only_punishes_triggering_segment(self) -> None:
        triggering = Segment(synapses={1: Synapse(1, delay=0.5, weight=0.5)})
        unrelated = Segment(synapses={2: Synapse(2, weight=0.5)})
        self.model.columns[0].neurons[0].segments = [triggering, unrelated]
        self.model.last_prediction_candidates = {
            0: [PredictionCandidate(0, 1.0, 0.0, triggering)]
        }
        self.model.previous_winners = {1: 0.0}

        self.model._punish_wrong_predictions(active_events={})

        self.assertAlmostEqual(triggering.synapses[1].weight, 0.49)
        self.assertEqual(triggering.synapses[1].age, 1)
        self.assertAlmostEqual(unrelated.synapses[2].weight, 0.5)
        self.assertEqual(unrelated.synapses[2].age, 0)

    def test_unpredicted_column_selects_one_winner(self) -> None:
        encoder = SSTDDiscreteEncoder(num_columns=4, k=1, seed=5)
        model = SequentialMemory(
            encoder=encoder,
            num_neurons_per_column=3,
        )
        active = model.observe("new-symbol")
        self.assertEqual(len(active), 1)
        self.assertEqual(len(model.previous_winners), 1)
        self.assertEqual(len(model.previous_active_cells), 3)

    def test_learning_tie_break_is_seeded(self) -> None:
        encoder_a = SSTDDiscreteEncoder(num_columns=4, k=1, seed=5)
        encoder_b = SSTDDiscreteEncoder(num_columns=4, k=1, seed=5)
        model_a = SequentialMemory(encoder_a, 3, tie_break_seed=17)
        model_b = SequentialMemory(encoder_b, 3, tie_break_seed=17)

        model_a.observe("new-symbol")
        model_b.observe("new-symbol")

        self.assertEqual(model_a.previous_winners, model_b.previous_winners)

    def test_new_segment_connects_to_winner_not_all_burst_cells(self) -> None:
        encoder = SSTDDiscreteEncoder(num_columns=4, k=1, seed=5)
        model = SequentialMemory(encoder=encoder, num_neurons_per_column=3)
        model.observe("first")
        first_winner = next(iter(model.previous_winners))

        model.predict_code()
        model.observe("second")
        segments = [
            segment
            for column in model.columns
            for neuron in column.neurons
            for segment in neuron.segments
        ]

        self.assertEqual(len(segments), 1)
        self.assertEqual(set(segments[0].synapses), {first_winner})

    def test_burst_cells_can_drive_a_learned_prediction(self) -> None:
        encoder = SSTDDiscreteEncoder(num_columns=3, k=1, seed=9)
        model = SequentialMemory(encoder=encoder, num_neurons_per_column=3)
        source = model._cell_id(1, 2)
        segment = Segment(
            synapses={source: Synapse(source, delay=0.5, weight=1.0)},
            target_time=1.0,
        )
        model._incoming_index[source] = [(0, 1, segment)]
        model.previous_active_cells = {
            model._cell_id(1, neuron): 0.0 for neuron in range(3)
        }
        model.previous_winners = {model._cell_id(1, 0): 0.0}

        model.predict_code()

        self.assertEqual(model.last_prediction_candidates[0][0].neuron_index, 1)

    def test_prediction_keeps_each_columns_own_neuron_winner(self) -> None:
        encoder = SSTDDiscreteEncoder(num_columns=3, k=1, seed=9)
        model = SequentialMemory(encoder=encoder, num_neurons_per_column=3)
        model.previous_winners = {1: 0.0}
        first = Segment(
            synapses={1: Synapse(1, delay=0.5, weight=1.0)}, target_time=1.0
        )
        second = Segment(
            synapses={1: Synapse(1, delay=0.5, weight=1.0)}, target_time=1.0
        )
        model._incoming_index[1] = [(0, 1, first), (1, 2, second)]

        model.predict_code()

        self.assertEqual(
            {
                column: candidates[0].neuron_index
                for column, candidates in model.last_prediction_candidates.items()
            },
            {0: 1, 1: 2},
        )

    def test_first_spike_inhibits_later_predictions_in_same_column(self) -> None:
        encoder = SSTDDiscreteEncoder(num_columns=3, k=2, seed=9)
        model = SequentialMemory(
            encoder=encoder,
            num_neurons_per_column=3,
            params=MemoryParams(intracolumn_inhibition=True),
        )
        model.previous_winners = {1: 0.0}
        early = Segment(
            synapses={1: Synapse(1, delay=0.5, weight=1.0)}, target_time=1.0
        )
        late = Segment(
            synapses={1: Synapse(1, delay=1.0, weight=1.0)}, target_time=1.5
        )
        model._incoming_index[1] = [(0, 1, early), (0, 2, late)]

        prediction = model.predict_code()

        self.assertIsNotNone(prediction)
        times = sorted(
            candidate.time for candidate in model.last_prediction_candidates[0]
        )
        self.assertEqual(len(times), 1)
        self.assertGreater(times[0], 0.0)
        self.assertLess(times[0], 0.05)


class Fig7ProtocolTests(unittest.TestCase):
    def test_forgetting_threshold_depends_on_prediction_task(self) -> None:
        self.assertEqual(paper_forgetting_threshold("single"), 25.0)
        self.assertEqual(paper_forgetting_threshold("multiple-2"), 50.0)
        self.assertEqual(paper_forgetting_threshold("multiple-4"), 50.0)

    def test_strict_tasks_have_all_paper_context_groups_and_endings(self) -> None:
        self.assertEqual(len(build_sequence_set("single")), 8)
        self.assertEqual(len(build_sequence_set("multiple-2")), 16)
        self.assertEqual(len(build_sequence_set("multiple-4")), 32)
        self.assertEqual(
            build_sequence_set("single")[0], ["6", "8", "7", "4", "2", "3", "0"]
        )

    def test_stream_draws_random_sequences_and_keeps_noise_in_stream(self) -> None:
        import random

        sequences = build_sequence_set("single")
        stream = make_stream(
            sequences=sequences,
            max_elements=500,
            noise_symbols=50000,
            rng=random.Random(12),
            modify_after_elements=-1,
        )
        self.assertGreater(len(set(stream.expected_by_index.values())), 2)
        self.assertTrue(any(symbol.startswith("noise_") for symbol in stream.symbols))
        self.assertEqual(len(stream.symbols), 500)
        self.assertGreater(len(stream.expected_by_index), 50)


class Fig9ProtocolTests(unittest.TestCase):
    def test_literal_v0_requires_four_initial_aligned_synapses(self) -> None:
        params = MemoryParams(response_scale=1.0).dynamics()
        peak = spike_response(params.kernel_peak_time, params)

        self.assertLess(3 * 0.5 * peak, 1.0)
        self.assertGreater(4 * 0.5 * peak, 1.0)

    def test_mape_matches_reference_58_normalization(self) -> None:
        self.assertAlmostEqual(mape([2.0, 10.0], [1.0, 10.0]), 1.0 / 11.0)

    def test_rolling_mape_uses_reference_global_target_scale(self) -> None:
        self.assertAlmostEqual(reference_rolling_mape([2.0, 4.0], 100.0), 0.03)

    def test_paper_defaults_enable_burst_and_intracolumn_inhibition(self) -> None:
        params = MemoryParams()
        self.assertTrue(params.burst_context)
        self.assertTrue(params.intracolumn_inhibition)
        self.assertTrue(params.continuous_dynamics)

    def test_correct_prediction_reinforces_without_l_match_gate(self) -> None:
        encoder = SSTDDiscreteEncoder(num_columns=2, k=1, seed=1)
        model = SequentialMemory(
            encoder=encoder,
            num_neurons_per_column=1,
            params=MemoryParams(l_match=4),
        )
        source = model._cell_id(0, 0)
        synapse = Synapse(source=source, delay=0.5, weight=0.5)
        segment = Segment(
            synapses={source: synapse},
            target_time=1.0,
        )
        model.columns[1].neurons[0].segments.append(segment)
        model.previous_active_cells = {source: 0.0}
        model.previous_winners = {source: 0.0}
        model.last_prediction_candidates = {
            1: [PredictionCandidate(0, 1.0, 0.0, segment)]
        }

        model.observe_code(SymbolCode((SpikeEvent(1, 0.0),)), learn=True)

        self.assertAlmostEqual(synapse.weight, 0.6)

    def test_projection_selects_k_distinct_columns_per_field(self) -> None:
        field = SSTDRealValueEncoder(3, 2, 0.0, 2.0)
        encoder = SSTDCompositeEncoder([field])
        model = SequentialMemory(encoder=encoder, num_neurons_per_column=2)
        segment = Segment()
        model.last_prediction_candidates = {
            0: [
                PredictionCandidate(0, 2.0, 0.0, segment),
                PredictionCandidate(0, 2.0, 0.5, segment),
            ],
            1: [PredictionCandidate(0, 1.0, 0.5, segment)],
        }

        projected = project_sparse_prediction(model, encoder)

        self.assertIsNotNone(projected)
        self.assertEqual(projected.columns, (0, 1))

    def test_projection_returns_a_coherent_real_value_code(self) -> None:
        field = SSTDRealValueEncoder(5, 2, 0.0, 4.0)
        encoder = SSTDCompositeEncoder([field])
        model = SequentialMemory(encoder=encoder, num_neurons_per_column=2)
        segment = Segment()
        model.last_prediction_candidates = {
            0: [PredictionCandidate(0, 10.0, 0.0, segment)],
            4: [PredictionCandidate(0, 9.0, 0.5, segment)],
        }

        projected = project_sparse_prediction(model, encoder)

        self.assertIsNotNone(projected)
        self.assertNotEqual(projected.columns, (0, 4))
        self.assertIn(projected, field.likelihood_codes())

    def test_ranked_projection_can_retain_multiple_legal_paths(self) -> None:
        field = SSTDRealValueEncoder(5, 2, 0.0, 4.0)
        encoder = SSTDCompositeEncoder([field])
        model = SequentialMemory(encoder=encoder, num_neurons_per_column=2)
        segment = Segment()
        model.last_prediction_candidates = {
            0: [PredictionCandidate(0, 10.0, 0.0, segment)],
            3: [PredictionCandidate(0, 9.0, 0.0, segment)],
        }

        ranked = rank_sparse_predictions(model, encoder, max_predictions=2)

        self.assertEqual(len(ranked), 2)
        self.assertTrue(all(code in field.likelihood_codes() for _score, code in ranked))

    def test_periodic_codebook_includes_wraparound_midpoint(self) -> None:
        field = SSTDPeriodicEncoder(8, 3, period=8.0)

        self.assertEqual(len(field.likelihood_grid()), 16)
        self.assertEqual(field.likelihood_grid()[-1][0], 7.5)

    def test_actual_record_predicts_before_online_learning(self) -> None:
        class RecordingModel:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def predict_code(self) -> None:
                self.calls.append("predict")

            def observe_code(self, code: SymbolCode, learn: bool) -> None:
                self.calls.append(f"observe:{learn}:{len(code.events)}")

        model = RecordingModel()
        learn_actual_code(model, SymbolCode((SpikeEvent(0, 0.0),)))  # type: ignore[arg-type]

        self.assertEqual(model.calls, ["predict", "observe:True:1"])

    def test_rollout_does_not_advance_learning_tie_rng(self) -> None:
        field = SSTDRealValueEncoder(3, 2, 0.0, 2.0)
        encoder = SSTDCompositeEncoder([field])
        model = SequentialMemory(encoder=encoder, num_neurons_per_column=2)
        before = model._learning_rng.getstate()

        def consume_learning_rng() -> None:
            model._learning_rng.random()
            return None

        model.predict_code = consume_learning_rng  # type: ignore[method-assign]

        rollout(model, encoder, steps=1)

        self.assertEqual(model._learning_rng.getstate(), before)

    def test_retrieval_advances_only_through_predicted_neurons(self) -> None:
        field = SSTDRealValueEncoder(3, 2, 0.0, 2.0)
        encoder = SSTDCompositeEncoder([field])
        model = SequentialMemory(encoder=encoder, num_neurons_per_column=3)
        segment = Segment()
        model.last_prediction_candidates = {
            0: [PredictionCandidate(2, 2.0, 0.0, segment)],
        }

        advanced = model.advance_prediction(
            SymbolCode((SpikeEvent(0, 0.0), SpikeEvent(1, 0.5)))
        )

        self.assertTrue(advanced)
        self.assertEqual(model.previous_active_cells, {2: 0.0})
        self.assertEqual(model.previous_winners, {2: 0.0})

    def test_known_external_context_bursts_only_when_unpredicted(self) -> None:
        field = SSTDRealValueEncoder(2, 1, 0.0, 1.0)
        encoder = SSTDCompositeEncoder([field])
        model = SequentialMemory(encoder=encoder, num_neurons_per_column=3)

        active = model.external_input_active_cells(
            SymbolCode((SpikeEvent(0, 0.0),))
        )

        self.assertEqual(active, {0: 0.0, 1: 0.0, 2: 0.0})


if __name__ == "__main__":
    unittest.main()
