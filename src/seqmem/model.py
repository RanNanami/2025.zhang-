from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

from .dynamics import (
    DSDynamicsParams,
    DSNeuronState,
    spike_response,
)
from .encoding import SSTDDiscreteEncoder, SpikeEvent, SymbolCode


@dataclass
class Synapse:
    source: int
    delay: float = 1.0
    weight: float = 0.5
    age: int = 0

    def forgetting_score(self, l_weight: float, l_age: float) -> float:
        return l_weight * (1.0 - self.weight) + l_age * self.age


@dataclass
class Segment:
    synapses: dict[int, Synapse] = field(default_factory=dict)
    active: bool = True
    target_time: float | None = None

    def overlap(self, active_sources: dict[int, float]) -> int:
        return sum(1 for source in active_sources if source in self.synapses)

    def score(self, active_sources: dict[int, float]) -> float:
        return sum(self.synapses[source].weight for source in active_sources if source in self.synapses)

    def timed_overlap(
        self,
        active_sources: dict[int, float],
        target_time: float,
        tolerance: float,
    ) -> int:
        return sum(
            1
            for source, source_time in active_sources.items()
            if (synapse := self.synapses.get(source)) is not None
            and abs(source_time + synapse.delay - target_time) <= tolerance
        )

    def response_score(
        self,
        active_sources: dict[int, float],
        target_arrival_time: float,
        dynamics: DSDynamicsParams,
    ) -> float:
        evaluation_time = target_arrival_time + dynamics.kernel_peak_time
        return dynamics.v_rest + sum(
            synapse.weight
            * spike_response(
                evaluation_time - (source_time + synapse.delay), dynamics
            )
            for source, source_time in active_sources.items()
            if (synapse := self.synapses.get(source)) is not None
        )


@dataclass
class Neuron:
    segments: list[Segment] = field(default_factory=list)


@dataclass
class MiniColumn:
    neurons: list[Neuron]

    @classmethod
    def create(cls, num_neurons: int) -> "MiniColumn":
        return cls(neurons=[Neuron() for _ in range(num_neurons)])

    def least_used_neuron_indices(self) -> list[int]:
        minimum = min(len(neuron.segments) for neuron in self.neurons)
        return [
            index
            for index, neuron in enumerate(self.neurons)
            if len(neuron.segments) == minimum
        ]


@dataclass(frozen=True)
class PredictionCandidate:
    neuron_index: int
    score: float
    time: float
    segment: Segment


@dataclass
class SegmentPredictionTrace:
    target_column: int
    target_neuron: int
    segment_id: str
    segment_identity: int
    segment_total_synapse_count: int
    active_source_count: int
    active_matched_synapse_count: int
    target_sstd_event_time: float | None
    sum_active_weights: float
    dendritic_threshold: float
    temporally_contributing_synapse_count: int = 0
    contributing_source_cell_ids: tuple[int, ...] = ()
    contributing_source_times: tuple[float, ...] = ()
    synaptic_arrival_times: tuple[float, ...] = ()
    contributing_weights: tuple[float, ...] = ()
    peak_dendritic_potential: float | None = None
    threshold_margin: float | None = None
    first_threshold_crossing_time: float | None = None
    predicted_soma_firing_time: float | None = None
    crossed_threshold: bool = False
    entered_raw_prediction: bool = False
    inhibited_intracolumn: bool = False
    inhibited_intercolumn: bool = False


@dataclass
class PredictionTrace:
    """Optional per-call diagnostics; discarded after the caller streams it."""

    segments: list[SegmentPredictionTrace] = field(default_factory=list)


@dataclass(frozen=True)
class ReinforcementTrace:
    scenario: str
    target_column: int
    target_neuron: int
    segment_identity: int
    contributing_synapse_count: int
    weights_before: tuple[float, ...]
    weights_after: tuple[float, ...]


@dataclass(frozen=True)
class TransientStateSnapshot:
    previous_active_cells: dict[int, float]
    previous_winners: dict[int, float]
    last_prediction_candidates: dict[int, list[PredictionCandidate]]
    last_symbol_ranking: list[tuple[int, int, str]]
    decode_rng_state: object
    learning_rng_state: object


@dataclass
class MemoryParams:
    w0: float = 0.5
    delta_w: float = 0.1
    delta_w_bad: float = 0.01
    l_match: int = 3
    dendrite_threshold: float = 1.0
    timing_tolerance: float = 0.03
    forgetting_threshold: float = 25.0
    l_age: float = 1.0
    l_weight: float = 10.0
    tau_m: float = 0.10
    tau_s: float = 0.02
    response_scale: float | None = None
    burst_context: bool = True
    intracolumn_inhibition: bool = True
    continuous_dynamics: bool = True
    integration_step: float = 0.005
    integration_voltage_tolerance: float = 2e-5
    cycle_period: float = 1.0

    def dynamics(self) -> DSDynamicsParams:
        return DSDynamicsParams(
            tau_m=self.tau_m,
            tau_s=self.tau_s,
            response_scale=self.response_scale,
            dendrite_threshold=self.dendrite_threshold,
            depolarization_duration=self.cycle_period / 2.0,
            oscillation_frequency=1.0 / self.cycle_period,
            refractory_duration=self.cycle_period,
        )


class SequentialMemory:
    """Fine-grid implementation of the paper's one-layer DS memory model.

    SSTD spike times are retained throughout prediction and matching. Distal
    spikes arrive half an oscillation cycle before the target soma spike, and
    the double-exponential response kernel is integrated through dendritic
    threshold crossing and phase-precessed soma firing.
    Online segment growth, reinforcement, punishment, ageing, and pruning follow
    the learning rules and parameterization reported in the paper.
    """

    SPIKE_RESPONSE_CACHE_LIMIT = 500_000

    def __init__(
        self,
        encoder: SSTDDiscreteEncoder,
        num_neurons_per_column: int = 10,
        params: MemoryParams | None = None,
        tie_break_seed: int = 0,
    ) -> None:
        self.encoder = encoder
        self.params = params or MemoryParams()
        self.columns = [
            MiniColumn.create(num_neurons_per_column) for _ in range(encoder.num_columns)
        ]
        self._incoming_index: dict[int, list[tuple[int, int, Segment]]] = {}
        self._dirty_incoming_sources: set[int] = set()
        self._dynamics = self.params.dynamics()
        self._kernel_peak_time = self._dynamics.kernel_peak_time
        self._spike_response_cache: dict[float, float] = {}
        self._predictive_soma_can_fire = self._predictive_soma_fires(1.0)
        self._event_times = self.encoder.event_times
        self._decode_rng = random.Random(tie_break_seed)
        self._learning_rng = random.Random(tie_break_seed + 1_000_003)
        self.previous_active_cells: dict[int, float] = {}
        self.previous_winners: dict[int, float] = {}
        self.last_prediction_candidates: dict[int, list[PredictionCandidate]] = {}
        self.last_symbol_ranking: list[tuple[int, int, str]] = []
        self.reinforcement_trace_callback: (
            Callable[[ReinforcementTrace], None] | None
        ) = None

    def __getstate__(self) -> dict[str, object]:
        """Exclude the reproducible numeric cache from model checkpoints."""

        state = self.__dict__.copy()
        state["_spike_response_cache"] = {}
        return state

    def reset_state(self) -> None:
        self.previous_active_cells = {}
        self.previous_winners = {}
        self.last_prediction_candidates = {}
        self.last_symbol_ranking = []

    def snapshot_transient_state(self) -> TransientStateSnapshot:
        """Capture retrieval state without copying long-term synaptic memory."""

        return TransientStateSnapshot(
            previous_active_cells=self.previous_active_cells.copy(),
            previous_winners=self.previous_winners.copy(),
            last_prediction_candidates={
                column: candidates.copy()
                for column, candidates in self.last_prediction_candidates.items()
            },
            last_symbol_ranking=self.last_symbol_ranking.copy(),
            decode_rng_state=self._decode_rng.getstate(),
            learning_rng_state=self._learning_rng.getstate(),
        )

    def restore_transient_state(self, snapshot: TransientStateSnapshot) -> None:
        """Restore a state captured by :meth:`snapshot_transient_state`."""

        self.previous_active_cells = snapshot.previous_active_cells.copy()
        self.previous_winners = snapshot.previous_winners.copy()
        self.last_prediction_candidates = {
            column: candidates.copy()
            for column, candidates in snapshot.last_prediction_candidates.items()
        }
        self.last_symbol_ranking = snapshot.last_symbol_ranking.copy()
        self._decode_rng.setstate(snapshot.decode_rng_state)
        self._learning_rng.setstate(snapshot.learning_rng_state)

    def _active_sources(self) -> dict[int, float]:
        # The fallback keeps manually constructed tests and callers compatible.
        if self.params.burst_context:
            return self.previous_active_cells or self.previous_winners
        return self.previous_winners

    def predict_code(self, trace: PredictionTrace | None = None) -> SymbolCode | None:
        """Return the next symbol code predicted from previous winners."""
        self.last_prediction_candidates = {}
        if trace is not None:
            trace.segments.clear()
        active_sources = self._active_sources()
        if not active_sources:
            return None

        candidates: dict[int, tuple[int, int, Segment, float, list[float]]] = {}
        for source, source_time in active_sources.items():
            for column_id, neuron_index, segment in self._live_incoming(source):
                key = id(segment)
                synapse = segment.synapses[source]
                upper_bound = synapse.weight
                contributions: list[float] = []
                if key in candidates:
                    upper_bound += candidates[key][3]
                    contributions = candidates[key][4]
                if segment.target_time is not None:
                    contributions.append(
                        synapse.weight
                        * self._cached_spike_response(
                            segment.target_time
                            - self.params.cycle_period / 2.0
                            + self._kernel_peak_time
                            - (source_time + synapse.delay)
                        )
                    )
                candidates[key] = (
                    column_id,
                    neuron_index,
                    segment,
                    upper_bound,
                    contributions,
                )

        trace_by_segment: dict[int, SegmentPredictionTrace] = {}
        if trace is not None:
            for column_id, neuron_index, segment, upper_bound, _ in candidates.values():
                segment_index = next(
                    index
                    for index, candidate_segment in enumerate(
                        self.columns[column_id].neurons[neuron_index].segments
                    )
                    if candidate_segment is segment
                )
                matched = [
                    (source, source_time, segment.synapses[source])
                    for source, source_time in active_sources.items()
                    if source in segment.synapses
                ]
                item = SegmentPredictionTrace(
                    target_column=column_id,
                    target_neuron=neuron_index,
                    segment_id=f"{column_id}:{neuron_index}:{segment_index}",
                    segment_identity=id(segment),
                    segment_total_synapse_count=len(segment.synapses),
                    active_source_count=len(active_sources),
                    active_matched_synapse_count=len(matched),
                    target_sstd_event_time=(
                        segment.target_time - self.params.cycle_period
                        if segment.target_time is not None
                        else None
                    ),
                    sum_active_weights=upper_bound,
                    dendritic_threshold=self.params.dendrite_threshold,
                )
                trace.segments.append(item)
                trace_by_segment[id(segment)] = item

        best_by_event: dict[tuple[int, float], tuple[int, float, float, Segment]] = {}
        raw_eligible_segments: set[int] = set()
        for (
            column_id,
            neuron_index,
            segment,
            upper_bound,
            contributions,
        ) in candidates.values():
            # The normalized response kernel never exceeds one. Segments whose
            # active weights cannot reach threshold need no timed evaluation.
            if self._dynamics.v_rest + upper_bound < self.params.dendrite_threshold:
                continue
            trace_item = trace_by_segment.get(id(segment))
            if self.params.continuous_dynamics:
                diagnostic: dict[str, object] | None = (
                    {} if trace_item is not None else None
                )
                continuous = self._continuous_segment_prediction(
                    segment, active_sources, diagnostic=diagnostic
                )
                if trace_item is not None and diagnostic is not None:
                    trace_item.peak_dendritic_potential = diagnostic.get(
                        "peak_dendritic_potential"
                    )  # type: ignore[assignment]
                    trace_item.first_threshold_crossing_time = diagnostic.get(
                        "first_threshold_crossing_time"
                    )  # type: ignore[assignment]
                    trace_item.predicted_soma_firing_time = diagnostic.get(
                        "predicted_soma_firing_time"
                    )  # type: ignore[assignment]
                    trace_item.temporally_contributing_synapse_count = int(
                        diagnostic.get("contributing_synapse_count", 0)
                    )
                    trace_item.contributing_source_cell_ids = tuple(
                        diagnostic.get("contributing_source_cell_ids", ())
                    )
                    trace_item.contributing_source_times = tuple(
                        diagnostic.get("contributing_source_times", ())
                    )
                    trace_item.synaptic_arrival_times = tuple(
                        diagnostic.get("synaptic_arrival_times", ())
                    )
                    trace_item.contributing_weights = tuple(
                        diagnostic.get("contributing_weights", ())
                    )
                    if trace_item.peak_dendritic_potential is not None:
                        trace_item.threshold_margin = (
                            trace_item.peak_dendritic_potential
                            - self.params.dendrite_threshold
                        )
                timed_scores = (continuous,) if continuous is not None else ()
            else:
                target_times = (
                    (segment.target_time,)
                    if segment.target_time is not None
                    else tuple(
                        self.params.cycle_period + offset
                        for offset in self._event_times
                    )
                )
                timed_scores = tuple(
                    (
                        target_time,
                        self._dynamics.v_rest + sum(contributions)
                        if segment.target_time is not None
                        else self._segment_score(
                            segment, active_sources, target_time
                        ),
                    )
                    for target_time in target_times
                )
            for target_time, score in timed_scores:
                if score < self.params.dendrite_threshold:
                    continue
                if trace_item is not None:
                    trace_item.crossed_threshold = True
                if (
                    not self.params.continuous_dynamics
                    and not self._predictive_soma_can_fire
                ):
                    continue
                if (
                    target_time < self.params.cycle_period
                    or target_time
                    > self.params.cycle_period
                    + self.params.cycle_period / 2.0
                    + self.params.timing_tolerance
                ):
                    continue
                raw_eligible_segments.add(id(segment))
                event_key = (column_id, round(target_time, 12))
                previous = best_by_event.get(event_key)
                if previous is None or score > previous[1]:
                    best_by_event[event_key] = (
                        neuron_index,
                        score,
                        target_time,
                        segment,
                    )

        if not best_by_event:
            return None

        if self.params.intracolumn_inhibition:
            winner_by_column: dict[int, tuple[int, float, float, Segment]] = {}
            for (column_id, _event_time), values in best_by_event.items():
                previous = winner_by_column.get(column_id)
                if previous is None or values[2] < previous[2]:
                    winner_by_column[column_id] = values
            selected_events = list(winner_by_column.items())
        else:
            selected_events = [
                (column_id, values)
                for (column_id, _event_time), values in best_by_event.items()
            ]

        for column_id, values in selected_events:
            neuron_index, score, target_time, segment = values
            self.last_prediction_candidates.setdefault(column_id, []).append(
                PredictionCandidate(
                    neuron_index=neuron_index,
                    score=score,
                    time=target_time - self.params.cycle_period,
                    segment=segment,
                )
            )
        if trace is not None:
            selected_segment_ids = {
                id(values[3]) for _column, values in selected_events
            }
            for item in trace.segments:
                item.entered_raw_prediction = (
                    item.segment_identity in selected_segment_ids
                )
                item.inhibited_intracolumn = (
                    item.segment_identity in raw_eligible_segments
                    and item.segment_identity not in selected_segment_ids
                )
        return SymbolCode(
            events=tuple(
                SpikeEvent(
                    column=column_id,
                    time=target_time - self.params.cycle_period,
                )
                for column_id,
                (_neuron_index, _score, target_time, _segment) in selected_events
            )
        )

    def select_prediction_events(
        self,
        predicted: SymbolCode,
        max_predictions_per_event: int = 1,
        trace: PredictionTrace | None = None,
    ) -> SymbolCode | None:
        """Apply diagnostic intercolumn inhibition to real prediction cells."""

        if max_predictions_per_event <= 0:
            return None
        raw_events = {
            (event.column, round(event.time, 12)) for event in predicted.events
        }
        selected: list[tuple[int, PredictionCandidate]] = []
        selected_segment_ids: set[int] = set()
        for event_time in self._event_times:
            choices = [
                (column, candidate)
                for column, candidates in self.last_prediction_candidates.items()
                for candidate in candidates
                if (column, round(candidate.time, 12)) in raw_events
                and abs(candidate.time - event_time)
                <= self.params.timing_tolerance
            ]
            choices.sort(
                key=lambda item: (
                    -item[1].score,
                    item[0],
                    item[1].neuron_index,
                    item[1].time,
                )
            )
            used_columns: set[int] = set()
            for column, candidate in choices:
                if column in used_columns:
                    continue
                used_columns.add(column)
                selected.append((column, candidate))
                selected_segment_ids.add(id(candidate.segment))
                if len(used_columns) >= max_predictions_per_event:
                    break
        if trace is not None:
            for item in trace.segments:
                item.inhibited_intercolumn = (
                    item.entered_raw_prediction
                    and item.segment_identity not in selected_segment_ids
                )
        if not selected:
            return None
        return SymbolCode(
            events=tuple(
                SpikeEvent(column=column, time=candidate.time)
                for column, candidate in selected
            )
        )

    def predict_symbol(
        self,
        min_overlap: int | None = None,
        candidate_symbols: set[str] | None = None,
    ) -> str | None:
        """Decode a predicted code using both mini-columns and firing order."""
        predictions = self.predict_symbols(
            max_predictions=1,
            min_overlap=min_overlap,
            candidate_symbols=candidate_symbols,
        )
        return predictions[0] if predictions else None

    def predict_symbols(
        self,
        max_predictions: int,
        min_overlap: int | None = None,
        candidate_symbols: set[str] | None = None,
    ) -> list[str]:
        """Return the strongest order-sensitive symbol predictions."""
        if max_predictions <= 0:
            return []
        predicted = self.predict_code()
        if predicted is None:
            return []
        return self.decode_symbols_from_prediction(
            predicted,
            max_predictions=max_predictions,
            min_overlap=min_overlap,
            candidate_symbols=candidate_symbols,
        )

    def decode_symbol_from_prediction(
        self,
        predicted: SymbolCode,
        min_overlap: int | None = None,
        candidate_symbols: set[str] | None = None,
    ) -> str | None:
        """Decode one symbol from an already computed neural prediction."""

        decoded = self.decode_symbols_from_prediction(
            predicted,
            max_predictions=1,
            min_overlap=min_overlap,
            candidate_symbols=candidate_symbols,
        )
        return decoded[0] if decoded else None

    def decode_symbols_from_prediction(
        self,
        predicted: SymbolCode,
        max_predictions: int,
        min_overlap: int | None = None,
        candidate_symbols: set[str] | None = None,
    ) -> list[str]:
        """Decode a raw prediction without predicting or advancing state."""

        if max_predictions <= 0:
            return []
        if min_overlap is None:
            min_overlap = 1

        predicted_times: dict[int, list[float]] = {}
        if self.last_prediction_candidates:
            allowed_events = {
                (event.column, round(event.time, 12)) for event in predicted.events
            }
            for event_time in self._event_times:
                choices = [
                    (candidate.score, column)
                    for column, candidates in self.last_prediction_candidates.items()
                    for candidate in candidates
                    if (column, round(candidate.time, 12)) in allowed_events
                    if abs(candidate.time - event_time)
                    <= self.params.timing_tolerance
                ]
                self._decode_rng.shuffle(choices)
                choices.sort(key=lambda item: item[0], reverse=True)
                used_at_time: set[int] = set()
                for _score, column in choices:
                    if column in used_at_time:
                        continue
                    used_at_time.add(column)
                    predicted_times.setdefault(column, []).append(event_time)
                    if len(used_at_time) == max_predictions:
                        break
        else:
            for event in predicted.events:
                predicted_times.setdefault(event.column, []).append(event.time)
        symbols = (
            candidate_symbols
            if candidate_symbols is not None
            else self._decode_candidates(predicted_times, min_overlap)
        )
        ranked: list[tuple[int, int, str]] = []
        # Sets have process-randomized iteration order. Sort before the seeded
        # tie shuffle so identical runs stay reproducible across processes.
        for symbol in sorted(symbols):
            code = self.encoder.encode(symbol)
            column_overlap = sum(
                event.column in predicted_times for event in code.events
            )
            timed_overlap = sum(
                event.column in predicted_times
                and any(
                    abs(predicted_time - event.time)
                    <= self.params.timing_tolerance
                    for predicted_time in predicted_times[event.column]
                )
                for event in code.events
            )
            if timed_overlap >= min_overlap:
                ranked.append((timed_overlap, column_overlap, symbol))
        self._decode_rng.shuffle(ranked)
        ranked.sort(key=lambda item: (-item[0], -item[1]))
        self.last_symbol_ranking = ranked.copy()
        return [symbol for _timed, _columns, symbol in ranked[:max_predictions]]

    def _decode_candidates(
        self, predicted_times: dict[int, list[float]], min_overlap: int
    ) -> set[str]:
        """Find full-vocabulary candidates without scanning every known symbol."""

        counts: dict[str, int] = {}
        for column, column_times in predicted_times.items():
            for event_time in self._event_times:
                if not any(
                    abs(predicted_time - event_time)
                    <= self.params.timing_tolerance
                    for predicted_time in column_times
                ):
                    continue
                for symbol in self.encoder.symbols_for_event(column, event_time):
                    counts[symbol] = counts.get(symbol, 0) + 1
        return {symbol for symbol, count in counts.items() if count >= min_overlap}

    def observe(self, symbol: str, learn: bool = True) -> dict[int, float]:
        """Feed one symbol, learn online, and return current winner ids."""
        code = self.encoder.encode(symbol)
        return self.observe_code(code, learn=learn)

    def observe_code(self, code: SymbolCode, learn: bool = True) -> dict[int, float]:
        """Feed an already encoded SSTD item into the sequential memory."""
        previous_active = self._active_sources()
        active_cells: dict[int, float] = {}
        learning_winners: dict[int, float] = {}

        for event in code.events:
            column_id = event.column
            column = self.columns[column_id]
            matching_predictions = [
                candidate
                for candidate in self.last_prediction_candidates.get(column_id, [])
                if abs(candidate.time - event.time) <= self.params.timing_tolerance
            ]
            predicted = (
                max(matching_predictions, key=lambda candidate: candidate.score)
                if matching_predictions
                else None
            )
            if predicted is None:
                matched = self._best_matching_neuron(
                    column_id, column, previous_active, event.time
                )
            else:
                matched = (
                    (predicted.neuron_index, predicted.segment)
                    if predicted.segment.active
                    else None
                )
            was_predicted = predicted is not None and matched is not None
            if matched is None:
                neuron_index = self._least_used_neuron_index(column)
                if learn:
                    self._grow_segment(
                        column_id, neuron_index, self.previous_winners, event.time
                    )
            else:
                neuron_index, segment = matched
                if not learn:
                    pass
                elif was_predicted:
                    # Paper scenario 1: a predictive neuron that subsequently
                    # receives its proximal input triggers learning directly.
                    self._reinforce_segment(
                        column_id,
                        neuron_index,
                        segment,
                        previous_active,
                        event.time,
                        growth_sources=self.previous_winners,
                        grow_missing=False,
                        depress_noncontributing=True,
                        scenario="scenario1",
                    )
                elif segment.timed_overlap(
                    previous_active,
                    self._dendritic_time(self.params.cycle_period + event.time),
                    self.params.timing_tolerance,
                ) >= self.params.l_match:
                    self._reinforce_segment(
                        column_id,
                        neuron_index,
                        segment,
                        previous_active,
                        event.time,
                        growth_sources=self.previous_winners,
                        grow_missing=True,
                        depress_noncontributing=False,
                        scenario="scenario2",
                    )
                else:
                    neuron_index = self._least_used_neuron_index(column)
                    self._grow_segment(
                        column_id, neuron_index, self.previous_winners, event.time
                    )

            winner_id = self._cell_id(column_id, neuron_index)
            learning_winners[winner_id] = event.time
            if was_predicted:
                active_cells[winner_id] = event.time
            else:
                for active_neuron in range(len(column.neurons)):
                    active_cells[
                        self._cell_id(column_id, active_neuron)
                    ] = event.time

        if learn:
            self._punish_wrong_predictions(
                {event.column: event.time for event in code.events}
            )
        self.previous_active_cells = active_cells
        self.previous_winners = learning_winners
        return learning_winners

    def step(self, symbol: str) -> str | None:
        prediction = self.predict_symbol()
        self.observe(symbol)
        return prediction

    def step_code(self, code: SymbolCode, learn: bool = True) -> SymbolCode | None:
        prediction = self.predict_code()
        self.observe_code(code, learn=learn)
        return prediction

    def advance_prediction(self, code: SymbolCode) -> bool:
        """Advance retrieval through neurons that actually entered prediction."""

        active_cells = self.prediction_active_cells(code)
        if not active_cells:
            return False
        self.previous_active_cells = active_cells
        self.previous_winners = active_cells.copy()
        return True

    def prediction_active_cells(self, code: SymbolCode) -> dict[int, float]:
        """Resolve an inhibited prediction code back to its firing neurons."""

        active_cells: dict[int, float] = {}
        for event in code.events:
            matching = [
                candidate
                for candidate in self.last_prediction_candidates.get(event.column, [])
                if abs(candidate.time - event.time) <= self.params.timing_tolerance
            ]
            if not matching:
                continue
            winner = max(matching, key=lambda candidate: candidate.score)
            active_cells[
                self._cell_id(event.column, winner.neuron_index)
            ] = event.time
        return active_cells

    def external_input_active_cells(self, code: SymbolCode) -> dict[int, float]:
        """Resolve known proximal context without applying synaptic learning."""

        active_cells: dict[int, float] = {}
        for event in code.events:
            matching = [
                candidate
                for candidate in self.last_prediction_candidates.get(event.column, [])
                if abs(candidate.time - event.time) <= self.params.timing_tolerance
            ]
            if matching:
                winner = max(matching, key=lambda candidate: candidate.score)
                active_cells[
                    self._cell_id(event.column, winner.neuron_index)
                ] = event.time
            else:
                for neuron_index in range(len(self.columns[event.column].neurons)):
                    active_cells[
                        self._cell_id(event.column, neuron_index)
                    ] = event.time
        return active_cells

    def _best_matching_neuron(
        self,
        column_id: int,
        column: MiniColumn,
        active_sources: dict[int, float],
        target_time: float,
    ) -> tuple[int, Segment] | None:
        if not active_sources:
            return None

        best: tuple[int, Segment] | None = None
        best_overlap = 0
        best_score = 0.0
        candidates: dict[int, tuple[int, Segment]] = {}
        for source in active_sources:
            for target_column, neuron_index, segment in self._live_incoming(source):
                if target_column == column_id and segment.active:
                    candidates[id(segment)] = (neuron_index, segment)

        for neuron_index, segment in candidates.values():
            soma_time = self.params.cycle_period + target_time
            score = self._segment_score(segment, active_sources, soma_time)
            timed_overlap = segment.timed_overlap(
                active_sources,
                self._dendritic_time(soma_time),
                self.params.timing_tolerance,
            )
            if (timed_overlap, score) > (best_overlap, best_score):
                best = (neuron_index, segment)
                best_overlap = timed_overlap
                best_score = score
            elif (
                best is not None
                and (timed_overlap, score) == (best_overlap, best_score)
                and self._learning_rng.random() < 0.5
            ):
                best = (neuron_index, segment)
        return best

    def _grow_segment(
        self,
        column_id: int,
        neuron_index: int,
        active_sources: dict[int, float],
        target_time: float,
    ) -> None:
        if not active_sources:
            return
        neuron = self.columns[column_id].neurons[neuron_index]
        segment = Segment(
            synapses={
                source: Synapse(
                    source=source,
                    delay=self.params.cycle_period / 2.0 + target_time - source_time,
                    weight=self.params.w0,
                    age=0,
                )
                for source, source_time in active_sources.items()
            },
            target_time=self.params.cycle_period + target_time,
        )
        neuron.segments.append(segment)
        for source in active_sources:
            self._incoming_index.setdefault(source, []).append(
                (column_id, neuron_index, segment)
            )

    def _segment_score(
        self,
        segment: Segment,
        active_sources: dict[int, float],
        target_time: float,
    ) -> float:
        dendritic_time = self._dendritic_time(target_time)
        evaluation_time = dendritic_time + self._kernel_peak_time
        contributions: list[float] = []
        for source, source_time in active_sources.items():
            synapse = segment.synapses.get(source)
            if synapse is None:
                continue
            elapsed = evaluation_time - (source_time + synapse.delay)
            contributions.append(
                synapse.weight * self._cached_spike_response(elapsed)
            )
        return self._dynamics.v_rest + sum(contributions)

    def _cached_spike_response(self, elapsed: float) -> float:
        cache_key = round(elapsed, 9)
        response = self._spike_response_cache.get(cache_key)
        if response is None:
            response = spike_response(cache_key, self._dynamics)
            self._remember_spike_response(cache_key, response)
        return response

    def _remember_spike_response(self, key: float, response: float) -> None:
        """Keep a bounded cache of reusable numerical integration responses."""

        limit = self.SPIKE_RESPONSE_CACHE_LIMIT
        if limit <= 0:
            return
        cache = self._spike_response_cache
        if len(cache) >= limit:
            cache.clear()
        cache[key] = response

    def _continuous_segment_prediction(
        self,
        segment: Segment,
        active_sources: dict[int, float],
        diagnostic: dict[str, object] | None = None,
    ) -> tuple[float, float] | None:
        """Integrate one distal segment through dendritic and soma firing."""

        matched = [
            (source, source_time, source_time + synapse.delay, synapse.weight)
            for source, source_time in active_sources.items()
            if (synapse := segment.synapses.get(source)) is not None
        ]
        if not matched:
            return None
        result = self._continuous_prediction_from_arrivals(
            [(arrival, weight) for _source, _time, arrival, weight in matched],
            diagnostic=diagnostic,
        )
        if diagnostic is not None:
            crossing = diagnostic.get("first_threshold_crossing_time")
            contributing = []
            if isinstance(crossing, float):
                for source, source_time, arrival, weight in matched:
                    response = spike_response(crossing - arrival, self._dynamics)
                    if weight * response > 1e-12:
                        contributing.append(
                            (source, source_time, arrival, weight)
                        )
            diagnostic.update(
                {
                    "contributing_synapse_count": len(contributing),
                    "contributing_source_cell_ids": tuple(
                        item[0] for item in contributing
                    ),
                    "contributing_source_times": tuple(
                        item[1] for item in contributing
                    ),
                    "synaptic_arrival_times": tuple(
                        item[2] for item in contributing
                    ),
                    "contributing_weights": tuple(
                        item[3] for item in contributing
                    ),
                }
            )
        return result

    def _continuous_prediction_from_arrivals(
        self,
        arrivals: list[tuple[float, float]],
        diagnostic: dict[str, object] | None = None,
    ) -> tuple[float, float] | None:
        """Integrate a canonical set of delayed, weighted distal spikes."""

        step = self.params.integration_step
        if step <= 0.0:
            raise ValueError("integration_step must be positive")

        effective_threshold = (
            self.params.dendrite_threshold
            - self.params.integration_voltage_tolerance
        )
        response_cache = self._spike_response_cache
        dynamics = self._dynamics

        def potential(time: float, *, remember: bool = True) -> float:
            total = 0
            for arrival, weight in arrivals:
                cache_key = round(time - arrival, 9)
                response = response_cache.get(cache_key)
                if response is None:
                    response = spike_response(cache_key, dynamics)
                    if remember:
                        self._remember_spike_response(cache_key, response)
                total += weight * response
            return dynamics.v_rest + total

        start = min(arrival for arrival, _weight in arrivals)
        stop = max(arrival for arrival, _weight in arrivals) + max(
            self.params.cycle_period / 2.0,
            5.0 * self.params.tau_m,
        )
        previous_time = start
        previous_potential = potential(start)
        crossing: float | None = None
        peak = previous_potential
        time = start + step
        while time <= stop + step / 2.0:
            voltage = potential(time)
            peak = max(peak, voltage)
            if (
                previous_potential < effective_threshold <= voltage
            ):
                low = previous_time
                high = time
                for _ in range(12):
                    middle = (low + high) / 2.0
                    # Bisection midpoints are effectively one-shot values. Not
                    # caching them preserves the calculation while preventing
                    # autonomous rollout from flooding the reusable grid cache.
                    if potential(middle, remember=False) >= effective_threshold:
                        high = middle
                    else:
                        low = middle
                crossing = high
                break
            previous_time = time
            previous_potential = voltage
            time += step
        if crossing is None:
            if diagnostic is not None:
                diagnostic.update(
                    {
                        "peak_dendritic_potential": peak,
                        "first_threshold_crossing_time": None,
                        "predicted_soma_firing_time": None,
                    }
                )
            return None

        if diagnostic is not None:
            diagnostic_peak = peak
            diagnostic_time = time + step
            while diagnostic_time <= stop + step / 2.0:
                diagnostic_peak = max(
                    diagnostic_peak,
                    potential(diagnostic_time, remember=False),
                )
                diagnostic_time += step
            diagnostic.update(
                {
                    "peak_dendritic_potential": diagnostic_peak,
                    "first_threshold_crossing_time": crossing,
                    "predicted_soma_firing_time": None,
                }
            )

        state = DSNeuronState()
        state.trigger_dendritic_spike(crossing)
        soma_time = max(crossing, self.params.cycle_period)
        soma_stop = crossing + self._dynamics.depolarization_duration
        while soma_time <= soma_stop + step / 2.0:
            if (
                state.membrane_potential(soma_time, self._dynamics)
                >= self._dynamics.soma_threshold
                - self.params.integration_voltage_tolerance
            ):
                if diagnostic is not None:
                    diagnostic["predicted_soma_firing_time"] = soma_time
                return soma_time, max(peak, self.params.dendrite_threshold)
            soma_time += step
        if (
            state.membrane_potential(soma_stop, self._dynamics)
            >= self._dynamics.soma_threshold
            - self.params.integration_voltage_tolerance
        ):
            if diagnostic is not None:
                diagnostic["predicted_soma_firing_time"] = soma_stop
            return soma_stop, max(peak, self.params.dendrite_threshold)
        return None

    def _dendritic_time(self, soma_time: float) -> float:
        return soma_time - self.params.cycle_period / 2.0

    def _predictive_soma_fires(self, target_time: float) -> bool:
        dynamics = self._dynamics
        state = DSNeuronState()
        state.trigger_dendritic_spike(self._dendritic_time(target_time))
        return state.try_fire(target_time, dynamics)

    def _reinforce_segment(
        self,
        column_id: int,
        neuron_index: int,
        segment: Segment,
        active_sources: dict[int, float],
        target_time: float,
        growth_sources: dict[int, float] | None = None,
        grow_missing: bool = False,
        depress_noncontributing: bool = True,
        scenario: str = "unspecified",
    ) -> None:
        weights_before = tuple(
            synapse.weight
            for _source, synapse in sorted(segment.synapses.items())
        )
        dendritic_time = self._dendritic_time(
            self.params.cycle_period + target_time
        )
        contributed = {
            source
            for source, source_time in active_sources.items()
            if (synapse := segment.synapses.get(source)) is not None
            and abs(source_time + synapse.delay - dendritic_time)
            <= self.params.timing_tolerance
        }

        if grow_missing:
            for source, source_time in (growth_sources or {}).items():
                if source in segment.synapses:
                    continue
                segment.synapses[source] = Synapse(
                    source=source,
                    delay=self.params.cycle_period / 2.0 + target_time - source_time,
                    weight=self.params.w0,
                    age=0,
                )
                contributed.add(source)
                self._incoming_index.setdefault(source, []).append(
                    (column_id, neuron_index, segment)
                )

        for source, synapse in segment.synapses.items():
            if source in contributed:
                synapse.weight = min(1.0, synapse.weight + self.params.delta_w)
                synapse.age = 0
            else:
                if depress_noncontributing:
                    synapse.weight = max(0.0, synapse.weight - self.params.delta_w)
                    synapse.age += 1

        neuron = self.columns[column_id].neurons[neuron_index]
        for other_segment in neuron.segments:
            if other_segment is segment:
                continue
            for synapse in other_segment.synapses.values():
                if depress_noncontributing:
                    synapse.weight = max(
                        0.0, synapse.weight - self.params.delta_w
                    )
                    synapse.age += 1

        if self.reinforcement_trace_callback is not None:
            self.reinforcement_trace_callback(
                ReinforcementTrace(
                    scenario=scenario,
                    target_column=column_id,
                    target_neuron=neuron_index,
                    segment_identity=id(segment),
                    contributing_synapse_count=len(contributed),
                    weights_before=weights_before,
                    weights_after=tuple(
                        synapse.weight
                        for _source, synapse in sorted(segment.synapses.items())
                    ),
                )
            )

        self._prune_neuron(neuron)

    def _punish_wrong_predictions(self, active_events: dict[int, float]) -> None:
        active_sources = self._active_sources()
        for column_id, candidates in self.last_prediction_candidates.items():
            actual_time = active_events.get(column_id)
            for candidate in candidates:
                predicted_time = candidate.time
                neuron_index = candidate.neuron_index
                segment = candidate.segment
                if (
                    actual_time is not None
                    and abs(actual_time - predicted_time)
                    <= self.params.timing_tolerance
                ):
                    continue
                soma_time = self.params.cycle_period + predicted_time
                dendritic_time = self._dendritic_time(soma_time)
                contributed = {
                    source
                    for source, source_time in active_sources.items()
                    if (synapse := segment.synapses.get(source)) is not None
                    and abs(source_time + synapse.delay - dendritic_time)
                    <= self.params.timing_tolerance
                }
                for source in contributed:
                    synapse = segment.synapses[source]
                    synapse.weight = max(
                        0.0, synapse.weight - self.params.delta_w_bad
                    )
                    synapse.age += 1
                self._prune_neuron(
                    self.columns[column_id].neurons[neuron_index]
                )

    def _prune_neuron(self, neuron: Neuron) -> None:
        kept_segments: list[Segment] = []
        for segment in neuron.segments:
            previous_sources = set(segment.synapses)
            segment.synapses = {
                source: synapse
                for source, synapse in segment.synapses.items()
                if synapse.forgetting_score(
                    self.params.l_weight, self.params.l_age
                )
                < self.params.forgetting_threshold
            }
            self._dirty_incoming_sources.update(
                previous_sources.difference(segment.synapses)
            )
            segment.active = bool(segment.synapses)
            if segment.active:
                kept_segments.append(segment)
        neuron.segments = kept_segments

    def _cell_id(self, column_id: int, neuron_index: int) -> int:
        return column_id * len(self.columns[column_id].neurons) + neuron_index

    def _least_used_neuron_index(self, column: MiniColumn) -> int:
        return self._learning_rng.choice(column.least_used_neuron_indices())

    def _live_incoming(self, source: int) -> list[tuple[int, int, Segment]]:
        entries = self._incoming_index.get(source, [])
        if source not in self._dirty_incoming_sources:
            return entries
        live = [
            entry
            for entry in entries
            if entry[2].active and source in entry[2].synapses
        ]
        self._incoming_index[source] = live
        self._dirty_incoming_sources.discard(source)
        return live
