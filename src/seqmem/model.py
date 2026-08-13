"""序列记忆网络的核心数据结构、预测、学习与状态推进。

本模块目前也是旧 checkpoint 的 pickle 类路径。重构时可以抽取纯 helper，
但在兼容迁移完成前，不能移动 Synapse、Segment、Neuron、MemoryParams 或
SequentialMemory 的定义。strict 路径对 RNG 调用顺序、浮点累加顺序和候选
排序敏感，相关循环不能仅为缩短代码而改写。
"""

from __future__ import annotations

import hashlib
import json
import math
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
    """One distal synapse from a previous active cell into a segment.

    中文调试提示：source 是 presynaptic cell id，delay 把 source spike time
    移到目标 dendrite 时间，weight/age 是长期记忆，会被学习规则修改。
    """

    source: int
    delay: float = 1.0
    weight: float = 0.5
    age: int = 0

    def forgetting_score(self, l_weight: float, l_age: float) -> float:
        return l_weight * (1.0 - self.weight) + l_age * self.age


@dataclass
class Segment:
    """Distal dendritic segment on one neuron.

    中文调试提示：segment 是“某个上下文 -> 某个目标 neuron/time”的记忆单元。
    synapses 是长期结构；active=False 代表被 forgetting/pruning 停用。
    """

    synapses: dict[int, Synapse] = field(default_factory=dict)
    active: bool = True
    target_time: float | None = None
    diagnostic_id: int | None = None
    creation_sentence_index: int | None = None
    creation_transition_index: int | None = None
    creation_target_column: int | None = None
    creation_target_time: float | None = None
    creation_target_neuron: int | None = None
    creation_source_cell_ids: tuple[int, ...] = ()
    creation_source_fingerprint: str | None = None
    scenario1_reinforcements: int = 0
    scenario2_reinforcements: int = 0

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
class SynapsePSPContribution:
    source_cell_id: int
    source_time: float
    arrival_time: float
    weight: float
    psp_contribution: float


@dataclass(frozen=True)
class PredictionCandidate:
    """A neuron that crossed dendritic threshold during one predict_code call.

    DEBUG WATCH: Scenario 1 必须使用同一个 PredictionCandidate 中保存的
    crossing time/contributions；不能重新预测后再判断贡献突触。
    """

    neuron_index: int
    score: float
    time: float
    segment: Segment
    dendritic_crossing_time: float | None = None
    crossing_synapse_contributions: tuple[SynapsePSPContribution, ...] = ()
    peak_dendritic_potential: float | None = None
    threshold_margin: float | None = None
    predicted_soma_firing_time: float | None = None


INTRACOLUMN_SELECTION_POLICIES = {
    "existing",
    "max_candidate_score",
    "max_response_peak",
    "max_contributor_count",
    "context_then_score",
}


@dataclass(frozen=True)
class IntracolumnCandidateTrace:
    """One real event winner considered by the per-column selector."""

    original_index: int
    column: int
    neuron_index: int
    segment_index: int
    segment_identity: int
    predicted_time: float
    candidate_score: float
    response_peak: float | None
    contributor_count: int
    positive_contributor_count: int
    current_context_jaccard: float


@dataclass(frozen=True)
class IntracolumnSelectionGroupTrace:
    """Read-only outcome of selecting one event winner for one column."""

    policy: str
    column: int
    candidates: tuple[IntracolumnCandidateTrace, ...]
    selected_candidate_original_index: int
    existing_policy_candidate_index: int
    selector_primary_key: str
    selector_secondary_keys: tuple[str, ...]
    tie_count: int
    tie_break_used: bool
    candidate_pool_fingerprint: str


@dataclass
class IntracolumnSelectionTrace:
    """Optional per-call trace; it never participates in model learning."""

    groups: list[IntracolumnSelectionGroupTrace] = field(default_factory=list)


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
    crossing_psp_contributions: tuple[float, ...] = ()
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


@dataclass
class PreselectionSegmentTrace:
    """Read-only state copied from one segment's existing prediction path."""

    target_column: int
    target_neuron: int
    segment_index: int
    inspection_order: int
    segment: Segment
    segment_identity: int
    active_source_count: int
    active_matched_synapse_count: int
    sum_active_weights: float
    dendritic_threshold: float
    response_computed: bool = False
    response_peak: float | None = None
    response_at_selected_time: float | None = None
    first_threshold_crossing_time: float | None = None
    predicted_soma_firing_time: float | None = None
    candidate_score_value: float | None = None
    contributor_count: int = 0
    positive_contributor_count: int = 0
    crossing_synapse_contributions: tuple[
        SynapsePSPContribution, ...
    ] = ()
    crossed_threshold: bool = False
    valid_firing_time: bool = False
    event_time_key: float | None = None
    became_event_winner: bool = False
    became_column_winner: bool = False
    became_prediction_candidate: bool = False
    furthest_stage_reached: str = "INSPECTED"
    elimination_stage: str = ""
    elimination_reason: str = ""
    winner_segment_identity: int | None = None
    rank_within_event_group: int | None = None
    rank_within_column_group: int | None = None
    winner_score_gap: float | None = None
    winner_time_gap: float | None = None
    tie_break_used: bool = False


@dataclass(frozen=True)
class PreselectionReplacementTrace:
    """One replacement performed by an existing prediction selection branch."""

    group_type: str
    target_column: int
    target_neuron: int | None
    stable_time_key: float | None
    previous_segment: Segment
    replacement_segment: Segment
    previous_segment_identity: int
    replacement_segment_identity: int
    comparison_field: str
    previous_value: float
    replacement_value: float
    previous_tie_break_value: tuple[int, int]
    replacement_tie_break_value: tuple[int, int]
    replacement_stage: str


@dataclass
class PreselectionTrace:
    """Optional pre-candidate funnel trace for one ``predict_code`` call."""

    segments: list[PreselectionSegmentTrace] = field(default_factory=list)
    replacements: list[PreselectionReplacementTrace] = field(
        default_factory=list
    )


@dataclass(frozen=True)
class ReinforcementTrace:
    scenario: str
    target_column: int
    target_neuron: int
    segment_identity: int
    contributing_synapse_count: int
    weights_before: tuple[float, ...]
    weights_after: tuple[float, ...]
    contribution_mode: str = "arrival-window"
    prediction_candidate_identity: int | None = None
    prediction_crossing_time: float | None = None
    actual_positive_synapse_count: int = 0
    actual_positive_weakened_count: int = 0
    strengthened_synapse_count: int = 0
    weakened_synapse_count: int = 0
    segment_diagnostic_id: int | None = None
    source_count_before: int = 0
    source_count_after: int = 0
    scenario1_count_before: int = 0
    scenario1_count_after: int = 0
    scenario2_count_before: int = 0
    scenario2_count_after: int = 0
    contributed_source_ids: tuple[int, ...] = ()
    strengthened_source_ids: tuple[int, ...] = ()
    weakened_source_ids: tuple[int, ...] = ()
    same_segment_noncontributor_ids: tuple[int, ...] = ()
    same_segment_noncontributor_aged_count: int = 0
    other_segment_synapse_count: int = 0
    other_segment_weakened_count: int = 0
    other_segment_aged_count: int = 0
    previous_winner_source_ids: tuple[int, ...] = ()
    growth_source_ids: tuple[int, ...] = ()
    added_source_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class MatchingCandidateTrace:
    """One candidate measured by the real Scenario-2 traversal."""

    target_neuron: int
    segment: Segment
    timed_overlap: int
    candidate_score: float


@dataclass(frozen=True)
class BurstPSPTrace:
    total_psp: float
    predicted_source_psp: float
    burst_only_psp: float
    unlabelled_psp: float
    predicted_without_burst_crosses: bool
    burst_only_crosses: bool
    classification: str


@dataclass(frozen=True)
class TransientStateSnapshot:
    """Read-only/evaluation snapshot of state that is not long-term memory.

    中文调试提示：这里保存 previous_active_cells、previous_winners、
    last_prediction_candidates、解码 RNG 和学习 RNG。segment/synapse 本体
    不会被复制，所以它只适合保护 rollout/evaluate 的瞬时污染。
    """

    previous_active_cells: dict[int, float]
    previous_winners: dict[int, float]
    last_prediction_candidates: dict[int, list[PredictionCandidate]]
    last_prediction_stats: dict[str, int | float]
    last_observe_stats: dict[str, int | float]
    last_symbol_ranking: list[tuple[int, int, str]]
    previous_predicted_sources: set[int]
    previous_burst_only_sources: set[int]
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
    scenario1_contribution_mode: str = "arrival-window"
    capture_prediction_contributions: bool = False
    synapse_delay_mode: str = "current-delay"
    capture_branch_diagnostics: bool = False
    # Opt-in identity/scope audit.  It records existing decisions only.
    capture_intralayer_parity_diagnostics: bool = False
    # Separate opt-in snapshot of prediction candidates before the temporal
    # confirmation gate.  This is audit metadata only and never selects a
    # candidate or changes the learning path.
    capture_temporal_confirmation_diagnostics: bool = False
    continuous_prediction_impl: str = "reference"

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


@dataclass
class ObservationEventTrace:
    """Read-only identity already selected for one proximal input event."""

    target_column: int
    target_time: float
    winner_neuron: int
    winner_cell_id: int
    scenario: str
    was_predicted: bool
    selected_segment: Segment | None = None
    reinforced_segment: Segment | None = None
    created_segment: Segment | None = None
    predicted_candidate_count_in_column: int = 0
    timing_matched_prediction_count: int = 0
    matching_segment_count: int = 0
    best_matching_overlap: int = 0
    best_matching_score: float = 0.0
    best_matching_active_synapse_count: int = 0
    existing_segment_count_in_column: int = 0
    existing_segment_count_on_selected_neuron: int = 0
    least_used_neuron_segment_count: int = 0
    previous_winner_count: int = 0
    previous_context_size: int = 0
    scenario_assignment_reason: str = ""
    predictive_neuron_ids: tuple[int, ...] = ()
    predictive_segment_count: int = 0
    existing_segment_count_per_neuron: tuple[int, ...] = ()
    best_matching_neuron: int | None = None
    best_matching_segment: Segment | None = None
    predicted_time_available: bool = False
    predicted_time_valid: bool = False
    matching_response_available: bool = False
    selected_candidate_score: float = 0.0
    matching_candidates: tuple[MatchingCandidateTrace, ...] = ()
    # Identity fields are populated from the already-selected candidate.
    # They are audit metadata, not additional selection logic.
    predicted_neuron_index: int | None = None
    predicted_cell_id: int | None = None
    predicted_segment: Segment | None = None
    predicted_segment_identity: int | None = None
    predicted_candidate_identity: int | None = None
    predicted_candidate_time: float | None = None
    predicted_crossing_time: float | None = None
    pre_gate_candidate_count: int = 0
    pre_gate_candidate_identity_ids: tuple[int, ...] = ()
    pre_gate_candidate_neuron_ids: tuple[int, ...] = ()
    pre_gate_candidate_segment_identities: tuple[int, ...] = ()
    pre_gate_candidate_times: tuple[float, ...] = ()
    pre_gate_candidate_scores: tuple[float, ...] = ()
    pre_gate_candidate_crossing_times: tuple[float | None, ...] = ()
    pre_gate_candidate_depolarization_states: tuple[str, ...] = ()
    pre_gate_candidate_time_gate_passed: tuple[bool, ...] = ()
    pre_gate_candidate_class: str = "NO_PREDICTIVE_CANDIDATE"
    pre_gate_timing_tolerance: float = 0.0


@dataclass
class BestMatchingTrace:
    """Values already computed by one _best_matching_neuron call."""

    candidate_segment_count: int = 0
    best_overlap: int = 0
    best_score: float = 0.0
    best_active_synapse_count: int = 0
    best_neuron: int | None = None
    best_segment: Segment | None = None
    candidates: list[MatchingCandidateTrace] = field(default_factory=list)


@dataclass
class ObservationTrace:
    """Optional observe output populated without recomputing any decision."""

    events: list[ObservationEventTrace] = field(default_factory=list)
    pre_observe_previous_winners: dict[int, float] = field(default_factory=dict)
    pre_observe_active_sources: dict[int, float] = field(default_factory=dict)
    capture_scenario_details: bool = False


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
    SCENARIO1_CONTRIBUTION_MODES = {
        "arrival-window",
        "continuous-positive",
        "continuous-causal",
    }
    PREDICTION_RANKING_MODES = {
        "current-score",
        "actual-peak",
        "earliest-crossing",
    }
    SYNAPSE_DELAY_MODES = {
        "current-delay",
        "peak-aligned-delay",
    }
    CONTINUOUS_PREDICTION_IMPLS = {
        "reference",
        "optimized_v1",
        "optimized_v2",
    }

    def __init__(
        self,
        encoder: SSTDDiscreteEncoder,
        num_neurons_per_column: int = 10,
        params: MemoryParams | None = None,
        tie_break_seed: int = 0,
    ) -> None:
        self.encoder = encoder
        self.params = params or MemoryParams()
        if (
            self.params.scenario1_contribution_mode
            not in self.SCENARIO1_CONTRIBUTION_MODES
        ):
            raise ValueError(
                "unsupported Scenario-1 contribution mode: "
                f"{self.params.scenario1_contribution_mode}"
            )
        if self.params.synapse_delay_mode not in self.SYNAPSE_DELAY_MODES:
            raise ValueError(
                "unsupported synapse delay mode: "
                f"{self.params.synapse_delay_mode}"
            )
        if self.params.continuous_prediction_impl not in self.CONTINUOUS_PREDICTION_IMPLS:
            raise ValueError(
                "unsupported continuous prediction implementation: "
                f"{self.params.continuous_prediction_impl}"
            )
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
        self.last_prediction_stats: dict[str, int | float] = {}
        self.last_observe_stats: dict[str, int | float] = {}
        self.last_symbol_ranking: list[tuple[int, int, str]] = []
        self.previous_predicted_sources: set[int] = set()
        self.previous_burst_only_sources: set[int] = set()
        self._diagnostic_sentence_index: int | None = None
        self._diagnostic_transition_index: int | None = None
        self._next_segment_diagnostic_id = 1
        self.reinforcement_trace_callback: (
            Callable[[ReinforcementTrace], None] | None
        ) = None
        # Debug-only hook.  It receives scalar snapshots around pruning and is
        # deliberately excluded from checkpoints by ``__getstate__``.
        self.prune_diagnostic_callback: (
            Callable[[dict[str, object]], None] | None
        ) = None
        # Runtime-only callback for the Fig.8 intralayer parity audit.  It is
        # deliberately excluded from checkpoints and never participates in
        # prediction, learning, or RNG decisions.
        self.intralayer_parity_callback: (
            Callable[[dict[str, object]], None] | None
        ) = None
        # Runtime-only breadcrumb hook.  It is intentionally excluded from
        # checkpoints so enabling diagnostics cannot change model state.
        self.runtime_debug_callback: (
            Callable[[str, dict[str, object]], None] | None
        ) = None
        self._runtime_debug_context: dict[str, object] = {}

    def __getstate__(self) -> dict[str, object]:
        """Exclude the reproducible numeric cache from model checkpoints."""

        state = self.__dict__.copy()
        state["_spike_response_cache"] = {}
        state["prune_diagnostic_callback"] = None
        state["intralayer_parity_callback"] = None
        state["runtime_debug_callback"] = None
        state["_runtime_debug_context"] = {}
        return state

    def _emit_runtime_debug(
        self,
        phase: str,
        details: dict[str, object] | None = None,
    ) -> None:
        """Emit an optional read-only runtime breadcrumb.

        The callback is deliberately isolated from the numerical path.  A
        diagnostics I/O failure must never change prediction, learning, or
        RNG behavior, so callback exceptions are ignored.
        """

        callback = getattr(self, "runtime_debug_callback", None)
        if callback is None:
            return
        try:
            callback(phase, dict(details or {}))
        except Exception:
            return

    def _emit_intralayer_parity(
        self,
        phase: str,
        details: dict[str, object],
    ) -> None:
        """Send an existing learning decision to an opt-in read-only audit."""

        callback = getattr(self, "intralayer_parity_callback", None)
        if not getattr(
            self.params, "capture_intralayer_parity_diagnostics", False
        ):
            return
        if callback is None:
            return
        try:
            callback({"phase": phase, **details})
        except Exception:
            # Diagnostics must never change the numerical or RNG path.
            return

    def _intralayer_parity_enabled(self) -> bool:
        return bool(
            getattr(self.params, "capture_intralayer_parity_diagnostics", False)
            and getattr(self, "intralayer_parity_callback", None) is not None
        )

    def reset_state(self) -> None:
        # STATE MUTATION: 只清空当前序列上下文，不删除任何已学 segment/synapse。
        self.previous_active_cells = {}
        self.previous_winners = {}
        self.last_prediction_candidates = {}
        self.last_prediction_stats = {}
        self.last_observe_stats = {}
        self.last_symbol_ranking = []
        self.previous_predicted_sources = set()
        self.previous_burst_only_sources = set()

    def snapshot_transient_state(self) -> TransientStateSnapshot:
        """Capture retrieval state without copying long-term synaptic memory.

        DEBUG WATCH: checkpoint/evaluate/rollout 前在这里下断点。若不同
        report_every 导致最终模型不同，通常是某个临时字段或 RNG 没恢复。
        """

        return TransientStateSnapshot(
            previous_active_cells=self.previous_active_cells.copy(),
            previous_winners=self.previous_winners.copy(),
            last_prediction_candidates={
                column: candidates.copy()
                for column, candidates in self.last_prediction_candidates.items()
            },
            last_prediction_stats=self.last_prediction_stats.copy(),
            last_observe_stats=self.last_observe_stats.copy(),
            last_symbol_ranking=self.last_symbol_ranking.copy(),
            previous_predicted_sources=self.previous_predicted_sources.copy(),
            previous_burst_only_sources=self.previous_burst_only_sources.copy(),
            decode_rng_state=self._decode_rng.getstate(),
            learning_rng_state=self._learning_rng.getstate(),
        )

    def restore_transient_state(self, snapshot: TransientStateSnapshot) -> None:
        """Restore a state captured by :meth:`snapshot_transient_state`.

        STATE MUTATION: 恢复的是 transient state，不回滚 synapse 权重。
        因此调用方必须保证 snapshot 期间没有 learn=True。
        """

        self.previous_active_cells = snapshot.previous_active_cells.copy()
        self.previous_winners = snapshot.previous_winners.copy()
        self.last_prediction_candidates = {
            column: candidates.copy()
            for column, candidates in snapshot.last_prediction_candidates.items()
        }
        self.last_prediction_stats = snapshot.last_prediction_stats.copy()
        self.last_observe_stats = snapshot.last_observe_stats.copy()
        self.last_symbol_ranking = snapshot.last_symbol_ranking.copy()
        self.previous_predicted_sources = (
            snapshot.previous_predicted_sources.copy()
        )
        self.previous_burst_only_sources = (
            snapshot.previous_burst_only_sources.copy()
        )
        self._decode_rng.setstate(snapshot.decode_rng_state)
        self._learning_rng.setstate(snapshot.learning_rng_state)

    def _active_sources(self) -> dict[int, float]:
        """Return the cells that provide lateral/distal context now.

        中文调试提示：burst_context=True 时，未预测列会把整列 active cells
        送入下一步；这是 Fig.8/Fig.9 raw columns 膨胀的重要观察点。
        """

        # The fallback keeps manually constructed tests and callers compatible.
        if self.params.burst_context:
            return self.previous_active_cells or self.previous_winners
        return self.previous_winners

    def set_segment_provenance_context(
        self,
        sentence_index: int | None,
        transition_index: int | None,
    ) -> None:
        """Set optional creation labels used only by diagnostic training."""

        self._diagnostic_sentence_index = sentence_index
        self._diagnostic_transition_index = transition_index

    @staticmethod
    def _finite_or(value: object, fallback: float) -> float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            if math.isfinite(numeric):
                return numeric
        return fallback

    def _intracolumn_event_winner_metadata(
        self,
        *,
        original_index: int,
        column_id: int,
        values: tuple[int, float, float, Segment],
        prediction_metadata: dict[int, dict[str, object]],
        active_source_ids: set[int],
    ) -> IntracolumnCandidateTrace:
        neuron_index, score, target_time, segment = values
        metadata = prediction_metadata.get(id(segment), {})
        contributions = tuple(
            item
            for item in metadata.get("crossing_synapse_contributions", ())
            if isinstance(item, SynapsePSPContribution)
        )
        segment_sources = set(segment.synapses)
        union = segment_sources | active_source_ids
        context_jaccard = (
            len(segment_sources & active_source_ids) / len(union)
            if union
            else 1.0
        )
        segment_index = next(
            index
            for index, candidate_segment in enumerate(
                self.columns[column_id].neurons[neuron_index].segments
            )
            if candidate_segment is segment
        )
        peak = metadata.get("peak_dendritic_potential")
        return IntracolumnCandidateTrace(
            original_index=original_index,
            column=column_id,
            neuron_index=neuron_index,
            segment_index=segment_index,
            segment_identity=id(segment),
            predicted_time=target_time,
            candidate_score=score,
            response_peak=(
                float(peak)
                if isinstance(peak, (int, float))
                and not isinstance(peak, bool)
                and math.isfinite(float(peak))
                else None
            ),
            contributor_count=len(contributions),
            positive_contributor_count=sum(
                item.psp_contribution > 0.0 for item in contributions
            ),
            current_context_jaccard=context_jaccard,
        )

    def _intracolumn_policy_key(
        self,
        candidate: IntracolumnCandidateTrace,
        policy: str,
    ) -> tuple[float, ...]:
        score = self._finite_or(candidate.candidate_score, float("-inf"))
        peak = self._finite_or(candidate.response_peak, float("-inf"))
        predicted_time = self._finite_or(
            candidate.predicted_time, float("inf")
        )
        context = self._finite_or(
            candidate.current_context_jaccard, float("-inf")
        )
        if policy == "existing":
            return (predicted_time,)
        if policy == "max_candidate_score":
            return (-score, predicted_time)
        if policy == "max_response_peak":
            return (-peak, -score, predicted_time)
        if policy == "max_contributor_count":
            return (
                -float(candidate.contributor_count),
                -peak,
                -score,
                predicted_time,
            )
        if policy == "context_then_score":
            return (-context, -score, predicted_time)
        raise ValueError(f"unsupported alternative policy: {policy}")

    def _select_intracolumn_event_winners(
        self,
        *,
        best_by_event: dict[
            tuple[int, float], tuple[int, float, float, Segment]
        ],
        policy: str,
        prediction_metadata: dict[int, dict[str, object]],
        active_sources: dict[int, float],
        selection_trace: IntracolumnSelectionTrace | None,
    ) -> list[tuple[int, tuple[int, float, float, Segment]]]:
        """Select one real event winner per column without recomputing it."""

        grouped: dict[
            int, list[tuple[int, tuple[int, float, float, Segment]]]
        ] = {}
        for original_index, (
            (column_id, _event_time),
            values,
        ) in enumerate(best_by_event.items()):
            grouped.setdefault(column_id, []).append((original_index, values))

        selected_events: list[
            tuple[int, tuple[int, float, float, Segment]]
        ] = []
        active_source_ids = set(active_sources)
        selector_keys = {
            "existing": (
                "predicted_time_asc",
                ("stable_original_order",),
            ),
            "max_candidate_score": (
                "candidate_score_desc",
                ("predicted_time_asc", "stable_original_order"),
            ),
            "max_response_peak": (
                "response_peak_desc",
                (
                    "candidate_score_desc",
                    "predicted_time_asc",
                    "stable_original_order",
                ),
            ),
            "max_contributor_count": (
                "contributor_count_desc",
                (
                    "response_peak_desc",
                    "candidate_score_desc",
                    "predicted_time_asc",
                    "stable_original_order",
                ),
            ),
            "context_then_score": (
                "current_context_jaccard_desc",
                (
                    "candidate_score_desc",
                    "predicted_time_asc",
                    "stable_original_order",
                ),
            ),
        }
        primary_key, secondary_keys = selector_keys[policy]

        for column_id, members in grouped.items():
            candidate_traces = tuple(
                self._intracolumn_event_winner_metadata(
                    original_index=original_index,
                    column_id=column_id,
                    values=values,
                    prediction_metadata=prediction_metadata,
                    active_source_ids=active_source_ids,
                )
                for original_index, values in members
            )
            existing_candidate = min(
                candidate_traces,
                key=lambda candidate: candidate.predicted_time,
            )
            ranked = sorted(
                candidate_traces,
                key=lambda candidate: (
                    *self._intracolumn_policy_key(candidate, policy),
                    candidate.original_index,
                ),
            )
            selected = ranked[0]
            selected_values = next(
                values
                for original_index, values in members
                if original_index == selected.original_index
            )
            selected_events.append((column_id, selected_values))

            if selection_trace is not None:
                selected_numeric_key = self._intracolumn_policy_key(
                    selected, policy
                )
                tie_count = sum(
                    self._intracolumn_policy_key(candidate, policy)
                    == selected_numeric_key
                    for candidate in candidate_traces
                )
                def stable_number(value: float | None) -> float | None:
                    return (
                        float(value)
                        if value is not None and math.isfinite(float(value))
                        else None
                    )

                fingerprint_payload = [
                    {
                        "original_index": candidate.original_index,
                        "column": candidate.column,
                        "neuron": candidate.neuron_index,
                        "segment_index": candidate.segment_index,
                        "predicted_time": stable_number(
                            candidate.predicted_time
                        ),
                        "candidate_score": stable_number(
                            candidate.candidate_score
                        ),
                        "response_peak": stable_number(
                            candidate.response_peak
                        ),
                        "contributor_count": candidate.contributor_count,
                        "positive_contributor_count": (
                            candidate.positive_contributor_count
                        ),
                        "current_context_jaccard": stable_number(
                            candidate.current_context_jaccard
                        ),
                    }
                    for candidate in candidate_traces
                ]
                fingerprint = hashlib.sha256(
                    json.dumps(
                        fingerprint_payload,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("ascii")
                ).hexdigest()
                selection_trace.groups.append(
                    IntracolumnSelectionGroupTrace(
                        policy=policy,
                        column=column_id,
                        candidates=candidate_traces,
                        selected_candidate_original_index=(
                            selected.original_index
                        ),
                        existing_policy_candidate_index=(
                            existing_candidate.original_index
                        ),
                        selector_primary_key=primary_key,
                        selector_secondary_keys=secondary_keys,
                        tie_count=tie_count,
                        tie_break_used=tie_count > 1,
                        candidate_pool_fingerprint=fingerprint,
                    )
                )
        return selected_events

    def predict_code(
        self,
        trace: PredictionTrace | None = None,
        preselection_trace: PreselectionTrace | None = None,
        intracolumn_selection_policy: str = "existing",
        intracolumn_selection_trace: IntracolumnSelectionTrace | None = None,
    ) -> SymbolCode | None:
        """Return the next symbol code predicted from previous context cells.

        中文调试提示：这是“神经预测”本体。它只读 previous_active_cells /
        previous_winners 和长期 segment/synapse，写 last_prediction_candidates
        作为本次预测候选；不学习、不 observe 外部输入。
        """

        # DEBUG WATCH: after predict_code 要看的核心字段是
        # last_prediction_candidates、返回 SymbolCode 的 event/column 数量、
        # trace.segments 中 crossed_threshold/entered_raw_prediction。
        self.last_prediction_candidates = {}
        self.last_prediction_stats = {}
        if trace is not None:
            trace.segments.clear()
        if preselection_trace is not None:
            preselection_trace.segments.clear()
            preselection_trace.replacements.clear()
        if intracolumn_selection_trace is not None:
            intracolumn_selection_trace.groups.clear()
        if intracolumn_selection_policy not in INTRACOLUMN_SELECTION_POLICIES:
            raise ValueError(
                "intracolumn selection policy must be one of: "
                + ", ".join(sorted(INTRACOLUMN_SELECTION_POLICIES))
            )
        active_sources = self._active_sources()
        if not active_sources:
            self.last_prediction_stats = {
                "active_source_count": 0,
                "candidate_segment_count": 0,
                "threshold_crossing_segment_count": 0,
                "accepted_candidate_count": 0,
                "raw_event_count": 0,
                "raw_predicted_column_count": 0,
            }
            return None

        candidates: dict[int, tuple[int, int, Segment, float, list[float]]] = {}
        # DEBUG WATCH: active_sources 是 lateral context。若这里已经很大，
        # 后面每个 source 都会查 incoming segment，候选数会迅速上升。
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

        preselection_by_segment: dict[int, PreselectionSegmentTrace] = {}
        if preselection_trace is not None:
            for (
                column_id,
                neuron_index,
                segment,
                upper_bound,
                _,
            ) in candidates.values():
                segment_index = next(
                    index
                    for index, candidate_segment in enumerate(
                        self.columns[column_id].neurons[
                            neuron_index
                        ].segments
                    )
                    if candidate_segment is segment
                )
                matched_count = sum(
                    source in segment.synapses for source in active_sources
                )
                item = PreselectionSegmentTrace(
                    target_column=column_id,
                    target_neuron=neuron_index,
                    segment_index=segment_index,
                    inspection_order=len(preselection_trace.segments),
                    segment=segment,
                    segment_identity=id(segment),
                    active_source_count=len(active_sources),
                    active_matched_synapse_count=matched_count,
                    sum_active_weights=upper_bound,
                    dendritic_threshold=self.params.dendrite_threshold,
                )
                preselection_trace.segments.append(item)
                preselection_by_segment[id(segment)] = item

        prediction_metadata: dict[int, dict[str, object]] = {}
        best_by_event: dict[tuple[int, float], tuple[int, float, float, Segment]] = {}
        raw_eligible_segments: set[int] = set()
        continuous_result_memo: (
            dict[
                tuple[tuple[float, float], ...],
                tuple[tuple[float, float] | None, dict[str, object]],
            ]
            | None
        ) = (
            {}
            if self.params.continuous_prediction_impl == "optimized_v2"
            else None
        )
        for (
            column_id,
            neuron_index,
            segment,
            upper_bound,
            contributions,
        ) in candidates.values():
            # The normalized response kernel never exceeds one. Segments whose
            # active weights cannot reach threshold need no timed evaluation.
            preselection_item = preselection_by_segment.get(id(segment))
            if self._dynamics.v_rest + upper_bound < self.params.dendrite_threshold:
                if preselection_item is not None:
                    preselection_item.elimination_stage = (
                        "UPPER_BOUND_ELIGIBILITY"
                    )
                    preselection_item.elimination_reason = (
                        "active_weight_upper_bound_below_threshold"
                    )
                continue
            trace_item = trace_by_segment.get(id(segment))
            if preselection_item is not None:
                preselection_item.response_computed = True
                preselection_item.furthest_stage_reached = "RESPONSE_COMPUTED"
            if self.params.continuous_dynamics:
                # PAPER-EXPLICIT: 连续 PSP 路径会寻找实际 dendritic threshold
                # crossing，再推算 soma firing time。诊断 trace 只记录，不改变排序。
                capture_contributions = (
                    trace_item is not None
                    or preselection_item is not None
                    or intracolumn_selection_policy != "existing"
                    or intracolumn_selection_trace is not None
                    or self.params.capture_prediction_contributions
                    or self.params.scenario1_contribution_mode
                    != "arrival-window"
                )
                diagnostic: dict[str, object] | None = (
                    {} if capture_contributions else None
                )
                continuous = self._continuous_segment_prediction(
                    segment,
                    active_sources,
                    diagnostic=diagnostic,
                    result_memo=continuous_result_memo,
                )
                if diagnostic is not None:
                    prediction_metadata[id(segment)] = diagnostic
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
                    crossing_contributions = tuple(
                        diagnostic.get("crossing_synapse_contributions", ())
                    )
                    trace_item.crossing_psp_contributions = tuple(
                        item.psp_contribution
                        for item in crossing_contributions
                        if isinstance(item, SynapsePSPContribution)
                    )
                    if trace_item.peak_dendritic_potential is not None:
                        trace_item.threshold_margin = (
                            trace_item.peak_dendritic_potential
                            - self.params.dendrite_threshold
                        )
                if preselection_item is not None and diagnostic is not None:
                    peak = diagnostic.get("peak_dendritic_potential")
                    crossing = diagnostic.get(
                        "first_threshold_crossing_time"
                    )
                    soma_time = diagnostic.get(
                        "predicted_soma_firing_time"
                    )
                    contributions_at_crossing = tuple(
                        item
                        for item in diagnostic.get(
                            "crossing_synapse_contributions",
                            (),
                        )
                        if isinstance(item, SynapsePSPContribution)
                    )
                    preselection_item.response_peak = (
                        peak if isinstance(peak, float) else None
                    )
                    preselection_item.first_threshold_crossing_time = (
                        crossing if isinstance(crossing, float) else None
                    )
                    preselection_item.predicted_soma_firing_time = (
                        soma_time if isinstance(soma_time, float) else None
                    )
                    preselection_item.contributor_count = len(
                        contributions_at_crossing
                    )
                    preselection_item.crossing_synapse_contributions = (
                        contributions_at_crossing
                    )
                    preselection_item.positive_contributor_count = sum(
                        item.psp_contribution > 0.0
                        for item in contributions_at_crossing
                    )
                    if isinstance(crossing, float):
                        preselection_item.crossed_threshold = True
                        preselection_item.furthest_stage_reached = (
                            "THRESHOLD_CROSSED"
                        )
                    elif continuous is None:
                        preselection_item.elimination_stage = (
                            "RESPONSE_COMPUTED"
                        )
                        preselection_item.elimination_reason = (
                            "below_dendritic_threshold"
                        )
                    if (
                        continuous is None
                        and isinstance(crossing, float)
                        and not isinstance(soma_time, float)
                    ):
                        preselection_item.elimination_stage = "SOMA_FIRING"
                        preselection_item.elimination_reason = (
                            "no_valid_soma_firing_time"
                        )
                if preselection_item is not None and continuous is not None:
                    preselection_item.response_at_selected_time = continuous[1]
                    preselection_item.candidate_score_value = continuous[1]
                timed_scores = (continuous,) if continuous is not None else ()
            else:
                # LOCAL CHOICE: event mode 使用固定 target_time 评分，主要用于
                # 历史对照；strict Fig.8/Fig.9 默认走 continuous_dynamics。
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
                if preselection_item is not None:
                    preselection_item.response_at_selected_time = score
                    preselection_item.candidate_score_value = score
                if score < self.params.dendrite_threshold:
                    if preselection_item is not None:
                        preselection_item.elimination_stage = (
                            "RESPONSE_COMPUTED"
                        )
                        preselection_item.elimination_reason = (
                            "below_dendritic_threshold"
                        )
                    continue
                if trace_item is not None:
                    trace_item.crossed_threshold = True
                if preselection_item is not None:
                    preselection_item.crossed_threshold = True
                    preselection_item.furthest_stage_reached = (
                        "THRESHOLD_CROSSED"
                    )
                if (
                    not self.params.continuous_dynamics
                    and not self._predictive_soma_can_fire
                ):
                    if preselection_item is not None:
                        preselection_item.elimination_stage = "SOMA_FIRING"
                        preselection_item.elimination_reason = (
                            "predictive_soma_cannot_fire"
                        )
                    continue
                if (
                    target_time < self.params.cycle_period
                    or target_time
                    > self.params.cycle_period
                    + self.params.cycle_period / 2.0
                    + self.params.timing_tolerance
                ):
                    if preselection_item is not None:
                        preselection_item.elimination_stage = (
                            "VALID_FIRING_WINDOW"
                        )
                        preselection_item.elimination_reason = (
                            "predicted_time_outside_valid_window"
                        )
                    continue
                raw_eligible_segments.add(id(segment))
                event_key = (column_id, round(target_time, 12))
                if preselection_item is not None:
                    preselection_item.valid_firing_time = True
                    preselection_item.event_time_key = event_key[1]
                    preselection_item.furthest_stage_reached = (
                        "VALID_FIRING_WINDOW"
                    )
                previous = best_by_event.get(event_key)
                if previous is None or score > previous[1]:
                    if (
                        previous is not None
                        and preselection_trace is not None
                    ):
                        previous_segment = previous[3]
                        previous_item = preselection_by_segment[
                            id(previous_segment)
                        ]
                        previous_item.elimination_stage = (
                            "EVENT_SCORE_SELECTION"
                        )
                        previous_item.elimination_reason = (
                            "lower_score_same_column_time"
                        )
                        previous_item.winner_segment_identity = id(segment)
                        preselection_trace.replacements.append(
                            PreselectionReplacementTrace(
                                group_type="column_time_event",
                                target_column=column_id,
                                target_neuron=None,
                                stable_time_key=event_key[1],
                                previous_segment=previous_segment,
                                replacement_segment=segment,
                                previous_segment_identity=id(
                                    previous_segment
                                ),
                                replacement_segment_identity=id(segment),
                                comparison_field="score",
                                previous_value=previous[1],
                                replacement_value=score,
                                previous_tie_break_value=(
                                    previous[0],
                                    previous_item.segment_index,
                                ),
                                replacement_tie_break_value=(
                                    neuron_index,
                                    (
                                        preselection_item.segment_index
                                        if preselection_item is not None
                                        else -1
                                    ),
                                ),
                                replacement_stage=(
                                    "EVENT_SCORE_SELECTION"
                                ),
                            )
                        )
                    # DEBUG WATCH: candidate selection。若许多候选 score 都贴近
                    # dendrite_threshold，这里的 winner 可能近似靠 tie/order 决定。
                    best_by_event[event_key] = (
                        neuron_index,
                        score,
                        target_time,
                        segment,
                    )
                elif preselection_item is not None:
                    preselection_item.elimination_stage = (
                        "EVENT_SCORE_SELECTION"
                    )
                    preselection_item.elimination_reason = (
                        "lower_or_equal_score_same_column_time"
                    )
                    preselection_item.winner_segment_identity = id(previous[3])
                    preselection_item.tie_break_used = score == previous[1]

        if not best_by_event:
            self.last_prediction_stats = {
                "active_source_count": len(active_sources),
                "candidate_segment_count": len(candidates),
                "threshold_crossing_segment_count": 0,
                "accepted_candidate_count": 0,
                "raw_event_count": 0,
                "raw_predicted_column_count": 0,
            }
            return None

        if preselection_trace is not None:
            for (column_id, event_time), winner in best_by_event.items():
                winner_item = preselection_by_segment[id(winner[3])]
                winner_item.became_event_winner = True
                winner_item.furthest_stage_reached = "EVENT_SCORE_WINNER"
                members = [
                    item
                    for item in preselection_trace.segments
                    if item.target_column == column_id
                    and item.valid_firing_time
                    and item.event_time_key == event_time
                ]
                ordered_members = sorted(
                    members,
                    key=lambda item: (
                        -(
                            item.candidate_score_value
                            if item.candidate_score_value is not None
                            else float("-inf")
                        ),
                        item.inspection_order,
                    ),
                )
                tie_count = sum(
                    item.candidate_score_value == winner[1]
                    for item in ordered_members
                )
                for rank, item in enumerate(ordered_members, start=1):
                    item.rank_within_event_group = rank
                    item.winner_segment_identity = id(winner[3])
                    if item.candidate_score_value is not None:
                        item.winner_score_gap = (
                            winner[1] - item.candidate_score_value
                        )
                    item.tie_break_used = tie_count > 1

        if (
            self.params.intracolumn_inhibition
            and intracolumn_selection_policy == "existing"
        ):
            # PAPER-EXPLICIT: 同一 mini-column 最终只保留最早 firing 的预测事件，
            # 防止一个列内多个 neuron 同时代表同一个输入列。
            winner_by_column: dict[int, tuple[int, float, float, Segment]] = {}
            for (column_id, _event_time), values in best_by_event.items():
                previous = winner_by_column.get(column_id)
                if previous is None or values[2] < previous[2]:
                    if (
                        previous is not None
                        and preselection_trace is not None
                    ):
                        previous_item = preselection_by_segment[
                            id(previous[3])
                        ]
                        replacement_item = preselection_by_segment[
                            id(values[3])
                        ]
                        previous_item.elimination_stage = (
                            "COLUMN_EARLIEST_SELECTION"
                        )
                        previous_item.elimination_reason = (
                            "later_firing_time_same_column"
                        )
                        previous_item.winner_segment_identity = id(values[3])
                        preselection_trace.replacements.append(
                            PreselectionReplacementTrace(
                                group_type="column",
                                target_column=column_id,
                                target_neuron=None,
                                stable_time_key=None,
                                previous_segment=previous[3],
                                replacement_segment=values[3],
                                previous_segment_identity=id(previous[3]),
                                replacement_segment_identity=id(values[3]),
                                comparison_field="predicted_time",
                                previous_value=previous[2],
                                replacement_value=values[2],
                                previous_tie_break_value=(
                                    previous[0],
                                    previous_item.inspection_order,
                                ),
                                replacement_tie_break_value=(
                                    values[0],
                                    replacement_item.inspection_order,
                                ),
                                replacement_stage=(
                                    "COLUMN_EARLIEST_SELECTION"
                                ),
                            )
                        )
                    winner_by_column[column_id] = values
                elif preselection_trace is not None:
                    item = preselection_by_segment[id(values[3])]
                    item.elimination_stage = "COLUMN_EARLIEST_SELECTION"
                    item.elimination_reason = (
                        "later_or_equal_firing_time_same_column"
                    )
                    item.winner_segment_identity = id(previous[3])
                    item.tie_break_used = values[2] == previous[2]
            selected_events = list(winner_by_column.items())
            if intracolumn_selection_trace is not None:
                # Trace the established result, but never use the helper's
                # independently computed return value on the strict path.
                traced_events = self._select_intracolumn_event_winners(
                    best_by_event=best_by_event,
                    policy="existing",
                    prediction_metadata=prediction_metadata,
                    active_sources=active_sources,
                    selection_trace=intracolumn_selection_trace,
                )
                assert [
                    (column, id(values[3])) for column, values in traced_events
                ] == [
                    (column, id(values[3])) for column, values in selected_events
                ]
        elif self.params.intracolumn_inhibition:
            selected_events = self._select_intracolumn_event_winners(
                best_by_event=best_by_event,
                policy=intracolumn_selection_policy,
                prediction_metadata=prediction_metadata,
                active_sources=active_sources,
                selection_trace=intracolumn_selection_trace,
            )
            assert {column for column, _values in selected_events} == {
                column for column, _event_time in best_by_event
            }
        else:
            selected_events = [
                (column_id, values)
                for (column_id, _event_time), values in best_by_event.items()
            ]

        if preselection_trace is not None:
            selected_by_column = {
                column_id: values
                for column_id, values in selected_events
            }
            event_winners = [
                preselection_by_segment[id(values[3])]
                for values in best_by_event.values()
            ]
            for column_id, selected in selected_by_column.items():
                selected_item = preselection_by_segment[id(selected[3])]
                selected_item.became_column_winner = True
                selected_item.became_prediction_candidate = True
                selected_item.furthest_stage_reached = (
                    "PREDICTION_CANDIDATE"
                )
                selected_item.elimination_stage = ""
                selected_item.elimination_reason = ""
                selected_item.winner_segment_identity = id(selected[3])
                members = [
                    item
                    for item in event_winners
                    if item.target_column == column_id
                ]
                ordered_members = sorted(
                    members,
                    key=lambda item: (
                        (
                            item.predicted_soma_firing_time
                            if item.predicted_soma_firing_time is not None
                            else float("inf")
                        ),
                        item.inspection_order,
                    ),
                )
                selected_time = selected[2]
                tie_count = sum(
                    item.predicted_soma_firing_time == selected_time
                    for item in ordered_members
                )
                for rank, item in enumerate(ordered_members, start=1):
                    item.rank_within_column_group = rank
                    if item.predicted_soma_firing_time is not None:
                        item.winner_time_gap = (
                            item.predicted_soma_firing_time - selected_time
                        )
                    item.tie_break_used = (
                        item.tie_break_used or tie_count > 1
                    )

        for column_id, values in selected_events:
            neuron_index, score, target_time, segment = values
            metadata = prediction_metadata.get(id(segment), {})
            crossing_time = metadata.get("first_threshold_crossing_time")
            peak_potential = metadata.get("peak_dendritic_potential")
            soma_time = metadata.get("predicted_soma_firing_time")
            self.last_prediction_candidates.setdefault(column_id, []).append(
                # DEBUG WATCH: 这里保存“同一次 predict_code”的 crossing metadata。
                # 后续 decode/advance/Scenario 1 都应复用它，不能重新预测。
                PredictionCandidate(
                    neuron_index=neuron_index,
                    score=score,
                    time=target_time - self.params.cycle_period,
                    segment=segment,
                    dendritic_crossing_time=(
                        crossing_time if isinstance(crossing_time, float) else None
                    ),
                    crossing_synapse_contributions=tuple(
                        item
                        for item in metadata.get(
                            "crossing_synapse_contributions", ()
                        )
                        if isinstance(item, SynapsePSPContribution)
                    ),
                    peak_dendritic_potential=(
                        peak_potential
                        if isinstance(peak_potential, float)
                        else None
                    ),
                    threshold_margin=(
                        peak_potential - self.params.dendrite_threshold
                        if isinstance(peak_potential, float)
                        else None
                    ),
                    predicted_soma_firing_time=(
                        soma_time if isinstance(soma_time, float) else None
                    ),
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
        self.last_prediction_stats = {
            "active_source_count": len(active_sources),
            "candidate_segment_count": len(candidates),
            "threshold_crossing_segment_count": len(raw_eligible_segments),
            "accepted_candidate_count": sum(
                len(candidates)
                for candidates in self.last_prediction_candidates.values()
            ),
            "raw_event_count": len(selected_events),
            "raw_predicted_column_count": len(
                {column_id for column_id, _values in selected_events}
            ),
        }
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
        ranking_mode: str = "current-score",
    ) -> SymbolCode | None:
        """Apply diagnostic intercolumn inhibition to real prediction cells."""

        if max_predictions_per_event <= 0:
            return None
        if ranking_mode not in self.PREDICTION_RANKING_MODES:
            raise ValueError(
                f"unsupported prediction ranking mode: {ranking_mode}"
            )
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
                    *self.prediction_candidate_sort_key(
                        item[1], ranking_mode
                    ),
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

    def prediction_candidate_sort_key(
        self,
        candidate: PredictionCandidate,
        ranking_mode: str = "current-score",
    ) -> tuple[float]:
        """Return a ground-truth-free ordering key for one saved candidate."""

        if ranking_mode == "current-score":
            return (-candidate.score,)
        if ranking_mode == "actual-peak":
            if candidate.peak_dendritic_potential is None:
                raise RuntimeError(
                    "actual-peak ranking requires prediction diagnostics"
                )
            return (-candidate.peak_dendritic_potential,)
        if ranking_mode == "earliest-crossing":
            if candidate.dendritic_crossing_time is None:
                raise RuntimeError(
                    "earliest-crossing ranking requires prediction diagnostics"
                )
            return (candidate.dendritic_crossing_time,)
        raise ValueError(
            f"unsupported prediction ranking mode: {ranking_mode}"
        )

    def select_coherent_prediction_events(
        self,
        predicted: SymbolCode,
        *,
        beam_width: int,
        lambda_coherence: float,
        lambda_support: float = 0.0,
        trace: PredictionTrace | None = None,
    ) -> SymbolCode | None:
        """Select a coherent set of real candidates without symbol feedback."""

        if beam_width <= 0:
            raise ValueError("beam_width must be positive")
        raw_events = {
            (event.column, round(event.time, 12)) for event in predicted.events
        }
        groups: list[list[tuple[int, PredictionCandidate, float, float]]] = []
        for event_time in self._event_times:
            choices = [
                (column, candidate)
                for column, candidates in self.last_prediction_candidates.items()
                for candidate in candidates
                if (column, round(candidate.time, 12)) in raw_events
                and abs(candidate.time - event_time)
                <= self.params.timing_tolerance
            ]
            if not choices:
                continue
            maximum = max(candidate.score for _column, candidate in choices)
            group: list[tuple[int, PredictionCandidate, float, float]] = []
            for column, candidate in choices:
                support = self.candidate_burst_psp_trace(candidate)
                denominator = support.total_psp
                predicted_fraction = (
                    (support.predicted_source_psp + support.unlabelled_psp)
                    / denominator
                    if denominator > 0.0
                    else 0.0
                )
                group.append(
                    (
                        column,
                        candidate,
                        candidate.score / maximum if maximum > 0.0 else 0.0,
                        predicted_fraction,
                    )
                )
            group.sort(
                key=lambda item: (
                    -item[2],
                    item[0],
                    item[1].neuron_index,
                    item[1].time,
                )
            )
            groups.append(group)

        beams: list[
            tuple[list[tuple[int, PredictionCandidate]], float, float]
        ] = [([], 0.0, 0.0)]
        for group in groups:
            expanded: list[
                tuple[
                    list[tuple[int, PredictionCandidate]],
                    float,
                    float,
                ]
            ] = []
            for selected, local_sum, support_sum in beams:
                used_columns = {column for column, _candidate in selected}
                for column, candidate, local_score, support in group:
                    if column in used_columns:
                        continue
                    expanded.append(
                        (
                            [*selected, (column, candidate)],
                            local_sum + local_score,
                            support_sum + support,
                        )
                    )
            if not expanded:
                continue
            expanded.sort(
                key=lambda beam: (
                    -self._coherent_beam_score(
                        beam,
                        lambda_coherence=lambda_coherence,
                        lambda_support=lambda_support,
                    ),
                    tuple(
                        (column, candidate.neuron_index, candidate.time)
                        for column, candidate in beam[0]
                    ),
                )
            )
            beams = expanded[:beam_width]
        if not beams or not beams[0][0]:
            return None
        selected = beams[0][0]
        if trace is not None:
            selected_segments = {id(candidate.segment) for _column, candidate in selected}
            for item in trace.segments:
                item.inhibited_intercolumn = (
                    item.entered_raw_prediction
                    and item.segment_identity not in selected_segments
                )
        return SymbolCode(
            events=tuple(
                SpikeEvent(column=column, time=candidate.time)
                for column, candidate in selected
            )
        )

    def _coherent_beam_score(
        self,
        beam: tuple[
            list[tuple[int, PredictionCandidate]],
            float,
            float,
        ],
        *,
        lambda_coherence: float,
        lambda_support: float,
    ) -> float:
        selected, local_sum, support_sum = beam
        source_sets = [set(candidate.segment.synapses) for _column, candidate in selected]
        pairwise: list[float] = []
        for index, left in enumerate(source_sets):
            for right in source_sets[index + 1 :]:
                union = left | right
                pairwise.append(len(left & right) / len(union) if union else 1.0)
        coherence = sum(pairwise) / len(pairwise) if pairwise else 0.0
        support = support_sum / len(selected) if selected else 0.0
        return (
            local_sum
            + lambda_coherence * coherence
            + lambda_support * support
        )

    def predict_symbol(
        self,
        min_overlap: int | None = None,
        candidate_symbols: set[str] | None = None,
    ) -> str | None:
        """Decode a predicted code using both mini-columns and firing order.

        中文调试提示：兼容旧接口，会先 predict_code() 一次，再 decode。
        如果你已经有 raw_code，请改用 decode_symbol_from_prediction()，避免重复预测。
        """
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
        """Return the strongest order-sensitive symbol predictions.

        DEBUG WATCH: 这个函数内部只能调用一次 predict_code()。断点统计调用次数
        时，decode_symbols_from_prediction() 不应该再次进入 predict_code()。
        """
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
        """Decode one symbol from an already computed neural prediction.

        STRICT PROTOCOL: decode 只把 raw neural activity 转成可读符号/数值；
        它不能 observe、不能改 previous_active_cells，也不能新建/强化 synapse。
        """

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
        """Decode a raw prediction without predicting or advancing state.

        DEBUG WATCH: 输入 predicted 必须来自同一次 predict_code()，排序依据是
        timed overlap 优先、column overlap 次之，再用 seeded RNG 做可复现 tie-break。
        """

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
        """Feed one symbol, learn online, and return current winner ids.

        中文调试提示：observe(symbol) 会先完整 SSTD 编码外部输入，所以在
        autonomous retrieval 阶段不能拿 decoded_symbol 调它。
        """
        code = self.encoder.encode(symbol)
        return self.observe_code(code, learn=learn)

    def observe_code(
        self,
        code: SymbolCode,
        learn: bool = True,
        observation_trace: ObservationTrace | None = None,
    ) -> dict[int, float]:
        """Feed an already encoded SSTD item into the sequential memory.

        中文调试提示：这是外部 proximal input 入口，也是 learn=True 时唯一
        会修改长期记忆结构/权重/age 的主路径。
        """

        # DEBUG WATCH: actual observe。previous_active 是上一周期 lateral
        # context；code.events 是这周期真实外部输入。
        previous_active = self._active_sources()
        if observation_trace is not None:
            observation_trace.events.clear()
            observation_trace.pre_observe_previous_winners = (
                self.previous_winners.copy()
            )
            observation_trace.pre_observe_active_sources = (
                previous_active.copy()
            )
        active_cells: dict[int, float] = {}
        learning_winners: dict[int, float] = {}
        predicted_sources: set[int] = set()
        burst_only_sources: set[int] = set()
        scenario_counts = {"scenario1": 0, "scenario2": 0, "scenario3": 0}

        for event in code.events:
            column_id = event.column
            column = self.columns[column_id]
            selected_segment: Segment | None = None
            reinforced_segment: Segment | None = None
            created_segment: Segment | None = None
            observe_scenario = ""
            scenario_assignment_reason = ""
            capture_scenario_details = bool(
                observation_trace is not None
                and observation_trace.capture_scenario_details
            )
            best_matching_trace = (
                BestMatchingTrace()
                if observation_trace is not None
                else None
            )
            existing_segment_count_in_column = (
                sum(len(neuron.segments) for neuron in column.neurons)
                if observation_trace is not None
                else 0
            )
            least_used_neuron_segment_count = (
                min(len(neuron.segments) for neuron in column.neurons)
                if observation_trace is not None
                else 0
            )
            predicted_candidates_in_column = self.last_prediction_candidates.get(
                column_id,
                [],
            )
            pre_gate_candidates = tuple(predicted_candidates_in_column)
            if not pre_gate_candidates:
                pre_gate_candidate_class = "NO_PREDICTIVE_CANDIDATE"
            elif len(pre_gate_candidates) == 1:
                pre_gate_candidate_class = "UNIQUE_PREDICTIVE_CANDIDATE"
            else:
                pre_gate_candidate_class = "MULTIPLE_PREDICTIVE_CANDIDATES"
            pre_gate_candidate_passes = tuple(
                abs(candidate.time - event.time)
                <= self.params.timing_tolerance
                for candidate in pre_gate_candidates
            )
            temporal_diagnostics_enabled = bool(
                getattr(
                    self.params,
                    "capture_temporal_confirmation_diagnostics",
                    False,
                )
            )
            pre_gate_metadata: dict[str, object] = {}
            if temporal_diagnostics_enabled:
                pre_gate_metadata = {
                    "pre_gate_candidate_count": len(pre_gate_candidates),
                    "pre_gate_candidate_identity_ids": tuple(
                        id(candidate) for candidate in pre_gate_candidates
                    ),
                    "pre_gate_candidate_neuron_ids": tuple(
                        candidate.neuron_index for candidate in pre_gate_candidates
                    ),
                    "pre_gate_candidate_segment_identities": tuple(
                        id(candidate.segment) for candidate in pre_gate_candidates
                    ),
                    "pre_gate_candidate_times": tuple(
                        candidate.time for candidate in pre_gate_candidates
                    ),
                    "pre_gate_candidate_scores": tuple(
                        candidate.score for candidate in pre_gate_candidates
                    ),
                    "pre_gate_candidate_crossing_times": tuple(
                        candidate.dendritic_crossing_time
                        for candidate in pre_gate_candidates
                    ),
                    "pre_gate_candidate_depolarization_states": tuple(
                        (
                            "PREDICTIVE_CANDIDATE_WITH_SOMA_TIME"
                            if candidate.predicted_soma_firing_time is not None
                            else (
                                "PREDICTIVE_CANDIDATE_WITH_CROSSING"
                                if candidate.dendritic_crossing_time is not None
                                else "PREDICTIVE_CANDIDATE_METADATA_UNRESOLVED"
                            )
                        )
                        for candidate in pre_gate_candidates
                    ),
                    "pre_gate_candidate_time_gate_passed": pre_gate_candidate_passes,
                    "pre_gate_candidate_class": pre_gate_candidate_class,
                    "pre_gate_timing_tolerance": self.params.timing_tolerance,
                }
            matching_predictions = [
                candidate
                for candidate, passes_gate in zip(
                    pre_gate_candidates, pre_gate_candidate_passes
                )
                if passes_gate
            ]
            predicted = (
                max(matching_predictions, key=lambda candidate: candidate.score)
                if matching_predictions
                else None
            )
            if predicted is None:
                # Scenario 2/3 候选入口：真实输入没有被 last_prediction_candidates
                # 准确预测时，尝试找同列里最匹配的已有 segment。
                if best_matching_trace is None:
                    matched = self._best_matching_neuron(
                        column_id,
                        column,
                        previous_active,
                        event.time,
                    )
                else:
                    matched = self._best_matching_neuron(
                        column_id,
                        column,
                        previous_active,
                        event.time,
                        trace=best_matching_trace,
                    )
            else:
                matched = (
                    (predicted.neuron_index, predicted.segment)
                    if predicted.segment.active
                    else None
                )
            was_predicted = predicted is not None and matched is not None
            if matched is None:
                # PAPER-EXPLICIT: Scenario 3。既没有预测，也没有足够匹配的
                # segment，就在 least-used neuron 上长一个新 segment。
                scenario_counts["scenario3"] += 1
                observe_scenario = "scenario3"
                if matching_predictions:
                    scenario_assignment_reason = (
                        "MATCHING_SEGMENT_NOT_ELIGIBLE"
                    )
                elif predicted_candidates_in_column:
                    scenario_assignment_reason = "PREDICTED_TIME_INVALID"
                elif existing_segment_count_in_column == 0:
                    scenario_assignment_reason = (
                        "NO_EXISTING_SEGMENT_IN_COLUMN"
                    )
                elif (
                    best_matching_trace is not None
                    and best_matching_trace.candidate_segment_count > 0
                ):
                    scenario_assignment_reason = (
                        "MATCHING_SEGMENT_NOT_ELIGIBLE"
                    )
                else:
                    scenario_assignment_reason = (
                        "EXISTING_SEGMENTS_BELOW_L_MATCH"
                    )
                neuron_index = self._least_used_neuron_index(column)
                if learn:
                    created_segment = self._grow_segment(
                        column_id,
                        neuron_index,
                        self.previous_winners,
                        event.time,
                        creation_scenario="scenario3",
                    )
                    selected_segment = created_segment
            else:
                neuron_index, segment = matched
                selected_segment = segment
                if not learn:
                    pass
                elif was_predicted:
                    # Paper scenario 1: a predictive neuron that subsequently
                    # receives its proximal input triggers learning directly.
                    # DEBUG WATCH: Scenario 1 branch。这里应复用 predicted
                    # PredictionCandidate，检查贡献突触如何被增强/减弱。
                    scenario_counts["scenario1"] += 1
                    observe_scenario = "scenario1"
                    scenario_assignment_reason = (
                        "CORRECTLY_PREDICTED_CELL_AVAILABLE"
                    )
                    reinforced_segment = segment
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
                        prediction_candidate=predicted,
                    )
                elif segment.timed_overlap(
                    previous_active,
                    self._dendritic_time(self.params.cycle_period + event.time),
                    self.params.timing_tolerance,
                ) >= self.params.l_match:
                    # PAPER-EXPLICIT: Scenario 2。没有提前预测成功，但存在
                    # 与当前输入匹配的旧 segment；会强化并补长缺失突触。
                    scenario_counts["scenario2"] += 1
                    observe_scenario = "scenario2"
                    scenario_assignment_reason = "MATCHING_SEGMENT_FOUND"
                    reinforced_segment = segment
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
                    # PAPER-EXPLICIT: Scenario 3 fallback。匹配 segment 不足
                    # L_match，转为新建 segment。
                    scenario_counts["scenario3"] += 1
                    observe_scenario = "scenario3"
                    scenario_assignment_reason = (
                        "EXISTING_SEGMENTS_BELOW_L_MATCH"
                    )
                    neuron_index = self._least_used_neuron_index(column)
                    created_segment = self._grow_segment(
                        column_id,
                        neuron_index,
                        self.previous_winners,
                        event.time,
                        creation_scenario="scenario3",
                    )
                    selected_segment = created_segment

            winner_id = self._cell_id(column_id, neuron_index)
            learning_winners[winner_id] = event.time
            if self._intralayer_parity_enabled():
                self._emit_intralayer_parity(
                    "observation_event",
                    {
                        "target_column": column_id,
                        "target_time": event.time,
                        "scenario": observe_scenario,
                        "was_predicted": was_predicted,
                        "actual_winner_neuron": neuron_index,
                        "actual_winner_cell_id": winner_id,
                        "predicted_neuron_index": (
                            predicted.neuron_index if predicted is not None else None
                        ),
                        "predicted_cell_id": (
                            self._cell_id(column_id, predicted.neuron_index)
                            if predicted is not None
                            else None
                        ),
                        "predicted_candidate_identity": (
                            id(predicted) if predicted is not None else None
                        ),
                        "predicted_candidate_time": (
                            predicted.time if predicted is not None else None
                        ),
                        "predicted_crossing_time": (
                            predicted.dendritic_crossing_time
                            if predicted is not None
                            else None
                        ),
                        "predicted_segment_identity": (
                            id(predicted.segment) if predicted is not None else None
                        ),
                        "selected_segment_identity": (
                            id(selected_segment) if selected_segment is not None else None
                        ),
                        "reinforced_segment_identity": (
                            id(reinforced_segment)
                            if reinforced_segment is not None
                            else None
                        ),
                        "created_segment_identity": (
                            id(created_segment) if created_segment is not None else None
                        ),
                        "previous_winner_ids": tuple(sorted(self.previous_winners)),
                        "previous_active_ids": tuple(sorted(previous_active)),
                        "scenario_assignment_reason": scenario_assignment_reason,
                        "predicted_candidate_count": len(
                            predicted_candidates_in_column
                        ),
                        "timing_matched_prediction_count": len(
                            matching_predictions
                        ),
                        **pre_gate_metadata,
                    },
                )
            if observation_trace is not None:
                observation_trace.events.append(
                    ObservationEventTrace(
                        target_column=column_id,
                        target_time=event.time,
                        winner_neuron=neuron_index,
                        winner_cell_id=winner_id,
                        scenario=observe_scenario,
                        was_predicted=was_predicted,
                        selected_segment=selected_segment,
                        reinforced_segment=reinforced_segment,
                        created_segment=created_segment,
                        predicted_candidate_count_in_column=len(
                            predicted_candidates_in_column
                        ),
                        timing_matched_prediction_count=len(
                            matching_predictions
                        ),
                        matching_segment_count=(
                            best_matching_trace.candidate_segment_count
                            if best_matching_trace is not None
                            else 0
                        ),
                        best_matching_overlap=(
                            best_matching_trace.best_overlap
                            if best_matching_trace is not None
                            else 0
                        ),
                        best_matching_score=(
                            best_matching_trace.best_score
                            if best_matching_trace is not None
                            else 0.0
                        ),
                        best_matching_active_synapse_count=(
                            best_matching_trace.best_active_synapse_count
                            if best_matching_trace is not None
                            else 0
                        ),
                        existing_segment_count_in_column=(
                            existing_segment_count_in_column
                        ),
                        existing_segment_count_on_selected_neuron=len(
                            column.neurons[neuron_index].segments
                        ) - (1 if created_segment is not None else 0),
                        least_used_neuron_segment_count=(
                            least_used_neuron_segment_count
                        ),
                        previous_winner_count=len(self.previous_winners),
                        previous_context_size=len(previous_active),
                        scenario_assignment_reason=scenario_assignment_reason,
                        predictive_neuron_ids=(
                            tuple(
                                sorted(
                                    {
                                        candidate.neuron_index
                                        for candidate in (
                                            predicted_candidates_in_column
                                        )
                                    }
                                )
                            )
                            if capture_scenario_details
                            else ()
                        ),
                        predictive_segment_count=(
                            len(
                                {
                                    id(candidate.segment)
                                    for candidate in (
                                        predicted_candidates_in_column
                                    )
                                }
                            )
                            if capture_scenario_details
                            else 0
                        ),
                        existing_segment_count_per_neuron=(
                            tuple(
                                len(neuron.segments)
                                - (
                                    1
                                    if (
                                        created_segment is not None
                                        and index == neuron_index
                                    )
                                    else 0
                                )
                                for index, neuron in enumerate(
                                    column.neurons
                                )
                            )
                            if capture_scenario_details
                            else ()
                        ),
                        best_matching_neuron=(
                            best_matching_trace.best_neuron
                            if best_matching_trace is not None
                            else None
                        ),
                        best_matching_segment=(
                            best_matching_trace.best_segment
                            if best_matching_trace is not None
                            else None
                        ),
                        predicted_time_available=bool(
                            predicted_candidates_in_column
                        ),
                        predicted_time_valid=bool(matching_predictions),
                        matching_response_available=bool(
                            best_matching_trace is not None
                            and best_matching_trace.candidate_segment_count
                        ),
                        selected_candidate_score=(
                            predicted.score
                            if predicted is not None
                            else (
                                best_matching_trace.best_score
                                if best_matching_trace is not None
                                else 0.0
                            )
                        ),
                        matching_candidates=(
                            tuple(best_matching_trace.candidates)
                            if (
                                best_matching_trace is not None
                                and capture_scenario_details
                            )
                            else ()
                        ),
                        predicted_neuron_index=(
                            predicted.neuron_index if predicted is not None else None
                        ),
                        predicted_cell_id=(
                            self._cell_id(column_id, predicted.neuron_index)
                            if predicted is not None
                            else None
                        ),
                        predicted_segment=(
                            predicted.segment if predicted is not None else None
                        ),
                        predicted_segment_identity=(
                            id(predicted.segment) if predicted is not None else None
                        ),
                        predicted_candidate_identity=(
                            id(predicted) if predicted is not None else None
                        ),
                        predicted_candidate_time=(
                            predicted.time if predicted is not None else None
                        ),
                        predicted_crossing_time=(
                            predicted.dendritic_crossing_time
                            if predicted is not None
                            else None
                        ),
                        pre_gate_candidate_count=(
                            len(pre_gate_candidates)
                            if temporal_diagnostics_enabled
                            else 0
                        ),
                        pre_gate_candidate_identity_ids=(
                            tuple(id(candidate) for candidate in pre_gate_candidates)
                            if temporal_diagnostics_enabled
                            else ()
                        ),
                        pre_gate_candidate_neuron_ids=(
                            tuple(
                                candidate.neuron_index
                                for candidate in pre_gate_candidates
                            )
                            if temporal_diagnostics_enabled
                            else ()
                        ),
                        pre_gate_candidate_segment_identities=(
                            tuple(
                                id(candidate.segment)
                                for candidate in pre_gate_candidates
                            )
                            if temporal_diagnostics_enabled
                            else ()
                        ),
                        pre_gate_candidate_times=(
                            tuple(
                                candidate.time for candidate in pre_gate_candidates
                            )
                            if temporal_diagnostics_enabled
                            else ()
                        ),
                        pre_gate_candidate_scores=(
                            tuple(
                                candidate.score for candidate in pre_gate_candidates
                            )
                            if temporal_diagnostics_enabled
                            else ()
                        ),
                        pre_gate_candidate_crossing_times=(
                            tuple(
                                candidate.dendritic_crossing_time
                                for candidate in pre_gate_candidates
                            )
                            if temporal_diagnostics_enabled
                            else ()
                        ),
                        pre_gate_candidate_depolarization_states=(
                            tuple(
                                (
                                    "PREDICTIVE_CANDIDATE_WITH_SOMA_TIME"
                                    if candidate.predicted_soma_firing_time is not None
                                    else (
                                        "PREDICTIVE_CANDIDATE_WITH_CROSSING"
                                        if candidate.dendritic_crossing_time is not None
                                        else "PREDICTIVE_CANDIDATE_METADATA_UNRESOLVED"
                                    )
                                )
                                for candidate in pre_gate_candidates
                            )
                            if temporal_diagnostics_enabled
                            else ()
                        ),
                        pre_gate_candidate_time_gate_passed=(
                            pre_gate_candidate_passes
                            if temporal_diagnostics_enabled
                            else ()
                        ),
                        pre_gate_candidate_class=(
                            pre_gate_candidate_class
                            if temporal_diagnostics_enabled
                            else "NO_PREDICTIVE_CANDIDATE"
                        ),
                        pre_gate_timing_tolerance=(
                            self.params.timing_tolerance
                            if temporal_diagnostics_enabled
                            else 0.0
                        ),
                    )
                )
            if was_predicted:
                active_cells[winner_id] = event.time
                predicted_sources.add(winner_id)
            else:
                # PAPER-EXPLICIT: unpredicted proximal input triggers burst。
                # DEBUG WATCH: 这里会把整列 neuron 加入 active_cells，是 Fig.8/9
                # raw-column explosion 的常见起点。
                for active_neuron in range(len(column.neurons)):
                    active_id = self._cell_id(column_id, active_neuron)
                    active_cells[active_id] = event.time
                    burst_only_sources.add(active_id)

        if learn:
            # STATE MUTATION: 惩罚错误预测会修改未兑现 segment 的权重/age。
            self._punish_wrong_predictions(
                {event.column: event.time for event in code.events}
            )
        self.previous_active_cells = active_cells
        self.previous_winners = learning_winners
        if self.params.capture_branch_diagnostics:
            self.previous_predicted_sources = predicted_sources
            self.previous_burst_only_sources = burst_only_sources
        else:
            self.previous_predicted_sources = set()
            self.previous_burst_only_sources = set()
        self.last_observe_stats = {
            **scenario_counts,
            "active_cell_count": len(active_cells),
            "winner_count": len(learning_winners),
            "predicted_cell_count": len(predicted_sources),
            "burst_cell_count": len(burst_only_sources),
        }
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
        """Advance retrieval through neurons that actually entered prediction.

        STRICT PROTOCOL: autonomous retrieval 只能用 raw_code 对应的预测细胞
        推进上下文。这里不会把 decoded word/value 重新编码，也不会学习。
        """

        active_cells = self.prediction_active_cells(code)
        if not active_cells:
            return False
        # STATE MUTATION: 只改 transient context。长期 segment/synapse 不变。
        self.previous_active_cells = active_cells
        self.previous_winners = active_cells.copy()
        if self.params.capture_branch_diagnostics:
            self.previous_predicted_sources = set(active_cells)
            self.previous_burst_only_sources = set()
        return True

    def prediction_active_cells(self, code: SymbolCode) -> dict[int, float]:
        """Resolve an inhibited prediction code back to its firing neurons.

        DEBUG WATCH: neural rollout 后 previous_active_cells 应等于这个返回值。
        若出现 decoded symbol 中其他列的整列 burst，说明走错到了 observe_code()。
        """

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
        """Resolve known proximal context without applying synaptic learning.

        中文调试提示：这是“外部输入但不学习”的 cell 激活规则。已预测事件选
        winner cell；未预测事件按论文 burst 整列 active。
        """

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
        *,
        trace: BestMatchingTrace | None = None,
    ) -> tuple[int, Segment] | None:
        """Find the best old segment for an unpredicted proximal event.

        DEBUG WATCH: Scenario 2/3 分界点。timed_overlap >= L_match 才能走
        Scenario 2，否则会退到新建 segment。
        """

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
        if trace is not None:
            trace.candidate_segment_count = len(candidates)

        for neuron_index, segment in candidates.values():
            soma_time = self.params.cycle_period + target_time
            score = self._segment_score(segment, active_sources, soma_time)
            timed_overlap = segment.timed_overlap(
                active_sources,
                self._dendritic_time(soma_time),
                self.params.timing_tolerance,
            )
            if trace is not None:
                trace.candidates.append(
                    MatchingCandidateTrace(
                        target_neuron=neuron_index,
                        segment=segment,
                        timed_overlap=timed_overlap,
                        candidate_score=score,
                    )
                )
            if (timed_overlap, score) > (best_overlap, best_score):
                best = (neuron_index, segment)
                best_overlap = timed_overlap
                best_score = score
                if trace is not None:
                    trace.best_neuron = neuron_index
                    trace.best_segment = segment
                    trace.best_active_synapse_count = sum(
                        source in segment.synapses
                        for source in active_sources
                    )
            elif (
                best is not None
                and (timed_overlap, score) == (best_overlap, best_score)
                and self._learning_rng.random() < 0.5
            ):
                best = (neuron_index, segment)
                if trace is not None:
                    trace.best_neuron = neuron_index
                    trace.best_segment = segment
                    trace.best_active_synapse_count = sum(
                        source in segment.synapses
                        for source in active_sources
                    )
        if trace is not None:
            trace.best_overlap = best_overlap
            trace.best_score = best_score
        return best

    def _grow_segment(
        self,
        column_id: int,
        neuron_index: int,
        active_sources: dict[int, float],
        target_time: float,
        *,
        creation_scenario: str = "unknown",
    ) -> Segment | None:
        """Create a new distal segment from current winners to a target event.

        PAPER-EXPLICIT: Scenario 3 和部分 Scenario 2 会长新突触。新突触的
        delay 由 source_time 与 target_time 决定，weight 从 w0 开始。
        """

        if not active_sources:
            return None
        neuron = self.columns[column_id].neurons[neuron_index]
        source_ids = tuple(sorted(active_sources))
        diagnostic_id: int | None = None
        if self.params.capture_branch_diagnostics:
            diagnostic_id = self._next_segment_diagnostic_id
            self._next_segment_diagnostic_id += 1
        segment = Segment(
            synapses={
                source: Synapse(
                    source=source,
                    delay=self._new_synapse_delay(target_time, source_time),
                    weight=self.params.w0,
                    age=0,
                )
                for source, source_time in active_sources.items()
            },
            target_time=self.params.cycle_period + target_time,
            diagnostic_id=diagnostic_id,
            creation_sentence_index=(
                self._diagnostic_sentence_index
                if self.params.capture_branch_diagnostics
                else None
            ),
            creation_transition_index=(
                self._diagnostic_transition_index
                if self.params.capture_branch_diagnostics
                else None
            ),
            creation_target_column=column_id if diagnostic_id is not None else None,
            creation_target_time=target_time if diagnostic_id is not None else None,
            creation_target_neuron=neuron_index if diagnostic_id is not None else None,
            creation_source_cell_ids=source_ids if diagnostic_id is not None else (),
            creation_source_fingerprint=(
                ",".join(map(str, source_ids))
                if diagnostic_id is not None
                else None
            ),
        )
        neuron.segments.append(segment)
        # STATE MUTATION: 长期记忆结构在这里真正增加；同时维护 incoming index，
        # 让后续 predict_code 可以从 source cell 快速找到目标 segment。
        for source in active_sources:
            self._incoming_index.setdefault(source, []).append(
                (column_id, neuron_index, segment)
            )
        self._emit_intralayer_parity(
            "segment_created",
            {
                "creation_scenario": creation_scenario,
                "target_column": column_id,
                "target_neuron": neuron_index,
                "segment_identity": id(segment),
                "segment_diagnostic_id": diagnostic_id,
                "source_ids": tuple(sorted(active_sources)),
                "previous_winner_ids": tuple(sorted(self.previous_winners)),
                "previous_active_ids": tuple(sorted(self.previous_active_cells)),
                "previous_predicted_ids": tuple(
                    sorted(self.previous_predicted_sources)
                ),
                "previous_burst_only_ids": tuple(
                    sorted(self.previous_burst_only_sources)
                ),
                "source_ids_equal_previous_winners": (
                    set(active_sources) == set(self.previous_winners)
                ),
            },
        )
        return segment

    def _new_synapse_delay(
        self,
        target_time: float,
        source_time: float,
    ) -> float:
        """Compute distal delay from source event time to target dendritic time.

        LOCAL CHOICE: peak-aligned-delay 只是 nonpaper diagnostic；strict 默认
        current-delay，不能静默改成补偿 PSP peak 的公式。
        """

        delay = self.params.cycle_period / 2.0 + target_time - source_time
        if self.params.synapse_delay_mode == "peak-aligned-delay":
            delay -= self._kernel_peak_time
        return delay

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
        result_memo: (
            dict[
                tuple[tuple[float, float], ...],
                tuple[tuple[float, float] | None, dict[str, object]],
            ]
            | None
        ) = None,
    ) -> tuple[float, float] | None:
        """Integrate one distal segment through dendritic and soma firing.

        DEBUG WATCH: 这里把 active synapse 的 arrival_time/weight 送进连续
        PSP 积分。diagnostic 打开时会保存 crossing 时每条突触的 PSP contribution。
        """

        matched = [
            (source, source_time, source_time + synapse.delay, synapse.weight)
            for source, source_time in active_sources.items()
            if (synapse := segment.synapses.get(source)) is not None
        ]
        if not matched:
            return None
        self._emit_runtime_debug(
            "continuous_segment_prediction_enter",
            {
                "segment_identity": id(segment),
                "segment_synapse_count": len(segment.synapses),
                "matched_synapse_count": len(matched),
                "arrival_count": len(matched),
                "continuous_impl": self.params.continuous_prediction_impl,
                "diagnostic_requested": diagnostic is not None,
                "result_memo_enabled": result_memo is not None,
            },
        )
        arrivals = [(arrival, weight) for _source, _time, arrival, weight in matched]
        memo_key = tuple(arrivals)
        cached_result = result_memo.get(memo_key) if result_memo is not None else None
        if cached_result is not None and (
            diagnostic is None or cached_result[1]
        ):
            result = cached_result[0]
            if diagnostic is not None:
                diagnostic.update(cached_result[1])
        else:
            result = self._continuous_prediction_from_arrivals(
                arrivals,
                diagnostic=diagnostic,
            )
            if result_memo is not None:
                result_memo[memo_key] = (
                    result,
                    dict(diagnostic) if diagnostic is not None else {},
                )
        if diagnostic is not None:
            crossing = diagnostic.get("first_threshold_crossing_time")
            crossing_contributions: list[SynapsePSPContribution] = []
            if isinstance(crossing, float):
                for source, source_time, arrival, weight in matched:
                    response = spike_response(crossing - arrival, self._dynamics)
                    crossing_contributions.append(
                        SynapsePSPContribution(
                            source_cell_id=source,
                            source_time=source_time,
                            arrival_time=arrival,
                            weight=weight,
                            psp_contribution=weight * response,
                        )
                    )
            contributing = [
                item
                for item in crossing_contributions
                if item.psp_contribution > 0.0
            ]
            diagnostic.update(
                {
                    "crossing_synapse_contributions": tuple(
                        crossing_contributions
                    ),
                    "contributing_synapse_count": len(contributing),
                    "contributing_source_cell_ids": tuple(
                        item.source_cell_id for item in contributing
                    ),
                    "contributing_source_times": tuple(
                        item.source_time for item in contributing
                    ),
                    "synaptic_arrival_times": tuple(
                        item.arrival_time for item in contributing
                    ),
                    "contributing_weights": tuple(
                        item.weight for item in contributing
                    ),
                }
            )
        self._emit_runtime_debug(
            "continuous_segment_prediction_exit",
            {
                "segment_identity": id(segment),
                "matched_synapse_count": len(matched),
                "result_available": result is not None,
                "crossing_time": (
                    diagnostic.get("first_threshold_crossing_time")
                    if diagnostic is not None
                    else None
                ),
                "soma_firing_time": (
                    diagnostic.get("predicted_soma_firing_time")
                    if diagnostic is not None
                    else None
                ),
            },
        )
        return result

    def _continuous_prediction_from_arrivals(
        self,
        arrivals: list[tuple[float, float]],
        diagnostic: dict[str, object] | None = None,
    ) -> tuple[float, float] | None:
        """Strict prediction path using the optimized equivalent integrator."""

        impl = self.params.continuous_prediction_impl
        if impl == "reference":
            return self.reference_continuous_prediction(
                arrivals,
                diagnostic=diagnostic,
            )
        if impl == "optimized_v1":
            return self.optimized_v1_continuous_prediction(
                arrivals,
                diagnostic=diagnostic,
            )
        return self.optimized_continuous_prediction(arrivals, diagnostic=diagnostic)

    def reference_continuous_prediction(
        self,
        arrivals: list[tuple[float, float]],
        diagnostic: dict[str, object] | None = None,
    ) -> tuple[float, float] | None:
        """Integrate a canonical set of delayed, weighted distal spikes.

        中文调试提示：先找 dendritic threshold crossing，再模拟 soma 是否在
        depolarization window 内放电。返回值是 (soma firing time, score/peak)。
        """

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
            # DEBUG WATCH: potential(time) 是所有 active synapse PSP 的和。
            # trace 开关不能改变这个求和顺序，否则会影响浮点可重复性。
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
                # DEBUG WATCH: threshold crossing。若 crossing 很晚，预测 event
                # time 可能超过 SSTD timing_tolerance，引发“列对但时间错”的 burst。
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

    def optimized_v1_continuous_prediction(
        self,
        arrivals: list[tuple[float, float]],
        diagnostic: dict[str, object] | None = None,
    ) -> tuple[float, float] | None:
        """Equivalent continuous integration with lower Python overhead.

        This keeps the same time grid, rounded cache key, spike-response
        formula, threshold tests, and accumulation order as
        reference_continuous_prediction(). The savings come only from local
        bindings and reusing identical rounded responses inside one potential
        evaluation.
        """

        step = self.params.integration_step
        if step <= 0.0:
            raise ValueError("integration_step must be positive")

        effective_threshold = (
            self.params.dendrite_threshold
            - self.params.integration_voltage_tolerance
        )
        response_cache = self._spike_response_cache
        cache_get = response_cache.get
        remember_response = self._remember_spike_response
        dynamics = self._dynamics
        spike_response_fn = spike_response
        round_fn = round
        v_rest = dynamics.v_rest
        threshold = self.params.dendrite_threshold

        arrival_times = tuple(arrival for arrival, _weight in arrivals)
        weights = tuple(weight for _arrival, weight in arrivals)

        def potential(time: float, *, remember: bool = True) -> float:
            total = 0
            local_responses: dict[float, float] = {}
            for arrival, weight in zip(arrival_times, weights):
                cache_key = round_fn(time - arrival, 9)
                response = local_responses.get(cache_key)
                if response is None:
                    response = cache_get(cache_key)
                    if response is None:
                        response = spike_response_fn(cache_key, dynamics)
                        if remember:
                            remember_response(cache_key, response)
                    local_responses[cache_key] = response
                total += weight * response
            return v_rest + total

        start = min(arrival_times)
        stop = max(arrival_times) + max(
            self.params.cycle_period / 2.0,
            5.0 * self.params.tau_m,
        )
        previous_time = start
        previous_potential = potential(start)
        crossing: float | None = None
        peak = previous_potential
        time = start + step
        half_step = step / 2.0
        while time <= stop + half_step:
            voltage = potential(time)
            peak = max(peak, voltage)
            if previous_potential < effective_threshold <= voltage:
                low = previous_time
                high = time
                for _ in range(12):
                    middle = (low + high) / 2.0
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
            while diagnostic_time <= stop + half_step:
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
        soma_stop = crossing + dynamics.depolarization_duration
        soma_threshold = dynamics.soma_threshold - self.params.integration_voltage_tolerance
        while soma_time <= soma_stop + half_step:
            if state.membrane_potential(soma_time, dynamics) >= soma_threshold:
                if diagnostic is not None:
                    diagnostic["predicted_soma_firing_time"] = soma_time
                return soma_time, max(peak, threshold)
            soma_time += step
        if state.membrane_potential(soma_stop, dynamics) >= soma_threshold:
            if diagnostic is not None:
                diagnostic["predicted_soma_firing_time"] = soma_stop
            return soma_stop, max(peak, threshold)
        return None

    def optimized_continuous_prediction(
        self,
        arrivals: list[tuple[float, float]],
        diagnostic: dict[str, object] | None = None,
    ) -> tuple[float, float] | None:
        """Equivalent continuous integration with conservative local binding.

        optimized_v2 deliberately keeps the same per-synapse cache lookup order
        as reference_continuous_prediction(). The first optimization round's
        per-potential local response dict is preserved as optimized_v1 for A/B,
        but this path avoids that short-lived dict when duplicate rounded
        deltas are rare.
        """

        step = self.params.integration_step
        if step <= 0.0:
            raise ValueError("integration_step must be positive")

        effective_threshold = (
            self.params.dendrite_threshold
            - self.params.integration_voltage_tolerance
        )
        response_cache = self._spike_response_cache
        cache_get = response_cache.get
        remember_response = self._remember_spike_response
        dynamics = self._dynamics
        spike_response_fn = spike_response
        round_fn = round
        v_rest = dynamics.v_rest
        threshold = self.params.dendrite_threshold
        arrivals_local = arrivals

        def potential(time: float, *, remember: bool = True) -> float:
            total = 0
            for arrival, weight in arrivals_local:
                cache_key = round_fn(time - arrival, 9)
                response = cache_get(cache_key)
                if response is None:
                    response = spike_response_fn(cache_key, dynamics)
                    if remember:
                        remember_response(cache_key, response)
                total += weight * response
            return v_rest + total

        start = min(arrival for arrival, _weight in arrivals_local)
        stop = max(arrival for arrival, _weight in arrivals_local) + max(
            self.params.cycle_period / 2.0,
            5.0 * self.params.tau_m,
        )
        previous_time = start
        previous_potential = potential(start)
        crossing: float | None = None
        peak = previous_potential
        time = start + step
        half_step = step / 2.0
        while time <= stop + half_step:
            voltage = potential(time)
            peak = max(peak, voltage)
            if previous_potential < effective_threshold <= voltage:
                low = previous_time
                high = time
                for _ in range(12):
                    middle = (low + high) / 2.0
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
            while diagnostic_time <= stop + half_step:
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
        soma_stop = crossing + dynamics.depolarization_duration
        soma_threshold = dynamics.soma_threshold - self.params.integration_voltage_tolerance
        while soma_time <= soma_stop + half_step:
            if state.membrane_potential(soma_time, dynamics) >= soma_threshold:
                if diagnostic is not None:
                    diagnostic["predicted_soma_firing_time"] = soma_time
                return soma_time, max(peak, threshold)
            soma_time += step
        if state.membrane_potential(soma_stop, dynamics) >= soma_threshold:
            if diagnostic is not None:
                diagnostic["predicted_soma_firing_time"] = soma_stop
            return soma_stop, max(peak, threshold)
        return None

    def _dendritic_time(self, soma_time: float) -> float:
        return soma_time - self.params.cycle_period / 2.0

    def _predictive_soma_fires(self, target_time: float) -> bool:
        dynamics = self._dynamics
        state = DSNeuronState()
        state.trigger_dendritic_spike(self._dendritic_time(target_time))
        return state.try_fire(target_time, dynamics)

    def scenario1_contributing_sources(
        self,
        candidate: PredictionCandidate,
        *,
        mode: str | None = None,
        active_sources: dict[int, float] | None = None,
        target_time: float | None = None,
    ) -> set[int]:
        """Select Scenario-1 sources without recomputing a prediction.

        STRICT PROTOCOL: 默认 arrival-window 是论文对齐规则。continuous-positive
        和 continuous-causal 只用于 nonpaper diagnostic，且必须读取同一个
        PredictionCandidate 保存的 crossing contributions。
        """

        selected_mode = mode or self.params.scenario1_contribution_mode
        if selected_mode not in self.SCENARIO1_CONTRIBUTION_MODES:
            raise ValueError(
                f"unsupported Scenario-1 contribution mode: {selected_mode}"
            )
        segment = candidate.segment
        if selected_mode == "arrival-window":
            # PAPER-EXPLICIT: 当前 strict 默认，按目标 dendritic time 的
            # timing_tolerance 窗口判断哪些 synapse contributed。
            if active_sources is None or target_time is None:
                raise ValueError(
                    "arrival-window requires active_sources and target_time"
                )
            dendritic_time = self._dendritic_time(
                self.params.cycle_period + target_time
            )
            return {
                source
                for source, source_time in active_sources.items()
                if (synapse := segment.synapses.get(source)) is not None
                and abs(source_time + synapse.delay - dendritic_time)
                <= self.params.timing_tolerance
            }

        positive = [
            item
            for item in candidate.crossing_synapse_contributions
            if item.psp_contribution > 0.0
            and item.source_cell_id in segment.synapses
        ]
        if selected_mode == "continuous-positive":
            # LOCAL CHOICE: 诊断模式，只选 crossing 时 PSP contribution > 0 的
            # active synapse，用来检查预测/学习贡献定义不一致。
            return {item.source_cell_id for item in positive}
        if candidate.dendritic_crossing_time is None:
            raise RuntimeError("continuous candidate has no threshold crossing")

        effective_threshold = (
            self.params.dendrite_threshold
            - self.params.integration_voltage_tolerance
        )
        potential = self._dynamics.v_rest
        causal: set[int] = set()
        for item in sorted(
            positive,
            key=lambda value: (
                -value.psp_contribution,
                value.source_cell_id,
            ),
        ):
            # LOCAL CHOICE: causal subset 是诊断统计，不改变预测本身。
            causal.add(item.source_cell_id)
            potential += item.psp_contribution
            if potential >= effective_threshold:
                break
        if potential < effective_threshold:
            raise RuntimeError(
                "saved prediction contributions do not reach threshold"
            )
        return causal

    def candidate_burst_psp_trace(
        self,
        candidate: PredictionCandidate,
    ) -> BurstPSPTrace:
        """Split saved crossing PSP by the active-cell origin labels."""

        predicted = 0.0
        burst_only = 0.0
        unlabelled = 0.0
        for item in candidate.crossing_synapse_contributions:
            if item.source_cell_id in self.previous_predicted_sources:
                predicted += item.psp_contribution
            elif item.source_cell_id in self.previous_burst_only_sources:
                burst_only += item.psp_contribution
            else:
                unlabelled += item.psp_contribution
        total = predicted + burst_only + unlabelled
        threshold = (
            self.params.dendrite_threshold
            - self.params.integration_voltage_tolerance
            - self._dynamics.v_rest
        )
        predicted_crosses = predicted + unlabelled >= threshold
        burst_crosses = burst_only >= threshold
        if predicted_crosses:
            classification = "predicted-supported"
        elif burst_crosses:
            classification = "burst-only"
        else:
            classification = "burst-assisted"
        return BurstPSPTrace(
            total_psp=total,
            predicted_source_psp=predicted,
            burst_only_psp=burst_only,
            unlabelled_psp=unlabelled,
            predicted_without_burst_crosses=predicted_crosses,
            burst_only_crosses=burst_crosses,
            classification=classification,
        )

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
        prediction_candidate: PredictionCandidate | None = None,
    ) -> None:
        """Apply learning to one chosen segment.

        中文调试提示：Scenario 1 会强化真实促成预测的突触并减弱非贡献突触；
        Scenario 2 可补长缺失 winner 突触；Scenario 3 不进这里而是 grow segment。
        forgetting/age 在后续 pruning 路径中使用。
        """

        scenario1_count_before = segment.scenario1_reinforcements
        scenario2_count_before = segment.scenario2_reinforcements
        source_count_before = len(segment.synapses)
        if segment.diagnostic_id is not None:
            if scenario == "scenario1":
                segment.scenario1_reinforcements += 1
            elif scenario == "scenario2":
                segment.scenario2_reinforcements += 1
        weights_before_by_source = {
            source: synapse.weight for source, synapse in segment.synapses.items()
        }
        weights_before = tuple(
            weight for _source, weight in sorted(weights_before_by_source.items())
        )
        neuron = self.columns[column_id].neurons[neuron_index]
        capture_scope = bool(
            self._intralayer_parity_enabled()
            or self.reinforcement_trace_callback is not None
        )
        if capture_scope:
            ages_before_by_source = {
                source: synapse.age
                for source, synapse in segment.synapses.items()
            }
            other_segment_weights_before = {
                (id(other_segment), source): synapse.weight
                for other_segment in neuron.segments
                if other_segment is not segment
                for source, synapse in other_segment.synapses.items()
            }
            other_segment_ages_before = {
                (id(other_segment), source): synapse.age
                for other_segment in neuron.segments
                if other_segment is not segment
                for source, synapse in other_segment.synapses.items()
            }
        else:
            ages_before_by_source = {}
            other_segment_weights_before = {}
            other_segment_ages_before = {}
        dendritic_time = self._dendritic_time(
            self.params.cycle_period + target_time
        )
        if scenario == "scenario1":
            # DEBUG WATCH: Scenario 1 branch。prediction_candidate_identity 必须
            # 对应 observe_code 中刚匹配到的 predicted candidate。
            if (
                prediction_candidate is None
                or prediction_candidate.segment is not segment
            ):
                raise RuntimeError(
                    "Scenario 1 requires its matching PredictionCandidate"
                )
            contributed = self.scenario1_contributing_sources(
                prediction_candidate,
                active_sources=active_sources,
                target_time=target_time,
            )
        else:
            # DEBUG WATCH: Scenario 2/other branch。这里仍按 arrival window
            # 从 active_sources 判断贡献突触。
            contributed = {
                source
                for source, source_time in active_sources.items()
                if (synapse := segment.synapses.get(source)) is not None
                and abs(source_time + synapse.delay - dendritic_time)
                <= self.params.timing_tolerance
            }

        if grow_missing:
            # STATE MUTATION: Scenario 2 会把 missing growth_sources 加成新突触。
            for source, source_time in (growth_sources or {}).items():
                if source in segment.synapses:
                    continue
                segment.synapses[source] = Synapse(
                    source=source,
                    delay=self._new_synapse_delay(target_time, source_time),
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
                    # STATE MUTATION: 非贡献突触被减弱并老化；这是排查
                    # “正确预测后被误减弱”的断点。
                    synapse.weight = max(0.0, synapse.weight - self.params.delta_w)
                    synapse.age += 1

        for other_segment in neuron.segments:
            if other_segment is segment:
                continue
            for synapse in other_segment.synapses.values():
                if depress_noncontributing:
                    synapse.weight = max(
                        0.0, synapse.weight - self.params.delta_w
                    )
                    synapse.age += 1

        if capture_scope:
            weights_after_by_source = {
                source: synapse.weight
                for source, synapse in segment.synapses.items()
            }
            strengthened = {
                source
                for source, before in weights_before_by_source.items()
                if weights_after_by_source.get(source, before) > before
            }
            weakened = {
                source
                for source, before in weights_before_by_source.items()
                if weights_after_by_source.get(source, before) < before
            }
            other_segment_weights_after = {
                (id(other_segment), source): synapse.weight
                for other_segment in neuron.segments
                if other_segment is not segment
                for source, synapse in other_segment.synapses.items()
            }
            other_segment_weakened_count = sum(
                other_segment_weights_after.get(key, before) < before
                for key, before in other_segment_weights_before.items()
            )
            same_segment_noncontributor_aged_count = sum(
                segment.synapses[source].age > ages_before_by_source[source]
                for source in set(weights_before_by_source) - contributed
            )
            other_segment_ages_after = {
                (id(other_segment), source): synapse.age
                for other_segment in neuron.segments
                if other_segment is not segment
                for source, synapse in other_segment.synapses.items()
            }
            other_segment_aged_count = sum(
                other_segment_ages_after.get(key, before) > before
                for key, before in other_segment_ages_before.items()
            )
        else:
            strengthened = set()
            weakened = set()
            other_segment_weakened_count = 0
            same_segment_noncontributor_aged_count = 0
            other_segment_aged_count = 0
        if self.reinforcement_trace_callback is not None:
            actual_positive = (
                {
                    item.source_cell_id
                    for item in prediction_candidate.crossing_synapse_contributions
                    if item.psp_contribution > 0.0
                }
                if prediction_candidate is not None
                else set()
            )
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
                    contribution_mode=(
                        self.params.scenario1_contribution_mode
                        if scenario == "scenario1"
                        else "arrival-window"
                    ),
                    prediction_candidate_identity=(
                        id(prediction_candidate)
                        if prediction_candidate is not None
                        else None
                    ),
                    prediction_crossing_time=(
                        prediction_candidate.dendritic_crossing_time
                        if prediction_candidate is not None
                        else None
                    ),
                    actual_positive_synapse_count=len(actual_positive),
                    actual_positive_weakened_count=len(
                        actual_positive.intersection(weakened)
                    ),
                    strengthened_synapse_count=len(strengthened),
                    weakened_synapse_count=len(weakened),
                    segment_diagnostic_id=segment.diagnostic_id,
                    source_count_before=source_count_before,
                    source_count_after=len(segment.synapses),
                    scenario1_count_before=scenario1_count_before,
                    scenario1_count_after=segment.scenario1_reinforcements,
                    scenario2_count_before=scenario2_count_before,
                    scenario2_count_after=segment.scenario2_reinforcements,
                    contributed_source_ids=tuple(sorted(contributed)),
                    strengthened_source_ids=tuple(sorted(strengthened)),
                    weakened_source_ids=tuple(sorted(weakened)),
                    same_segment_noncontributor_ids=tuple(
                        sorted(set(weights_before_by_source) - contributed)
                    ),
                    same_segment_noncontributor_aged_count=(
                        same_segment_noncontributor_aged_count
                    ),
                    other_segment_synapse_count=len(
                        other_segment_weights_before
                    ),
                    other_segment_weakened_count=other_segment_weakened_count,
                    other_segment_aged_count=other_segment_aged_count,
                    previous_winner_source_ids=tuple(
                        sorted(self.previous_winners)
                    ),
                    growth_source_ids=tuple(sorted(growth_sources or {})),
                    added_source_ids=tuple(
                        sorted(set(segment.synapses) - set(weights_before_by_source))
                    ),
                )
            )

        self._emit_intralayer_parity(
            "reinforcement_event",
            {
                "scenario": scenario,
                "target_column": column_id,
                "target_time": target_time,
                "target_neuron": neuron_index,
                "segment_identity": id(segment),
                "prediction_candidate_identity": (
                    id(prediction_candidate)
                    if prediction_candidate is not None
                    else None
                ),
                "prediction_neuron": (
                    prediction_candidate.neuron_index
                    if prediction_candidate is not None
                    else None
                ),
                "prediction_segment_identity": (
                    id(prediction_candidate.segment)
                    if prediction_candidate is not None
                    else None
                ),
                "causal_segment_identity_matches": (
                    prediction_candidate is not None
                    and prediction_candidate.segment is segment
                ),
                "actual_winner_neuron": neuron_index,
                "causal_neuron_identity_matches": (
                    prediction_candidate is not None
                    and prediction_candidate.neuron_index == neuron_index
                ),
                "contributed_source_ids": tuple(sorted(contributed)),
                "strengthened_source_ids": tuple(sorted(strengthened)),
                "weakened_source_ids": tuple(sorted(weakened)),
                "same_segment_noncontributor_ids": tuple(
                    sorted(set(weights_before_by_source) - contributed)
                ),
                "same_segment_noncontributor_aged_count": (
                    same_segment_noncontributor_aged_count
                ),
                "other_segment_synapse_count": len(other_segment_weights_before),
                "other_segment_weakened_count": other_segment_weakened_count,
                "other_segment_aged_count": other_segment_aged_count,
                "previous_winner_source_ids": tuple(
                    sorted(self.previous_winners)
                ),
                "growth_source_ids": tuple(sorted(growth_sources or {})),
                "added_source_ids": tuple(
                    sorted(set(segment.synapses) - set(weights_before_by_source))
                ),
                "depression_requested": depress_noncontributing,
                "same_segment_noncontributors_expected": (
                    tuple(sorted(set(weights_before_by_source) - contributed))
                    if depress_noncontributing
                    else ()
                ),
                "other_segment_synapses_expected": (
                    len(other_segment_weights_before)
                    if depress_noncontributing
                    else 0
                ),
            },
        )

        self._prune_neuron(neuron)

    def _punish_wrong_predictions(self, active_events: dict[int, float]) -> None:
        self._emit_prune_diagnostic(
            "PUNISH_WRONG_PREDICTIONS_ENTER",
            target_column=None,
            neuron_index=None,
            segment_count=sum(
                len(neuron.segments)
                for column in self.columns
                for neuron in column.neurons
            ),
            container_length=len(self.last_prediction_candidates),
        )
        active_sources = self._active_sources()
        for column_id, candidates in self.last_prediction_candidates.items():
            actual_time = active_events.get(column_id)
            for candidate in candidates:
                predicted_time = candidate.time
                neuron_index = candidate.neuron_index
                segment = candidate.segment
                self._emit_prune_diagnostic(
                    "PUNISH_WRONG_PREDICTIONS_TARGET_SELECTED",
                    target_column=column_id,
                    neuron_index=neuron_index,
                    candidate_segment=segment,
                    segment_count=len(
                        self.columns[column_id].neurons[neuron_index].segments
                    ),
                    container_length=len(candidates),
                )
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
                self._emit_intralayer_parity(
                    "wrong_prediction_punishment",
                    {
                        "target_column": column_id,
                        "predicted_neuron": neuron_index,
                        "predicted_candidate_identity": id(candidate),
                        "predicted_candidate_score": candidate.score,
                        "predicted_segment_identity": id(segment),
                        "predicted_time": predicted_time,
                        "predicted_crossing_time": candidate.dendritic_crossing_time,
                        "actual_time": actual_time,
                        "actual_column_event_present": actual_time is not None,
                        "contributed_source_ids": tuple(sorted(contributed)),
                        "punishment_reason": (
                            "NO_PROXIMAL_EVENT"
                            if actual_time is None
                            else "PROXIMAL_TIME_MISMATCH"
                        ),
                    },
                )
                for source in contributed:
                    synapse = segment.synapses[source]
                    synapse.weight = max(
                        0.0, synapse.weight - self.params.delta_w_bad
                    )
                    synapse.age += 1
                self._emit_prune_diagnostic(
                    "PUNISH_WRONG_PREDICTIONS_BEFORE_PRUNE",
                    target_column=column_id,
                    neuron_index=neuron_index,
                    candidate_segment=segment,
                    segment_count=len(
                        self.columns[column_id].neurons[neuron_index].segments
                    ),
                    container_length=len(segment.synapses),
                )
                self._prune_neuron(
                    self.columns[column_id].neurons[neuron_index],
                    target_column=column_id,
                    neuron_index=neuron_index,
                    candidate_segment=segment,
                )
        self._emit_prune_diagnostic(
            "PUNISH_WRONG_PREDICTIONS_EXIT",
            target_column=None,
            neuron_index=None,
            segment_count=sum(
                len(neuron.segments)
                for column in self.columns
                for neuron in column.neurons
            ),
            container_length=len(self.last_prediction_candidates),
        )

    def _emit_prune_diagnostic(
        self,
        phase: str,
        *,
        target_column: int | None,
        neuron_index: int | None,
        candidate_segment: Segment | None = None,
        segment_count: int = 0,
        container_length: int = 0,
    ) -> None:
        # Older strict checkpoints predate this optional field.
        callback = getattr(self, "prune_diagnostic_callback", None)
        if callback is None:
            return
        stable_ids = tuple(
            str(segment.diagnostic_id)
            for column in self.columns
            for neuron in column.neurons
            for segment in neuron.segments
            if segment.diagnostic_id is not None
        )
        segment_id_hash = hashlib.sha256(
            ",".join(sorted(stable_ids)).encode("ascii")
        ).hexdigest()
        callback(
            {
                "phase": phase,
                "encoded_column": target_column,
                "target_column": target_column,
                "neuron_index": neuron_index,
                "stable_neuron_id": (
                    f"column:{target_column}:neuron:{neuron_index}"
                    if target_column is not None and neuron_index is not None
                    else ""
                ),
                "stable_segment_id": (
                    candidate_segment.diagnostic_id
                    if candidate_segment is not None
                    else None
                ),
                "segment_count": segment_count,
                "container_length": container_length,
                "segment_id_hash": segment_id_hash,
            }
        )

    def _prune_neuron(
        self,
        neuron: Neuron,
        *,
        target_column: int | None = None,
        neuron_index: int | None = None,
        candidate_segment: Segment | None = None,
    ) -> None:
        self._emit_prune_diagnostic(
            "PRUNE_NEURON_ENTER",
            target_column=target_column,
            neuron_index=neuron_index,
            candidate_segment=candidate_segment,
            segment_count=len(neuron.segments),
            container_length=len(neuron.segments),
        )
        self._emit_prune_diagnostic(
            "PRUNE_NEURON_BEFORE_SEGMENT_SCAN",
            target_column=target_column,
            neuron_index=neuron_index,
            candidate_segment=candidate_segment,
            segment_count=len(neuron.segments),
            container_length=len(neuron.segments),
        )
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
        self._emit_prune_diagnostic(
            "PRUNE_NEURON_AFTER_SEGMENT_SCAN",
            target_column=target_column,
            neuron_index=neuron_index,
            candidate_segment=candidate_segment,
            segment_count=len(neuron.segments),
            container_length=len(kept_segments),
        )
        self._emit_prune_diagnostic(
            "PRUNE_NEURON_BEFORE_CONTAINER_MUTATION",
            target_column=target_column,
            neuron_index=neuron_index,
            candidate_segment=candidate_segment,
            segment_count=len(neuron.segments),
            container_length=len(kept_segments),
        )
        neuron.segments = kept_segments
        self._emit_prune_diagnostic(
            "PRUNE_NEURON_AFTER_CONTAINER_MUTATION",
            target_column=target_column,
            neuron_index=neuron_index,
            candidate_segment=candidate_segment,
            segment_count=len(neuron.segments),
            container_length=len(neuron.segments),
        )
        self._emit_prune_diagnostic(
            "PRUNE_NEURON_BEFORE_DIAGNOSTIC_CALLBACK",
            target_column=target_column,
            neuron_index=neuron_index,
            candidate_segment=candidate_segment,
            segment_count=len(neuron.segments),
            container_length=len(neuron.segments),
        )
        self._emit_prune_diagnostic(
            "PRUNE_NEURON_AFTER_DIAGNOSTIC_CALLBACK",
            target_column=target_column,
            neuron_index=neuron_index,
            candidate_segment=candidate_segment,
            segment_count=len(neuron.segments),
            container_length=len(neuron.segments),
        )
        self._emit_prune_diagnostic(
            "PRUNE_NEURON_EXIT",
            target_column=target_column,
            neuron_index=neuron_index,
            candidate_segment=candidate_segment,
            segment_count=len(neuron.segments),
            container_length=len(neuron.segments),
        )

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
