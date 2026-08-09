"""Read-only provenance audit for Fig.9 autonomous rollout context.

The audit keeps source identity and activation provenance separate.  A source
cell can have appeared in the actual prefix and still be activated again by a
rollout step; that case is recorded as mixed rather than silently relabelled
as historical or generated.  Formal runs write one bounded row per candidate.
Source rows are an optional smoke-test-only view and are never required by the
strict prediction path.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from experiments.diagnostics.fig9_candidate_context_oracle import (
    stable_candidate_event_id,
)


VERSION = "fig9-autonomous-context-provenance-v1"
TRAJECTORY = "autonomous_rollout"
PRE_ACTUAL = "PRE_ROLLOUT_ACTUAL_HISTORY"
PRE_PREDICTED = "PRE_ROLLOUT_PREDICTED_HISTORY"
GENERATED_PREFIX = "ROLLOUT_STEP"
UNKNOWN = "UNKNOWN_ORIGIN"
ACTUAL_ROOT = "ACTUAL_ROOT"
GENERATED_ROOT = "GENERATED_ROOT"
MIXED_ROOT = "MIXED_ROOT"
SOURCE_LEVELS = {"summary", "candidate", "source"}
MAX_SOURCE_ROWS = 32


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _mean(values: Sequence[float]) -> float | str:
    return statistics.fmean(values) if values else "NA"


def _std(values: Sequence[float]) -> float | str:
    return statistics.pstdev(values) if len(values) > 1 else (0.0 if values else "NA")


def _age_summary(values: Sequence[float]) -> dict[str, object]:
    clean = [value for value in values if math.isfinite(value) and value >= 0.0]
    if not clean:
        return {
            "mean": "NA",
            "std": "NA",
            "recent5_fraction": "NA",
            "compactness": "NA",
        }
    deviation = statistics.pstdev(clean) if len(clean) > 1 else 0.0
    return {
        "mean": statistics.fmean(clean),
        "std": deviation,
        "recent5_fraction": _ratio(sum(value < 5.0 for value in clean), len(clean)),
        "compactness": 1.0 / (1.0 + deviation),
    }


def _stable_rollout_id(
    *, stream_label: str, anchor_timestamp: str, seed: int, trajectory_ordinal: int
) -> str:
    payload = {
        "version": VERSION,
        "stream": stream_label,
        "anchor_timestamp": anchor_timestamp,
        "seed": int(seed),
        "trajectory_ordinal": int(trajectory_ordinal),
    }
    return _sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


@dataclass
class _SourceState:
    identity_actual: bool
    activation_origin: str
    origin_step: int | None
    root_class: str
    generated_depth: int
    reactivation_count: int
    propagated_from_actual: bool
    propagated_from_generated: bool
    last_activation_step: int


@dataclass
class _CsvSink:
    path: Path
    compress: bool
    handle: object | None = None
    writer: csv.DictWriter | None = None
    row_count: int = 0

    def write(self, row: Mapping[str, object]) -> None:
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.compress:
                self.handle = gzip.open(self.path, "at", encoding="utf-8", newline="")
            else:
                self.handle = self.path.open("a", encoding="utf-8", newline="")
            self.writer = csv.DictWriter(
                self.handle,
                fieldnames=list(row.keys()),
                extrasaction="ignore",
            )
            if self.path.stat().st_size == 0:
                self.writer.writeheader()
        self.writer.writerow(row)
        self.handle.flush()  # type: ignore[union-attr]
        self.row_count += 1

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()  # type: ignore[union-attr]


@dataclass
class AutonomousContextProvenanceTracker:
    """Capture causal provenance without feeding any result back to the model."""

    stream_label: str
    level: str = "candidate"
    seed: int = 0
    actual_last_activation: dict[int, int] = field(default_factory=dict)
    trajectory_ordinal: int = 0
    candidate_row_count: int = 0
    source_row_count: int = 0
    step_rows: list[dict[str, object]] = field(default_factory=list)
    origin_rows: list[dict[str, object]] = field(default_factory=list)
    survival_rows: list[dict[str, object]] = field(default_factory=list)
    _candidate_sink: _CsvSink | None = None
    _source_sink: _CsvSink | None = None
    _states: dict[int, _SourceState] = field(default_factory=dict)
    _pre_rollout_actual_ids: set[int] = field(default_factory=set)
    _rollout_id: str = ""
    _anchor_timestamp: str = ""
    _anchor_record_index: int = 0
    _step_contributor_sources: dict[int, set[int]] = field(default_factory=dict)
    _step_actual_sources: dict[int, set[int]] = field(default_factory=dict)
    _pending_candidate_rows: list[dict[str, object]] = field(default_factory=list)
    _pending_source_rows: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.level not in SOURCE_LEVELS:
            raise ValueError(f"unsupported autonomous provenance level: {self.level}")

    def enable_streaming(self, output_dir: Path) -> None:
        self._candidate_sink = _CsvSink(
            output_dir / "autonomous_candidate_provenance_trace.csv.gz",
            compress=True,
        )
        if self.level == "source":
            self._source_sink = _CsvSink(
                output_dir / "autonomous_source_provenance_trace.csv.gz",
                compress=True,
            )
        for row in self._pending_candidate_rows:
            self._candidate_sink.write(row)
        for row in self._pending_source_rows:
            if self._source_sink is not None:
                self._source_sink.write(row)
        self._pending_candidate_rows.clear()
        self._pending_source_rows.clear()

    def checkpoint_payload(self) -> dict[str, object]:
        return {
            "version": VERSION,
            "stream_label": self.stream_label,
            "level": self.level,
            "seed": self.seed,
            "actual_last_activation": {
                str(key): value for key, value in self.actual_last_activation.items()
            },
            "trajectory_ordinal": self.trajectory_ordinal,
            "candidate_row_count": self.candidate_row_count,
            "source_row_count": self.source_row_count,
        }

    @classmethod
    def from_checkpoint_payload(
        cls,
        payload: object,
        *,
        stream_label: str,
        level: str,
        seed: int,
    ) -> "AutonomousContextProvenanceTracker":
        if not isinstance(payload, Mapping) or payload.get("version") != VERSION:
            return cls(stream_label=stream_label, level=level, seed=seed)
        return cls(
            stream_label=stream_label,
            level=str(payload.get("level", level)),
            seed=_int(payload.get("seed"), seed),
            actual_last_activation={
                int(key): int(value)
                for key, value in dict(payload.get("actual_last_activation", {})).items()
            },
            trajectory_ordinal=_int(payload.get("trajectory_ordinal")),
            candidate_row_count=_int(payload.get("candidate_row_count")),
            source_row_count=_int(payload.get("source_row_count")),
        )

    def record_actual_observation(
        self, active_sources: Iterable[int], winners: Iterable[int], record_index: int
    ) -> None:
        """Store actual observation history after learning has completed."""

        for source in set(active_sources) | set(winners):
            self.actual_last_activation[int(source)] = int(record_index)

    def begin_rollout(
        self,
        *,
        anchor_record_index: int,
        anchor_timestamp: str,
        pre_rollout_active: Iterable[int],
        pre_rollout_predicted: Iterable[int],
        pre_rollout_burst: Iterable[int],
    ) -> None:
        if self._rollout_id and self._step_contributor_sources:
            self.survival_rows.extend(self._current_survival_rows())
        self.trajectory_ordinal += 1
        self._anchor_record_index = int(anchor_record_index)
        self._anchor_timestamp = str(anchor_timestamp)
        self._rollout_id = _stable_rollout_id(
            stream_label=self.stream_label,
            anchor_timestamp=self._anchor_timestamp,
            seed=self.seed,
            trajectory_ordinal=self.trajectory_ordinal,
        )
        active = {int(source) for source in pre_rollout_active}
        predicted = {int(source) for source in pre_rollout_predicted}
        burst = {int(source) for source in pre_rollout_burst}
        self._pre_rollout_actual_ids = set(active)
        self._states = {}
        for source in sorted(active):
            origin = PRE_PREDICTED if source in predicted else PRE_ACTUAL
            self._states[source] = _SourceState(
                identity_actual=True,
                activation_origin=origin,
                origin_step=None,
                root_class=ACTUAL_ROOT,
                generated_depth=0,
                reactivation_count=0,
                propagated_from_actual=False,
                propagated_from_generated=False,
                last_activation_step=0,
            )
        # A burst label is retained as a provenance fact but does not alter
        # the identity class or prediction path.
        for source in burst:
            if source in self._states and source not in predicted:
                self._states[source].activation_origin = PRE_ACTUAL
        self._step_contributor_sources = {}
        self._step_actual_sources = {}

    @staticmethod
    def _positive_contributors(candidate: object) -> tuple[object, ...]:
        return tuple(
            item
            for item in getattr(candidate, "crossing_synapse_contributions", ())
            if (_finite(getattr(item, "psp_contribution", None)) or 0.0) > 0.0
        )

    def _state_counts(
        self, source_ids: Sequence[int], horizon_step: int
    ) -> dict[str, object]:
        states = [self._states.get(source) for source in source_ids]
        states = [state for state in states if state is not None]
        total = len(source_ids)
        actual_identity = [state for state in states if state.identity_actual]
        pre_actual = [state for state in states if state.activation_origin == PRE_ACTUAL]
        pre_predicted = [state for state in states if state.activation_origin == PRE_PREDICTED]
        generated = [state for state in states if state.activation_origin.startswith(GENERATED_PREFIX)]
        actual_root = [state for state in states if state.root_class == ACTUAL_ROOT]
        generated_root = [state for state in states if state.root_class == GENERATED_ROOT]
        mixed_root = [state for state in states if state.root_class == MIXED_ROOT]
        unknown = total - len(states)
        actual_ages = [
            self._anchor_record_index - self.actual_last_activation[source]
            for source in source_ids
            if source in self.actual_last_activation
            and self.actual_last_activation[source] <= self._anchor_record_index
        ]
        any_ages = [
            horizon_step - state.last_activation_step
            for state in states
        ]
        root_ages = [
            self._anchor_record_index - self.actual_last_activation[source]
            for source, state in zip(source_ids, [self._states.get(s) for s in source_ids])
            if state is not None
            and state.identity_actual
            and source in self.actual_last_activation
        ]
        actual_summary = _age_summary(actual_ages)
        any_summary = _age_summary(any_ages)
        root_summary = _age_summary(root_ages)
        depths = [state.generated_depth for state in generated]
        reactivated = [state for state in states if state.reactivation_count > 0]
        propagated_actual = [state for state in states if state.propagated_from_actual]
        propagated_generated = [state for state in states if state.propagated_from_generated]
        origin_step_counts = {
            f"origin_step{step}_count": sum(
                state.origin_step == step for state in states
            )
            for step in range(1, 6)
        }
        return {
            "unique_source_count": total,
            "source_identity_actual_history_count": len(actual_identity),
            "source_identity_actual_history_fraction": _ratio(len(actual_identity), total),
            "source_identity_predicted_history_count": len(pre_predicted),
            "source_identity_unknown_count": unknown,
            "activation_pre_rollout_actual_count": len(pre_actual),
            "activation_pre_rollout_actual_fraction": _ratio(len(pre_actual), total),
            "activation_pre_rollout_predicted_count": len(pre_predicted),
            "activation_pre_rollout_predicted_fraction": _ratio(len(pre_predicted), total),
            "activation_rollout_generated_count": len(generated),
            "activation_rollout_generated_fraction": _ratio(len(generated), total),
            "activation_unknown_count": unknown,
            "activation_unknown_fraction": _ratio(unknown, total),
            "activation_propagated_from_actual_count": len(propagated_actual),
            "activation_propagated_from_actual_fraction": _ratio(len(propagated_actual), total),
            "activation_propagated_from_generated_count": len(propagated_generated),
            "activation_propagated_from_generated_fraction": _ratio(len(propagated_generated), total),
            "actual_root_fraction": _ratio(len(actual_root), total),
            "generated_root_fraction": _ratio(len(generated_root), total),
            "mixed_root_fraction": _ratio(len(mixed_root), total),
            "unknown_root_fraction": _ratio(unknown, total),
            "same_rollout_reactivation_count": len(reactivated),
            "same_rollout_reactivation_fraction": _ratio(len(reactivated), total),
            "self_generated_depth_mean": _mean(depths),
            "self_generated_depth_max": max(depths, default=0),
            "recent5_any_fraction": any_summary["recent5_fraction"],
            "recent5_actual_fraction": actual_summary["recent5_fraction"],
            "recent5_root_fraction": root_summary["recent5_fraction"],
            "compactness_any_activation": any_summary["compactness"],
            "compactness_actual_history_age": actual_summary["compactness"],
            "compactness_actual_root_age": root_summary["compactness"],
            "mean_any_activation_age": any_summary["mean"],
            "mean_actual_history_age": actual_summary["mean"],
            "mean_actual_root_age": root_summary["mean"],
            **origin_step_counts,
            **{
                f"origin_step{step}_fraction": _ratio(
                    origin_step_counts[f"origin_step{step}_count"], total
                )
                for step in range(1, 6)
            },
        }

    def _emit_candidate(self, row: dict[str, object]) -> None:
        if self._candidate_sink is None:
            self._pending_candidate_rows.append(row)
            self.candidate_row_count += 1
            return
        self._candidate_sink.write(row)
        self.candidate_row_count += 1

    def _emit_source(self, row: dict[str, object]) -> None:
        if self._source_sink is None:
            if self.level == "source":
                self._pending_source_rows.append(row)
            return
        self._source_sink.write(row)
        self.source_row_count += 1

    def _source_output_state(
        self,
        *,
        source_id: int,
        candidate_sources: Sequence[int],
        horizon_step: int,
    ) -> _SourceState:
        parent_states = [self._states.get(source) for source in candidate_sources]
        parent_states = [state for state in parent_states if state is not None]
        identity_actual = source_id in self._pre_rollout_actual_ids
        had_generated_parent = any(
            state.activation_origin.startswith(GENERATED_PREFIX)
            for state in parent_states
        )
        had_actual_parent = any(
            state.activation_origin in {PRE_ACTUAL, PRE_PREDICTED}
            for state in parent_states
        )
        previous = self._states.get(source_id)
        depth = (
            max((state.generated_depth for state in parent_states), default=0) + 1
            if parent_states
            else 1
        )
        return _SourceState(
            identity_actual=identity_actual,
            activation_origin=f"{GENERATED_PREFIX}{horizon_step}_GENERATED",
            origin_step=horizon_step,
            root_class=MIXED_ROOT if identity_actual else GENERATED_ROOT,
            generated_depth=depth,
            reactivation_count=(previous.reactivation_count + 1 if previous else 0),
            propagated_from_actual=had_actual_parent or any(
                state.propagated_from_actual for state in parent_states
            ),
            propagated_from_generated=had_generated_parent or any(
                state.propagated_from_generated for state in parent_states
            ),
            last_activation_step=horizon_step,
        )

    def record_step(
        self,
        *,
        model: object,
        raw: object | None,
        propagated: object | None,
        next_active_cells: Mapping[int, float],
        competition_result: object | None,
        actual_record_index: int,
        horizon_step: int,
        field_for_column,
        target_columns: Iterable[int],
        target_column_by_field: Mapping[str, int],
        input_index: int,
        input_timestamp: str,
        target_timestamp: str,
        registry: object | None = None,
    ) -> None:
        target_set = {int(value) for value in target_columns}
        candidates = [] if raw is None else [
            (int(column), candidate)
            for column, values in getattr(model, "last_prediction_candidates", {}).items()
            for candidate in values
        ]
        decisions = {
            id(decision.candidate.candidate): decision
            for decision in getattr(competition_result, "decisions", ())
        }
        emitted_ids = {
            id(item.candidate)
            for item in getattr(competition_result, "emitted_candidates", ())
        }
        raw_event_count = len(getattr(raw, "events", ()) or ())
        raw_columns = len({event.column for event in getattr(raw, "events", ()) or ()})
        emitted_columns = len({event.column for event in getattr(propagated, "events", ()) or ()})
        local_rows: list[dict[str, object]] = []
        next_active_ids = {int(cell_id) for cell_id in next_active_cells}
        candidate_by_cell: dict[int, object] = {}
        neurons_per_column = len(model.columns[0].neurons) if getattr(model, "columns", ()) else 0
        for column, candidate in candidates:
            candidate_by_cell.setdefault(
                int(column) * neurons_per_column + int(candidate.neuron_index),
                candidate,
            )
        source_union: set[int] = set()
        actual_union: set[int] = set()
        generated_union: set[int] = set()
        origin_counts: dict[str, int] = defaultdict(int)
        target_available = False
        target_selected = False
        for ordinal, (column, candidate) in enumerate(candidates):
            positive = self._positive_contributors(candidate)
            source_ids = tuple(sorted({int(item.source_cell_id) for item in positive}))
            counts = self._state_counts(source_ids, horizon_step)
            source_union.update(source_ids)
            actual_union.update(
                source for source in source_ids
                if self._states.get(source) is not None
                and self._states[source].identity_actual
            )
            generated_union.update(
                source for source in source_ids
                if self._states.get(source) is not None
                and self._states[source].activation_origin.startswith(GENERATED_PREFIX)
            )
            for source in source_ids:
                state = self._states.get(source)
                origin_counts[state.activation_origin if state else UNKNOWN] += 1
            field = field_for_column(column)
            origin = (
                registry.provenance_for(candidate.segment)
                if registry is not None
                else None
            )
            stable_segment_id = (
                getattr(origin, "segment_provenance_id", "")
                if origin is not None
                else ""
            )
            decision = decisions.get(id(candidate))
            emitted = id(candidate) in emitted_ids
            if competition_result is None:
                emitted = any(
                    event.column == column
                    and abs(float(event.time) - float(candidate.time)) <= getattr(model.params, "timing_tolerance", 0.0)
                    for event in getattr(propagated, "events", ()) or ()
                )
            is_target = column in target_set
            candidate_cell = (
                int(column) * len(model.columns[0].neurons)
                + int(candidate.neuron_index)
            )
            selected_intracolumn = candidate_cell in next_active_ids
            target_available = target_available or is_target
            target_selected = target_selected or (is_target and emitted)
            row = {
                "row_unit": "CANDIDATE_SEGMENT_PROVENANCE",
                "diagnostic_version": VERSION,
                "read_only": True,
                "stream_label": self.stream_label,
                "rollout_id": self._rollout_id,
                "anchor_record_index": self._anchor_record_index,
                "anchor_timestamp": self._anchor_timestamp,
                "input_index": input_index,
                "input_timestamp": input_timestamp,
                "target_timestamp": target_timestamp,
                "horizon_step": horizon_step,
                "field": field,
                "candidate_event_id": stable_candidate_event_id(
                    trajectory_kind=TRAJECTORY,
                    actual_record_index=actual_record_index,
                    horizon_step=horizon_step,
                    field=field,
                    observed_encoded_column="",
                    candidate_target_column=column,
                    candidate_target_neuron=int(candidate.neuron_index),
                    stable_segment_provenance_id=stable_segment_id,
                    candidate_generation_ordinal=ordinal,
                    observed_event_time=candidate.time,
                ),
                "candidate_target_column": column,
                "candidate_target_neuron": int(candidate.neuron_index),
                "stable_segment_provenance_id": stable_segment_id,
                "oracle_target_column": target_column_by_field.get(field, "NA"),
                "is_target_column_candidate": is_target,
                "candidate_score": candidate.score,
                "response_peak": candidate.peak_dendritic_potential if candidate.peak_dendritic_potential is not None else "NA",
                "contributor_count": len(source_ids),
                "unique_source_count": counts["unique_source_count"],
                "source_id_fingerprint": _sha256_text(",".join(map(str, source_ids))),
                "selected_intracolumn": selected_intracolumn,
                "selected_after_intercolumn": emitted,
                "suppressed_intracolumn": False,
                "suppressed_intercolumn": bool(decision is not None and not decision.emitted),
                "emitted": emitted,
                "predicted_time": candidate.time,
                "raw_column_count": raw_columns,
                "emitted_column_count": emitted_columns,
                **counts,
                "provenance_missing_reason": (
                    "NO_VALID_CONTRIBUTOR" if not source_ids else ""
                ),
                "source_identity_has_actual_history": bool(counts["source_identity_actual_history_count"]),
                "current_activation_origin_summary": ";".join(
                    sorted(set(
                        self._states[source].activation_origin
                        for source in source_ids
                        if source in self._states
                    ))
                ) or UNKNOWN,
                "source_context_domain": "actual_prefix_plus_autonomous_suffix",
            }
            local_rows.append(row)
            self._emit_candidate(row)
            if self.level == "source" and len(source_ids) <= MAX_SOURCE_ROWS:
                for source in positive:
                    state = self._states.get(int(source.source_cell_id))
                    self._emit_source({
                        "row_unit": "SOURCE_ACTIVATION_PROVENANCE",
                        "diagnostic_version": VERSION,
                        "rollout_id": self._rollout_id,
                        "candidate_event_id": row["candidate_event_id"],
                        "horizon_step": horizon_step,
                        "field": field,
                        "source_cell_id": int(source.source_cell_id),
                        "source_identity_has_actual_history": bool(state and state.identity_actual),
                        "current_activation_origin": state.activation_origin if state else UNKNOWN,
                        "root_class": state.root_class if state else UNKNOWN,
                        "origin_step": state.origin_step if state else "NA",
                        "generated_depth": state.generated_depth if state else "NA",
                        "propagated_from_actual": bool(state and state.propagated_from_actual),
                        "propagated_from_generated": bool(state and state.propagated_from_generated),
                        "contribution": source.psp_contribution,
                    })
        self._step_contributor_sources[horizon_step] = set(source_union)
        self._step_actual_sources[horizon_step] = set(actual_union)
        step_row = {
            "row_unit": "ROLLOUT_STEP_AGGREGATE",
            "diagnostic_version": VERSION,
            "stream_label": self.stream_label,
            "rollout_id": self._rollout_id,
            "anchor_record_index": self._anchor_record_index,
            "anchor_timestamp": self._anchor_timestamp,
            "horizon_step": horizon_step,
            "raw_column_count": raw_columns,
            "emitted_column_count": emitted_columns,
            "candidate_count": len(local_rows),
            "total_contributors": sum(_int(row.get("contributor_count")) for row in local_rows),
            "actual_history_contributors": sum(_int(row.get("source_identity_actual_history_count")) for row in local_rows),
            "generated_contributors": sum(_int(row.get("activation_rollout_generated_count")) for row in local_rows),
            "actual_history_fraction": _ratio(
                sum(_int(row.get("source_identity_actual_history_count")) for row in local_rows),
                sum(_int(row.get("contributor_count")) for row in local_rows),
            ),
            "generated_fraction": _ratio(
                sum(_int(row.get("activation_rollout_generated_count")) for row in local_rows),
                sum(_int(row.get("contributor_count")) for row in local_rows),
            ),
            "actual_root_fraction": _mean([float(row["actual_root_fraction"]) for row in local_rows]) if local_rows else "NA",
            "generated_root_fraction": _mean([float(row["generated_root_fraction"]) for row in local_rows]) if local_rows else "NA",
            "mean_self_generated_depth": _mean([
                float(row["self_generated_depth_mean"])
                for row in local_rows
                if _finite(row.get("self_generated_depth_mean")) is not None
            ]),
            "max_self_generated_depth": max((_int(row.get("self_generated_depth_max")) for row in local_rows), default=0),
            "unique_actual_sources": len(actual_union),
            "unique_generated_sources": len(generated_union),
            "candidate_score_mean": _mean([float(row["candidate_score"]) for row in local_rows]),
            "candidate_score_std": _std([float(row["candidate_score"]) for row in local_rows]),
            "target_candidate_available": target_available,
            "target_candidate_selected": target_selected,
            "live_segments": len(getattr(model, "columns", ())) and sum(
                len(getattr(neuron, "segments", ()))
                for column in model.columns
                for neuron in column.neurons
            ),
            "live_synapses": sum(
                len(getattr(segment, "synapses", ()))
                for column in getattr(model, "columns", ())
                for neuron in column.neurons
                for segment in getattr(neuron, "segments", ())
            ),
            "origin_pre_rollout_actual_count": origin_counts.get(PRE_ACTUAL, 0),
            "origin_pre_rollout_predicted_count": origin_counts.get(PRE_PREDICTED, 0),
            **{
                f"origin_step{step}_count": sum(
                    count for origin, count in origin_counts.items()
                    if origin == f"{GENERATED_PREFIX}{step}_GENERATED"
                )
                for step in range(1, 6)
            },
            "origin_unknown_count": origin_counts.get(UNKNOWN, 0),
        }
        self.step_rows.append(step_row)
        self.origin_rows.append(step_row.copy())
        next_states: dict[int, _SourceState] = {}
        for cell_id in next_active_cells:
            matching = candidate_by_cell.get(int(cell_id))
            if matching is None:
                next_states[int(cell_id)] = _SourceState(
                    identity_actual=int(cell_id) in self._pre_rollout_actual_ids,
                    activation_origin=f"{GENERATED_PREFIX}{horizon_step}_GENERATED",
                    origin_step=horizon_step,
                    root_class=(MIXED_ROOT if int(cell_id) in self._pre_rollout_actual_ids else GENERATED_ROOT),
                    generated_depth=1,
                    reactivation_count=(self._states[int(cell_id)].reactivation_count + 1 if int(cell_id) in self._states else 0),
                    propagated_from_actual=False,
                    propagated_from_generated=True,
                    last_activation_step=horizon_step,
                )
            else:
                source_ids = tuple(int(item.source_cell_id) for item in self._positive_contributors(matching))
                next_states[int(cell_id)] = self._source_output_state(
                    source_id=int(cell_id),
                    candidate_sources=source_ids,
                    horizon_step=horizon_step,
                )
        self._states = next_states

    def close(self) -> None:
        if self._candidate_sink is not None:
            self._candidate_sink.close()
        if self._source_sink is not None:
            self._source_sink.close()

    def write_run_outputs(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(output_dir / "autonomous_rollout_provenance_by_step.csv", self.step_rows)
        _write_csv(output_dir / "rollout_origin_step_matrix.csv", self.origin_rows)
        _write_csv(output_dir / "actual_source_survival_matrix.csv", self._survival_rows())
        _write_csv(output_dir / "actual_history_retention.csv", self._retention_rows())
        _write_csv(output_dir / "self_generated_fraction.csv", self._generated_fraction_rows())
        _write_csv(output_dir / "self_generated_depth.csv", self._depth_rows())

    def _survival_rows(self) -> list[dict[str, object]]:
        return [*self.survival_rows, *self._current_survival_rows()]

    def _current_survival_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for from_step, source_set in sorted(self._step_actual_sources.items()):
            for to_step in range(from_step, 6):
                later = self._step_contributor_sources.get(to_step, set())
                rows.append({
                    "row_unit": "ACTUAL_SOURCE_SURVIVAL",
                    "stream_label": self.stream_label,
                    "rollout_id": self._rollout_id,
                    "from_step": from_step,
                    "to_step": to_step,
                    "source_count_from": len(source_set),
                    "source_count_survived": len(source_set & later),
                    "survival_fraction": _ratio(len(source_set & later), len(source_set)),
                })
        return rows

    def _retention_rows(self) -> list[dict[str, object]]:
        return [
            {
                "row_unit": "ACTUAL_HISTORY_RETENTION",
                "stream_label": self.stream_label,
                "rollout_id": row.get("rollout_id"),
                "horizon_step": row.get("horizon_step"),
                "retention_fraction": row.get("actual_history_fraction", "NA"),
                "absolute_actual_history_count": row.get("actual_history_contributors", 0),
                "absolute_total_contributor_count": row.get("total_contributors", 0),
            }
            for row in self.step_rows
        ]

    def _generated_fraction_rows(self) -> list[dict[str, object]]:
        return [
            {
                "row_unit": "SELF_GENERATED_FRACTION",
                "stream_label": self.stream_label,
                "rollout_id": row.get("rollout_id"),
                "horizon_step": row.get("horizon_step"),
                "generated_fraction": row.get("generated_fraction", "NA"),
                "absolute_generated_count": row.get("generated_contributors", 0),
                "absolute_total_contributor_count": row.get("total_contributors", 0),
            }
            for row in self.step_rows
        ]

    def _depth_rows(self) -> list[dict[str, object]]:
        return [
            {
                "row_unit": "SELF_GENERATED_DEPTH",
                "stream_label": self.stream_label,
                "rollout_id": row.get("rollout_id"),
                "horizon_step": row.get("horizon_step"),
                "mean_depth": row.get("mean_self_generated_depth", "NA"),
                "max_depth": row.get("max_self_generated_depth", 0),
            }
            for row in self.step_rows
        ]


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["row_unit"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_protocol(path: Path, level: str) -> None:
    path.write_text(
        json.dumps(protocol(level), indent=2, sort_keys=True), encoding="utf-8"
    )


def protocol(level: str = "candidate") -> dict[str, object]:
    return {
        "version": VERSION,
        "diagnostic_only": True,
        "read_only": True,
        "default_enabled": False,
        "level": level,
        "candidate_row_unit": "CANDIDATE_SEGMENT_PROVENANCE",
        "source_row_unit": "SOURCE_ACTIVATION_PROVENANCE",
        "complexity": "O(number_of_contributors) per candidate",
        "source_rows_formal_250": False,
        "max_source_rows_per_candidate": MAX_SOURCE_ROWS,
        "source_identity_separate_from_activation_origin": True,
        "uses_compensation": False,
        "uses_future_covariates": False,
        "uses_ground_truth_for_selection": False,
        "learns_during_rollout": False,
        "strict_default_unchanged": True,
    }
