"""Read-only segment-context composition tracing for Fig.9.

This module observes the same pre-observation state already copied by the
match-overlap diagnostic.  It never calls prediction, matching, observation,
learning, or random-number generators.  The output is intentionally
diagnostic: source incidence and IDF-like values describe the live network at
the moment of a match; they are not used to rank or select a candidate.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from experiments.diagnostics.fig9_candidate_score_trace import field_for_column
from experiments.diagnostics.fig9_match_overlap import (
    MatchOverlapCapture,
    _cell_column,
    _cell_neuron,
)
from experiments.diagnostics.fig9_oracle_candidate import FieldColumnRanges
from seqmem.model import ObservationTrace, SequentialMemory, Segment


LEVELS = {"summary", "match", "source"}
VERSION = "fig9-segment-context-composition-v1"
MARKERS = {
    "diagnostic_only": True,
    "offline_analysis_only": True,
    "segment_context_composition_diagnostic": True,
    "selection_behavior_changed": False,
    "prediction_behavior_changed": False,
    "learning_behavior_changed": False,
    "ground_truth_used_for_selection": False,
    "trajectory_kind": "actual_observation",
}


def _number(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _truth(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _entropy(values: Iterable[object]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    if not total or len(counts) <= 1:
        return 0.0
    return -sum(
        (count / total) * math.log(count / total)
        for count in counts.values()
        if count
    )


def _quantile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _incidence_bin(value: int, thresholds: tuple[float, ...]) -> str:
    q25, q50, q75, q90 = thresholds
    if value <= q25:
        return "RARE"
    if value <= q50:
        return "MEDIUM"
    if value <= q75:
        return "COMMON"
    if value <= q90:
        return "VERY_COMMON"
    return "VERY_COMMON"


def _segment_id(registry: object, segment: Segment) -> str:
    provenance = registry.provenance_for(segment)
    if provenance is not None:
        return provenance.segment_provenance_id
    if segment.diagnostic_id is not None:
        return f"diagnostic:{segment.diagnostic_id}"
    return f"object:{id(segment)}"


def _origin_labels(
    source: int,
    *,
    winners: set[int],
    active: set[int],
    predicted: set[int],
    burst: set[int],
) -> tuple[str, ...]:
    labels: list[str] = []
    if source in winners:
        labels.append("ACTUAL_WINNER")
    if source in burst:
        labels.append("ACTUAL_BURST")
    if source in predicted:
        labels.append("PREDICTED")
        labels.append("PROPAGATED")
    if source in active and not labels:
        labels.append("PROPAGATED")
    return tuple(labels or ("UNKNOWN",))


@dataclass
class _SourceHistory:
    activation_count: int = 0
    recent_activation_count: int = 0
    actual_activation_count: int = 0
    autonomous_activation_count: int = 0
    last_actual_record: int | None = None
    last_any_record: int | None = None


@dataclass
class _Snapshot:
    record_index: int
    active_ids: set[int]
    winner_ids: set[int]
    predicted_ids: set[int]
    burst_ids: set[int]
    source_stats: dict[int, dict[str, object]]
    segment_sources: dict[int, tuple[int, ...]]
    segment_meta: dict[int, dict[str, object]]
    segment_id_by_object: dict[int, str]
    l_match: int


@dataclass
class SegmentContextCompositionTracker:
    """Accumulate auditable source/segment rows without touching the model."""

    registry: object
    level: str = "match"
    source_history: dict[int, _SourceHistory] = field(default_factory=dict)
    match_rows: list[dict[str, object]] = field(default_factory=list)
    source_rows: list[dict[str, object]] = field(default_factory=list)
    quality_rows: list[dict[str, object]] = field(default_factory=list)
    event_rows: list[dict[str, object]] = field(default_factory=list)
    temporal_rows: list[dict[str, object]] = field(default_factory=list)
    l2_only_rows: list[dict[str, object]] = field(default_factory=list)
    _pending: _Snapshot | None = None
    _source_incidence_history: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))
    _stream_output_dir: Path | None = None
    _stream_compress: bool = False
    _stream_batch_size: int = 5000
    _written_counts: Counter[str] = field(default_factory=Counter)
    _written_paths: dict[str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise ValueError("segment context composition level must be summary, match, or source")

    def enable_streaming(
        self,
        output_dir: Path,
        *,
        compress: bool,
        batch_size: int = 5000,
    ) -> None:
        """Write bounded diagnostic batches so source-level traces stay small."""

        self._stream_output_dir = output_dir
        self._stream_compress = compress
        self._stream_batch_size = batch_size

    def _flush_dataset(self, stem: str, rows: list[dict[str, object]]) -> None:
        if self._stream_output_dir is None or not rows:
            return
        if stem == "l2_only_context_composition":
            passenger_rows = [
                row for row in rows if row.get("field") == "passenger"
            ]
            if passenger_rows:
                passenger_path = self._stream_output_dir / (
                    "passenger_l2_only_context_composition.csv"
                )
                if self._stream_compress:
                    passenger_path = passenger_path.with_suffix(
                        passenger_path.suffix + ".gz"
                    )
                _append_rows(
                    passenger_path,
                    passenger_rows,
                    compress=self._stream_compress,
                )
                self._written_counts["passenger_l2_only_context_composition"] += len(
                    passenger_rows
                )
                self._written_paths[
                    "passenger_l2_only_context_composition"
                ] = passenger_path
        path = self._stream_output_dir / f"{stem}.csv"
        if self._stream_compress:
            path = path.with_suffix(path.suffix + ".gz")
        _append_rows(path, rows, compress=self._stream_compress)
        self._written_counts[stem] += len(rows)
        self._written_paths[stem] = path
        rows.clear()

    def _flush_if_needed(self) -> None:
        if self._stream_output_dir is None:
            return
        for stem, rows in (
            ("segment_context_match_trace", self.match_rows),
            ("segment_context_source_trace", self.source_rows),
            ("segment_context_quality", self.quality_rows),
            ("segment_context_event_summary", self.event_rows),
            ("l2_only_context_composition", self.l2_only_rows),
        ):
            if len(rows) >= self._stream_batch_size:
                self._flush_dataset(stem, rows)

    def capture_pre_observation(
        self,
        *,
        model: SequentialMemory,
        ranges: FieldColumnRanges,
        actual_record_index: int,
    ) -> None:
        """Copy live segments and current context before the real observation."""

        winners = set(model.previous_winners)
        active = set(
            model.previous_active_cells or model.previous_winners
            if model.params.burst_context
            else model.previous_winners
        )
        predicted = set(getattr(self.registry, "current_predicted_sources", ()))
        burst = set(getattr(self.registry, "current_burst_sources", ()))

        segment_sources: dict[int, tuple[int, ...]] = {}
        segment_meta: dict[int, dict[str, object]] = {}
        segment_id_by_object: dict[int, str] = {}
        source_segments: dict[int, set[str]] = defaultdict(set)
        source_target_columns: dict[int, set[int]] = defaultdict(set)
        source_target_neurons: dict[int, set[str]] = defaultdict(set)
        source_fields: dict[int, set[str]] = defaultdict(set)
        source_creation_records: dict[int, list[int]] = defaultdict(list)
        live_segment_count = 0

        for column_id, column in enumerate(model.columns):
            target_field = field_for_column(column_id, ranges)
            for neuron_index, neuron in enumerate(column.neurons):
                for segment in neuron.segments:
                    if not segment.active:
                        continue
                    live_segment_count += 1
                    object_id = id(segment)
                    sources = tuple(sorted(int(source) for source in segment.synapses))
                    segment_sources[object_id] = sources
                    segment_id = _segment_id(self.registry, segment)
                    segment_id_by_object[object_id] = segment_id
                    provenance = self.registry.provenance_for(segment)
                    creation_index = (
                        provenance.creation_transition_index
                        if provenance is not None
                        else segment.creation_transition_index
                    )
                    segment_meta[object_id] = {
                        "segment_id": segment_id,
                        "target_column": column_id,
                        "target_neuron": neuron_index,
                        "target_field": target_field,
                        "creation_record": creation_index if creation_index is not None else "",
                        "reinforcement_count": (
                            segment.scenario1_reinforcements
                            + segment.scenario2_reinforcements
                        ),
                        "synapse_count": len(sources),
                    }
                    for source in sources:
                        source_segments[source].add(segment_id)
                        source_target_columns[source].add(column_id)
                        source_target_neurons[source].add(f"{column_id}:{neuron_index}")
                        source_fields[source].add(target_field)
                        if creation_index is not None:
                            source_creation_records[source].append(int(creation_index))

        incidences = [len(values) for values in source_segments.values()]
        thresholds = tuple(
            _quantile([float(value) for value in incidences], fraction)
            for fraction in (0.25, 0.50, 0.75, 0.90)
        )
        source_stats: dict[int, dict[str, object]] = {}
        for source in sorted(source_segments):
            history = self.source_history.setdefault(source, _SourceHistory())
            incidence = len(source_segments[source])
            labels = _origin_labels(
                source,
                winners=winners,
                active=active,
                predicted=predicted,
                burst=burst,
            )
            source_stats[source] = {
                "source_id": source,
                "source_column": _cell_column(model, source),
                "source_neuron": _cell_neuron(model, source),
                "source_field": field_for_column(_cell_column(model, source), ranges),
                "source_creation_record": min(source_creation_records.get(source, ()), default=""),
                "source_last_actual_record": history.last_actual_record if history.last_actual_record is not None else "",
                "source_age": (
                    actual_record_index - min(source_creation_records.get(source, ()))
                    if source_creation_records.get(source)
                    else ""
                ),
                "source_origin_type": ";".join(labels),
                "source_segment_incidence": incidence,
                "source_live_segment_incidence": incidence,
                "source_field_segment_incidence": len(source_fields[source]),
                "source_target_column_incidence": len(source_target_columns[source]),
                "source_target_neuron_incidence": len(source_target_neurons[source]),
                "source_activation_frequency": history.activation_count,
                "source_recent_activation_frequency": history.recent_activation_count,
                "source_actual_activation_frequency": history.actual_activation_count,
                "source_autonomous_activation_frequency": history.autonomous_activation_count,
                "source_idf": math.log((live_segment_count + 1) / (incidence + 1)),
                "source_normalized_incidence": _ratio(incidence, live_segment_count),
                "source_incidence_bin": _incidence_bin(incidence, thresholds),
                "source_incidence_q25": thresholds[0],
                "source_incidence_q50": thresholds[1],
                "source_incidence_q75": thresholds[2],
                "source_incidence_q90": thresholds[3],
                "source_currently_active": source in active,
                "source_currently_winner": source in winners,
                "source_currently_predicted": source in predicted,
                "source_currently_burst": source in burst,
            }
            self._source_incidence_history[source].append(incidence)

        self._pending = _Snapshot(
            record_index=actual_record_index,
            active_ids=active,
            winner_ids=winners,
            predicted_ids=predicted,
            burst_ids=burst,
            source_stats=source_stats,
            segment_sources=segment_sources,
            segment_meta=segment_meta,
            segment_id_by_object=segment_id_by_object,
            l_match=model.params.l_match,
        )

    def consume_transition(
        self,
        *,
        capture: MatchOverlapCapture,
        completed: MatchOverlapCapture,
        actual_record_index: int,
        stream_label: str,
    ) -> None:
        """Join pre-match composition to the already-finished observation."""

        snapshot = self._pending
        if snapshot is None:
            raise RuntimeError("composition transition was consumed without a snapshot")
        column_by_key = {
            (int(row["actual_record_index"]), int(row["encoded_column"])): row
            for row in completed.column_rows
        }
        completed_by_object = {
            int(raw["_segment_object_id"]): row
            for raw, row in zip(capture.segment_rows, completed.segment_rows)
        }
        sources_by_object: dict[int, list[Mapping[str, object]]] = defaultdict(list)
        for row in capture.source_rows:
            sources_by_object[int(row["_segment_object_id"])].append(row)

        for raw, finished in zip(capture.segment_rows, completed.segment_rows):
            object_id = int(raw["_segment_object_id"])
            source_rows = sources_by_object.get(object_id, [])
            source_ids = tuple(
                int(row["source_cell_stable_id"]) for row in source_rows
            )
            overlap_sources = tuple(
                int(row["source_cell_stable_id"])
                for row in source_rows
                if _truth(row.get("contributes_to_actual_overlap"))
            )
            missing_sources = tuple(
                source
                for source in source_ids
                if source not in snapshot.active_ids
            )
            quality = self._quality(
                source_ids=source_ids,
                overlap_sources=overlap_sources,
                missing_sources=missing_sources,
                target_field=str(finished.get("field", "")),
                snapshot=snapshot,
            )
            column_key = (
                int(finished["actual_record_index"]),
                int(finished["encoded_column"]),
            )
            column = column_by_key.get(column_key, {})
            match_row = {
                **MARKERS,
                "stream_label": stream_label,
                "actual_record_index": actual_record_index,
                "field": finished.get("field", ""),
                "encoded_column": finished.get("encoded_column", ""),
                "L_match": snapshot.l_match,
                "observe_scenario": finished.get("observe_scenario", ""),
                "scenario_assignment_reason": finished.get("scenario_assignment_reason", ""),
                "segment_provenance_id": finished.get("segment_provenance_id", ""),
                "target_column": finished.get("encoded_column", ""),
                "target_neuron": finished.get("segment_target_neuron", ""),
                "segment_creation_index": finished.get("segment_creation_transition_index", ""),
                "segment_reinforcement_count": snapshot.segment_meta.get(object_id, {}).get("reinforcement_count", ""),
                "context_size": len(snapshot.active_ids),
                "overlap_count": finished.get("actual_overlap", 0),
                "passes_L2": _truth(finished.get("passes_L2")),
                "passes_L3": _truth(finished.get("passes_L3")),
                "passes_L4": _truth(finished.get("passes_L4")),
                "l2_only": _truth(finished.get("passes_L2")) and not _truth(finished.get("passes_L3")),
                "l3_only": _truth(finished.get("passes_L3")) and not _truth(finished.get("passes_L4")),
                "selected": _truth(finished.get("selected_for_scenario")),
                "reinforced": _truth(finished.get("reinforced")),
                "selected_as_best_matching": _truth(finished.get("selected_as_best_matching")),
                "matching_segment_count": column.get("segments_meeting_L_match", ""),
                "matching_neuron_count": (
                    1 if _truth(column.get("multiple_neurons_meet_L_match")) is False and _number(column.get("segments_meeting_L_match")) else ""
                ),
                "ambiguous": _truth(column.get("multiple_segments_meet_L_match")) or _truth(column.get("multiple_neurons_meet_L_match")),
                "overlap_source_ids": json.dumps(list(overlap_sources)),
                "overlap_source_count": len(overlap_sources),
                "missing_context_source_count": len(missing_sources),
                **self._quality_projection(quality, prefix="overlap", sources=overlap_sources),
                **self._quality_projection(quality, prefix="missing", sources=missing_sources),
            }
            self.match_rows.append(match_row)
            self.quality_rows.append(
                {
                    **MARKERS,
                    "stream_label": stream_label,
                    "actual_record_index": actual_record_index,
                    "field": finished.get("field", ""),
                    "segment_provenance_id": finished.get("segment_provenance_id", ""),
                    "target_column": finished.get("encoded_column", ""),
                    "target_neuron": finished.get("segment_target_neuron", ""),
                    "observe_scenario": finished.get("observe_scenario", ""),
                    "selected": _truth(finished.get("selected_for_scenario")),
                    "reinforced": _truth(finished.get("reinforced")),
                    "matching": _truth(finished.get("meets_L_match")),
                    "ambiguous": _truth(column.get("multiple_segments_meet_L_match"))
                    or _truth(column.get("multiple_neurons_meet_L_match")),
                    "segment_creation_index": finished.get("segment_creation_transition_index", ""),
                    "segment_reinforcement_count": snapshot.segment_meta.get(object_id, {}).get("reinforcement_count", ""),
                    "context_size": len(snapshot.active_ids),
                    "unique_source_count": len(source_ids),
                    **quality,
                }
            )
            if _truth(finished.get("passes_L2")) and not _truth(finished.get("passes_L3")):
                self.l2_only_rows.append(dict(match_row))

            if self.level == "source":
                for source_row in capture.source_rows:
                    if int(source_row["_segment_object_id"]) != object_id:
                        continue
                    source = int(source_row["source_cell_stable_id"])
                    stats = snapshot.source_stats.get(source, {})
                    self.source_rows.append(
                        {
                            **MARKERS,
                            "stream_label": stream_label,
                            "actual_record_index": actual_record_index,
                            "field": finished.get("field", ""),
                            "segment_provenance_id": finished.get("segment_provenance_id", ""),
                            "observe_scenario": finished.get("observe_scenario", ""),
                            "target_column": finished.get("encoded_column", ""),
                            "target_neuron": finished.get("segment_target_neuron", ""),
                            "source_cell_id": source,
                            "source_overlaps": _truth(source_row.get("contributes_to_actual_overlap")),
                            "source_selected_segment": _truth(finished.get("selected_for_scenario")),
                            "source_reinforced_segment": _truth(finished.get("reinforced")),
                            **dict(source_row),
                            **stats,
                        }
                    )

        self.event_rows.append(
            {
                **MARKERS,
                "stream_label": stream_label,
                "actual_record_index": actual_record_index,
                "field": completed.column_rows[0].get("field", "") if completed.column_rows else "",
                "context_size": len(snapshot.active_ids),
                "live_segment_count": len(snapshot.segment_sources),
                "candidate_segment_count": len(completed.segment_rows),
                "l2_only_count": sum(row["l2_only"] for row in self.match_rows if int(row["actual_record_index"]) == actual_record_index),
                "l3_only_count": sum(row["l3_only"] for row in self.match_rows if int(row["actual_record_index"]) == actual_record_index),
                "matching_segment_count": sum(_truth(row.get("meets_L_match")) for row in completed.segment_rows),
                "ambiguous_event_count": sum(
                    _truth(column.get("multiple_segments_meet_L_match"))
                    for column in completed.column_rows
                ),
                "mean_source_incidence": self._event_mean_source_incidence(snapshot),
                "mean_source_idf": self._event_mean_source_idf(snapshot),
            }
        )
        self._flush_if_needed()
        self._pending = None

    def record_observation(self, *, model: SequentialMemory, actual_record_index: int) -> None:
        """Update offline activation history after the real observation returns."""

        winners = set(model.previous_winners)
        active = set(model.previous_active_cells)
        predicted = set(getattr(self.registry, "current_predicted_sources", ()))
        burst = set(getattr(self.registry, "current_burst_sources", ()))
        for source in active | winners | predicted | burst:
            history = self.source_history.setdefault(source, _SourceHistory())
            history.activation_count += 1
            history.recent_activation_count += 1
            history.last_any_record = actual_record_index
            if source in winners or source in burst:
                history.actual_activation_count += 1
                history.last_actual_record = actual_record_index
            if source in predicted and source not in winners and source not in burst:
                history.autonomous_activation_count += 1
        if actual_record_index >= 20:
            for history in self.source_history.values():
                history.recent_activation_count = max(0, history.recent_activation_count - 1)

    def _quality(
        self,
        *,
        source_ids: Sequence[int],
        overlap_sources: Sequence[int],
        missing_sources: Sequence[int],
        target_field: str,
        snapshot: _Snapshot,
    ) -> dict[str, object]:
        stats = [snapshot.source_stats[source] for source in source_ids if source in snapshot.source_stats]
        if not stats:
            return {
                "mean_source_incidence": 0.0,
                "median_source_incidence": 0.0,
                "max_source_incidence": 0,
                "mean_source_idf": 0.0,
                "min_source_idf": 0.0,
                "max_source_idf": 0.0,
                "rare_source_fraction": 0.0,
                "common_source_fraction": 0.0,
                "very_common_source_fraction": 0.0,
                "same_field_fraction": 0.0,
                "cross_field_fraction": 0.0,
                "burst_fraction": 0.0,
                "predicted_fraction": 0.0,
                "winner_fraction": 0.0,
                "propagated_fraction": 0.0,
                "actual_history_fraction": 0.0,
                "unknown_fraction": 0.0,
                "context_field_entropy": 0.0,
                "context_column_entropy": 0.0,
                "context_neuron_entropy": 0.0,
            }
        incidences = [_number(item.get("source_segment_incidence")) for item in stats]
        idfs = [_number(item.get("source_idf")) for item in stats]
        bins = [str(item.get("source_incidence_bin", "")) for item in stats]
        labels = [str(item.get("source_origin_type", "")) for item in stats]
        fields = [str(item.get("source_field", "")) for item in stats]
        columns = [str(item.get("source_column", "")) for item in stats]
        neurons = [f"{item.get('source_column')}:{item.get('source_neuron')}" for item in stats]
        return {
            "mean_source_incidence": statistics.mean(incidences),
            "median_source_incidence": statistics.median(incidences),
            "max_source_incidence": max(incidences),
            "mean_source_idf": statistics.mean(idfs),
            "min_source_idf": min(idfs),
            "max_source_idf": max(idfs),
            "rare_source_fraction": _ratio(bins.count("RARE"), len(bins)),
            "common_source_fraction": _ratio(bins.count("COMMON"), len(bins)),
            "very_common_source_fraction": _ratio(bins.count("VERY_COMMON"), len(bins)),
            "same_field_fraction": _ratio(sum(field == target_field for field in fields), len(fields)),
            "cross_field_fraction": _ratio(sum(field != target_field for field in fields), len(fields)),
            "burst_fraction": _ratio(sum("ACTUAL_BURST" in label for label in labels), len(labels)),
            "predicted_fraction": _ratio(sum("PREDICTED" in label for label in labels), len(labels)),
            "winner_fraction": _ratio(sum("ACTUAL_WINNER" in label for label in labels), len(labels)),
            "propagated_fraction": _ratio(sum("PROPAGATED" in label for label in labels), len(labels)),
            "actual_history_fraction": _ratio(
                sum(_number(item.get("source_actual_activation_frequency")) > 0 for item in stats),
                len(stats),
            ),
            "unknown_fraction": _ratio(sum("UNKNOWN" in label for label in labels), len(labels)),
            "context_field_entropy": _entropy(fields),
            "context_column_entropy": _entropy(columns),
            "context_neuron_entropy": _entropy(neurons),
        }

    def _quality_projection(
        self,
        quality: Mapping[str, object],
        *,
        prefix: str,
        sources: Sequence[int],
    ) -> dict[str, object]:
        if not sources or self._pending is None:
            return {
                f"{prefix}_mean_incidence": 0.0,
                f"{prefix}_median_incidence": 0.0,
                f"{prefix}_max_incidence": 0,
                f"{prefix}_mean_idf": 0.0,
                f"{prefix}_min_idf": 0.0,
                f"{prefix}_rare_fraction": 0.0,
                f"{prefix}_common_fraction": 0.0,
                f"{prefix}_very_common_fraction": 0.0,
            }
        stats = [self._pending.source_stats[source] for source in sources if source in self._pending.source_stats]
        incidences = [_number(item.get("source_segment_incidence")) for item in stats]
        idfs = [_number(item.get("source_idf")) for item in stats]
        bins = [str(item.get("source_incidence_bin", "")) for item in stats]
        return {
            f"{prefix}_mean_incidence": statistics.mean(incidences) if incidences else 0.0,
            f"{prefix}_median_incidence": statistics.median(incidences) if incidences else 0.0,
            f"{prefix}_max_incidence": max(incidences, default=0),
            f"{prefix}_mean_idf": statistics.mean(idfs) if idfs else 0.0,
            f"{prefix}_min_idf": min(idfs, default=0.0),
            f"{prefix}_rare_fraction": _ratio(bins.count("RARE"), len(bins)),
            f"{prefix}_common_fraction": _ratio(bins.count("COMMON"), len(bins)),
            f"{prefix}_very_common_fraction": _ratio(bins.count("VERY_COMMON"), len(bins)),
        }

    def _event_mean_source_incidence(self, snapshot: _Snapshot) -> float:
        values = [_number(item.get("source_segment_incidence")) for item in snapshot.source_stats.values()]
        return statistics.mean(values) if values else 0.0

    def _event_mean_source_idf(self, snapshot: _Snapshot) -> float:
        values = [_number(item.get("source_idf")) for item in snapshot.source_stats.values()]
        return statistics.mean(values) if values else 0.0

    def summary(self) -> dict[str, object]:
        return {
            **MARKERS,
            "version": VERSION,
            "level": self.level,
            "match_rows": len(self.match_rows) + self._written_counts["segment_context_match_trace"],
            "source_rows": len(self.source_rows) + self._written_counts["segment_context_source_trace"],
            "quality_rows": len(self.quality_rows) + self._written_counts["segment_context_quality"],
            "event_rows": len(self.event_rows) + self._written_counts["segment_context_event_summary"],
            "l2_only_rows": len(self.l2_only_rows) + self._written_counts["l2_only_context_composition"],
            "source_count": len(self.source_history),
            "streamed_rows": dict(self._written_counts),
        }

    def write_outputs(self, output_dir: Path, *, compress: bool) -> dict[str, str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}
        datasets = {
            "segment_context_match_trace": self.match_rows,
            "segment_context_source_trace": self.source_rows,
            "segment_context_quality": self.quality_rows,
            "segment_context_event_summary": self.event_rows,
            "l2_only_context_composition": self.l2_only_rows,
        }
        if self._stream_output_dir is not None:
            for stem, rows in datasets.items():
                self._flush_dataset(stem, rows)
            for stem, path in self._written_paths.items():
                paths[stem] = str(path)
        else:
            for stem, rows in datasets.items():
                if not rows:
                    continue
                path = output_dir / f"{stem}.csv"
                if compress:
                    path = path.with_suffix(path.suffix + ".gz")
                _write_rows(path, rows, compress=compress)
                paths[stem] = str(path)
            passenger_rows = [
                row for row in self.l2_only_rows if row.get("field") == "passenger"
            ]
            if passenger_rows:
                path = output_dir / "passenger_l2_only_context_composition.csv"
                if compress:
                    path = path.with_suffix(path.suffix + ".gz")
                _write_rows(path, passenger_rows, compress=compress)
                paths["passenger_l2_only_context_composition"] = str(path)
        summary_path = output_dir / "segment_context_composition_summary.json"
        summary_path.write_text(json.dumps(self.summary(), indent=2, sort_keys=True), encoding="utf-8")
        paths["summary"] = str(summary_path)
        return paths


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]], *, compress: bool) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if compress else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _append_rows(path: Path, rows: Sequence[Mapping[str, object]], *, compress: bool) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if compress else open
    write_header = not path.exists() or path.stat().st_size == 0
    with opener(path, "at", encoding="utf-8", newline="") as handle:
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def protocol(*, level: str, compressed: bool) -> dict[str, object]:
    """Return a machine-readable statement of the diagnostic boundary."""

    return {
        **MARKERS,
        "version": VERSION,
        "level": level,
        "compressed": compressed,
        "matching_expression": "count of unique source cell IDs with arrival inside timing_tolerance",
        "source_expression": "log((N_segments + 1) / (source_segment_incidence + 1))",
        "L_match_used_by_model": True,
        "source_statistics_not_used_by_model": True,
        "actual_observation_trace_only": True,
        "autonomous_rollout_trace": "not captured by this diagnostic; kept separate from actual observation",
    }
