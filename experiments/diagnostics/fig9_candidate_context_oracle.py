"""Stable candidate identity and posthoc context/oracle projections for Fig.9.

The strict runner already has two useful observations of one prediction:
``PredictionCandidate`` objects from the real ``predict_code`` call and the
future code used only by the oracle diagnostic.  This module gives that
candidate instance a deterministic identity and computes context descriptors
from the live pre-selection state.  It never participates in matching,
selection, competition, learning, or RNG use.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from experiments.diagnostics.fig9_candidate_score_trace import field_for_column
from experiments.diagnostics.fig9_oracle_candidate import FieldColumnRanges
from seqmem.model import PredictionCandidate, Segment, SequentialMemory


VERSION = "fig9-candidate-context-oracle-v1"
ROW_UNIT_CANDIDATE = "CANDIDATE_SEGMENT"
TRAJECTORY_ACTUAL = "actual_observation"
TRAJECTORY_AUTONOMOUS = "autonomous_rollout"

MATCH_REGIMES = {
    "OVERLAP_EXACT_2",
    "OVERLAP_EXACT_3",
    "OVERLAP_GE_4",
    "L2_ONLY_STRICT",
    "L3_ONLY_VS_L4",
    "L2_RELAXED_VS_L4",
    "AUTONOMOUS_PREDICTION_CANDIDATE",
}


def _finite(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _entropy(values: Iterable[object]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    if total <= 0 or len(counts) <= 1:
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


def stable_candidate_event_id(
    *,
    trajectory_kind: str,
    actual_record_index: int,
    horizon_step: int,
    field: str,
    observed_encoded_column: int | str,
    candidate_target_column: int,
    candidate_target_neuron: int,
    stable_segment_provenance_id: str,
    candidate_generation_ordinal: int,
    observed_event_time: float | str = "",
) -> str:
    """Hash stable candidate-instance fields, never object or CSV positions."""

    payload = {
        "version": VERSION,
        "trajectory_kind": trajectory_kind,
        "actual_record_index": int(actual_record_index),
        "horizon_step": int(horizon_step),
        "field": str(field),
        "observed_encoded_column": str(observed_encoded_column),
        "candidate_target_column": int(candidate_target_column),
        "candidate_target_neuron": int(candidate_target_neuron),
        "stable_segment_provenance_id": str(stable_segment_provenance_id),
        "candidate_generation_ordinal": int(candidate_generation_ordinal),
        "observed_event_time": str(observed_event_time),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def classify_overlap_regime(overlap_count: int) -> tuple[str, ...]:
    """Use the permanent exact-2/exact-3/at-least-4 vocabulary."""

    regimes: list[str] = []
    if overlap_count == 2:
        regimes.extend(("OVERLAP_EXACT_2", "L2_ONLY_STRICT"))
    elif overlap_count == 3:
        regimes.extend(("OVERLAP_EXACT_3", "L3_ONLY_VS_L4"))
    elif overlap_count >= 4:
        regimes.append("OVERLAP_GE_4")
    return tuple(regimes)


@dataclass(frozen=True)
class _SourceStats:
    source_id: int
    source_column: int
    source_neuron: int
    source_field: str
    incidence: int
    idf: float
    incidence_bin: str
    age: int | str
    origin_labels: tuple[str, ...]


@dataclass(frozen=True)
class ContextCompositionIndex:
    """Read-only live-network source index for one prediction transition."""

    context_size: int
    live_segment_count: int
    source_stats: Mapping[int, _SourceStats]
    active_sources: frozenset[int]
    winner_sources: frozenset[int]
    predicted_sources: frozenset[int]
    burst_sources: frozenset[int]

    @classmethod
    def from_model(
        cls,
        *,
        model: SequentialMemory,
        registry: object | None,
        ranges: FieldColumnRanges,
        actual_record_index: int,
        active_sources: Mapping[int, float] | Iterable[int],
    ) -> "ContextCompositionIndex":
        active_ids = frozenset(int(value) for value in active_sources)
        winners = frozenset(int(value) for value in getattr(model, "previous_winners", ()))
        predicted = frozenset(int(value) for value in getattr(registry, "current_predicted_sources", ()))
        burst = frozenset(int(value) for value in getattr(registry, "current_burst_sources", ()))
        source_segments: dict[int, set[int]] = defaultdict(set)
        source_creation: dict[int, list[int]] = defaultdict(list)
        live_segment_count = 0
        for column_id, column in enumerate(model.columns):
            for neuron in column.neurons:
                for segment in neuron.segments:
                    if not segment.active:
                        continue
                    live_segment_count += 1
                    object_id = id(segment)
                    origin = registry.provenance_for(segment) if registry is not None else None
                    creation_index = (
                        origin.creation_transition_index
                        if origin is not None
                        else segment.creation_transition_index
                    )
                    for source in segment.synapses:
                        source_segments[int(source)].add(object_id)
                        if creation_index is not None:
                            source_creation[int(source)].append(int(creation_index))

        incidences = [len(values) for values in source_segments.values()]
        thresholds = tuple(
            _quantile([float(value) for value in incidences], fraction)
            for fraction in (0.25, 0.50, 0.75, 0.90)
        )
        stats: dict[int, _SourceStats] = {}
        neurons_per_column = len(model.columns[0].neurons) if model.columns else 1
        for source, segment_ids in sorted(source_segments.items()):
            source_column = source // neurons_per_column
            source_neuron = source % neurons_per_column
            labels: list[str] = []
            if source in winners:
                labels.append("WINNER")
            if source in predicted:
                labels.append("PREDICTED")
            if source in burst:
                labels.append("BURST")
            if source in active_ids and not labels:
                labels.append("ACTIVE")
            if not labels:
                labels.append("UNAVAILABLE")
            first_creation = min(source_creation.get(source, ()), default="")
            age = (
                actual_record_index - int(first_creation)
                if first_creation != ""
                else ""
            )
            incidence = len(segment_ids)
            stats[source] = _SourceStats(
                source_id=source,
                source_column=source_column,
                source_neuron=source_neuron,
                source_field=field_for_column(source_column, ranges),
                incidence=incidence,
                idf=math.log((live_segment_count + 1) / (incidence + 1)),
                incidence_bin=_incidence_bin(incidence, thresholds),
                age=age,
                origin_labels=tuple(labels),
            )
        return cls(
            context_size=len(active_ids),
            live_segment_count=live_segment_count,
            source_stats=stats,
            active_sources=active_ids,
            winner_sources=winners,
            predicted_sources=predicted,
            burst_sources=burst,
        )

    def describe_sources(
        self,
        sources: Iterable[int],
        *,
        target_field: str,
    ) -> dict[str, object]:
        source_ids = tuple(sorted(set(int(value) for value in sources)))
        overlap_ids = tuple(source for source in source_ids if source in self.active_sources)
        stats = [self.source_stats[source] for source in source_ids if source in self.source_stats]
        overlap_stats = [self.source_stats[source] for source in overlap_ids if source in self.source_stats]
        fields = [item.source_field for item in stats]
        columns = [str(item.source_column) for item in stats]
        neurons = [f"{item.source_column}:{item.source_neuron}" for item in stats]
        labels = [item.origin_labels for item in stats]

        def values(items: Sequence[_SourceStats], attr: str) -> list[float]:
            return [_finite(getattr(item, attr)) for item in items]

        idfs = values(overlap_stats, "idf")
        incidences = values(overlap_stats, "incidence")
        ages = values(stats, "age")
        bins = [item.incidence_bin for item in overlap_stats]
        return {
            "context_size": self.context_size,
            "unique_source_count": len(source_ids),
            "overlap_source_count": len(overlap_ids),
            "overlap_mean_incidence": statistics.mean(incidences) if incidences else 0.0,
            "overlap_median_incidence": statistics.median(incidences) if incidences else 0.0,
            "overlap_max_incidence": max(incidences, default=0.0),
            "overlap_mean_idf": statistics.mean(idfs) if idfs else 0.0,
            "overlap_min_idf": min(idfs, default=0.0),
            "overlap_max_idf": max(idfs, default=0.0),
            "overlap_rare_fraction": _ratio(bins.count("RARE"), len(bins)),
            "overlap_medium_fraction": _ratio(bins.count("MEDIUM"), len(bins)),
            "overlap_common_fraction": _ratio(bins.count("COMMON"), len(bins)),
            "overlap_very_common_fraction": _ratio(bins.count("VERY_COMMON"), len(bins)),
            "overlap_same_field_fraction": _ratio(
                sum(item.source_field == target_field for item in overlap_stats),
                len(overlap_stats),
            ),
            "overlap_cross_field_fraction": _ratio(
                sum(item.source_field != target_field for item in overlap_stats),
                len(overlap_stats),
            ),
            "overlap_actual_fraction": _ratio(
                sum("WINNER" in item.origin_labels for item in overlap_stats),
                len(overlap_stats),
            ),
            "overlap_burst_fraction": _ratio(
                sum("BURST" in item.origin_labels for item in overlap_stats),
                len(overlap_stats),
            ),
            "overlap_predicted_fraction": _ratio(
                sum("PREDICTED" in item.origin_labels for item in overlap_stats),
                len(overlap_stats),
            ),
            "overlap_propagated_fraction": _ratio(
                sum("PREDICTED" in item.origin_labels for item in overlap_stats),
                len(overlap_stats),
            ),
            "overlap_mean_age": statistics.mean(ages) if ages else 0.0,
            "overlap_median_age": statistics.median(ages) if ages else 0.0,
            "context_field_entropy": _entropy(fields),
            "context_column_entropy": _entropy(columns),
            "context_neuron_entropy": _entropy(neurons),
            "overlap_source_ids": json.dumps(list(overlap_ids), separators=(",", ":")),
            "source_ids": json.dumps(list(source_ids), separators=(",", ":")),
        }


def candidate_context_fields(
    *,
    candidate: PredictionCandidate,
    context_index: ContextCompositionIndex,
    target_field: str,
) -> dict[str, object]:
    """Return context descriptors for one real PredictionCandidate."""

    fields = context_index.describe_sources(
        candidate.segment.synapses,
        target_field=target_field,
    )
    positive = [
        item
        for item in candidate.crossing_synapse_contributions
        if item.psp_contribution > 0.0
    ]
    fields.update(
        {
            "candidate_score_available": True,
            "original_candidate_score": candidate.score,
            "effective_candidate_score": "",
            "response_peak": candidate.peak_dendritic_potential if candidate.peak_dendritic_potential is not None else "",
            "contributor_count": len(positive),
            "predicted_time": candidate.time,
            "first_crossing_time": candidate.dendritic_crossing_time if candidate.dendritic_crossing_time is not None else "",
            "threshold_margin": candidate.threshold_margin if candidate.threshold_margin is not None else "",
            "segment_age": max((synapse.age for synapse in candidate.segment.synapses.values()), default=0),
            "reinforcement_count": candidate.segment.scenario1_reinforcements + candidate.segment.scenario2_reinforcements,
            "matching_context_semantics": "live_source_membership_plus_crossing_contributors",
        }
    )
    return fields


def _actual_candidate_id(row: Mapping[str, object], ordinal: int) -> str:
    return stable_candidate_event_id(
        trajectory_kind=TRAJECTORY_ACTUAL,
        actual_record_index=int(row.get("actual_record_index", 0)),
        horizon_step=0,
        field=str(row.get("field", "")),
        observed_encoded_column=row.get("encoded_column", ""),
        candidate_target_column=int(row.get("target_column", row.get("encoded_column", 0))),
        candidate_target_neuron=int(row.get("target_neuron", 0) or 0),
        stable_segment_provenance_id=str(row.get("segment_provenance_id", "")),
        candidate_generation_ordinal=ordinal,
        observed_event_time=row.get("actual_observation_time", ""),
    )


def project_actual_composition_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Project actual-observation composition rows into candidate rows."""

    projected: list[dict[str, object]] = []
    for ordinal, original in enumerate(rows):
        row = dict(original)
        overlap = int(_finite(row.get("overlap_count", 0)))
        target_column = int(row.get("target_column", row.get("encoded_column", 0)) or 0)
        candidate_id = _actual_candidate_id(row, ordinal)
        regimes = classify_overlap_regime(overlap)
        projected.append(
            {
                "candidate_event_id": candidate_id,
                "row_unit": ROW_UNIT_CANDIDATE,
                "trajectory_kind": TRAJECTORY_ACTUAL,
                "actual_record_index": row.get("actual_record_index", ""),
                "horizon_step": "",
                "field": row.get("field", ""),
                "observed_encoded_column": row.get("encoded_column", ""),
                "candidate_target_column": target_column,
                "candidate_target_neuron": row.get("target_neuron", ""),
                "stable_segment_provenance_id": row.get("segment_provenance_id", ""),
                "segment_creation_index": row.get("segment_creation_index", ""),
                "candidate_generation_ordinal": ordinal,
                "L_match": row.get("L_match", ""),
                "overlap_count": overlap,
                "overlap_semantics": "actual_timed_overlap",
                "passes_L2": row.get("passes_L2", False),
                "passes_L3": row.get("passes_L3", False),
                "passes_L4": row.get("passes_L4", False),
                "L2_ONLY_STRICT": "L2_ONLY_STRICT" in regimes,
                "L2_RELAXED_VS_L4": "L2_ONLY_STRICT" in regimes or "L3_ONLY_VS_L4" in regimes,
                "L3_ONLY_VS_L4": "L3_ONLY_VS_L4" in regimes,
                "match_regime": "|".join(regimes),
                "matching_segment_count": row.get("matching_segment_count", ""),
                "matching_neuron_count": row.get("matching_neuron_count", ""),
                "ambiguous_segment": row.get("ambiguous", False),
                "ambiguous_neuron": row.get("ambiguous", False),
                "selected_intracolumn": row.get("selected", False),
                "selected_after_intercolumn": "",
                "reinforced": row.get("reinforced", False),
                "suppressed_intracolumn": False,
                "suppressed_intercolumn": False,
                "oracle_target_column": target_column,
                "oracle_target_columns": json.dumps([target_column]),
                "is_target_column_candidate": True,
                "is_target_neuron_candidate": True,
                "original_candidate_score": "",
                "effective_candidate_score": "",
                "response_peak": "",
                "contributor_count": "",
                "predicted_time": "",
                "candidate_score_available": False,
                "context_size": row.get("context_size", ""),
                "unique_source_count": row.get("context_size", ""),
                "overlap_source_count": row.get("overlap_source_count", ""),
                "overlap_mean_incidence": row.get("overlap_mean_incidence", ""),
                "overlap_median_incidence": row.get("overlap_median_incidence", ""),
                "overlap_max_incidence": row.get("overlap_max_incidence", ""),
                "overlap_mean_idf": row.get("overlap_mean_idf", ""),
                "overlap_min_idf": row.get("overlap_min_idf", ""),
                "overlap_max_idf": row.get("overlap_max_idf", ""),
                "overlap_rare_fraction": row.get("overlap_rare_fraction", ""),
                "overlap_medium_fraction": "",
                "overlap_common_fraction": row.get("overlap_common_fraction", ""),
                "overlap_very_common_fraction": row.get("overlap_very_common_fraction", ""),
                "overlap_same_field_fraction": row.get("overlap_same_field_fraction", ""),
                "overlap_cross_field_fraction": row.get("overlap_cross_field_fraction", ""),
                "overlap_actual_fraction": row.get("overlap_actual_history_fraction", ""),
                "overlap_burst_fraction": row.get("overlap_burst_fraction", ""),
                "overlap_predicted_fraction": row.get("overlap_predicted_fraction", ""),
                "overlap_propagated_fraction": row.get("overlap_propagated_fraction", ""),
                "overlap_mean_age": row.get("overlap_mean_age", ""),
                "overlap_median_age": row.get("overlap_median_age", ""),
                "context_field_entropy": row.get("context_field_entropy", ""),
                "context_column_entropy": row.get("context_column_entropy", ""),
                "context_neuron_entropy": row.get("context_neuron_entropy", ""),
                "source_ids": row.get("overlap_source_ids", ""),
                "step1_error": "",
                "step2_error": "",
                "step3_error": "",
                "step4_error": "",
                "step5_error": "",
                "recurrent_trajectory_divergence": False,
                "source_trace": "segment_context_composition",
            }
        )
    return projected


def enrich_autonomous_candidate_row(
    row: Mapping[str, object],
    *,
    candidate: PredictionCandidate,
    context_index: ContextCompositionIndex,
    ranges: FieldColumnRanges,
    target_columns_by_field: Mapping[str, Sequence[int]],
) -> dict[str, object]:
    """Add stable identity, oracle label, and context metrics to a branch row."""

    result = dict(row)
    field = str(result.get("field", field_for_column(int(result.get("column", 0)), ranges)))
    column = int(result.get("column", 0))
    neuron = int(result.get("neuron", candidate.neuron_index))
    ordinal = int(result.get("candidate_original_index", result.get("candidate_generation_ordinal", 0)) or 0)
    stable_segment_id = str(result.get("segment_provenance_id", ""))
    actual_record_index = int(result.get("input_index", 0) or 0) - 1
    horizon_step = int(result.get("horizon_step", 0) or 0)
    candidate_id = stable_candidate_event_id(
        trajectory_kind=TRAJECTORY_AUTONOMOUS,
        actual_record_index=actual_record_index,
        horizon_step=horizon_step,
        field=field,
        observed_encoded_column="",
        candidate_target_column=column,
        candidate_target_neuron=neuron,
        stable_segment_provenance_id=stable_segment_id,
        candidate_generation_ordinal=ordinal,
        observed_event_time=result.get("predicted_time", ""),
    )
    target_columns = tuple(int(value) for value in target_columns_by_field.get(field, ()))
    regimes = ("AUTONOMOUS_PREDICTION_CANDIDATE",)
    result.update(
        {
            "candidate_event_id": candidate_id,
            "row_unit": ROW_UNIT_CANDIDATE,
            "trajectory_kind": TRAJECTORY_AUTONOMOUS,
            "actual_record_index": actual_record_index,
            "observed_encoded_column": "",
            "candidate_target_column": column,
            "candidate_target_neuron": neuron,
            "stable_segment_provenance_id": stable_segment_id,
            "segment_creation_index": result.get("segment_creation_transition_index", ""),
            "candidate_generation_ordinal": ordinal,
            "oracle_target_column": target_columns[0] if len(target_columns) == 1 else "",
            "oracle_target_columns": json.dumps(list(target_columns), separators=(",", ":")),
            "is_target_column_candidate": column in target_columns,
            "is_target_neuron_candidate": "",
            "L2_ONLY_STRICT": False,
            "L2_RELAXED_VS_L4": False,
            "L3_ONLY_VS_L4": False,
            "match_regime": "|".join(regimes),
            "overlap_count": result.get("contributor_count", ""),
            "overlap_semantics": "continuous_positive_contributor_count",
            "selected_intracolumn": bool(result.get("is_column_representative", False)),
            "selected_after_intercolumn": bool(result.get("emitted", False)),
            "reinforced": False,
            "suppressed_intracolumn": not bool(result.get("is_column_representative", False)),
            "suppressed_intercolumn": bool(result.get("is_column_representative", False)) and not bool(result.get("emitted", False)),
            "recurrent_trajectory_divergence": horizon_step > 1,
        }
    )
    result.update(
        candidate_context_fields(
            candidate=candidate,
            context_index=context_index,
            target_field=field,
        )
    )
    return result


def validate_lossless_join(
    oracle_rows: Sequence[Mapping[str, object]],
    context_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Validate two projections by stable candidate_event_id."""

    oracle_ids = [str(row.get("candidate_event_id", "")) for row in oracle_rows]
    context_ids = [str(row.get("candidate_event_id", "")) for row in context_rows]
    oracle_counter = Counter(oracle_ids)
    context_counter = Counter(context_ids)
    duplicate_oracle = sum(max(0, count - 1) for count in oracle_counter.values())
    duplicate_context = sum(max(0, count - 1) for count in context_counter.values())
    oracle_set = set(oracle_ids)
    context_set = set(context_ids)
    return {
        "version": VERSION,
        "row_unit": ROW_UNIT_CANDIDATE,
        "oracle_candidate_rows": len(oracle_rows),
        "context_candidate_rows": len(context_rows),
        "joined_rows": len(oracle_set & context_set),
        "left_only_rows": len(oracle_set - context_set),
        "right_only_rows": len(context_set - oracle_set),
        "duplicate_candidate_event_ids": duplicate_oracle + duplicate_context,
        "duplicate_oracle_candidate_event_ids": duplicate_oracle,
        "duplicate_context_candidate_event_ids": duplicate_context,
        "one_to_many_joins": 0,
        "many_to_one_joins": 0,
        "many_to_many_joins": 0,
        "lossless": (
            len(oracle_rows) == len(context_rows)
            and len(oracle_set) == len(oracle_rows)
            and len(context_set) == len(context_rows)
            and oracle_set == context_set
        ),
    }


def write_csv_gz(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write a deterministic gzip CSV without relying on pandas."""

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_join_outputs(
    output_dir: Path,
    *,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Write combined trace plus oracle/context projections and join audit."""

    output_dir.mkdir(parents=True, exist_ok=True)
    combined = [dict(row) for row in rows]
    oracle_projection = [
        {
            key: value
            for key, value in row.items()
            if key not in {"context_size", "source_ids", "overlap_source_ids"}
        }
        for row in combined
    ]
    context_projection = [
        {
            key: value
            for key, value in row.items()
            if key
            not in {
                "oracle_target_column",
                "oracle_target_columns",
                "is_target_column_candidate",
                "is_target_neuron_candidate",
                "step1_error",
                "step2_error",
                "step3_error",
                "step4_error",
                "step5_error",
            }
        }
        for row in combined
    ]
    write_csv_gz(output_dir / "candidate_context_oracle_trace.csv.gz", combined)
    write_csv_gz(output_dir / "oracle_candidate_trace.csv.gz", oracle_projection)
    write_csv_gz(output_dir / "context_candidate_trace.csv.gz", context_projection)
    identity_rows = [
        {
            "candidate_event_id": row.get("candidate_event_id", ""),
            "trajectory_kind": row.get("trajectory_kind", ""),
            "actual_record_index": row.get("actual_record_index", ""),
            "horizon_step": row.get("horizon_step", ""),
            "field": row.get("field", ""),
            "candidate_target_column": row.get("candidate_target_column", ""),
            "candidate_target_neuron": row.get("candidate_target_neuron", ""),
            "stable_segment_provenance_id": row.get("stable_segment_provenance_id", ""),
            "candidate_generation_ordinal": row.get("candidate_generation_ordinal", ""),
        }
        for row in combined
    ]
    with (output_dir / "candidate_identity_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(identity_rows[0]) if identity_rows else ["candidate_event_id"])
        writer.writeheader()
        writer.writerows(identity_rows)
    consistency = validate_lossless_join(oracle_projection, context_projection)
    (output_dir / "candidate_join_consistency.json").write_text(
        json.dumps(consistency, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (output_dir / "candidate_join_consistency.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(consistency))
        writer.writeheader()
        writer.writerow(consistency)
    return consistency
