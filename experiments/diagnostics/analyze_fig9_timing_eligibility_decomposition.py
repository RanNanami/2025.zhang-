"""Offline analysis of Fig.9 source timing and eligibility evidence."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from experiments.diagnostics.analyze_fig9_match_overlap_decomposition import (
    classify_record_range,
)


def _find(run_dir: Path, stem: str) -> Path | None:
    for suffix in (".csv.gz", ".csv"):
        path = run_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def _read(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _truth(value: object) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


def _number(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _observation_key(row: Mapping[str, object]) -> tuple[str, ...]:
    return (
        str(row.get("actual_record_index", "")),
        str(row.get("timestamp", "")),
        str(row.get("field", "")),
        str(row.get("encoded_column", "")),
    )


def _segment_key(row: Mapping[str, object]) -> tuple[str, ...]:
    return _observation_key(row) + (
        str(row.get("segment_provenance_id", "")),
        str(row.get("segment_target_neuron", "")),
    )


def _reason(row: Mapping[str, object]) -> str:
    return str(
        row.get("source_loss_reason_primary")
        or row.get("source_loss_reason")
        or "UNKNOWN_TIMING_ELIGIBILITY_REASON"
    )


def _summary(
    rows: Sequence[Mapping[str, object]],
    group_fields: Sequence[str],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, object]]] = defaultdict(list)
    scope_totals: Counter[tuple[str, ...]] = Counter()
    for row in rows:
        for_reason = _reason(row)
        scope_key = tuple(
            classify_record_range(row.get("actual_record_index"))
            if field == "record_range"
            else str(row.get(field, ""))
            for field in group_fields
        )
        scope_totals[scope_key] += 1
        key = scope_key + (
            for_reason,
            str(row.get("source_loss_reason_secondary", "")),
        )
        grouped[key].append(row)
    output = []
    for key, group in sorted(grouped.items()):
        scope = {
            field: key[index] for index, field in enumerate(group_fields)
        }
        denominator = scope_totals[key[: len(group_fields)]]
        output.append(
            {
                **scope,
                "primary_reason": key[-2],
                "secondary_reason": key[-1],
                "source_row_count": len(group),
                "source_row_denominator": denominator,
                "rate": len(group) / denominator if denominator else 0.0,
                "denominator_scope": (
                    "pre_observation_source_synapse_rows_in_selected_scope"
                ),
                "unique_observation_count": len(
                    {_observation_key(row) for row in group}
                ),
                "unique_segment_count": len(
                    {_segment_key(row) for row in group}
                ),
                "mean_reasons_per_observation": (
                    len(group)
                    / len({_observation_key(row) for row in group})
                ),
                "mean_reasons_per_segment": (
                    len(group) / len({_segment_key(row) for row in group})
                ),
                "source_column_active_rate": sum(
                    _truth(row.get("source_column_currently_active"))
                    for row in group
                )
                / len(group),
                "exact_neuron_active_rate": sum(
                    _truth(row.get("exact_source_neuron_currently_active"))
                    for row in group
                )
                / len(group),
                "previous_winners_membership_rate": sum(
                    _truth(row.get("source_in_previous_winners"))
                    for row in group
                )
                / len(group),
                "predicted_membership_rate": sum(
                    _truth(row.get("source_in_predicted_cells"))
                    for row in group
                )
                / len(group),
                "burst_membership_rate": sum(
                    _truth(row.get("source_in_burst_cells"))
                    for row in group
                )
                / len(group),
            }
        )
    return output


def _bootstrap(
    rows: Sequence[Mapping[str, object]],
    *,
    samples: int,
    seed: int,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    observations: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for row in rows:
        observations[_observation_key(row)][_reason(row)] += 1
    groups = list(observations.values())
    reasons = sorted({_reason(row) for row in rows})
    output = []
    for reason in reasons:
        estimate = sum(group[reason] for group in groups) / len(rows)
        draws = []
        for _ in range(samples):
            sampled = [rng.choice(groups) for _ in groups]
            denominator = sum(sum(group.values()) for group in sampled)
            numerator = sum(group[reason] for group in sampled)
            draws.append(numerator / denominator if denominator else 0.0)
        draws.sort()
        output.append(
            {
                "scope": "all_source_rows",
                "primary_reason": reason,
                "numerator": sum(group[reason] for group in groups),
                "denominator": len(rows),
                "estimate": estimate,
                "ci_low": draws[int(0.025 * (samples - 1))],
                "ci_high": draws[int(0.975 * (samples - 1))],
                "bootstrap_unit": "encoded_field_column_observation",
                "bootstrap_samples": samples,
                "bootstrap_seed": seed,
            }
        )
    return output


def _simple_boolean_summary(
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[str],
) -> list[dict[str, object]]:
    return [
        {
            "metric": field,
            "numerator": sum(_truth(row.get(field)) for row in rows),
            "denominator": len(rows),
            "rate": (
                sum(_truth(row.get(field)) for row in rows) / len(rows)
                if rows
                else 0.0
            ),
            "denominator_scope": "pre_observation_source_synapse_rows",
        }
        for field in fields
    ]


def _counterfactual_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    segment_groups: dict[
        tuple[str, ...], list[Mapping[str, object]]
    ] = defaultdict(list)
    for row in rows:
        segment_groups[_segment_key(row)].append(row)
    observations: dict[
        tuple[str, ...],
        list[tuple[tuple[str, ...], list[Mapping[str, object]]]],
    ] = defaultdict(list)
    for key, group in segment_groups.items():
        observations[key[:4]].append((key, group))
    modes = {
        "ACTUAL_TIMED_OVERLAP": lambda row: _truth(
            row.get("contributes_to_actual_overlap")
        ),
        "ACTIVE_EXACT_IDENTITY_IGNORE_TIMING": lambda row: _truth(
            row.get("source_in_active_cells")
        ),
        "PREDICTED_EXACT_IDENTITY_IGNORE_TIMING": lambda row: _truth(
            row.get("source_in_predicted_cells")
        ),
        "EXPLICIT_CONTEXT_EXCLUSIONS_ONLY": lambda row: (
            _truth(row.get("contributes_to_actual_overlap"))
            or _reason(row) == "SOURCE_ACTIVE_NOT_IN_MATCH_CONTEXT"
        ),
    }
    output = []
    for label, predicate in modes.items():
        best_values = []
        match_count = 0
        ambiguous = 0
        multi_neuron = 0
        for segments in observations.values():
            values = [
                (sum(predicate(row) for row in group), key[-1])
                for key, group in segments
            ]
            best_values.append(max((value for value, _ in values), default=0))
            passing = [(value, neuron) for value, neuron in values if value >= 4]
            match_count += bool(passing)
            ambiguous += len(passing) > 1
            multi_neuron += len({neuron for _, neuron in passing}) > 1
        denominator = len(observations)
        output.append(
            {
                "counterfactual_mode": label,
                "observation_count": denominator,
                "mean_best_overlap": (
                    statistics.fmean(best_values) if best_values else 0.0
                ),
                "L4_match_count": match_count,
                "L4_match_rate": (
                    match_count / denominator if denominator else 0.0
                ),
                "L4_ambiguity_count": ambiguous,
                "L4_ambiguity_rate": (
                    ambiguous / denominator if denominator else 0.0
                ),
                "L4_multiple_neuron_ambiguity_count": multi_neuron,
                "L4_multiple_neuron_ambiguity_rate": (
                    multi_neuron / denominator if denominator else 0.0
                ),
                "teacher_reference_compatibility": (
                    "unavailable_in_source_trace"
                ),
                "trajectory_changed": False,
                "ground_truth_used_for_selection": False,
            }
        )
    return output


def _named_scope_summary(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    scopes = {
        "ALL_FIELDS": list(rows),
        "PASSENGER": [
            row for row in rows if row.get("field") == "passenger"
        ],
        "PASSENGER_SCENARIO3": [
            row
            for row in rows
            if row.get("field") == "passenger"
            and row.get("observe_scenario") == "scenario3"
        ],
        "PASSENGER_200_244_SCENARIO3": [
            row
            for row in rows
            if row.get("field") == "passenger"
            and row.get("observe_scenario") == "scenario3"
            and classify_record_range(row.get("actual_record_index"))
            == "200-244"
        ],
        "WEEKDAY_SCENARIO3": [
            row
            for row in rows
            if row.get("field") == "weekday"
            and row.get("observe_scenario") == "scenario3"
        ],
        "TIME_SCENARIO3": [
            row
            for row in rows
            if row.get("field") == "time"
            and row.get("observe_scenario") == "scenario3"
        ],
    }
    output = []
    for label, group in scopes.items():
        for row in _summary(group, ()):
            output.append({"scope": label, **row})
    return output


def _named_scope_bootstrap(
    rows: Sequence[Mapping[str, object]],
    *,
    samples: int,
    seed: int,
) -> list[dict[str, object]]:
    scopes = {
        "ALL_FIELDS": list(rows),
        "PASSENGER": [
            row for row in rows if row.get("field") == "passenger"
        ],
        "PASSENGER_SCENARIO3": [
            row
            for row in rows
            if row.get("field") == "passenger"
            and row.get("observe_scenario") == "scenario3"
        ],
        "PASSENGER_200_244_SCENARIO3": [
            row
            for row in rows
            if row.get("field") == "passenger"
            and row.get("observe_scenario") == "scenario3"
            and classify_record_range(row.get("actual_record_index"))
            == "200-244"
        ],
        "WEEKDAY_SCENARIO3": [
            row
            for row in rows
            if row.get("field") == "weekday"
            and row.get("observe_scenario") == "scenario3"
        ],
        "TIME_SCENARIO3": [
            row
            for row in rows
            if row.get("field") == "time"
            and row.get("observe_scenario") == "scenario3"
        ],
    }
    output = []
    for index, (label, group) in enumerate(scopes.items()):
        if not group:
            continue
        for row in _bootstrap(
            group, samples=samples, seed=seed + index
        ):
            output.append({**row, "scope": label})
    return output


def analyze(
    *,
    run_dir: Path,
    output_dir: Path,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    source_path = _find(run_dir, "match_overlap_source_trace")
    rows = _read(source_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    reason_summary = _named_scope_summary(rows)
    by_field = _summary(rows, ("field",))
    by_range = _summary(rows, ("record_range",))
    by_scenario = _summary(rows, ("observe_scenario",))
    late = [
        row
        for row in rows
        if row.get("field") == "passenger"
        and row.get("observe_scenario") == "scenario3"
        and classify_record_range(row.get("actual_record_index")) == "200-244"
    ]
    _write(output_dir / "timing_eligibility_reason_summary.csv", reason_summary)
    _write(output_dir / "timing_eligibility_reason_by_field.csv", by_field)
    _write(
        output_dir / "timing_eligibility_reason_by_record_range.csv", by_range
    )
    _write(
        output_dir / "timing_eligibility_reason_by_scenario.csv", by_scenario
    )
    _write(
        output_dir / "passenger_late_scenario3_reason_summary.csv",
        _summary(late, ()),
    )
    _write(
        output_dir / "timing_window_summary.csv",
        _simple_boolean_summary(
            rows,
            (
                "arrival_in_matching_window",
                "timing_used_by_actual_overlap",
                "source_predicted_time_available",
                "source_predicted_time_valid",
            ),
        ),
    )
    _write(
        output_dir / "context_membership_exclusion_summary.csv",
        _simple_boolean_summary(
            rows,
            (
                "source_in_matching_context",
                "source_in_active_cells",
                "source_in_previous_winners",
                "source_in_predicted_cells",
                "source_in_burst_cells",
            ),
        ),
    )
    _write(
        output_dir / "segment_eligibility_summary.csv",
        _simple_boolean_summary(
            rows,
            ("segment_considered_by_matching", "segment_eligible"),
        ),
    )
    _write(
        output_dir / "synapse_eligibility_summary.csv",
        _simple_boolean_summary(
            rows,
            ("synapse_considered_by_matching", "synapse_eligible"),
        ),
    )
    _write(
        output_dir / "timing_eligibility_bootstrap_ci.csv",
        _named_scope_bootstrap(
            rows,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
    )
    passenger_scenario3 = [
        row
        for row in rows
        if row.get("field") == "passenger"
        and row.get("observe_scenario") == "scenario3"
    ]
    _write(
        output_dir / "context_membership_counterfactual_summary.csv",
        _counterfactual_rows(
            late if late else passenger_scenario3
        ),
    )

    observations = {_observation_key(row) for row in rows}
    segments = {_segment_key(row) for row in rows}
    segment_counts = Counter(_segment_key(row) for row in rows)
    declared_counts: dict[tuple[str, ...], int] = {}
    for row in rows:
        value = _number(row.get("segment_synapse_count"))
        if value is not None:
            declared_counts[_segment_key(row)] = int(value)
    count_mismatches = [
        key
        for key, count in segment_counts.items()
        if key in declared_counts and declared_counts[key] != count
    ]
    unknown_ranges = sum(
        classify_record_range(row.get("actual_record_index"))
        == "UNKNOWN_RECORD_RANGE"
        for row in rows
    )
    unknown_reasons = sum(
        _reason(row)
        in {
            "UNKNOWN_SOURCE_LOSS",
            "UNKNOWN_TIMING_ELIGIBILITY_REASON",
        }
        for row in rows
    )
    is_late_target = bool(rows) and all(
        row.get("field") == "passenger"
        and row.get("observe_scenario") == "scenario3"
        and 200 <= int(row["actual_record_index"]) <= 244
        for row in rows
    )
    coverage_passed = (
        not is_late_target
        or (
            len(observations) == 329
            and len(segments) == 2437
            and not count_mismatches
        )
    )
    manifest = {
        "source_trace_path": str(source_path),
        "source_rows": len(rows),
        "unique_observation_count": len(observations),
        "unique_segment_count": len(segments),
        "record_index_min": min(
            (int(row["actual_record_index"]) for row in rows), default=None
        ),
        "record_index_max": max(
            (int(row["actual_record_index"]) for row in rows), default=None
        ),
        "late_target_scope_detected": is_late_target,
        "expected_late_observations": 329,
        "expected_late_segments": 2437,
        "model_execution_unfiltered": True,
    }
    checks = {
        "passed": coverage_passed and unknown_ranges == 0,
        "coverage_passed": coverage_passed,
        "unknown_record_range_rows": unknown_ranges,
        "unknown_reason_rows": unknown_reasons,
        "unknown_reason_rate": (
            unknown_reasons / len(rows) if rows else 0.0
        ),
        "segment_synapse_count_mismatch_count": len(count_mismatches),
        "segment_synapse_count_mismatch_keys": count_mismatches[:100],
    }
    (output_dir / "source_trace_coverage_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "timing_eligibility_consistency_checks.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True), encoding="utf-8"
    )
    reasons = Counter(_reason(row) for row in rows)
    dominant = reasons.most_common(1)[0][0] if reasons else "unavailable"
    if dominant == "SOURCE_COLUMN_NOT_ACTIVE":
        next_step = "CONTEXT_TRAJECTORY_DIAGNOSTIC_REQUIRED"
    elif dominant in {
        "SOURCE_ACTIVE_NOT_IN_MATCH_CONTEXT",
        "SOURCE_PREDICTED_NOT_IN_MATCH_CONTEXT",
        "SOURCE_BURST_NOT_IN_MATCH_CONTEXT",
        "SOURCE_STATE_CAPTURE_UNAVAILABLE",
        "UNKNOWN_TIMING_ELIGIBILITY_REASON",
    }:
        next_step = "MATCHING_SEMANTICS_FIX_REQUIRED"
    else:
        next_step = "REAL_LMATCH2_ABLATION"
    report = f"""# Fig.9 Timing / Eligibility Decomposition

This is a read-only diagnostic. It does not modify matching, prediction,
learning, L_match, burst context, RNG, or strict defaults.

- Source rows: {len(rows)}
- Unique observations: {len(observations)}
- Unique segments: {len(segments)}
- Unknown reason rate: {checks['unknown_reason_rate']:.6f}
- Dominant primary reason: `{dominant}`
- Recommended next step: `{next_step}`

The repository's actual overlap uses exact source-cell membership **and**
`source_time + synaptic_delay` within the dendritic target time plus/minus
`timing_tolerance`. Weight and PSP score rank candidate segments but do not
make an individual source contribute to `timed_overlap`. With strict
`burst_context=True`, matching uses `previous_active_cells` (falling back to
`previous_winners` only when active cells are empty), not winner-only context.

Rates in the CSV files use pre-observation source-synapse rows in the named
scope. This small diagnostic cannot establish the late 200-244 distribution
unless the manually filtered 250-record trace passes the 329-observation and
2437-segment coverage checks.

The context-membership counterfactual is fixed-state and read-only. It never
uses ground truth to select a segment and never produces a new MAPE. Identity
modes that ignore arrival timing are diagnostic upper bounds, not alternate
model trajectories.
"""
    (output_dir / "FIG9_TIMING_ELIGIBILITY_DECOMPOSITION_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    summary = {
        **manifest,
        **checks,
        "dominant_primary_reason": dominant,
        "recommended_next_step": next_step,
    }
    (output_dir / "timing_eligibility_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    if not checks["passed"]:
        raise AssertionError(
            "timing eligibility coverage/consistency checks failed"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()
    analyze(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )


if __name__ == "__main__":
    main()
