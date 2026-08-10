"""Offline Fig.9 L2/L4 ambiguity and sparsity analysis.

This module only reads completed run artifacts.  It never loads a model, calls
``predict_code``/``observe_code``, or uses a target label to change a decision.
The trace contains actual-observation and autonomous-rollout rows separately;
all joins in this file preserve that distinction.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


RANGES = ((200, 249), (250, 299), (300, 349), (350, 399), (400, 449), (450, 494))
STAGES = (
    "match_ambiguity",
    "intracolumn_ambiguity",
    "score_ambiguity",
    "neuron_ambiguity",
    "intercolumn_ambiguity",
    "reuse_ambiguity",
    "context_ambiguity",
)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_gzip_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Iterable[dict[str, object]], fields: list[str] | None = None) -> None:
    materialized = list(rows)
    if not materialized and not fields:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fields or list(materialized[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_trace_copy(path: Path, runs: Iterable[dict[str, object]]) -> None:
    """Copy de-duplicated trace rows with an explicit L_match column."""

    materialized: list[dict[str, object]] = []
    for run in runs:
        trace = run["trace"]
        assert isinstance(trace, list)
        materialized.extend({"L_match": run["L_match"], **row} for row in trace)
    if not materialized:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def number(value: object, default: float = math.nan) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def integer(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def values(rows: Iterable[dict[str, object]], field: str) -> list[float]:
    return [value for value in (number(row.get(field)) for row in rows) if math.isfinite(value)]


def mean(rows: Iterable[dict[str, object]], field: str) -> float:
    items = values(rows, field)
    return statistics.fmean(items) if items else 0.0


def quantile(items: list[float], fraction: float) -> float:
    if not items:
        return 0.0
    ordered = sorted(items)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summary_stats(rows: Iterable[dict[str, object]], field: str) -> dict[str, object]:
    items = values(rows, field)
    return {
        "n": len(items),
        "mean": statistics.fmean(items) if items else 0.0,
        "p50": quantile(items, 0.50),
        "p90": quantile(items, 0.90),
        "p99": quantile(items, 0.99),
    }


def pearson(rows: Iterable[dict[str, object]], x_field: str, y_field: str) -> dict[str, object]:
    pairs = [
        (number(row.get(x_field)), number(row.get(y_field)))
        for row in rows
    ]
    pairs = [(x, y) for x, y in pairs if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return {"n": len(pairs), "pearson": 0.0}
    x_bar = statistics.fmean(x for x, _ in pairs)
    y_bar = statistics.fmean(y for _, y in pairs)
    numerator = sum((x - x_bar) * (y - y_bar) for x, y in pairs)
    denominator = math.sqrt(
        sum((x - x_bar) ** 2 for x, _ in pairs)
        * sum((y - y_bar) ** 2 for _, y in pairs)
    )
    return {"n": len(pairs), "pearson": numerator / denominator if denominator else 0.0}


def slope(rows: Iterable[dict[str, object]], x_field: str, y_field: str) -> float:
    pairs = [
        (number(row.get(x_field)), number(row.get(y_field)))
        for row in rows
    ]
    pairs = [(x, y) for x, y in pairs if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return 0.0
    x_bar = statistics.fmean(x for x, _ in pairs)
    y_bar = statistics.fmean(y for _, y in pairs)
    denominator = sum((x - x_bar) ** 2 for x, _ in pairs)
    return sum((x - x_bar) * (y - y_bar) for x, y in pairs) / denominator if denominator else 0.0


def range_name(index: int) -> str:
    for start, end in RANGES:
        if start <= index <= end:
            return f"{start}-{end}"
    return "outside_requested_ranges"


def dedupe_trace(rows: Iterable[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    """Remove process-retry duplicates without collapsing distinct candidates."""

    raw_rows = list(rows)
    unique: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    for row in raw_rows:
        key = (
            row.get("trajectory_kind", ""),
            row.get("actual_record_index", ""),
            row.get("horizon_step", ""),
            row.get("event_number", ""),
            row.get("candidate_number", ""),
            row.get("candidate_segment_id", ""),
            row.get("selected_segment_id", ""),
            row.get("column", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique, len(raw_rows) - len(unique)


def load_run(l_match: int, run_dir: Path) -> dict[str, object]:
    trace_path = run_dir / "ambiguity_event_trace.csv.gz"
    raw_trace = read_gzip_csv(trace_path)
    trace, duplicate_rows = dedupe_trace(raw_trace)
    protocol = read_json(run_dir / "original_protocol.json")
    summary = read_json(run_dir / "original_summary.json")
    density = read_csv(run_dir / "original_density_trace.csv")
    return {
        "L_match": l_match,
        "run_dir": str(run_dir),
        "trace": trace,
        "trace_rows_raw": len(raw_trace),
        "trace_duplicate_rows": duplicate_rows,
        "protocol": protocol,
        "summary": summary,
        "density": density,
    }


def actual_events(run: dict[str, object]) -> list[dict[str, object]]:
    trace = run["trace"]
    assert isinstance(trace, list)
    groups: dict[tuple[int, int, int], dict[str, object]] = {}
    for row in trace:
        if row.get("trajectory_kind") != "actual_observation":
            continue
        key = (
            integer(row.get("actual_record_index")),
            integer(row.get("horizon_step")),
            integer(row.get("event_number")),
        )
        group = groups.setdefault(
            key,
            {
                "L_match": run["L_match"],
                "trajectory_kind": "actual_observation",
                "actual_record_index": key[0],
                "horizon_step": key[1],
                "event_number": key[2],
                "range": range_name(key[0]),
                "field": row.get("field", ""),
                "column": row.get("column", ""),
                "scenario": row.get("scenario", ""),
                "candidate_count": 0,
                "candidate_segment_ids": set(),
                "candidate_neuron_ids": set(),
            },
        )
        segment_id = str(row.get("candidate_segment_id", ""))
        if segment_id:
            group["candidate_count"] = integer(group["candidate_count"]) + 1
            group["candidate_segment_ids"].add(segment_id)  # type: ignore[union-attr]
            if str(row.get("candidate_neuron", "")):
                group["candidate_neuron_ids"].add(str(row.get("candidate_neuron")))  # type: ignore[union-attr]
        for field in (
            "matching_segment_count",
            "matching_neuron_count",
            "matching_column_count",
            "within_column_candidate_segment_count",
            "within_column_candidate_neuron_count",
            "local_top1_score",
            "local_top2_score",
            "local_score_margin",
            "normalized_score_margin",
            "candidate_score_entropy",
            "top_score_tie_count",
            "context_source_count",
            "selected_overlap",
        ):
            if field not in group or group[field] in (None, ""):
                group[field] = row.get(field, "")
        group["context_signature"] = row.get("context_signature", "")
    output = []
    for group in groups.values():
        group["candidate_segment_count_distinct"] = len(group.pop("candidate_segment_ids"))  # type: ignore[arg-type]
        group["candidate_neuron_count_distinct"] = len(group.pop("candidate_neuron_ids"))  # type: ignore[arg-type]
        output.append(group)
    return sorted(output, key=lambda row: (integer(row.get("actual_record_index")), integer(row.get("event_number"))))


def autonomous_rows(run: dict[str, object]) -> list[dict[str, object]]:
    trace = run["trace"]
    assert isinstance(trace, list)
    return [
        {
            "L_match": run["L_match"],
            **row,
            "actual_record_index": integer(row.get("actual_record_index")),
            "horizon_step": integer(row.get("horizon_step")),
            "range": range_name(integer(row.get("actual_record_index"))),
        }
        for row in trace
        if row.get("trajectory_kind") == "autonomous_rollout"
    ]


def density_step_rows(run: dict[str, object], step: int | None = None) -> list[dict[str, str]]:
    density = run["density"]
    assert isinstance(density, list)
    return [
        row
        for row in density
        if step is None or integer(row.get("horizon_step")) == step
    ]


def step_mape(rows: Iterable[dict[str, str]]) -> float:
    rows = list(rows)
    errors = sum(abs(number(row.get("absolute_error"), 0.0)) for row in rows)
    targets = sum(abs(number(row.get("actual_future_passenger"), number(row.get("target"), 0.0))) for row in rows)
    return errors / targets if targets else 0.0


def range_rows(run: dict[str, object]) -> list[dict[str, object]]:
    events = actual_events(run)
    autonomous = autonomous_rows(run)
    output: list[dict[str, object]] = []
    for start, end in RANGES:
        name = f"{start}-{end}"
        event_rows = [row for row in events if start <= integer(row.get("actual_record_index")) <= end]
        auto_rows = [row for row in autonomous if start <= integer(row.get("actual_record_index")) <= end]
        density = [
            row
            for row in density_step_rows(run)
            if start <= integer(row.get("record_index")) <= end
        ]
        output.append(
            {
                "L_match": run["L_match"],
                "range": name,
                "record_start": start,
                "record_end": end,
                "actual_event_count": len(event_rows),
                "autonomous_row_count": len(auto_rows),
                "mean_matching_segment_count": mean(event_rows, "matching_segment_count"),
                "mean_matching_neuron_count": mean(event_rows, "matching_neuron_count"),
                "mean_within_column_actual": mean(event_rows, "within_column_candidate_segment_count"),
                "mean_within_column_autonomous": mean(auto_rows, "within_column_candidate_segment_count"),
                "mean_score_margin_autonomous": mean(auto_rows, "local_score_margin"),
                "top_tie_fraction_autonomous": (
                    sum(integer(row.get("top_score_tie_count")) > 1 for row in auto_rows) / len(auto_rows)
                    if auto_rows else 0.0
                ),
                "mean_context_source_count": mean(event_rows, "context_source_count"),
                "mean_step1_MAPE": step_mape(
                    row for row in density if integer(row.get("horizon_step")) == 1
                ),
                "mean_step2_MAPE": step_mape(
                    row for row in density if integer(row.get("horizon_step")) == 2
                ),
                "mean_step3_MAPE": step_mape(
                    row for row in density if integer(row.get("horizon_step")) == 3
                ),
                "mean_step4_MAPE": step_mape(
                    row for row in density if integer(row.get("horizon_step")) == 4
                ),
                "mean_step5_MAPE": step_mape(
                    row for row in density if integer(row.get("horizon_step")) == 5
                ),
                "mean_raw_columns_step5": mean(
                    [row for row in density if integer(row.get("horizon_step")) == 5],
                    "raw_predicted_column_count",
                ),
                "mean_emitted_columns_step5": mean(
                    [row for row in density if integer(row.get("horizon_step")) == 5],
                    "emitted_column_count",
                ),
            }
        )
    return output


def grouped_rows(rows: Iterable[dict[str, object]], group_fields: tuple[str, ...], metric_fields: tuple[str, ...]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field, "") for field in group_fields)].append(row)
    output: list[dict[str, object]] = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        result = {field: value for field, value in zip(group_fields, key)}
        result["row_count"] = len(group)
        for field in metric_fields:
            result[f"{field}_mean"] = mean(group, field)
            result[f"{field}_p50"] = quantile(values(group, field), 0.5)
            result[f"{field}_p90"] = quantile(values(group, field), 0.9)
        output.append(result)
    return output


def reuse_rows(run: dict[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    trace = run["trace"]
    assert isinstance(trace, list)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in trace:
        if row.get("trajectory_kind") == "actual_observation" and row.get("candidate_segment_id"):
            groups[str(row["candidate_segment_id"])].append(row)
    summary: list[dict[str, object]] = []
    breadth: list[dict[str, object]] = []
    for segment_id, rows in sorted(groups.items()):
        records = {integer(row.get("actual_record_index")) for row in rows}
        contexts = {row.get("context_signature", "") for row in rows if row.get("context_signature")}
        fields = {row.get("field", "") for row in rows}
        scenarios = {row.get("scenario", "") for row in rows}
        recent = max((integer(row.get("recent_50_match_count_running")) for row in rows), default=0)
        summary.append(
            {
                "L_match": run["L_match"],
                "segment_id": segment_id,
                "matching_event_count": len(rows),
                "distinct_actual_records": len(records),
                "distinct_context_signatures": len(contexts),
                "recent_50_match_count": recent,
                "first_record_index": min(records) if records else "",
                "last_record_index": max(records) if records else "",
                "selected_count": sum(str(row.get("selected_candidate", "")).lower() == "true" for row in rows),
                "mean_candidate_overlap": mean(rows, "candidate_overlap"),
                "mean_segment_age": mean(rows, "segment_age"),
                "mean_reinforcement_count": mean(rows, "candidate_segment_reinforcement_count"),
            }
        )
        breadth.append(
            {
                "L_match": run["L_match"],
                "segment_id": segment_id,
                "field_breadth": len(fields),
                "fields": "|".join(sorted(str(value) for value in fields)),
                "scenario_breadth": len(scenarios),
                "scenarios": "|".join(sorted(str(value) for value in scenarios)),
                "context_breadth": len(contexts),
            }
        )
    return summary, breadth


def target_false_rows(run: dict[str, object]) -> list[dict[str, object]]:
    """Report the post-hoc labeling boundary without inventing target labels."""

    trace = run["trace"]
    assert isinstance(trace, list)
    return [
        {
            "L_match": run["L_match"],
            "trajectory_kind": kind,
            "comparison": "UNAVAILABLE_WITHOUT_ORACLE_CANDIDATE_TRACE",
            "target_reuse_count": "",
            "false_reuse_count": "",
            "ground_truth_used_online": False,
            "reason": (
                "The ambiguity trace records actual matched candidates and autonomous "
                "selector rows, but does not contain post-hoc oracle target labels."
            ),
        }
        for kind in ("actual_observation", "autonomous_rollout")
    ]


def score_margin_rows(run: dict[str, object]) -> list[dict[str, object]]:
    rows = autonomous_rows(run)
    output: list[dict[str, object]] = []
    for kind in ("autonomous_rollout", "actual_observation"):
        selected = rows if kind == "autonomous_rollout" else actual_events(run)
        output.append(
            {
                "L_match": run["L_match"],
                "trajectory_kind": kind,
                "candidate_groups": len(selected),
                "top1": summary_stats(selected, "local_top1_score"),
                "top2": summary_stats(selected, "local_top2_score"),
                "margin": summary_stats(selected, "local_score_margin"),
                "normalized_margin": summary_stats(selected, "normalized_score_margin"),
                "entropy": summary_stats(selected, "candidate_score_entropy"),
                "tie_fraction": (
                    sum(integer(row.get("top_score_tie_count")) > 1 for row in selected) / len(selected)
                    if selected else 0.0
                ),
            }
        )
    return output


def score_error_rows(run: dict[str, object]) -> list[dict[str, object]]:
    density_by_key = {
        (integer(row.get("record_index")), integer(row.get("horizon_step"))): row
        for row in density_step_rows(run)
    }
    rows = []
    for source in autonomous_rows(run):
        density = density_by_key.get((integer(source.get("actual_record_index")), integer(source.get("horizon_step"))))
        if density is None:
            continue
        rows.append(
            {
                **source,
                "absolute_percentage_error": density.get("absolute_percentage_error", ""),
                "absolute_error": density.get("absolute_error", ""),
            }
        )
    return rows


def native_rows(run: dict[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    run_dir = Path(str(run["run_dir"]))
    attempts: list[dict[str, object]] = []
    for path in sorted(run_dir.glob("process_*.json")):
        try:
            payload = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        payload = dict(payload)
        payload["L_match"] = run["L_match"]
        payload["attempt_file"] = path.name
        payload["is_native_crash"] = integer(payload.get("exit_code")) == 3221225477
        attempts.append(payload)
    progress = []
    for line in (run_dir / "progress.jsonl").read_text(encoding="utf-8").splitlines() if (run_dir / "progress.jsonl").exists() else []:
        try:
            progress.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    crashes = [row for row in attempts if row.get("is_native_crash")]
    by_size = []
    for crash in crashes:
        last = integer(crash.get("last_completed_index"))
        candidates = [row for row in progress if integer(row.get("current_index")) <= last]
        nearest = max(candidates, key=lambda row: integer(row.get("current_index")), default={})
        by_size.append(
            {
                "L_match": run["L_match"],
                "attempt_file": crash.get("attempt_file", ""),
                "last_completed_index": last,
                "last_native_phase": crash.get("last_native_phase", ""),
                "segment_count": nearest.get("segment_count", ""),
                "synapse_count": nearest.get("synapse_count", ""),
                "candidate_segment_count": nearest.get("candidate_segment_count", ""),
                "working_set_memory_mb": nearest.get("working_set_memory_mb", ""),
            }
        )
    return attempts, by_size


def paired_bootstrap(l2: dict[str, object], l4: dict[str, object], samples: int = 2000, seed: int = 11) -> dict[str, object]:
    def by_index(run: dict[str, object]) -> dict[int, float]:
        return {
            integer(row.get("record_index")): number(row.get("absolute_percentage_error"), 0.0)
            for row in density_step_rows(run, 5)
        }
    left = by_index(l2)
    right = by_index(l4)
    values = [right[index] - left[index] for index in sorted(set(left) & set(right))]
    if not values:
        return {"n": 0, "mean_L4_minus_L2": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = random.Random(seed)
    boot = [statistics.fmean(values[rng.randrange(len(values))] for _ in values) for _ in range(samples)]
    return {
        "n": len(values),
        "mean_L4_minus_L2": statistics.fmean(values),
        "ci_low": quantile(boot, 0.025),
        "ci_high": quantile(boot, 0.975),
    }


def choose_conclusion(l2: dict[str, object], l4: dict[str, object], ranges: list[dict[str, object]]) -> tuple[str, str, list[str]]:
    l2_ranges = [row for row in ranges if integer(row.get("L_match")) == 2]
    l4_ranges = [row for row in ranges if integer(row.get("L_match")) == 4]
    if not l2_ranges or not l4_ranges:
        return "REAL_LMATCH2_ABLATION", "REAL_LMATCH2_ABLATION", ["Insufficient paired range evidence."]
    early = number(l2_ranges[0].get("mean_step5_MAPE"), 0.0) - number(l4_ranges[0].get("mean_step5_MAPE"), 0.0)
    late = number(l2_ranges[-1].get("mean_step5_MAPE"), 0.0) - number(l4_ranges[-1].get("mean_step5_MAPE"), 0.0)
    l2_match = statistics.fmean(number(row.get("mean_matching_segment_count"), 0.0) for row in l2_ranges)
    l4_match = statistics.fmean(number(row.get("mean_matching_segment_count"), 0.0) for row in l4_ranges)
    l2_margin = statistics.fmean(number(row.get("mean_score_margin_autonomous"), 0.0) for row in l2_ranges)
    l4_margin = statistics.fmean(number(row.get("mean_score_margin_autonomous"), 0.0) for row in l4_ranges)
    if l2_match > l4_match * 1.10 and late > early:
        return (
            "SEGMENT_CONTEXT_REPRESENTATION_FIX_REQUIRED",
            "SEGMENT_CONTEXT_REPRESENTATION_FIX_REQUIRED",
            ["L2 matching ambiguity is higher and its late error gap worsens; this is diagnostic evidence, not a model fix."],
        )
    if l2_margin < l4_margin * 0.90:
        return (
            "INTRACOLUMN_RETENTION_FIX_REQUIRED",
            "INTRACOLUMN_RETENTION_FIX_REQUIRED",
            ["L2 autonomous score margins are smaller; inspect local candidate retention before changing strict selection."],
        )
    return (
        "REAL_LMATCH2_ABLATION",
        "REAL_LMATCH2_ABLATION",
        ["The available trace does not isolate one mechanism strongly enough for a code change."],
    )


def analyze(l2_dir: Path, l4_dir: Path, output_dir: Path, bootstrap_samples: int = 2000, bootstrap_seed: int = 11) -> dict[str, object]:
    l2 = load_run(2, l2_dir)
    l4 = load_run(4, l4_dir)
    runs = (l2, l4)
    range_summary = [row for run in runs for row in range_rows(run)]
    field_rows = [
        row
        for run in runs
        for kind, source in (("actual_observation", actual_events(run)), ("autonomous_rollout", autonomous_rows(run)))
        for row in grouped_rows(
            [{"L_match": run["L_match"], "trajectory_kind": kind, **item} for item in source],
            ("L_match", "trajectory_kind", "field"),
            ("matching_segment_count", "within_column_candidate_segment_count", "local_score_margin", "context_source_count"),
        )
    ]
    scenario_rows = grouped_rows(
        [
            {"L_match": run["L_match"], **row}
            for run in runs
            for row in actual_events(run)
        ],
        ("L_match", "scenario"),
        ("matching_segment_count", "candidate_count", "context_source_count"),
    )
    age_rows = []
    reinforcement_rows = []
    for run in runs:
        trace = run["trace"]
        assert isinstance(trace, list)
        actual = [row for row in trace if row.get("trajectory_kind") == "actual_observation" and row.get("candidate_segment_id")]
        for bucket, predicate in (
            ("0-49", lambda value: value < 50),
            ("50-99", lambda value: 50 <= value < 100),
            ("100-199", lambda value: 100 <= value < 200),
            ("200+", lambda value: value >= 200),
        ):
            subset = [row for row in actual if predicate(number(row.get("segment_age"), -1))]
            age_rows.append({"L_match": run["L_match"], "segment_age_bucket": bucket, "row_count": len(subset), "mean_matching_segment_count": mean(subset, "matching_segment_count"), "mean_score_margin": mean(subset, "local_score_margin")})
        for bucket, predicate in (
            ("0", lambda value: value == 0),
            ("1", lambda value: value == 1),
            ("2-3", lambda value: 2 <= value <= 3),
            ("4-7", lambda value: 4 <= value <= 7),
            ("8-15", lambda value: 8 <= value <= 15),
            ("16+", lambda value: value >= 16),
        ):
            subset = [row for row in actual if predicate(number(row.get("candidate_segment_reinforcement_count"), -1))]
            reinforcement_rows.append({"L_match": run["L_match"], "reinforcement_bucket": bucket, "row_count": len(subset), "mean_matching_segment_count": mean(subset, "matching_segment_count"), "mean_score_margin": mean(subset, "local_score_margin")})
    reuse = [reuse_rows(run) for run in runs]
    reuse_summary = [row for pair in reuse for row in pair[0]]
    reuse_breadth = [row for pair in reuse for row in pair[1]]
    target_false = [row for run in runs for row in target_false_rows(run)]
    score_summary_rows = [row for run in runs for row in score_margin_rows(run)]
    score_margin_range = [
        {
            "L_match": run["L_match"],
            "range": range_name(integer(row.get("actual_record_index"))),
            "trajectory_kind": "autonomous_rollout",
            **row,
        }
        for run in runs
        for row in autonomous_rows(run)
    ]
    score_margin_range = grouped_rows(
        score_margin_range,
        ("L_match", "range", "trajectory_kind"),
        ("local_top1_score", "local_top2_score", "local_score_margin", "normalized_score_margin", "candidate_score_entropy", "top_score_tie_count"),
    )
    score_margin_field = grouped_rows(
        [
            {"L_match": run["L_match"], **row}
            for run in runs
            for row in autonomous_rows(run)
        ],
        ("L_match", "field"),
        ("local_top1_score", "local_top2_score", "local_score_margin", "normalized_score_margin", "candidate_score_entropy", "top_score_tie_count"),
    )
    score_error = []
    for run in runs:
        joined = score_error_rows(run)
        score_error.append(
            {
                "L_match": run["L_match"],
                "margin_vs_step_error": pearson(joined, "local_score_margin", "absolute_percentage_error"),
                "entropy_vs_step_error": pearson(joined, "candidate_score_entropy", "absolute_percentage_error"),
                "tie_count_vs_step_error": pearson(joined, "top_score_tie_count", "absolute_percentage_error"),
                "joined_rows": len(joined),
            }
        )
    tradeoff = []
    for left, right in zip(
        [row for row in range_summary if integer(row.get("L_match")) == 2],
        [row for row in range_summary if integer(row.get("L_match")) == 4],
    ):
        tradeoff.append(
            {
                "range": left["range"],
                "local_step1_MAPE_L2_minus_L4": number(left.get("mean_step1_MAPE"), 0.0) - number(right.get("mean_step1_MAPE"), 0.0),
                "recurrent_step2_MAPE_L2_minus_L4": number(left.get("mean_step2_MAPE"), 0.0) - number(right.get("mean_step2_MAPE"), 0.0),
                "recurrent_step3_5_MAPE_L2_minus_L4": statistics.fmean(number(left.get(f"mean_step{step}_MAPE"), 0.0) - number(right.get(f"mean_step{step}_MAPE"), 0.0) for step in (3, 4, 5)),
                "match_ambiguity_L2_minus_L4": number(left.get("mean_matching_segment_count"), 0.0) - number(right.get("mean_matching_segment_count"), 0.0),
                "intracolumn_ambiguity_L2_minus_L4": number(left.get("mean_within_column_autonomous"), 0.0) - number(right.get("mean_within_column_autonomous"), 0.0),
                "score_margin_L2_minus_L4": number(left.get("mean_score_margin_autonomous"), 0.0) - number(right.get("mean_score_margin_autonomous"), 0.0),
                "raw_column_density_L2_minus_L4": number(left.get("mean_raw_columns_step5"), 0.0) - number(right.get("mean_raw_columns_step5"), 0.0),
            }
        )
    sparsity = [
        {
            "L_match": row["L_match"],
            "range": row["range"],
            "match_ambiguity": row["mean_matching_segment_count"],
            "intracolumn_ambiguity": row["mean_within_column_autonomous"],
            "score_margin": row["mean_score_margin_autonomous"],
            "raw_column_density_step5": row["mean_raw_columns_step5"],
            "emitted_column_density_step5": row["mean_emitted_columns_step5"],
            "sparsity_tradeoff_note": "lower raw/emitted activity is not treated as better without error comparison",
        }
        for row in range_summary
    ]
    native = [native_rows(run) for run in runs]
    native_summary = []
    native_by_size = []
    for attempt_rows, size_rows in native:
        crashes = [row for row in attempt_rows if row.get("is_native_crash")]
        native_summary.append({"L_match": attempt_rows[0].get("L_match") if attempt_rows else "", "attempt_count": len(attempt_rows), "native_crash_count": len(crashes), "exit_codes": "|".join(str(row.get("exit_code")) for row in crashes), "last_phases": "|".join(str(row.get("last_native_phase")) for row in crashes)})
        native_by_size.extend(size_rows)
    summaries = []
    for run in runs:
        summary = run["summary"]
        assert isinstance(summary, dict)
        summaries.append({"L_match": run["L_match"], "MAPE": summary.get("mape", ""), "coverage": summary.get("coverage", ""), "predictions": summary.get("predictions", ""), "final_segments": summary.get("final_segment_count", ""), "final_synapses": summary.get("final_synapse_count", ""), "trace_rows_raw": run["trace_rows_raw"], "trace_duplicate_rows": run["trace_duplicate_rows"], "model_fingerprint": summary.get("final_model_fingerprint", ""), "RNG_fingerprint": summary.get("final_rng_fingerprint", "")})
    primary, recommended, reasons = choose_conclusion(l2, l4, range_summary)
    bootstrap = paired_bootstrap(l2, l4, bootstrap_samples, bootstrap_seed)
    mechanism = [{"mechanism": "new_model_mechanism", "status": "NOT_STARTED", "reason": "No mechanism is started by this read-only diagnostic.", "strict_default_changed": False}]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_trace_copy(output_dir / "ambiguity_event_trace.csv.gz", runs)
    write_rows(output_dir / "ambiguity_range_summary.csv", range_summary)
    write_rows(output_dir / "ambiguity_by_field.csv", field_rows)
    write_rows(output_dir / "ambiguity_by_scenario.csv", scenario_rows)
    write_rows(output_dir / "ambiguity_by_segment_age.csv", age_rows)
    write_rows(output_dir / "ambiguity_by_reinforcement_count.csv", reinforcement_rows)
    write_rows(output_dir / "segment_reuse_summary.csv", reuse_summary)
    write_rows(output_dir / "segment_reuse_breadth.csv", reuse_breadth)
    write_rows(output_dir / "target_false_reuse_comparison.csv", target_false)
    write_rows(output_dir / "score_margin_summary.csv", score_summary_rows)
    write_rows(output_dir / "score_margin_by_range.csv", score_margin_range)
    write_rows(output_dir / "score_margin_by_field.csv", score_margin_field)
    write_rows(output_dir / "score_margin_error_association.csv", score_error)
    write_rows(output_dir / "local_recurrent_tradeoff.csv", tradeoff)
    write_rows(output_dir / "ambiguity_sparsity_tradeoff.csv", sparsity)
    write_rows(output_dir / "native_crash_summary.csv", native_summary)
    write_rows(output_dir / "native_crash_by_network_size.csv", native_by_size)
    write_rows(output_dir / "mechanism_results.csv", mechanism)
    write_rows(output_dir / "l2_l4_ambiguity_summary.csv", summaries)
    final = {
        "protocol": {
            "diagnostic": "Fig.9 L2 ambiguity accumulation vs recurrent sparsity trade-off",
            "model_math_changed": False,
            "ground_truth_used_online": False,
            "actual_and_autonomous_separated": True,
            "l2_match_class_labels": ["L2_ONLY_STRICT", "L3_ONLY_VS_L4", "L2_RELAXED_VS_L4"],
            "optional_1000": "NOT_RUN",
        },
        "summaries": summaries,
        "bootstrap": bootstrap,
        "primary_conclusion": primary,
        "recommended_next_step": recommended,
        "secondary_findings": reasons[:2],
        "trace_completeness": {
            "L2_raw_rows": l2["trace_rows_raw"],
            "L4_raw_rows": l4["trace_rows_raw"],
            "L2_duplicate_rows_removed": l2["trace_duplicate_rows"],
            "L4_duplicate_rows_removed": l4["trace_duplicate_rows"],
        },
        "mechanism_started": False,
    }
    write_json(output_dir / "FINAL_L2_AMBIGUITY_SUMMARY.json", final)
    write_rows(output_dir / "EXPERIMENT_LEDGER.csv", [{"stage": "offline_ambiguity_analysis", "L2_run": l2_dir, "L4_run": l4_dir, "model_math_changed": False, "optional_1000": "NOT_RUN", "primary_conclusion": primary}])
    report = [
        "# Fig.9 L2 Ambiguity Accumulation Report",
        "",
        "This report is a read-only diagnostic. It is not a strict-paper reproduction result and does not start a model mechanism.",
        "",
        f"- Primary conclusion: `{primary}`",
        f"- Recommended next step: `{recommended}`",
        f"- Paired step-5 MAPE bootstrap, L4 minus L2: `{number(bootstrap.get('mean_L4_minus_L2'), 0.0):.6f}` ({number(bootstrap.get('ci_low'), 0.0):.6f}, {number(bootstrap.get('ci_high'), 0.0):.6f})",
        "- Optional 1000-record run: `NOT_RUN`",
        "- Online ground-truth selection: `False`",
        "",
        "## Result Completeness",
        "",
        "| L_match | MAPE | coverage | predictions | final segments | final synapses | raw trace rows | duplicate rows removed |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        report.append(f"| {row['L_match']} | {number(row['MAPE'], 0.0):.9f} | {number(row['coverage'], 0.0):.6f} | {row['predictions']} | {row['final_segments']} | {row['final_synapses']} | {row['trace_rows_raw']} | {row['trace_duplicate_rows']} |")
    report += [
        "",
        "## Interpretation",
        "",
        "Actual-observation rows describe candidates matched during real proximal learning. Autonomous-rollout rows describe the same-call intracolumn candidate groups during read-only recurrent prediction. They are never pooled into one trajectory.",
        "The target-vs-false reuse file is explicitly unavailable when no post-hoc oracle candidate trace was recorded; no target label is inferred from an online row.",
        "",
        "## Mechanism Gate",
        "",
        "No new mechanism was started. The trace is used to decide whether a later bounded diagnostic is justified; strict defaults remain unchanged.",
        "",
        "## Native Stability",
        "",
        "Native-crash counts are runtime observations and are not interpreted as model-quality evidence. See `native_crash_summary.csv` and `native_crash_by_network_size.csv`.",
    ]
    (output_dir / "FIG9_L2_AMBIGUITY_ACCUMULATION_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l2-dir", type=Path, required=True)
    parser.add_argument("--l4-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=11)
    args = parser.parse_args()
    analyze(args.l2_dir, args.l4_dir, args.output_dir, args.bootstrap_samples, args.bootstrap_seed)


if __name__ == "__main__":
    main()
