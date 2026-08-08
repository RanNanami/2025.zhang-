"""Offline analysis for Fig.9 segment-context composition traces.

The analyzer deliberately runs after the experiment.  It can describe
associations between context composition and errors, but it cannot alter the
model or choose a target using future observations.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


RECORD_RANGES = (
    (0, 49, "0-49"),
    (50, 99, "50-99"),
    (100, 149, "100-149"),
    (150, 199, "150-199"),
    (200, 244, "200-244"),
)


def _find(run_dir: Path, stem: str) -> Path | None:
    for suffix in (".csv", ".csv.gz"):
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


def _iter_rows(path: Path | None):
    if path is None:
        return
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def _summarize_source(path: Path | None) -> tuple[list[dict[str, object]], int]:
    """Aggregate the potentially multi-gigabyte decompressed source trace."""

    groups: dict[str, dict[str, object]] = defaultdict(
        lambda: {"rows": 0, "sources": set(), "incidence": [], "idf": [], "rare": 0, "common": 0, "very_common": 0}
    )
    total = 0
    for row in _iter_rows(path):
        total += 1
        field = str(row.get("source_field", ""))
        group = groups[field]
        group["rows"] = int(group["rows"]) + 1
        group["sources"].add(row.get("source_cell_stable_id"))  # type: ignore[union-attr]
        group["incidence"].append(_number(row.get("source_segment_incidence")))  # type: ignore[union-attr]
        group["idf"].append(_number(row.get("source_idf")))  # type: ignore[union-attr]
        group["rare"] = int(group["rare"]) + (row.get("source_incidence_bin") == "RARE")
        group["common"] = int(group["common"]) + (row.get("source_incidence_bin") == "COMMON")
        group["very_common"] = int(group["very_common"]) + (row.get("source_incidence_bin") == "VERY_COMMON")
    output = []
    for field, group in sorted(groups.items()):
        count = int(group["rows"])
        incidences = group["incidence"]
        idfs = group["idf"]
        output.append(
            {
                "source_field": field,
                "rows": count,
                "unique_sources": len(group["sources"]),
                "mean_incidence": statistics.mean(incidences) if incidences else 0.0,
                "mean_idf": statistics.mean(idfs) if idfs else 0.0,
                "rare_fraction": int(group["rare"]) / count if count else 0.0,
                "common_fraction": int(group["common"]) / count if count else 0.0,
                "very_common_fraction": int(group["very_common"]) / count if count else 0.0,
            }
        )
    return output, total


def _write(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _number(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _truth(value: object) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


def _mean(rows: Sequence[Mapping[str, object]], key: str) -> float:
    values = [_number(row.get(key)) for row in rows]
    return statistics.mean(values) if values else 0.0


def _rate(rows: Sequence[Mapping[str, object]], key: str) -> float:
    return sum(_truth(row.get(key)) for row in rows) / len(rows) if rows else 0.0


def _record_range(value: object) -> str:
    index = int(_number(value))
    for start, stop, label in RECORD_RANGES:
        if start <= index <= stop:
            return label
    return "UNKNOWN"


def _group(rows: Iterable[Mapping[str, object]], key: str) -> dict[str, list[Mapping[str, object]]]:
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, ""))].append(row)
    return groups


def _bootstrap_mean_ci(values: Sequence[float], *, seed: int = 11, rounds: int = 500) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    for _ in range(rounds):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(statistics.mean(sample))
    means.sort()
    return means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]


def _bootstrap_difference_ci(left: Sequence[float], right: Sequence[float]) -> tuple[float, float]:
    if not left or not right:
        return (0.0, 0.0)
    rng = random.Random(17)
    differences = []
    for _ in range(500):
        left_sample = [left[rng.randrange(len(left))] for _ in left]
        right_sample = [right[rng.randrange(len(right))] for _ in right]
        differences.append(statistics.mean(left_sample) - statistics.mean(right_sample))
    differences.sort()
    return differences[12], differences[487]


def _write_group_summary(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    key: str,
) -> list[dict[str, object]]:
    summaries = []
    for value, group in sorted(_group(rows, key).items()):
        summaries.append(
            {
                key: value,
                "rows": len(group),
                "mean_overlap": _mean(group, "overlap_count"),
                "mean_source_idf": _mean(group, "overlap_mean_idf"),
                "mean_source_incidence": _mean(group, "overlap_mean_incidence"),
                "rare_fraction": _mean(group, "overlap_rare_fraction"),
                "common_fraction": _mean(group, "overlap_common_fraction"),
                "very_common_fraction": _mean(group, "overlap_very_common_fraction"),
                "ambiguous_rate": _rate(group, "ambiguous"),
                "selected_rate": _rate(group, "selected"),
                "reinforced_rate": _rate(group, "reinforced"),
            }
        )
    _write(path, summaries)
    return summaries


def _write_quality_summary(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    key: str,
) -> list[dict[str, object]]:
    summaries = []
    for value, group in sorted(_group(rows, key).items()):
        summaries.append(
            {
                key: value,
                "rows": len(group),
                "mean_context_size": _mean(group, "context_size"),
                "mean_unique_source_count": _mean(group, "unique_source_count"),
                "mean_source_incidence": _mean(group, "mean_source_incidence"),
                "mean_source_idf": _mean(group, "mean_source_idf"),
                "rare_fraction": _mean(group, "rare_source_fraction"),
                "common_fraction": _mean(group, "common_source_fraction"),
                "very_common_fraction": _mean(group, "very_common_source_fraction"),
                "same_field_fraction": _mean(group, "same_field_fraction"),
                "cross_field_fraction": _mean(group, "cross_field_fraction"),
                "field_entropy": _mean(group, "context_field_entropy"),
                "column_entropy": _mean(group, "context_column_entropy"),
                "neuron_entropy": _mean(group, "context_neuron_entropy"),
                "matching_rate": _rate(group, "matching"),
                "ambiguous_rate": _rate(group, "ambiguous"),
            }
        )
    _write(path, summaries)
    return summaries


def analyze(run_dir: Path, output_dir: Path | None = None) -> dict[str, object]:
    """Analyze one composition run and write auditable aggregate tables."""

    analysis_dir = output_dir or run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    match_rows = _read(_find(run_dir, "segment_context_match_trace"))
    l2_rows = _read(_find(run_dir, "l2_only_context_composition"))
    source_field_rows, source_row_count = _summarize_source(
        _find(run_dir, "segment_context_source_trace")
    )
    quality_rows = _read(_find(run_dir, "segment_context_quality"))
    event_rows = _read(_find(run_dir, "segment_context_event_summary"))
    prediction_rows = _read(_find(run_dir, "original_predictions"))

    stage_funnel = [
        {"stage": "candidate", "rows": len(match_rows)},
        {"stage": "passes_L2", "rows": sum(_truth(row.get("passes_L2")) for row in match_rows)},
        {"stage": "passes_L3", "rows": sum(_truth(row.get("passes_L3")) for row in match_rows)},
        {"stage": "passes_L4", "rows": sum(_truth(row.get("passes_L4")) for row in match_rows)},
        {"stage": "selected", "rows": sum(_truth(row.get("selected")) for row in match_rows)},
        {"stage": "reinforced", "rows": sum(_truth(row.get("reinforced")) for row in match_rows)},
    ]
    _write(analysis_dir / "stage_funnel.csv", stage_funnel)

    l2_by_field = _write_group_summary(analysis_dir / "l2_only_by_field.csv", l2_rows, key="field")
    l2_with_range = [dict(row, record_range=_record_range(row.get("actual_record_index"))) for row in l2_rows]
    _write(analysis_dir / "l2_only_by_record_range.csv", l2_with_range)
    _write_group_summary(analysis_dir / "l2_only_by_range_summary.csv", l2_with_range, key="record_range")
    _write_quality_summary(analysis_dir / "context_quality_by_field.csv", quality_rows, key="field")
    _write_quality_summary(analysis_dir / "context_quality_by_record_range.csv", [
        dict(row, record_range=_record_range(row.get("actual_record_index"))) for row in quality_rows
    ], key="record_range")
    _write_group_summary(analysis_dir / "temporal_context_degradation.csv", [
        dict(row, record_range=_record_range(row.get("actual_record_index"))) for row in match_rows
    ], key="record_range")
    _write_group_summary(analysis_dir / "l2_only_by_reinforcement.csv", l2_rows, key="segment_reinforcement_count")

    _write(analysis_dir / "source_frequency_by_field.csv", source_field_rows)

    prediction_error_by_index = {
        int(_number(row.get("input_index"))): _number(row.get("normalized_absolute_error"))
        for row in prediction_rows
    }
    error_join = []
    for row in l2_rows:
        index = int(_number(row.get("actual_record_index")))
        error_join.append({
            **row,
            "step1_normalized_error": prediction_error_by_index.get(index, ""),
        })
    known_errors = sorted(_number(row["step1_normalized_error"]) for row in error_join if row["step1_normalized_error"] != "")
    low_cut = known_errors[int(0.25 * (len(known_errors) - 1))] if known_errors else 0.0
    high_cut = known_errors[int(0.75 * (len(known_errors) - 1))] if known_errors else 0.0
    for row in error_join:
        error = row["step1_normalized_error"]
        row["posthoc_error_group"] = (
            "UNKNOWN" if error == "" else "LOW_ERROR" if _number(error) <= low_cut else "HIGH_ERROR" if _number(error) >= high_cut else "MEDIUM_ERROR"
        )
    _write(analysis_dir / "l2_only_error_association.csv", error_join)
    error_association = _write_group_summary(analysis_dir / "l2_only_error_association_summary.csv", error_join, key="posthoc_error_group")

    target_false_status = "unavailable_without_oracle_candidate_trace"
    target_false_path = run_dir / "oracle_candidate_trace.csv"
    if target_false_path.exists() or (run_dir / "oracle_candidate_trace.csv.gz").exists():
        target_false_status = "available_in_existing_oracle_trace; not used for selection"

    low_rows = [row for row in error_join if row["posthoc_error_group"] == "LOW_ERROR"]
    high_rows = [row for row in error_join if row["posthoc_error_group"] == "HIGH_ERROR"]
    low_idf = [_number(row.get("overlap_mean_idf")) for row in low_rows]
    high_idf = [_number(row.get("overlap_mean_idf")) for row in high_rows]
    low_common = [_number(row.get("overlap_common_fraction")) for row in low_rows]
    high_common = [_number(row.get("overlap_common_fraction")) for row in high_rows]
    idf_ci = _bootstrap_difference_ci(low_idf, high_idf)
    common_ci = _bootstrap_difference_ci(low_common, high_common)

    l2_rare = _mean(l2_rows, "overlap_rare_fraction")
    l2_common = _mean(l2_rows, "overlap_common_fraction")
    l2_very_common = _mean(l2_rows, "overlap_very_common_fraction")
    l2_ambiguous = _rate(l2_rows, "ambiguous")
    if l2_rare > l2_common + 0.10 and l2_ambiguous < 0.50:
        hypothesis = "H1_SPARSE_MEANINGFUL_MATCHING"
    elif l2_common + l2_very_common > l2_rare + 0.20 and l2_ambiguous >= 0.50:
        hypothesis = "H2_SHARED_CONTEXT_AMBIGUITY"
    else:
        hypothesis = "H3_MIXED_REGIME"

    report = {
        "run_dir": str(run_dir),
        "rows": {
            "match": len(match_rows),
            "l2_only": len(l2_rows),
            "source": source_row_count,
            "quality": len(quality_rows),
            "events": len(event_rows),
        },
        "stage_funnel": stage_funnel,
        "l2_only": {
            "mean_idf": _mean(l2_rows, "overlap_mean_idf"),
            "mean_incidence": _mean(l2_rows, "overlap_mean_incidence"),
            "rare_fraction": l2_rare,
            "common_fraction": l2_common,
            "very_common_fraction": l2_very_common,
            "ambiguous_rate": l2_ambiguous,
            "fields": l2_by_field,
        },
        "posthoc_error_association": {
            "low_minus_high_mean_idf_ci95": idf_ci,
            "low_minus_high_common_fraction_ci95": common_ci,
            "low_rows": len(low_rows),
            "high_rows": len(high_rows),
            "warning": "association only; Step1 error was not used by the model",
        },
        "target_vs_false": {"status": target_false_status},
        "hypothesis": hypothesis,
        "strict_invariants": {
            "diagnostic_only": True,
            "selection_behavior_changed": False,
            "prediction_behavior_changed": False,
            "learning_behavior_changed": False,
            "ground_truth_used_for_selection": False,
        },
    }
    (analysis_dir / "segment_context_analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _markdown(report: Mapping[str, object]) -> str:
    l2 = report["l2_only"]
    assoc = report["posthoc_error_association"]
    return "\n".join(
        [
            "# Fig.9 Segment Context Composition Analysis",
            "",
            f"- Run: `{report['run_dir']}`",
            f"- Primary hypothesis: **{report['hypothesis']}**",
            "- All error grouping is posthoc association; it never changes model selection.",
            "",
            "## L2-only context",
            f"- rows: {report['rows']['l2_only']}",
            f"- mean overlap source IDF: {l2['mean_idf']:.6f}",
            f"- mean source incidence: {l2['mean_incidence']:.6f}",
            f"- rare/common/very-common fraction: {l2['rare_fraction']:.4f} / {l2['common_fraction']:.4f} / {l2['very_common_fraction']:.4f}",
            f"- ambiguous rate: {l2['ambiguous_rate']:.4f}",
            "",
            "## Error association",
            f"- LOW minus HIGH IDF bootstrap CI95: {assoc['low_minus_high_mean_idf_ci95']}",
            f"- LOW minus HIGH common-fraction bootstrap CI95: {assoc['low_minus_high_common_fraction_ci95']}",
            f"- target-vs-false status: {report['target_vs_false']['status']}",
            "",
            "## Interpretation boundary",
            "This report diagnoses representation and sharedness. It does not claim that an IDF weighting or gating mechanism is a strict paper reproduction.",
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    report = analyze(args.run_dir, args.output_dir)
    output_dir = args.output_dir or args.run_dir / "analysis"
    (output_dir / "FIG9_SEGMENT_CONTEXT_REPRESENTATION_REPORT.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
