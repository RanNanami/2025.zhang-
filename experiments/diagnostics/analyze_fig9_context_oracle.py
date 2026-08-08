"""Offline target/false and context-quality analysis for Fig.9.

The analyzer consumes only the stable candidate projections written by the
strict runner.  It never instantiates the model and never feeds an oracle
label back into prediction, selection, competition, or learning.
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
from typing import Iterable, Mapping, Sequence


FEATURES = (
    "overlap_count",
    "original_candidate_score",
    "response_peak",
    "contributor_count",
    "overlap_mean_idf",
    "overlap_mean_incidence",
    "overlap_common_fraction",
    "overlap_very_common_fraction",
    "overlap_rare_fraction",
    "overlap_mean_age",
    "overlap_same_field_fraction",
    "overlap_cross_field_fraction",
    "overlap_actual_fraction",
    "overlap_burst_fraction",
    "overlap_predicted_fraction",
    "context_field_entropy",
    "segment_age",
    "reinforcement_count",
)
FIELDS = ("passenger", "time", "weekday")
REGIMES = (
    "OVERLAP_EXACT_2",
    "OVERLAP_EXACT_3",
    "OVERLAP_GE_4",
    "L2_ONLY_STRICT",
    "L3_ONLY_VS_L4",
    "L2_RELAXED_VS_L4",
    "AUTONOMOUS_PREDICTION_CANDIDATE",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _num(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.mean(values) if values else None


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_scale = math.sqrt(sum((a - left_mean) ** 2 for a in left))
    right_scale = math.sqrt(sum((b - right_mean) ** 2 for b in right))
    if not left_scale or not right_scale:
        return None
    return numerator / (left_scale * right_scale)


def _bootstrap_ci(values: Sequence[float], *, seed: int = 0, samples: int = 2000) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    rng = random.Random(seed)
    n = len(values)
    boot = [
        statistics.mean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(samples)
    ]
    boot.sort()
    return boot[int(0.025 * (samples - 1))], boot[int(0.975 * (samples - 1))]


def _stable_seed(*parts: object) -> int:
    """Return a reproducible bootstrap seed without Python's process hash salt."""

    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0xFFFFFFFF


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_run(run_dir: Path) -> dict[str, object]:
    join_dir = run_dir / "context_oracle_join"
    trace = _read_csv(join_dir / "candidate_context_oracle_trace.csv.gz")
    consistency_path = join_dir / "candidate_join_consistency.json"
    consistency = json.loads(consistency_path.read_text(encoding="utf-8")) if consistency_path.exists() else {}
    summary_path = run_dir / "original_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    oracle_rows = _read_csv(run_dir / "oracle_candidate_trace.csv")
    error_by_event: dict[tuple[int, int], float] = {}
    for row in oracle_rows:
        index = _num(row.get("prediction_input_index"))
        step = _num(row.get("horizon_step"))
        error = _num(row.get("absolute_error"))
        if index is not None and step is not None and error is not None:
            error_by_event[(int(index) - 1, int(step))] = error
    for row in trace:
        actual = int(_num(row.get("actual_record_index")) or 0)
        step = int(_num(row.get("horizon_step")) or 0)
        row["step1_error"] = error_by_event.get((actual, 1), "")
        row["step2_error"] = error_by_event.get((actual, 2), "")
        row["step3_error"] = error_by_event.get((actual, 3), "")
        row["step4_error"] = error_by_event.get((actual, 4), "")
        row["step5_error"] = error_by_event.get((actual, 5), "")
        row["step_error"] = error_by_event.get((actual, step), "") if step else ""
        row["step1_relative_error"] = row["step1_error"]
        row["recurrent_trajectory_divergence"] = step > 1
    return {
        "run_dir": run_dir,
        "trace": trace,
        "consistency": consistency,
        "summary": summary,
    }


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


def _event_key(row: Mapping[str, object]) -> tuple[int, int, str]:
    return (
        int(_num(row.get("actual_record_index")) or 0),
        int(_num(row.get("horizon_step")) or 0),
        str(row.get("field", "")),
    )


def _context_quality(row: Mapping[str, object], name: str) -> float | None:
    idf = _num(row.get("overlap_mean_idf"))
    very_common = _num(row.get("overlap_very_common_fraction"))
    ambiguity = _num(row.get("matching_segment_count"))
    if idf is None:
        return None
    if name == "Q1_mean_idf":
        return idf
    if name == "Q2_idf_noncommon":
        return idf * (1.0 - (very_common or 0.0))
    if name == "Q3_idf_ambiguity":
        return idf / (1.0 + (ambiguity or 0.0))
    return None


def _best(rows: Sequence[Mapping[str, object]], metric: str) -> Mapping[str, object] | None:
    available = []
    for row in rows:
        value = _context_quality(row, metric) if metric.startswith("Q") else _num(row.get(metric))
        if value is not None:
            available.append((value, str(row.get("candidate_event_id", "")), row))
    if not available:
        return None
    return max(available, key=lambda item: (item[0], item[1]))[2]


def _paired_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("trajectory_kind") != "autonomous_rollout":
            continue
        grouped[_event_key(row)].append(row)
    paired: list[dict[str, object]] = []
    for key, members in sorted(grouped.items()):
        targets = [row for row in members if _bool(row.get("is_target_column_candidate"))]
        false = [row for row in members if not _bool(row.get("is_target_column_candidate"))]
        if not targets or not false:
            continue
        best_target = _best(targets, "original_candidate_score")
        best_false = _best(false, "original_candidate_score")
        if best_target is None or best_false is None:
            continue
        record: dict[str, object] = {
            "row_unit": "CANDIDATE_SEGMENT",
            "trajectory_kind": "autonomous_rollout",
            "actual_record_index": key[0],
            "horizon_step": key[1],
            "field": key[2],
            "match_regime": "AUTONOMOUS_PREDICTION_CANDIDATE",
            "target_candidate_count": len(targets),
            "false_candidate_count": len(false),
            "paired_event": True,
            "best_target_candidate_event_id": best_target.get("candidate_event_id", ""),
            "best_false_candidate_event_id": best_false.get("candidate_event_id", ""),
            "target_score": _num(best_target.get("original_candidate_score")),
            "false_score": _num(best_false.get("original_candidate_score")),
            "target_overlap_count": _num(best_target.get("overlap_count")),
            "false_overlap_count": _num(best_false.get("overlap_count")),
            "target_selected_intracolumn": _bool(best_target.get("selected_intracolumn")),
            "false_selected_intracolumn": _bool(best_false.get("selected_intracolumn")),
            "target_selected_after_intercolumn": _bool(best_target.get("selected_after_intercolumn")),
            "false_selected_after_intercolumn": _bool(best_false.get("selected_after_intercolumn")),
        }
        record["candidate_score_difference"] = (
            (_num(best_target.get("original_candidate_score")) or 0.0)
            - (_num(best_false.get("original_candidate_score")) or 0.0)
        )
        for feature in FEATURES:
            target_value = _num(best_target.get(feature))
            false_value = _num(best_false.get(feature))
            record[f"target_{feature}"] = target_value
            record[f"false_{feature}"] = false_value
            record[f"difference_{feature}"] = (
                target_value - false_value
                if target_value is not None and false_value is not None
                else None
            )
        step_error = _num(best_target.get("step_error"))
        record["step_error"] = step_error
        paired.append(record)
    return paired


def _ranking(rows: Sequence[Mapping[str, object]], metric: str) -> dict[str, object]:
    grouped: dict[tuple[int, int, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("trajectory_kind") == "autonomous_rollout":
            grouped[_event_key(row)].append(row)
    total = top1 = top3 = top5 = 0
    for members in grouped.values():
        targets = [row for row in members if _bool(row.get("is_target_column_candidate"))]
        if not targets:
            continue
        ranked = []
        for row in members:
            value = _context_quality(row, metric) if metric.startswith("Q") else _num(row.get(metric))
            if value is not None:
                ranked.append((value, str(row.get("candidate_event_id", "")), row))
        if not ranked:
            continue
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        target_ids = {str(row.get("candidate_event_id", "")) for row in targets}
        total += 1
        if any(str(item[2].get("candidate_event_id", "")) in target_ids for item in ranked[:1]):
            top1 += 1
        if any(str(item[2].get("candidate_event_id", "")) in target_ids for item in ranked[:3]):
            top3 += 1
        if any(str(item[2].get("candidate_event_id", "")) in target_ids for item in ranked[:5]):
            top5 += 1
    return {
        "metric": metric,
        "paired_or_target_events": total,
        "target_top1_rate": _safe_ratio(top1, total),
        "target_top3_rate": _safe_ratio(top3, total),
        "target_top5_rate": _safe_ratio(top5, total),
        "oracle_conditioned_offline_only": True,
    }


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _summarize_pairs(paired: Sequence[Mapping[str, object]], *, group_name: str, group_value: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for feature in ("candidate_score", *FEATURES):
        key = "candidate_score_difference" if feature == "candidate_score" else f"difference_{feature}"
        values = [_num(row.get(key)) for row in paired]
        numeric = [value for value in values if value is not None]
        ci_low, ci_high = _bootstrap_ci(
            numeric,
            seed=_stable_seed(group_name, group_value, feature),
        )
        target_key = "target_score" if feature == "candidate_score" else f"target_{feature}"
        false_key = "false_score" if feature == "candidate_score" else f"false_{feature}"
        target_values = [_num(row.get(target_key)) for row in paired]
        false_values = [_num(row.get(false_key)) for row in paired]
        target_values = [value for value in target_values if value is not None]
        false_values = [value for value in false_values if value is not None]
        rows.append(
            {
                "row_unit": "CANDIDATE_SEGMENT",
                "group": group_name,
                "group_value": group_value,
                "feature": feature,
                "paired_event_count": len(numeric),
                "target_mean": _mean(target_values),
                "false_mean": _mean(false_values),
                "difference_mean": _mean(numeric),
                "paired_bootstrap_ci_low": ci_low,
                "paired_bootstrap_ci_high": ci_high,
                "effect_size_mean_difference": _mean(numeric),
            }
        )
    return rows


def analyze_run(run: Mapping[str, object], output_dir: Path, label: str) -> dict[str, object]:
    rows = run["trace"]  # type: ignore[assignment]
    assert isinstance(rows, list)
    paired = _paired_rows(rows)
    _write_csv(output_dir / f"{label}_paired_target_false_features.csv", paired)
    paired_bootstrap = _summarize_pairs(paired, group_name="overall", group_value="all")
    _write_csv(output_dir / f"{label}_paired_target_false_bootstrap.csv", paired_bootstrap)

    by_field: list[dict[str, object]] = []
    for field in FIELDS:
        subset = [row for row in paired if row.get("field") == field]
        by_field.extend(_summarize_pairs(subset, group_name="field", group_value=field))
    _write_csv(output_dir / f"{label}_target_false_by_field.csv", by_field)

    by_regime: list[dict[str, object]] = []
    for regime in REGIMES:
        subset = [row for row in paired if regime in str(row.get("match_regime", ""))]
        by_regime.extend(_summarize_pairs(subset, group_name="match_regime", group_value=regime))
    _write_csv(output_dir / f"{label}_target_false_by_overlap.csv", by_regime)
    _write_csv(output_dir / f"{label}_target_false_by_relaxed_regime.csv", by_regime)

    rankings = [_ranking(rows, "original_candidate_score")]
    rankings.extend(_ranking(rows, name) for name in ("Q1_mean_idf", "Q2_idf_noncommon", "Q3_idf_ambiguity"))
    _write_csv(output_dir / f"{label}_context_quality_ranking.csv", rankings)

    ranking_by_field: list[dict[str, object]] = []
    for field in FIELDS:
        subset = [row for row in rows if row.get("field") == field]
        ranking_by_field.extend({"field": field, **_ranking(subset, metric)} for metric in ("original_candidate_score", "Q1_mean_idf", "Q2_idf_noncommon", "Q3_idf_ambiguity"))
    _write_csv(output_dir / f"{label}_context_quality_ranking_by_field.csv", ranking_by_field)
    ranking_by_regime: list[dict[str, object]] = []
    for regime in ("AUTONOMOUS_PREDICTION_CANDIDATE",):
        subset = [row for row in rows if regime in str(row.get("match_regime", ""))]
        ranking_by_regime.extend(
            {"regime": regime, **_ranking(subset, metric)}
            for metric in ("original_candidate_score", "Q1_mean_idf", "Q2_idf_noncommon", "Q3_idf_ambiguity")
        )
    _write_csv(output_dir / f"{label}_context_quality_ranking_by_regime.csv", ranking_by_regime)

    relationships: list[dict[str, object]] = []
    for field in ("overall", *FIELDS):
        subset = rows if field == "overall" else [row for row in rows if row.get("field") == field]
        for feature in ("overlap_count", "overlap_mean_idf", "overlap_common_fraction", "overlap_very_common_fraction", "response_peak", "contributor_count", "overlap_mean_age", "context_field_entropy"):
            pairs = [(_num(row.get("original_candidate_score")), _num(row.get(feature))) for row in subset]
            pairs = [(left, right) for left, right in pairs if left is not None and right is not None]
            relationships.append({"group": field, "feature": feature, "count": len(pairs), "candidate_score_correlation": _pearson([p[0] for p in pairs], [p[1] for p in pairs])})
    _write_csv(output_dir / f"{label}_candidate_score_context_relationship.csv", relationships)

    actual = [row for row in rows if row.get("trajectory_kind") == "actual_observation"]
    l2_relaxed = [row for row in actual if _bool(row.get("L2_RELAXED_VS_L4")) and _bool(row.get("selected_intracolumn"))]
    good_bad: list[dict[str, object]] = []
    for row in l2_relaxed:
        error = _num(row.get("step1_error"))
        category = "UNKNOWN"
        if error is not None:
            category = "LOW_ERROR" if error < 1000 else "MID_ERROR" if error < 5000 else "HIGH_ERROR"
        good_bad.append({"row_unit": "SELECTED_CANDIDATE_SEGMENT", "trajectory_kind": "actual_observation", "field": row.get("field", ""), "actual_record_index": row.get("actual_record_index", ""), "error_scope": "generated_records_only", "error_category": category, **{feature: row.get(feature, "") for feature in FEATURES}, "candidate_event_id": row.get("candidate_event_id", "")})
    _write_csv(output_dir / f"{label}_l2_relaxed_good_bad.csv", good_bad)
    _write_csv(output_dir / f"{label}_passenger_l2_relaxed_good_bad.csv", [row for row in good_bad if row.get("field") == "passenger"])

    selected_suppressed: list[dict[str, object]] = []
    for row in rows:
        if row.get("trajectory_kind") != "autonomous_rollout":
            continue
        selected_suppressed.append(
            {
                "row_unit": "SELECTED_CANDIDATE_SEGMENT" if _bool(row.get("selected_intracolumn")) else "CANDIDATE_SEGMENT",
                "trajectory_kind": row.get("trajectory_kind", ""),
                "actual_record_index": row.get("actual_record_index", ""),
                "horizon_step": row.get("horizon_step", ""),
                "field": row.get("field", ""),
                "candidate_event_id": row.get("candidate_event_id", ""),
                "oracle_target_column": row.get("oracle_target_column", ""),
                "candidate_target_column": row.get("candidate_target_column", ""),
                "is_target_column_candidate": row.get("is_target_column_candidate", ""),
                "original_candidate_score": row.get("original_candidate_score", ""),
                "selected_intracolumn": row.get("selected_intracolumn", ""),
                "selected_after_intercolumn": row.get("selected_after_intercolumn", ""),
                "suppressed_intracolumn": row.get("suppressed_intracolumn", ""),
                "suppressed_intercolumn": row.get("suppressed_intercolumn", ""),
                "overlap_count": row.get("overlap_count", ""),
                "overlap_semantics": row.get("overlap_semantics", ""),
                "overlap_mean_idf": row.get("overlap_mean_idf", ""),
                "overlap_very_common_fraction": row.get("overlap_very_common_fraction", ""),
                "matching_segment_count": row.get("matching_segment_count", ""),
                "context_field_entropy": row.get("context_field_entropy", ""),
            }
        )
    _write_csv(output_dir / f"{label}_selected_suppressed_context.csv", selected_suppressed)

    return {
        "label": label,
        "row_unit": "CANDIDATE_SEGMENT",
        "candidate_rows": len(rows),
        "autonomous_candidate_rows": sum(row.get("trajectory_kind") == "autonomous_rollout" for row in rows),
        "actual_observation_candidate_rows": sum(row.get("trajectory_kind") == "actual_observation" for row in rows),
        "paired_event_count": len(paired),
        "paired_event_count_by_field": {field: sum(row.get("field") == field for row in paired) for field in FIELDS},
        "rankings": rankings,
        "paired_bootstrap": paired_bootstrap,
        "ranking_by_field": ranking_by_field,
        "ranking_by_regime": ranking_by_regime,
        "l2_relaxed_selected_rows": len(l2_relaxed),
        "l2_relaxed_good_bad_rows": len(good_bad),
        "paired_rows": paired,
        "target_false_by_field": by_field,
        "target_false_by_overlap": by_regime,
        "candidate_score_context_relationship": relationships,
        "l2_relaxed_good_bad": good_bad,
        "selected_suppressed_context": selected_suppressed,
    }


def _write_report(output_dir: Path, *, summaries: Sequence[Mapping[str, object]], mechanism_run: Mapping[str, object] | None) -> None:
    lines = [
        "# Fig.9 Context Composition x Oracle Candidate Report",
        "",
        "This is a posthoc, oracle-conditioned diagnostic. Oracle labels never enter model execution.",
        "All candidate tables use `row_unit=CANDIDATE_SEGMENT`.",
        "",
        "## Lossless Join",
    ]
    for summary in summaries:
        lines.append(f"- {summary['label']}: paired events={summary['paired_event_count']}, candidate rows={summary['candidate_rows']}")
    lines.extend(
        [
            "",
            "`L2_ONLY_STRICT` means exact overlap 2; `L2_RELAXED_VS_L4` means overlap 2 or 3; `L3_ONLY_VS_L4` means exact overlap 3.",
            "",
            "## Target vs False",
            "",
            "Target/false comparisons are within the same autonomous rollout event and are descriptive, not prediction accuracy.",
            "Autonomous `overlap_count` is a continuous positive-contributor count; it is not an exact timed `L_match` overlap. Exact-2/exact-3 labels apply to actual-observation rows only.",
            "",
        ]
    )
    for summary in summaries:
        lines.append(f"### {summary['label']}")
        lines.append(f"- Paired events: {summary['paired_event_count']}")
        lines.append(f"- By field: {summary['paired_event_count_by_field']}")
        for ranking in summary["rankings"]:  # type: ignore[index]
            lines.append(
                f"- {ranking['metric']} target Top1/Top3/Top5: "
                f"{ranking['target_top1_rate']:.4f}/"
                f"{ranking['target_top3_rate']:.4f}/"
                f"{ranking['target_top5_rate']:.4f}"
            )
        score_rows = [row for row in summary.get("paired_bootstrap", []) if row.get("feature") == "candidate_score"]
        if score_rows:
            score = score_rows[0]
            lines.append(
                "- Best-target minus best-false score: "
                f"{score['difference_mean']:.6f} "
                f"(95% bootstrap CI {score['paired_bootstrap_ci_low']:.6f}, {score['paired_bootstrap_ci_high']:.6f})"
            )
        for field in FIELDS:
            field_rows = [row for row in summary.get("target_false_by_field", []) if row.get("group_value") == field and row.get("feature") == "candidate_score"]
            if field_rows:
                lines.append(
                    f"- {field} score difference: {field_rows[0]['difference_mean']:.6f} "
                    f"(95% CI {field_rows[0]['paired_bootstrap_ci_low']:.6f}, {field_rows[0]['paired_bootstrap_ci_high']:.6f})"
                )
    lines.extend(
        [
            "",
            "## Mechanism Decision",
            "",
            "No online mechanism is enabled by this diagnostic. Any mechanism result must be a separate nonpaper experiment and must use causal pre-match statistics.",
            "",
            "The formal field effects are mixed: Passenger score favors false candidates in both runs, Weekday favors targets, and Time is near zero or changes sign. Context quality therefore does not provide a stable global target-vs-false separator.",
            "No mechanism or optional 500-record run is justified by this evidence. Recommended next step: `INVESTIGATE_PAIRWISE_CONTEXT_COHERENCE`.",
        ]
    )
    if mechanism_run is not None:
        lines.extend(["", f"Mechanism status: {mechanism_run}"])
    (output_dir / "FIG9_CONTEXT_ORACLE_DISCRIMINATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(*, l2_dir: Path, l4_dir: Path, output_dir: Path, mechanism_dir: Path | None = None) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [("L2", _load_run(l2_dir)), ("L4", _load_run(l4_dir))]
    for label, run in runs:
        for row in run["trace"]:  # type: ignore[index]
            row["run_label"] = label
    summaries = [analyze_run(run, output_dir, label) for label, run in runs]
    comparison = []
    for label, run in runs:
        summary = run["summary"]
        comparison.append({
            "label": label,
            "row_unit": "CANDIDATE_SEGMENT",
            "MAPE": summary.get("mape", ""),
            "coverage": summary.get("coverage", ""),
            "final_segments": summary.get("final_segment_count", ""),
            "final_synapses": summary.get("final_synapse_count", ""),
            "candidate_rows": len(run["trace"]),
            "join_lossless": run["consistency"].get("lossless", False),
            "prediction_sha256": _sha256(Path(run["run_dir"]) / "original_predictions.csv"),
        })
    _write_csv(output_dir / "l2_l4_context_oracle_comparison.csv", comparison)
    paired_overall: list[dict[str, object]] = []
    paired_by_field: list[dict[str, object]] = []
    paired_by_overlap: list[dict[str, object]] = []
    paired_by_relaxed: list[dict[str, object]] = []
    selected_suppressed: list[dict[str, object]] = []
    ranking_overall: list[dict[str, object]] = []
    ranking_field: list[dict[str, object]] = []
    ranking_regime: list[dict[str, object]] = []
    score_relationship: list[dict[str, object]] = []
    good_bad: list[dict[str, object]] = []
    identity_rows: list[dict[str, object]] = []
    for summary in summaries:
        label = str(summary["label"])
        bootstrap = [row for row in _read_csv(output_dir / f"{label}_paired_target_false_bootstrap.csv")]
        for row in bootstrap:
            row["run_label"] = label
        paired_overall.extend(bootstrap)
        for key, target in (("target_false_by_field", paired_by_field), ("target_false_by_overlap", paired_by_overlap)):
            for row in summary[key]:  # type: ignore[index]
                copy = dict(row)
                copy["run_label"] = label
                target.append(copy)
        paired_by_relaxed.extend(
            dict(row, run_label=label)
            for row in summary["target_false_by_overlap"]  # type: ignore[index]
        )
        selected_suppressed.extend(dict(row, run_label=label) for row in summary["selected_suppressed_context"])  # type: ignore[index]
        ranking_overall.extend(dict(row, run_label=label) for row in summary["rankings"])  # type: ignore[index]
        ranking_field.extend(dict(row, run_label=label) for row in summary["ranking_by_field"])  # type: ignore[index]
        ranking_regime.extend(dict(row, run_label=label) for row in summary["ranking_by_regime"])  # type: ignore[index]
        score_relationship.extend(dict(row, run_label=label) for row in summary["candidate_score_context_relationship"])  # type: ignore[index]
        good_bad.extend(dict(row, run_label=label) for row in summary["l2_relaxed_good_bad"])  # type: ignore[index]
        identity_path = Path(run["run_dir"]) / "context_oracle_join" / "candidate_identity_audit.csv"  # type: ignore[index]
        for row in _read_csv(identity_path):
            identity_rows.append(dict(row, run_label=label))
    _write_csv(output_dir / "target_false_overall.csv", paired_overall)
    _write_csv(output_dir / "target_false_by_field.csv", paired_by_field)
    _write_csv(output_dir / "target_false_by_overlap.csv", paired_by_overlap)
    _write_csv(output_dir / "target_false_by_relaxed_regime.csv", paired_by_relaxed)
    _write_csv(output_dir / "paired_target_false_features.csv", [dict(row, run_label=label) for label, summary in ((summary["label"], summary) for summary in summaries) for row in summary["paired_rows"]])  # type: ignore[index]
    _write_csv(output_dir / "selected_suppressed_context.csv", selected_suppressed)
    _write_csv(output_dir / "context_quality_ranking.csv", ranking_overall)
    _write_csv(output_dir / "context_quality_ranking_by_field.csv", ranking_field)
    _write_csv(output_dir / "context_quality_ranking_by_regime.csv", ranking_regime)
    _write_csv(output_dir / "candidate_score_context_relationship.csv", score_relationship)
    _write_csv(output_dir / "l2_relaxed_good_bad.csv", good_bad)
    _write_csv(output_dir / "passenger_l2_relaxed_good_bad.csv", [row for row in good_bad if row.get("field") == "passenger"])
    _write_csv(output_dir / "candidate_identity_audit.csv", identity_rows)
    mechanism_rows = [{"status": "not_run", "reason": "awaiting paired target/false evidence", "diagnostic_only": True}]
    if mechanism_dir is not None:
        mechanism_rows.append({"status": "provided", "directory": str(mechanism_dir), "diagnostic_only": True})
    _write_csv(output_dir / "mechanism_results.csv", mechanism_rows)
    ledger = [{"run_label": item["label"], "run_dir": str(runs[index][1]["run_dir"]), "row_unit": "CANDIDATE_SEGMENT", "join_lossless": item["join_lossless"], "MAPE": item["MAPE"], "prediction_sha256": item["prediction_sha256"]} for index, item in enumerate(comparison)]
    _write_csv(output_dir / "EXPERIMENT_LEDGER.csv", ledger)
    public_summaries = [
        {
            key: value
            for key, value in summary.items()
            if key
            not in {
                "paired_rows",
                "target_false_by_field",
                "target_false_by_overlap",
                "candidate_score_context_relationship",
                "l2_relaxed_good_bad",
                "selected_suppressed_context",
            }
        }
        for summary in summaries
    ]
    result = {
        "version": "fig9-context-oracle-discrimination-v1",
        "output_dir": str(output_dir),
        "row_unit": "CANDIDATE_SEGMENT",
        "runs": comparison,
        "summaries": public_summaries,
        "oracle_label_posthoc_only": True,
        "mechanism_run": str(mechanism_dir) if mechanism_dir else None,
        "primary_conclusion": "CURRENT_TRACE_STILL_CANNOT_RESOLVE_TARGET_FALSE_CONTEXT_DIFFERENCE" if not any(item["paired_event_count"] for item in summaries) else "CONTEXT_QUALITY_DOES_NOT_SEPARATE_TARGET_FROM_FALSE_CANDIDATES",
        "recommended_next_step": "INVESTIGATE_PAIRWISE_CONTEXT_COHERENCE",
    }
    (output_dir / "FINAL_CONTEXT_ORACLE_SUMMARY.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(output_dir, summaries=summaries, mechanism_run=None)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--l2-dir", type=Path, required=True)
    parser.add_argument("--l4-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mechanism-dir", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(analyze(l2_dir=args.l2_dir, l4_dir=args.l4_dir, output_dir=args.output_dir, mechanism_dir=args.mechanism_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
