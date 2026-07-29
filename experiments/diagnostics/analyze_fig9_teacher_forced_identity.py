"""Offline alignment of autonomous Fig.9 candidates to future observations."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from experiments.diagnostics.fig9_teacher_forced_identity import (
    DIAGNOSTIC_MARKERS,
    winner_set,
)


CLASSIFICATIONS = {
    "A": "TARGET_COLUMN_ABSENT_FROM_PRESELECTION",
    "B": "TARGET_COLUMN_BELOW_THRESHOLD",
    "C": "TARGET_COLUMN_NO_VALID_FIRING_TIME",
    "D": "TARGET_COLUMN_CANDIDATE_MISSING_REFERENCE_NEURON",
    "E": "REFERENCE_NEURON_PRESENT_BUT_NOT_SELECTED",
    "F": "REFERENCE_NEURON_CANDIDATE_BUT_SUPPRESSED",
    "G": "TARGET_COLUMN_EMITTED_WRONG_NEURON",
    "H": "REFERENCE_NEURON_EMITTED",
    "I": "REFERENCE_UNAVAILABLE",
}


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _float(value: object) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: object) -> int | None:
    try:
        if value in ("", None):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _cell_set(value: object) -> set[int]:
    raw = str(value or "").strip()
    return {int(item) for item in raw.split()} if raw else set()


def _jaccard(left: set[int], right: set[int]) -> float | None:
    if not left and not right:
        return None
    return len(left & right) / len(left | right)


def _find_trace(run_dir: Path, stem: str) -> Path | None:
    for name in (f"{stem}.csv", f"{stem}.csv.gz"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    return None


def _read_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
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
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_creation_sources(run_dir: Path) -> dict[str, set[int]]:
    paths = sorted(run_dir.glob("*.branch_provenance.json"))
    result: dict[str, set[int]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("segments", ()):
            segment_id = str(row.get("segment_provenance_id", ""))
            if segment_id:
                result[segment_id] = {
                    int(value)
                    for value in row.get("creation_source_cell_ids", ())
                }
    return result


def _segment_key(row: Mapping[str, object]) -> tuple[int, str, int, int]:
    return (
        int(row["input_index"]),
        str(row["target_timestamp"]),
        int(row["horizon_step"]),
        int(row["target_column"]),
    )


def classify_identity(
    *,
    funnel_outcome: str,
    reference_available: bool,
    target_column_candidate: bool,
    target_column_emitted: bool,
    reference_neuron_crossed: bool,
    reference_neuron_candidate: bool,
    reference_neuron_emitted: bool,
) -> str:
    """Classify one target column using observed pipeline stages only."""

    if not reference_available:
        return CLASSIFICATIONS["I"]
    if funnel_outcome == "TARGET_NO_SEGMENT_INSPECTED":
        return CLASSIFICATIONS["A"]
    if funnel_outcome == "TARGET_SEGMENT_BELOW_THRESHOLD":
        return CLASSIFICATIONS["B"]
    if funnel_outcome == "TARGET_SEGMENT_CROSSED_BUT_NO_VALID_FIRING_TIME":
        return CLASSIFICATIONS["C"]
    if reference_neuron_emitted:
        return CLASSIFICATIONS["H"]
    if target_column_emitted:
        return CLASSIFICATIONS["G"]
    if reference_neuron_candidate:
        return CLASSIFICATIONS["F"]
    if reference_neuron_crossed:
        return CLASSIFICATIONS["E"]
    if target_column_candidate:
        return CLASSIFICATIONS["D"]
    return CLASSIFICATIONS["E"]


def align_identity_rows(
    *,
    observation_rows: Sequence[Mapping[str, object]],
    funnel_rows: Sequence[Mapping[str, object]],
    segment_rows: Sequence[Mapping[str, object]],
    policy_label: str,
    creation_sources_by_segment: Mapping[str, set[int]] | None = None,
) -> list[dict[str, object]]:
    """Join autonomous steps to later real observations by timestamp and column."""

    references = {
        (
            str(row["actual_timestamp"]),
            str(row["field"]),
            int(row["target_column"]),
        ): row
        for row in observation_rows
    }
    segments_by_key: dict[
        tuple[int, str, int, int],
        list[Mapping[str, object]],
    ] = defaultdict(list)
    for row in segment_rows:
        segments_by_key[_segment_key(row)].append(row)
    creation_sources_by_segment = creation_sources_by_segment or {}
    aligned: list[dict[str, object]] = []
    for funnel in funnel_rows:
        input_index = int(funnel["input_index"])
        target_timestamp = str(funnel["target_timestamp"])
        horizon_step = int(funnel["horizon_step"])
        outcomes = json.loads(str(funnel["target_column_outcomes"]))
        for target in outcomes:
            field = str(target["field"])
            column = int(target["column"])
            outcome = str(target["outcome"])
            reference = references.get((target_timestamp, field, column))
            winners = winner_set(reference or {})
            segments = segments_by_key.get(
                (input_index, target_timestamp, horizon_step, column),
                [],
            )
            crossing = [
                row for row in segments if _bool(row.get("crossed_threshold"))
            ]
            candidates = [
                row
                for row in segments
                if _bool(row.get("became_prediction_candidate"))
            ]
            emitted = [
                row
                for row in segments
                if _bool(row.get("emitted_after_competition"))
            ]
            reference_crossing = [
                row
                for row in crossing
                if _int(row.get("target_neuron")) in winners
            ]
            reference_candidates = [
                row
                for row in candidates
                if _int(row.get("target_neuron")) in winners
            ]
            reference_emitted = [
                row
                for row in emitted
                if _int(row.get("target_neuron")) in winners
            ]
            target_column_candidate = bool(candidates) or "BECAME_CANDIDATE" in outcome or outcome == "TARGET_SEGMENT_EMITTED"
            target_column_emitted = bool(emitted) or outcome == "TARGET_SEGMENT_EMITTED"
            target_column_raw = (
                any(_bool(row.get("appeared_in_raw_code")) for row in segments)
                or target_column_candidate
            )
            reference_available = bool(
                reference and _bool(reference.get("observed_winner_available"))
            )
            classification = classify_identity(
                funnel_outcome=outcome,
                reference_available=reference_available,
                target_column_candidate=target_column_candidate,
                target_column_emitted=target_column_emitted,
                reference_neuron_crossed=bool(reference_crossing),
                reference_neuron_candidate=bool(reference_candidates),
                reference_neuron_emitted=bool(reference_emitted),
            )
            selected_rows = (
                reference_emitted
                or reference_candidates
                or emitted
                or candidates
                or reference_crossing
                or crossing
            )
            autonomous = selected_rows[0] if selected_rows else {}
            reference_segment_id = (
                str(reference.get("selected_segment_id", ""))
                if reference
                else ""
            )
            autonomous_segment_id = str(
                autonomous.get("segment_provenance_id", "")
            )
            created_after_observation = bool(
                reference
                and _bool(reference.get("scenario3"))
                and reference.get("created_segment_id")
            )
            exact_segment_applicable = bool(
                reference_segment_id
                and autonomous_segment_id
                and not created_after_observation
            )
            creation_exact = bool(
                reference
                and reference.get("creation_source_fingerprint")
                and autonomous.get("creation_source_fingerprint")
                and reference.get("creation_source_fingerprint")
                == autonomous.get("creation_source_fingerprint")
            )
            current_exact = bool(
                reference
                and reference.get("current_source_fingerprint")
                and autonomous.get("current_source_fingerprint")
                and reference.get("current_source_fingerprint")
                == autonomous.get("current_source_fingerprint")
            )
            teacher_creation = (
                _cell_set(reference.get("creation_source_cell_ids", ""))
                if reference
                else set()
            )
            autonomous_creation = creation_sources_by_segment.get(
                autonomous_segment_id,
                set(),
            )
            expected_target_index = input_index + horizon_step - 1
            actual_target_index = (
                _int(reference.get("actual_record_index"))
                if reference
                else None
            )
            aligned.append(
                {
                    **DIAGNOSTIC_MARKERS,
                    "policy": policy_label,
                    "input_index": input_index,
                    "target_record_index": (
                        actual_target_index
                        if actual_target_index is not None
                        else ""
                    ),
                    "target_timestamp": target_timestamp,
                    "horizon_step": horizon_step,
                    "field": field,
                    "target_column": column,
                    "funnel_outcome": outcome,
                    "classification": classification,
                    "reference_available": reference_available,
                    "winner_set_size": len(winners),
                    "reference_winner_neurons": " ".join(
                        map(str, sorted(winners))
                    ),
                    "target_index_mapping_valid": (
                        actual_target_index == expected_target_index
                        if actual_target_index is not None
                        else ""
                    ),
                    "target_column_crossing_recall": bool(crossing),
                    "target_column_raw_recall": target_column_raw,
                    "target_column_candidate_recall": target_column_candidate,
                    "target_column_emitted_recall": target_column_emitted,
                    "reference_neuron_present_in_inspected": (
                        bool(reference_crossing)
                    ),
                    "reference_neuron_threshold_crossing_recall": (
                        bool(reference_crossing)
                    ),
                    "reference_neuron_candidate_recall": (
                        bool(reference_candidates)
                    ),
                    "reference_neuron_emitted_recall": (
                        bool(reference_emitted)
                    ),
                    "correct_column_wrong_neuron": (
                        target_column_emitted and not reference_emitted
                    ),
                    "autonomous_neuron": autonomous.get("target_neuron", ""),
                    "autonomous_segment_id": autonomous_segment_id,
                    "teacher_forced_segment_id": reference_segment_id,
                    "teacher_forced_observe_scenario": (
                        reference.get("observe_scenario", "")
                        if reference
                        else ""
                    ),
                    "teacher_forced_segment_created_only_after_observation": (
                        created_after_observation
                    ),
                    "exact_segment_match_applicable": (
                        exact_segment_applicable
                    ),
                    "exact_segment_match": (
                        autonomous_segment_id == reference_segment_id
                        if exact_segment_applicable
                        else ""
                    ),
                    "creation_source_fingerprint_exact_match": creation_exact,
                    "creation_source_fingerprint_jaccard": (
                        _jaccard(teacher_creation, autonomous_creation)
                    ),
                    "current_source_fingerprint_exact_match": current_exact,
                    "current_source_fingerprint_jaccard": (
                        1.0 if current_exact else ""
                    ),
                    "same_creation_transition": (
                        str(autonomous.get("creation_transition_index", ""))
                        == str(
                            reference.get(
                                "segment_creation_transition_index",
                                "",
                            )
                        )
                        if reference
                        and autonomous.get("creation_transition_index", "")
                        != ""
                        and reference.get(
                            "segment_creation_transition_index",
                            "",
                        )
                        != ""
                        else ""
                    ),
                    "same_support_class": (
                        autonomous.get("source_support_class")
                        == reference.get("source_support_class")
                        if reference
                        and autonomous.get("source_support_class")
                        and reference.get("source_support_class")
                        else ""
                    ),
                }
            )
    return aligned


def _rate(rows: Sequence[Mapping[str, object]], key: str) -> float:
    values = [
        _bool(row[key])
        for row in rows
        if row.get(key, "") not in ("", None)
    ]
    return sum(values) / len(values) if values else 0.0


def _mean_numeric(rows: Sequence[Mapping[str, object]], key: str) -> float:
    values = [
        value
        for row in rows
        if (value := _float(row.get(key))) is not None
    ]
    return sum(values) / len(values) if values else 0.0


def summarize_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    group_fields: Sequence[str],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    output: list[dict[str, object]] = []
    for group, items in sorted(grouped.items(), key=lambda pair: repr(pair[0])):
        available = [row for row in items if _bool(row["reference_available"])]
        emitted_correct_columns = [
            row
            for row in available
            if _bool(row["target_column_emitted_recall"])
        ]
        candidate_correct_columns = [
            row
            for row in available
            if _bool(row["target_column_candidate_recall"])
        ]
        payload = dict(zip(group_fields, group))
        payload.update(
            {
                "target_column_count": len(items),
                "reference_available_rate": _rate(
                    items,
                    "reference_available",
                ),
                "target_column_raw_recall": _rate(
                    available,
                    "target_column_raw_recall",
                ),
                "target_column_candidate_recall": _rate(
                    available,
                    "target_column_candidate_recall",
                ),
                "target_column_emitted_recall": _rate(
                    available,
                    "target_column_emitted_recall",
                ),
                "reference_neuron_crossing_recall": _rate(
                    available,
                    "reference_neuron_threshold_crossing_recall",
                ),
                "reference_neuron_candidate_recall": _rate(
                    available,
                    "reference_neuron_candidate_recall",
                ),
                "reference_neuron_emitted_recall": _rate(
                    available,
                    "reference_neuron_emitted_recall",
                ),
                "conditional_neuron_match_given_candidate": _rate(
                    candidate_correct_columns,
                    "reference_neuron_candidate_recall",
                ),
                "conditional_neuron_match_given_emitted": _rate(
                    emitted_correct_columns,
                    "reference_neuron_emitted_recall",
                ),
                "wrong_neuron_with_correct_column_rate": _rate(
                    emitted_correct_columns,
                    "correct_column_wrong_neuron",
                ),
                "exact_segment_match_rate": _rate(
                    available,
                    "exact_segment_match",
                ),
                "creation_source_exact_match_rate": _rate(
                    available,
                    "creation_source_fingerprint_exact_match",
                ),
                "creation_source_jaccard": _mean_numeric(
                    available,
                    "creation_source_fingerprint_jaccard",
                ),
                "current_source_exact_match_rate": _rate(
                    available,
                    "current_source_fingerprint_exact_match",
                ),
                "same_creation_transition_rate": _rate(
                    available,
                    "same_creation_transition",
                ),
                "same_support_class_rate": _rate(
                    available,
                    "same_support_class",
                ),
            }
        )
        output.append(payload)
    return output


def bootstrap_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    samples: int,
    seed: int,
) -> list[dict[str, object]]:
    """Bootstrap rollout inputs so columns from one rollout stay together."""

    by_rollout: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_rollout[int(row["input_index"])].append(row)
    rollout_ids = sorted(by_rollout)
    if not rollout_ids or samples <= 0:
        return []
    rng = random.Random(seed)
    metrics = (
        "target_column_emitted_recall",
        "reference_neuron_candidate_recall",
        "reference_neuron_emitted_recall",
        "correct_column_wrong_neuron",
    )
    draws: dict[str, list[float]] = {metric: [] for metric in metrics}
    for _ in range(samples):
        sample_rows: list[Mapping[str, object]] = []
        for _index in rollout_ids:
            selected = rollout_ids[rng.randrange(len(rollout_ids))]
            sample_rows.extend(by_rollout[selected])
        available = [
            row for row in sample_rows if _bool(row["reference_available"])
        ]
        for metric in metrics:
            draws[metric].append(_rate(available, metric))
    output = []
    for metric, values in draws.items():
        values.sort()
        low = values[int(0.025 * (len(values) - 1))]
        high = values[int(0.975 * (len(values) - 1))]
        output.append(
            {
                "metric": metric,
                "bootstrap_samples": samples,
                "bootstrap_seed": seed,
                "mean": sum(values) / len(values),
                "ci_2_5": low,
                "ci_97_5": high,
            }
        )
    return output


def analyze(
    *,
    run_dir: Path,
    output_dir: Path,
    policy_label: str,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
) -> dict[str, object]:
    observations = _read_csv(
        _find_trace(run_dir, "teacher_forced_observation_trace")
    )
    funnels = _read_csv(_find_trace(run_dir, "preselection_funnel_trace"))
    segments = _read_csv(_find_trace(run_dir, "preselection_segment_trace"))
    rows = align_identity_rows(
        observation_rows=observations,
        funnel_rows=funnels,
        segment_rows=segments,
        policy_label=policy_label,
        creation_sources_by_segment=_load_creation_sources(run_dir),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "teacher_forced_identity_rows.csv", rows)
    overall = summarize_rows(rows, group_fields=("policy",))
    by_field = summarize_rows(
        rows,
        group_fields=("policy", "field"),
    )
    by_horizon = summarize_rows(
        rows,
        group_fields=("policy", "horizon_step"),
    )
    by_field_horizon = summarize_rows(
        rows,
        group_fields=("policy", "horizon_step", "field"),
    )
    _write_csv(
        output_dir / "teacher_forced_column_summary.csv",
        by_field_horizon,
    )
    _write_csv(
        output_dir / "teacher_forced_neuron_summary.csv",
        by_field_horizon,
    )
    _write_csv(
        output_dir / "teacher_forced_segment_summary.csv",
        by_field_horizon,
    )
    _write_csv(output_dir / "teacher_forced_field_summary.csv", by_field)
    _write_csv(
        output_dir / "teacher_forced_horizon_summary.csv",
        by_horizon,
    )
    loss_rows = [
        {
            "policy": policy_label,
            "classification": label,
            "count": sum(row["classification"] == label for row in rows),
            "fraction": (
                sum(row["classification"] == label for row in rows) / len(rows)
                if rows
                else 0.0
            ),
        }
        for label in CLASSIFICATIONS.values()
    ]
    _write_csv(
        output_dir / "teacher_forced_loss_stage_summary.csv",
        loss_rows,
    )
    bootstrap = bootstrap_rows(
        rows,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    _write_csv(output_dir / "teacher_forced_bootstrap_ci.csv", bootstrap)
    mapping_rows = [
        row
        for row in rows
        if row["target_index_mapping_valid"] not in ("", None)
    ]
    summary = {
        **DIAGNOSTIC_MARKERS,
        "policy": policy_label,
        "reference_definition": "future_observed_teacher_forced_reference",
        "target_columns": len(rows),
        "observation_reference_rows": len(observations),
        "target_index_mapping_valid_rate": _rate(
            mapping_rows,
            "target_index_mapping_valid",
        ),
        "overall": overall[0] if overall else {},
        "by_field": by_field,
        "by_horizon": by_horizon,
        "loss_stages": loss_rows,
        "current_source_jaccard_limitation": (
            "Only exact current-source fingerprints are available in existing "
            "autonomous traces; non-exact current-source Jaccard is left blank."
        ),
    }
    (output_dir / "teacher_forced_identity_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(output_dir, summary)
    return summary


def _write_report(output_dir: Path, summary: Mapping[str, object]) -> None:
    overall = summary.get("overall", {})
    fields = summary.get("by_field", [])
    horizons = summary.get("by_horizon", [])
    lines = [
        "# Fig.9 Teacher-Forced Winner Identity Diagnostic",
        "",
        "This is a nonpaper, offline-only diagnostic. The reference is the "
        "winner identity produced when the same target record is later observed "
        "in the normal online stream. It never affects prediction or selection.",
        "",
        "## Overall",
        "",
        f"- Reference available rate: {float(overall.get('reference_available_rate', 0.0)):.6f}",
        f"- Target column emitted recall: {float(overall.get('target_column_emitted_recall', 0.0)):.6f}",
        f"- Reference neuron crossing recall: {float(overall.get('reference_neuron_crossing_recall', 0.0)):.6f}",
        f"- Reference neuron candidate recall: {float(overall.get('reference_neuron_candidate_recall', 0.0)):.6f}",
        f"- Reference neuron emitted recall: {float(overall.get('reference_neuron_emitted_recall', 0.0)):.6f}",
        f"- Correct-column wrong-neuron rate: {float(overall.get('wrong_neuron_with_correct_column_rate', 0.0)):.6f}",
        f"- Exact segment match rate: {float(overall.get('exact_segment_match_rate', 0.0)):.6f}",
        "",
        "## Field Summary",
        "",
        "| field | column emitted | neuron candidate | neuron emitted | wrong neuron given emitted |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in fields:
        lines.append(
            f"| {row['field']} | "
            f"{float(row['target_column_emitted_recall']):.6f} | "
            f"{float(row['reference_neuron_candidate_recall']):.6f} | "
            f"{float(row['reference_neuron_emitted_recall']):.6f} | "
            f"{float(row['wrong_neuron_with_correct_column_rate']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Horizon Summary",
            "",
            "| step | column emitted | neuron candidate | neuron emitted | wrong neuron given emitted |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in horizons:
        lines.append(
            f"| {row['horizon_step']} | "
            f"{float(row['target_column_emitted_recall']):.6f} | "
            f"{float(row['reference_neuron_candidate_recall']):.6f} | "
            f"{float(row['reference_neuron_emitted_recall']):.6f} | "
            f"{float(row['wrong_neuron_with_correct_column_rate']):.6f} |"
        )
    lines.extend(
        [
            "",
            "Step 1 compares the initial autonomous branch to a future observed "
            "reference. Steps 2-5 also include accumulated recurrent trajectory "
            "divergence and must not be interpreted as a single selector error.",
            "",
            "Scenario 3 segments created only after the real target observation "
            "are excluded from exact-segment-match applicability.",
            "",
            "Non-exact current-source Jaccard is unavailable because the existing "
            "autonomous traces retain current-source fingerprints but not the "
            "full current source set; those values remain blank.",
        ]
    )
    (output_dir / "FIG9_TEACHER_FORCED_IDENTITY_REPORT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy-label", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze(
        run_dir=Path(args.run_dir),
        output_dir=Path(args.output_dir),
        policy_label=args.policy_label,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )


if __name__ == "__main__":
    main()
