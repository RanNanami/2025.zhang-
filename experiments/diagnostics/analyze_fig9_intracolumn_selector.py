"""Offline comparison for the nonpaper Fig.9 intracolumn selector ablation."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
from collections import defaultdict
from pathlib import Path


def _find(run_dir: Path, stem: str) -> Path | None:
    for suffix in (".csv", ".csv.gz", ".json"):
        direct = run_dir / f"{stem}{suffix}"
        if direct.exists():
            return direct
    for candidate in run_dir.rglob(f"{stem}.*"):
        if candidate.suffix in {".csv", ".gz", ".json"}:
            return candidate
    return None


def _read_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    if path.suffix == ".gz":
        handle = gzip.open(path, "rt", encoding="utf-8", newline="")
    else:
        handle = path.open("r", encoding="utf-8", newline="")
    with handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(
    path: Path,
    rows: list[dict[str, object]],
    *,
    fieldnames: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames or (list(rows[0]) if rows else ["status"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows or [{"status": "unavailable"}])


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(value: object) -> bool:
    return str(value).lower() == "true"


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def _strict_mape(rows: list[dict[str, str]]) -> float:
    usable = [
        row
        for row in rows
        if row.get("absolute_error", "") != ""
        and row.get("target", "") != ""
    ]
    denominator = sum(abs(_float(row["target"])) for row in usable)
    return (
        sum(_float(row["absolute_error"]) for row in usable) / denominator
        if denominator
        else 0.0
    )


def _group_summary(
    rows: list[dict[str, str]],
    key: str,
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get(key, "")].append(row)
    output: list[dict[str, object]] = []
    for label, members in sorted(grouped.items()):
        changed = sum(
            _bool(row.get("selected_differs_from_existing")) for row in members
        )
        output.append(
            {
                key: label,
                "candidate_groups": len(members),
                "multi_candidate_groups": sum(
                    int(row.get("group_candidate_count", "0")) > 1
                    for row in members
                ),
                "groups_differing_from_existing": changed,
                "change_rate": _ratio(changed, len(members)),
                "mean_candidates_per_group": _ratio(
                    sum(
                        int(row.get("group_candidate_count", "0"))
                        for row in members
                    ),
                    len(members),
                ),
                "tie_rate": _ratio(
                    sum(_bool(row.get("tie_break_used")) for row in members),
                    len(members),
                ),
            }
        )
    return output


def _density_summary(
    rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row.get("horizon_step", "0"))].append(row)
    output: list[dict[str, object]] = []
    for step, members in sorted(grouped.items()):
        raw = sum(_float(row.get("raw_predicted_column_count")) for row in members)
        emitted = sum(_float(row.get("emitted_column_count")) for row in members)
        output.append(
            {
                "horizon_step": step,
                "rows": len(members),
                "mape": _strict_mape(members),
                "mean_raw_columns": _ratio(raw, len(members)),
                "mean_emitted_columns": _ratio(emitted, len(members)),
                "suppression_ratio": _ratio(raw - emitted, raw),
                "coverage": _ratio(
                    sum(not _bool(row.get("prediction_missing")) for row in members),
                    len(members),
                ),
            }
        )
    return output


def _step1_pool_map(
    rows: list[dict[str, str]],
) -> dict[tuple[int, int], str]:
    return {
        (int(row["input_index"]), int(row["column"])): row[
            "candidate_pool_fingerprint"
        ]
        for row in rows
        if int(row.get("horizon_step", "0")) == 1
    }


def _bootstrap_mape(
    rows: list[dict[str, str]],
    *,
    samples: int,
    seed: int,
    policy: str,
) -> list[dict[str, object]]:
    usable = [
        row
        for row in rows
        if row.get("absolute_error", "") != ""
        and row.get("target", "") != ""
    ]
    if not usable or samples <= 0:
        return []
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        draw = [usable[rng.randrange(len(usable))] for _row in usable]
        estimates.append(_strict_mape(draw))
    estimates.sort()
    return [
        {
            "policy": policy,
            "metric": "MAPE",
            "bootstrap_samples": samples,
            "estimate": _strict_mape(usable),
            "ci_low": estimates[int(0.025 * (samples - 1))],
            "ci_high": estimates[int(0.975 * (samples - 1))],
            "bootstrap_unit": "density_trace_row",
        }
    ]


def _reference_identity_summary(
    selector: list[dict[str, str]],
    references: list[dict[str, str]],
) -> list[dict[str, object]]:
    selected = {
        (
            int(row["input_index"]),
            int(row["horizon_step"]),
            int(row["column"]),
        ): row
        for row in selector
    }
    joined: list[tuple[dict[str, str], dict[str, str], bool, bool]] = []
    for reference in references:
        applicability = reference.get("reference_applicability", "")
        if not applicability.startswith("PREEXISTING_"):
            continue
        key = (
            int(reference["input_index"]),
            int(reference["horizon_step"]),
            int(reference["target_column"]),
        )
        chosen = selected.get(key)
        if chosen is None:
            continue
        reference_neurons = {
            int(value)
            for value in reference.get("reference_neurons", "").split()
            if value
        }
        candidate_match = int(chosen["selected_neuron"]) in reference_neurons
        emitted_match = candidate_match and _bool(
            chosen.get("emitted_after_competition")
        )
        joined.append(
            (reference, chosen, candidate_match, emitted_match)
        )
    output: list[dict[str, object]] = []
    scopes = (
        ("all", lambda row: True),
        ("passenger", lambda row: row[0].get("field") == "passenger"),
        (
            "passenger_step1",
            lambda row: row[0].get("field") == "passenger"
            and int(row[0]["horizon_step"]) == 1,
        ),
    )
    for scope, include in scopes:
        members = [row for row in joined if include(row)]
        candidate_matches = sum(row[2] for row in members)
        emitted_matches = sum(row[3] for row in members)
        output.append(
            {
                "scope": scope,
                "post_hoc_analysis_labels_only": True,
                "reference_rows": len(members),
                "reference_neuron_candidate_recall": _ratio(
                    candidate_matches, len(members)
                ),
                "reference_neuron_emitted_recall": _ratio(
                    emitted_matches, len(members)
                ),
                "correct_column_wrong_neuron_rate": _ratio(
                    len(members) - candidate_matches, len(members)
                ),
                "exact_segment_match": "",
                "similar_segment_match": "",
                "segment_match_status": (
                    "unavailable: selector run did not enable provenance trace"
                ),
            }
        )
    if not joined:
        return [
            {
                "scope": "all",
                "post_hoc_analysis_labels_only": True,
                "reference_rows": 0,
                "reference_neuron_candidate_recall": "",
                "reference_neuron_emitted_recall": "",
                "correct_column_wrong_neuron_rate": "",
                "exact_segment_match": "",
                "similar_segment_match": "",
                "segment_match_status": (
                    "unavailable: no compatible input/step/column reference join"
                ),
            }
        ]
    return output


def analyze(
    *,
    run_dir: Path,
    baseline_dir: Path,
    output_dir: Path,
    policy_label: str,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
) -> dict[str, object]:
    selector = _read_csv(_find(run_dir, "intracolumn_selection_trace"))
    density = _read_csv(
        _find(run_dir, "original_density_trace")
        or _find(run_dir, "competition_trace")
    )
    baseline_selector = _read_csv(
        _find(baseline_dir, "intracolumn_selection_trace")
    )
    baseline_density = _read_csv(
        _find(baseline_dir, "original_density_trace")
        or _find(baseline_dir, "competition_trace")
    )
    summary = _read_json(_find(run_dir, "original_summary"))
    baseline_summary = _read_json(_find(baseline_dir, "original_summary"))
    reference_rows = _read_csv(
        _find(baseline_dir, "reference_neuron_funnel_trace")
    )

    policy_summary = [
        {
            "policy": policy_label,
            "MAPE": summary.get("mape", ""),
            "final_rolling_MAPE": summary.get("final_rolling_mape", ""),
            "coverage": summary.get("coverage", ""),
            "mean_raw_columns": summary.get("mean_raw_column_count", ""),
            "peak_raw_columns": summary.get("peak_raw_column_count", ""),
            "candidate_groups": len(selector),
            "groups_differing_from_existing": sum(
                _bool(row.get("selected_differs_from_existing"))
                for row in selector
            ),
            "candidate_group_change_rate": _ratio(
                sum(
                    _bool(row.get("selected_differs_from_existing"))
                    for row in selector
                ),
                len(selector),
            ),
        }
    ]
    horizon = _density_summary(density)
    field = _group_summary(selector, "field")
    identity = _reference_identity_summary(selector, reference_rows)
    for row in identity:
        row["policy"] = policy_label
    density_output = horizon

    current_step1 = _step1_pool_map(selector)
    baseline_step1 = _step1_pool_map(baseline_selector)
    shared_step1 = set(current_step1) & set(baseline_step1)
    step1_pool_equal = (
        all(current_step1[key] == baseline_step1[key] for key in shared_step1)
        if shared_step1
        else ""
    )
    comparison = [
        {
            "policy": policy_label,
            "baseline_policy": "existing",
            "MAPE": summary.get("mape", ""),
            "baseline_MAPE": baseline_summary.get("mape", ""),
            "MAPE_difference": (
                _float(summary.get("mape")) - _float(baseline_summary.get("mape"))
                if summary and baseline_summary
                else ""
            ),
            "coverage_difference": (
                _float(summary.get("coverage"))
                - _float(baseline_summary.get("coverage"))
                if summary and baseline_summary
                else ""
            ),
            "step1_shared_candidate_groups": len(shared_step1),
            "step1_candidate_pool_fingerprint_parity": step1_pool_equal,
            "step1_parity_status": (
                "verified"
                if step1_pool_equal is True
                else "failed"
                if step1_pool_equal is False
                else "baseline selector trace unavailable"
            ),
            "step2_to_5_interpretation": (
                "recurrent trajectory divergence; not a fixed-pool reorder"
            ),
        }
    ]
    bootstrap = _bootstrap_mape(
        density,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
        policy=policy_label,
    )

    _write_csv(output_dir / "intracolumn_policy_summary.csv", policy_summary)
    _write_csv(output_dir / "intracolumn_horizon_summary.csv", horizon)
    _write_csv(output_dir / "intracolumn_field_summary.csv", field)
    _write_csv(output_dir / "intracolumn_identity_summary.csv", identity)
    _write_csv(output_dir / "intracolumn_density_summary.csv", density_output)
    _write_csv(
        output_dir / "intracolumn_policy_comparison.csv", comparison
    )
    _write_csv(output_dir / "intracolumn_bootstrap_ci.csv", bootstrap)

    report = [
        "# Fig.9 Intracolumn Selector Diagnostic",
        "",
        "This is a nonpaper diagnostic local-choice experiment.",
        "",
        f"- Policy: `{policy_label}`",
        f"- MAPE: `{summary.get('mape', 'unavailable')}`",
        f"- Coverage: `{summary.get('coverage', 'unavailable')}`",
        f"- Candidate groups: `{len(selector)}`",
        (
            "- Groups differing from existing: "
            f"`{policy_summary[0]['groups_differing_from_existing']}`"
        ),
        (
            "- Step-1 raw candidate-pool parity: "
            f"`{comparison[0]['step1_parity_status']}`"
        ),
        "",
        "Step 1 can isolate local neuron identity when a baseline selector trace "
        "is available. Steps 2-5 include recurrent context divergence.",
        "",
        (
            "Teacher/reference identity labels are joined only after prediction. "
            f"Compatible joined rows: `{identity[0]['reference_rows']}`."
        ),
        "No identity is inferred from the final model.",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "INTRACOLUMN_SELECTOR_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    result = {
        "policy": policy_label,
        "selector_rows": len(selector),
        "density_rows": len(density),
        "baseline_density_rows": len(baseline_density),
        "step1_candidate_pool_parity": step1_pool_equal,
        "diagnostic_only": True,
    }
    (output_dir / "intracolumn_analysis_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-label", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze(
        run_dir=args.run_dir,
        baseline_dir=args.baseline_dir,
        output_dir=args.output_dir,
        policy_label=args.policy_label,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )


if __name__ == "__main__":
    main()
