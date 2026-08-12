"""Read-only metric and evaluation-window audit for Fig.9 artifacts.

This module never imports the strict runner's training loop.  It reads saved
prediction CSV files and protocol metadata, then recomputes metrics and fixed
windows without changing model state, RNG state, or historical summaries.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from .metrics import (
    absolute_percentage_errors,
    mean_absolute_error,
    median_absolute_percentage_error,
    mape as repository_mape,
    percentile_absolute_percentage_error,
    root_mean_squared_error,
    standard_mape,
    wape,
)


EXPECTED_DATA_SHA256 = "092d957f5bb0d2cd62f85098ed2268114a47b4738a5f4b29ea6be4be7349fc4d"
PAPER_FIGURE_VALUE_ESTIMATE = 0.10
BOOTSTRAP_SEED = 11
BOOTSTRAP_SAMPLES = 1000
ALLOWED_STRICT_RELATIVE_PATHS = {
    "results/fig9_strict/stage_a_250/original_predictions.csv",
    "results/fig9_strict/diagnostic_500_20260727_205411/original_predictions.csv",
    "results/fig9_diagnostics/lmatch_stack_2x2_20260810/250/STRICT_RAW_L4/original_predictions.csv",
    "results/fig9_diagnostics/lmatch_stack_500_completion_20260810_resume2/500/STRICT_RAW_L4/original_predictions.csv",
}


@dataclass(frozen=True)
class Prediction:
    """One valid prediction-target pair with its source metadata."""

    prediction: float
    target: float
    timestamp: str
    input_index: int
    target_timestamp: str = ""


@dataclass(frozen=True)
class Artifact:
    """A prediction artifact plus the protocol facts needed for filtering."""

    path: Path
    protocol_path: Path | None
    summary_path: Path | None
    rows: tuple[Prediction, ...]
    attempted: int
    protocol: dict[str, object]
    summary: dict[str, object]
    anchor_semantics: str
    valid: bool
    comparable: bool
    exclusion_reason: str

    @property
    def name(self) -> str:
        return str(self.path)


def _float(value: object, default: float | None = None) -> float | None:
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: object, default: int | None = None) -> int | None:
    number = _float(value)
    return int(number) if number is not None else default


def _read_json(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _paired_paths(prediction_path: Path) -> tuple[Path | None, Path | None]:
    directory = prediction_path.parent
    protocol = directory / "original_protocol.json"
    if not protocol.exists():
        protocol = directory / "protocol.json"
    summary = directory / "original_summary.json"
    if not summary.exists():
        summary = directory / "summary.json"
    return (
        protocol if protocol.exists() else None,
        summary if summary.exists() else None,
    )


def _parse_strict_predictions(path: Path) -> tuple[tuple[Prediction, ...], int]:
    rows: list[Prediction] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        raw_rows = list(reader)
    for row in raw_rows:
        prediction = _float(row.get("prediction"))
        target = _float(row.get("target"))
        if prediction is None or target is None:
            continue
        rows.append(
            Prediction(
                prediction=prediction,
                target=target,
                timestamp=str(row.get("input_timestamp", "")),
                input_index=_int(row.get("input_index"), len(rows) + 1) or 0,
                target_timestamp=str(row.get("target_timestamp", "")),
            )
        )
    return tuple(rows), len(raw_rows)


def _parse_anchor_b(path: Path) -> tuple[tuple[Prediction, ...], int]:
    rows: list[Prediction] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        raw_rows = list(reader)
    for row in raw_rows:
        if str(row.get("cell", "")) != "B":
            continue
        if str(row.get("in_common_anchor_set", "False")).lower() != "true":
            continue
        prediction = _float(row.get("prediction"))
        target = _float(row.get("ground_truth"))
        if prediction is None or target is None:
            continue
        rows.append(
            Prediction(
                prediction=prediction,
                target=target,
                timestamp=str(row.get("labelled_anchor_timestamp", "")),
                input_index=_int(row.get("anchor_index"), len(rows)) or 0,
                target_timestamp=str(row.get("evaluation_target_timestamp", "")),
            )
        )
    return tuple(rows), len(raw_rows)


def _anchor_b_artifact(path: Path, root: Path) -> Artifact:
    """Load the opt-in B diagnostic with its unique-anchor denominator."""

    rows, _ = _parse_anchor_b(path)
    attempted = len({row.input_index for row in rows})
    protocol_path = path.parent / "anchor_protocol.json"
    protocol = _read_json(protocol_path)
    protocol.update(
        {
            "data_file_sha256": EXPECTED_DATA_SHA256,
            "data_file_limit": 250,
            "warmup_length": 200,
            "prediction_horizon": 5,
            "continuous_impl": protocol.get("continuous_impl", "reference"),
        }
    )
    summary = _read_json(path.parent / "FINAL_ANCHOR_PARITY_SUMMARY.json")
    return Artifact(
        path=path,
        protocol_path=protocol_path,
        summary_path=path.parent / "FINAL_ANCHOR_PARITY_SUMMARY.json",
        rows=rows,
        attempted=attempted,
        protocol=protocol,
        summary=summary,
        anchor_semantics="observe_current_then_5",
        valid=bool(rows),
        comparable=bool(rows) and len(rows) == 45,
        exclusion_reason="" if len(rows) == 45 else "incomplete_common_anchor_set",
    )


def _artifact_comparability(
    path: Path,
    protocol: dict[str, object],
    summary: dict[str, object],
) -> tuple[bool, str]:
    reasons: list[str] = []
    if _int(protocol.get("L_match")) != 4:
        reasons.append("L_match_not_4")
    if _int(protocol.get("warmup_length")) != 200:
        reasons.append("warmup_not_200")
    if _int(protocol.get("prediction_horizon")) != 5:
        reasons.append("horizon_not_5")
    if protocol.get("propagation_mode", "raw") != "raw":
        reasons.append("propagation_not_raw")
    if protocol.get("continuous_impl", "reference") != "reference":
        reasons.append("continuous_impl_not_reference")
    if protocol.get("data_file_sha256", EXPECTED_DATA_SHA256) != EXPECTED_DATA_SHA256:
        reasons.append("data_sha256_mismatch")
    if summary.get("uses_compensation") is True or protocol.get("uses_compensation") is True:
        reasons.append("compensation_enabled")
    if protocol.get("uses_future_covariates") is True:
        reasons.append("future_covariates_enabled")
    relative = path.as_posix().lower()
    marker = "/results/"
    if marker in relative:
        relative = relative[relative.index(marker) + 1 :]
    if relative not in {value.lower() for value in ALLOWED_STRICT_RELATIVE_PATHS}:
        reasons.append("not_in_paper_constrained_raw_l4_allowlist")
    if not protocol or not summary:
        reasons.append("missing_protocol_or_summary")
    return not reasons, ";".join(reasons)


def discover_artifacts(root: Path) -> list[Artifact]:
    """Discover prediction artifacts and conservatively mark comparability."""

    artifacts: list[Artifact] = []
    for prediction_path in sorted(root.rglob("original_predictions.csv")):
        protocol_path, summary_path = _paired_paths(prediction_path)
        protocol = _read_json(protocol_path)
        summary = _read_json(summary_path)
        rows, attempted = _parse_strict_predictions(prediction_path)
        comparable, reason = _artifact_comparability(
            prediction_path, protocol, summary
        )
        artifacts.append(
            Artifact(
                path=prediction_path,
                protocol_path=protocol_path,
                summary_path=summary_path,
                rows=rows,
                attempted=attempted,
                protocol=protocol,
                summary=summary,
                anchor_semantics="predict_before_observe",
                valid=bool(rows) and attempted >= len(rows),
                comparable=comparable,
                exclusion_reason=reason,
            )
        )

    anchor_path = root / "results" / "fig9_diagnostics" / "anchor_parity_abcd_20260813_000500" / "anchor_parity_predictions.csv"
    if anchor_path.exists():
        artifacts.append(_anchor_b_artifact(anchor_path, root))
    return artifacts


def metric_values(rows: Sequence[Prediction]) -> dict[str, float | int]:
    """Compute all audit metrics from one fixed row set."""

    predictions = [row.prediction for row in rows]
    targets = [row.target for row in rows]
    return {
        "n": len(rows),
        "standard_mape": standard_mape(predictions, targets),
        "wape": wape(predictions, targets),
        "repo_metric": repository_mape(predictions, targets),
        "mae": mean_absolute_error(predictions, targets),
        "rmse": root_mean_squared_error(predictions, targets),
        "median_ape": median_absolute_percentage_error(predictions, targets),
        "p90_ape": percentile_absolute_percentage_error(predictions, targets, 0.90),
        "zero_target_count": sum(row.target == 0.0 for row in rows),
    }


def _bootstrap_ci(rows: Sequence[Prediction], metric: str) -> tuple[float, float]:
    if not rows:
        return 0.0, 0.0
    rng = random.Random(BOOTSTRAP_SEED)
    values: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        values.append(float(metric_values(sample)[metric]))
    values.sort()
    return values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]


def fixed_windows(rows: Sequence[Prediction]) -> list[tuple[str, int, int, tuple[Prediction, ...]]]:
    """Return predefined positional windows; no result-based selection occurs."""

    n = len(rows)
    boundaries = {
        "full": (0, n),
        "first_25": (0, n // 4),
        "second_25": (n // 4, n // 2),
        "third_25": (n // 2, (3 * n) // 4),
        "final_25": ((3 * n) // 4, n),
        "first_half": (0, n // 2),
        "second_half": (n // 2, n),
    }
    for block_size in (50, 100):
        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            if end - start == block_size:
                boundaries[f"block_{block_size}_{start // block_size + 1}"] = (start, end)
    return [
        (name, start, end, tuple(rows[start:end]))
        for name, (start, end) in boundaries.items()
        if end > start
    ]


def learning_curve(
    rows: Sequence[Prediction],
    *,
    warmup: int,
    anchor_semantics: str,
) -> list[dict[str, object]]:
    """Build cumulative and fixed rolling curves from saved rows."""

    output: list[dict[str, object]] = []
    cumulative: dict[str, float | int] | None = None
    for index in range(len(rows)):
        prefix = rows[: index + 1]
        row = rows[index]
        cumulative = metric_values(prefix)
        for window in (25, 50, 100):
            recent = prefix[-window:]
            metrics = metric_values(recent)
            output.append(
                {
                    "prediction_index": index + 1,
                    "records_seen": warmup + index,
                    "timestamp": row.target_timestamp or row.timestamp,
                    "anchor_semantics": anchor_semantics,
                    "rolling_window": window,
                    "cumulative_standard_mape": cumulative["standard_mape"],
                    "cumulative_repo_metric": cumulative["repo_metric"],
                    "cumulative_wape": cumulative["wape"],
                    "rolling_standard_mape": metrics["standard_mape"],
                    "rolling_repo_metric": metrics["repo_metric"],
                    "rolling_wape": metrics["wape"],
                }
            )
    return output


def inventory_row(artifact: Artifact) -> dict[str, object]:
    rows = artifact.rows
    protocol = artifact.protocol
    summary = artifact.summary
    return {
        "path": str(artifact.path),
        "commit": protocol.get("git_commit_sha", ""),
        "limit": protocol.get("data_file_limit", summary.get("records_used", "")),
        "warmup": protocol.get("warmup_length", ""),
        "anchor_semantics": artifact.anchor_semantics,
        "prediction_horizon": protocol.get("prediction_horizon", ""),
        "L_match": protocol.get("L_match", ""),
        "forgetting": protocol.get("forgetting_threshold", ""),
        "competition": protocol.get("competition_mode", protocol.get("competition", "off")),
        "selector": protocol.get("intracolumn_selection", protocol.get("selector", "default")),
        "predictions": len(rows),
        "attempted_rows": artifact.attempted,
        "coverage": len(rows) / artifact.attempted if artifact.attempted else 0.0,
        "first_target_timestamp": rows[0].target_timestamp or rows[0].timestamp if rows else "",
        "last_target_timestamp": rows[-1].target_timestamp or rows[-1].timestamp if rows else "",
        "valid": artifact.valid,
        "comparable": artifact.comparable,
        "exclusion_reason": artifact.exclusion_reason,
        "model_fingerprint": summary.get("final_model_fingerprint", ""),
        "rng_fingerprint": summary.get("final_rng_fingerprint", ""),
        "data_sha256": protocol.get("data_file_sha256", ""),
    }


def metric_rows(artifact: Artifact) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, start, end, window in fixed_windows(artifact.rows):
        values = metric_values(window)
        low, high = _bootstrap_ci(window, "standard_mape")
        repo_low, repo_high = _bootstrap_ci(window, "repo_metric")
        rows.append(
            {
                "artifact": str(artifact.path),
                "anchor_semantics": artifact.anchor_semantics,
                "window": name,
                "start_index": start,
                "end_index_exclusive": end,
                **values,
                "standard_mape_ci_low": low,
                "standard_mape_ci_high": high,
                "repo_metric_ci_low": repo_low,
                "repo_metric_ci_high": repo_high,
            }
        )
    return rows


def window_rows(artifact: Artifact) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    warmup = _int(artifact.protocol.get("warmup_length"), 0) or 0
    for name, start, end, rows in fixed_windows(artifact.rows):
        values = metric_values(rows)
        output.append(
            {
                "artifact": str(artifact.path),
                "anchor_semantics": artifact.anchor_semantics,
                "window_name": name,
                "start_index": start,
                "end_index_exclusive": end,
                "start_timestamp": rows[0].target_timestamp or rows[0].timestamp,
                "end_timestamp": rows[-1].target_timestamp or rows[-1].timestamp,
                "n": len(rows),
                "training_records_seen_at_start": warmup + start,
                "training_records_seen_at_end": warmup + end - 1,
                "standard_mape": values["standard_mape"],
                "repo_metric": values["repo_metric"],
                "wape": values["wape"],
                "coverage_within_artifact": len(rows) / artifact.attempted if artifact.attempted else 0.0,
            }
        )
    return output


def gap_rows(artifacts: Sequence[Artifact]) -> list[dict[str, object]]:
    """Describe measured effects without assigning unknown residuals."""

    comparable = [artifact for artifact in artifacts if artifact.comparable]
    rows: list[dict[str, object]] = []
    base = next(
        (artifact for artifact in comparable if artifact.anchor_semantics == "predict_before_observe" and len(artifact.rows) == 45),
        None,
    )
    corrected = next(
        (artifact for artifact in comparable if artifact.anchor_semantics == "observe_current_then_5"),
        None,
    )
    long_run = next(
        (artifact for artifact in comparable if len(artifact.rows) == 295),
        None,
    )
    if base:
        values = metric_values(base.rows)
        rows.append({"effect": "current_repo_vs_standard", "from": values["repo_metric"], "to": values["standard_mape"], "delta": values["standard_mape"] - values["repo_metric"], "status": "MEASURED", "note": "Same 45 valid prediction rows; ratio-of-sums versus pointwise MAPE."})
    if base and corrected:
        a = metric_values(base.rows)
        b = metric_values(corrected.rows)
        rows.append({"effect": "anchor_protocol_A_to_B_repo_metric", "from": a["repo_metric"], "to": b["repo_metric"], "delta": b["repo_metric"] - a["repo_metric"], "status": "MEASURED", "note": "Existing isolated B diagnostic; not strict default."})
        rows.append({"effect": "anchor_protocol_A_to_B_standard_mape", "from": a["standard_mape"], "to": b["standard_mape"], "delta": b["standard_mape"] - a["standard_mape"], "status": "MEASURED", "note": "Existing isolated B diagnostic; not strict default."})
    if base and long_run:
        a = metric_values(base.rows)
        b = metric_values(long_run.rows)
        rows.append({"effect": "250_to_500_standard_mape", "from": a["standard_mape"], "to": b["standard_mape"], "delta": b["standard_mape"] - a["standard_mape"], "status": "MEASURED", "note": "Descriptive short-run extension, not convergence evidence."})
        rows.append({"effect": "250_to_500_repo_metric", "from": a["repo_metric"], "to": b["repo_metric"], "delta": b["repo_metric"] - a["repo_metric"], "status": "MEASURED", "note": "Descriptive short-run extension, not convergence evidence."})
    rows.append({"effect": "paper_fig9_visual_value_to_current", "from": PAPER_FIGURE_VALUE_ESTIMATE, "to": metric_values(base.rows)["repo_metric"] if base else "", "delta": "", "status": "ESTIMATED", "note": "Figure-scale estimate only; not a tabulated target."})
    rows.append({"effect": "unexplained_paper_gap_residual", "from": "", "to": "", "delta": "", "status": "UNKNOWN", "note": "Not filled by residual arithmetic; protocol and model differences remain unresolved."})
    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_audit_report(root: Path, output_dir: Path) -> dict[str, object]:
    """Read existing artifacts and write the complete parity audit."""

    artifacts = discover_artifacts(root)
    inventory = [inventory_row(artifact) for artifact in artifacts]
    comparable = [artifact for artifact in artifacts if artifact.comparable]
    metric_comparison: list[dict[str, object]] = []
    evaluation_windows: list[dict[str, object]] = []
    curves: list[dict[str, object]] = []
    for artifact in comparable:
        metric_comparison.extend(metric_rows(artifact))
        evaluation_windows.extend(window_rows(artifact))
        curves.extend(
            learning_curve(
                artifact.rows,
                warmup=_int(artifact.protocol.get("warmup_length"), 0) or 0,
                anchor_semantics=artifact.anchor_semantics,
            )
        )
    gaps = gap_rows(artifacts)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(output_dir / "strict_l4_prediction_artifact_inventory.csv", inventory)
    _write_csv(output_dir / "evaluation_metric_comparison.csv", metric_comparison)
    _write_csv(output_dir / "evaluation_window_comparison.csv", evaluation_windows)
    _write_csv(output_dir / "evaluation_learning_curve.csv", curves)
    _write_csv(output_dir / "mape_gap_decomposition.csv", gaps)

    base = next((a for a in comparable if a.anchor_semantics == "predict_before_observe" and len(a.rows) == 45), None)
    standard = metric_values(base.rows) if base else {}
    repo_value = float(standard.get("repo_metric", 0.0)) if standard else None
    standard_value = float(standard.get("standard_mape", 0.0)) if standard else None
    summary = {
        "diagnostic": "FIG9_EVALUATION_METRIC_PARITY",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_math_changed": False,
        "strict_default_changed": False,
        "paper_fig9_mape_formula": "NOT_SPECIFIED_IN_ZHANG_TEXT; [58] defines ratio-of-sums",
        "reference_58": {
            "title": "Continuous online sequence learning with an unsupervised neural network model",
            "authors": "Yuwei Cui, Subutai Ahmad, Jeff Hawkins",
            "formula": "sum(abs(y - y_hat)) / sum(abs(y))",
            "source": "https://arxiv.org/abs/1512.05463",
        },
        "current_repo_metric": repo_value,
        "standard_mape": standard_value,
        "current_repo_metric_is_wape_like": True,
        "strict_250_prediction_count": len(next((a.rows for a in comparable if len(a.rows) == 45), ())),
        "strict_500_prediction_count": len(next((a.rows for a in comparable if len(a.rows) == 295), ())),
        "evaluation_range": "PAPER_EVALUATION_RANGE_UNSPECIFIED",
        "warmup_200_provenance": "DIAGNOSTIC_CHOICE / LOCAL_IMPLEMENTATION; paper does not specify warmup",
        "long_term_convergence": "LONG_TERM_CONVERGENCE_NOT_ESTABLISHED",
        "primary_conclusion": "CURRENT_REPO_METRIC_DIFFERS_FROM_STANDARD_MAPE",
        "recommended_next_step": "WAIT_FOR_AUTHOR_PROTOCOL_CLARIFICATION",
        "comparable_artifacts": [inventory_row(artifact) for artifact in comparable],
        "excluded_artifact_count": sum(not artifact.comparable for artifact in artifacts),
        "gap_rows": gaps,
    }
    (output_dir / "FINAL_EVALUATION_PARITY_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "CURRENT_REPO_METRIC_DEFINITION.md").write_text(
        """# Current repository metric definition

The historical `mape` field is computed by `experiments/fig9/metrics.py:mape`, called from `experiments/fig9_strict_reproduction.py` when the final summary is assembled.

```text
absolute_error = sum(abs(prediction - target) for paired rows)
repo_metric = absolute_error / sum(abs(target) for paired rows)
```

This is an aggregate ratio-of-sums, equivalent to WAPE for the paired rows. It is not the pointwise mean MAPE. It is not multiplied by 100. The implementation uses `zip`, so unmatched tail rows are ignored. Missing predictions are removed upstream and reported through coverage; they are not inserted as error terms. Zero targets remain in the numerator and denominator uses their absolute value, so a zero target contributes an absolute error but no denominator scale. The strict summary aggregates all valid prediction rows, not only the final horizon step. The five-step horizon describes the forecast distance.

The legacy field and historical values are intentionally preserved. This audit adds standard MAPE, WAPE, MAE, RMSE, median APE, and p90 APE as offline calculations only.
""",
        encoding="utf-8",
    )
    (output_dir / "PAPER_REFERENCE_METRIC_AUDIT.md").write_text(
        """# Paper and reference metric audit

## Zhang et al. 2025

Fig.9 says that mean absolute percentage error (MAPE) [58] is used to compare the most likely prediction with the actual passenger value. The Fig.9 text does not state the denominator, averaging convention, zero-target handling, percentage scaling, evaluation start/end, warmup, cumulative versus rolling aggregation, or whether the reported curve is a stable-phase value. Therefore the Zhang evaluation range is `PAPER_EVALUATION_RANGE_UNSPECIFIED`.

The paper states that the taxi stream has 17,520 half-hour records and that the forecast horizon is five steps (2.5 hours). It does not state that only 250 or 500 records should be evaluated.

## Reference [58]

Reference [58] is Yuwei Cui, Subutai Ahmad, and Jeff Hawkins, *Continuous online sequence learning with an unsupervised neural network model*, Neural Computation 28(11), 2016. Its Appendix metric is `sum(abs(y - y_hat)) / sum(abs(y))`, which is a ratio-of-sums/WAPE-like metric despite being called MAPE in that source. This supports the legacy repository formula, but it does not prove that every detail of Zhang Fig.9 used the same evaluation window.

Source used for the reference audit: https://arxiv.org/abs/1512.05463

## Warmup provenance

The strict configuration default is 5904 records in `experiments/fig9_strict_reproduction.py`. The 200-record value is explicitly supplied by local short-run scripts and written into their protocol JSON, so it is classified as `DIAGNOSTIC_CHOICE / LOCAL_IMPLEMENTATION`, not `PAPER_EXPLICIT`.
""",
        encoding="utf-8",
    )
    report = "# Fig.9 evaluation metric parity report\n\n"
    report += "## Scope\n\n"
    report += "This is a read-only audit of saved prediction artifacts. No model, encoder, decoder, learning rule, RNG path, strict default, or checkpoint was changed. Historical strict values remain intact.\n\n"
    report += "## Core numbers\n\n"
    report += f"- Strict raw L4, 250 records, warmup 200: 45 valid predictions; legacy repository metric = {repo_value:.12f}; standard pointwise MAPE = {standard_value:.12f}.\n" if base else "- No comparable 250-row artifact was found.\n"
    if repo_value is not None and standard_value is not None:
        report += f"- The metric-definition gap on the same rows is {standard_value - repo_value:.12f}; this does not move the result to the paper's approximate figure scale of 0.10.\n"
    long_run = next((a for a in comparable if len(a.rows) == 295), None)
    if long_run:
        long_values = metric_values(long_run.rows)
        report += f"- Same-family 500-record artifact: 295 valid predictions; legacy repository metric = {long_values['repo_metric']:.12f}; standard pointwise MAPE = {long_values['standard_mape']:.12f}.\n"
    report += "- For limit 250, the count is `250 - 200 - 5 = 45`; for limit 500 it is `500 - 200 - 5 = 295`. The loop has no second off-by-one in these counts.\n\n"
    report += "## Window and long-term findings\n\n"
    report += "Fixed first/second/third/final quarters, halves, and complete blocks are written to `evaluation_window_comparison.csv`; cumulative and rolling 25/50/100 curves are written to `evaluation_learning_curve.csv`. Windows were predefined by position and were not selected by score.\n\n"
    report += "The 500-record result is descriptive evidence of a short extension, not a longitudinal convergence result. No compatible 800/1000/2000-or-longer raw L4 artifact was accepted by the conservative filter, so `LONG_TERM_CONVERGENCE_NOT_ESTABLISHED`.\n\n"
    report += "## Allowed conclusion\n\n"
    report += "`CURRENT_REPO_METRIC_DIFFERS_FROM_STANDARD_MAPE`\n\n"
    report += "The repository's historical metric is consistent with the formula in reference [58], but the Zhang paper does not document enough evaluation-window detail to claim numerical parity with Fig.9.\n\n"
    report += "## Recommended next step\n\n"
    report += "`WAIT_FOR_AUTHOR_PROTOCOL_CLARIFICATION`\n\n"
    report += "Do not run another large experiment solely to chase the approximate 0.10 figure until the evaluation range and the intended use of reference [58] are clarified.\n"
    (output_dir / "FIG9_EVALUATION_METRIC_PARITY_REPORT.md").write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = write_audit_report(args.root, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
