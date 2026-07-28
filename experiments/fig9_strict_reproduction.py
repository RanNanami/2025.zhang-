"""Strict Fig.9 taxi adaptation runner.

中文调试提示：这个文件是当前 Fig.9 严格协议入口。主循环按
“读出租车记录 -> 对当前上下文做 5 步自主预测 -> 解码 passenger_count
-> 再把真实当前记录 observe/learn 进去”的顺序执行。
"""

from __future__ import annotations

import argparse
import cProfile
import csv
import hashlib
import io
import json
import pickle
import pstats
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.fig9 import (  # noqa: E402
    TaxiRecord,
    learn_actual_code,
    mape,
    plot_adaptation,
    read_records,
    record_values,
    reference_rolling_mape,
    write_predictions,
)
from experiments.diagnostics.fig9_competitive_inhibition import (  # noqa: E402
    CompetitionResult,
    CompetitionSettings,
    candidates_for_prediction,
    compete_prediction_candidates,
    emitted_prediction_code,
    summarize_competition,
)
from seqmem.encoding import (  # noqa: E402
    SSTDCompositeEncoder,
    SSTDPeriodicEncoder,
    SSTDRealValueEncoder,
    SymbolCode,
)
from seqmem.model import MemoryParams, SequentialMemory  # noqa: E402


PAPER_CHANGE_DATE = datetime(2015, 4, 1)
CONTINUOUS_IMPL_VERSIONS = {
    "reference": "reference-v1",
    "optimized_v1": "optimized-v1-local-bindings-local-response-memo",
    "optimized_v2": "optimized-v2-exact-arrivals-memo",
}


@dataclass(frozen=True)
class Fig9StrictConfig:
    horizon: int = 5
    warmup: int = 5904
    rolling_window: int = 400
    seed: int = 0
    weekday_columns: int = 30
    time_columns: int = 58
    passenger_columns: int = 482
    k: int = 10
    neurons_per_column: int = 32
    l_match: int = 4
    forgetting_threshold: float = 65.0
    passenger_min: float = 0.0
    passenger_max: float = 40000.0
    response_scale: float | None = None
    continuous_dynamics: bool = True
    integration_step: float = 0.005
    burst_context: bool = True
    intracolumn_inhibition: bool = True
    propagation_mode: str = "raw"
    scenario1_rule: str = "direct-reinforce-predicted-segment"
    use_future_covariates: bool = False
    reencode_decoded_value: bool = False
    rollout_learning: bool = False
    restore_transient_state: bool = True
    strict_label: str = "strict"
    continuous_prediction_impl: str = "reference"


@dataclass(frozen=True)
class RolloutResult:
    code: SymbolCode | None
    raw_event_counts: tuple[int, ...]
    raw_column_counts: tuple[int, ...]
    diagnostics: tuple[dict[str, object], ...] = ()
    emitted_event_counts: tuple[int, ...] = ()
    emitted_column_counts: tuple[int, ...] = ()


@dataclass
class StrictRunState:
    encoder: SSTDCompositeEncoder
    model: SequentialMemory
    fingerprint: dict[str, object]
    stream_label: str
    data_path: str
    limit: int
    next_index: int
    predictions: list[float] = field(default_factory=list)
    targets: list[float] = field(default_factory=list)
    rows: list[dict[str, object]] = field(default_factory=list)
    recent_errors: deque[float] = field(default_factory=deque)
    missing: int = 0
    raw_event_counts: list[int] = field(default_factory=list)
    raw_column_counts: list[int] = field(default_factory=list)
    density_rows: list[dict[str, object]] = field(default_factory=list)
    interval_rows: list[dict[str, object]] = field(default_factory=list)
    started_at: float = 0.0


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return ordered[index]


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left, right)
    )
    left_var = sum((a - left_mean) ** 2 for a in left)
    right_var = sum((b - right_mean) ** 2 for b in right)
    if left_var == 0.0 or right_var == 0.0:
        return 0.0
    return numerator / (left_var * right_var) ** 0.5


def field_column_counts(code: SymbolCode | None, config: Fig9StrictConfig) -> dict[str, int]:
    if code is None:
        return {"weekday": 0, "time": 0, "passenger": 0}
    weekday_stop = config.weekday_columns
    time_stop = config.weekday_columns + config.time_columns
    columns = {event.column for event in code.events}
    return {
        "weekday": sum(0 <= column < weekday_stop for column in columns),
        "time": sum(weekday_stop <= column < time_stop for column in columns),
        "passenger": sum(time_stop <= column < config.weekday_columns + config.time_columns + config.passenger_columns for column in columns),
    }


def synapse_count(model: SequentialMemory) -> int:
    return sum(
        len(segment.synapses)
        for column in model.columns
        for neuron in column.neurons
        for segment in neuron.segments
    )


def passenger_decode_details(
    passenger_encoder: SSTDRealValueEncoder,
    code: SymbolCode,
    timing_tolerance: float,
) -> tuple[float, int]:
    predicted_times: dict[int, list[float]] = {}
    for event in code.events:
        if (
            passenger_encoder.column_offset
            <= event.column
            < passenger_encoder.column_offset + passenger_encoder.num_columns
        ):
            predicted_times.setdefault(event.column, []).append(event.time)
    if not predicted_times:
        raise ValueError("code contains no columns from this encoder.")
    best_score = (-1, -1)
    best_values: list[float] = []
    for value, candidate_code in passenger_encoder.likelihood_grid():
        column_overlap = sum(
            event.column in predicted_times for event in candidate_code.events
        )
        timed_overlap = sum(
            event.column in predicted_times
            and any(
                abs(predicted_time - event.time) <= timing_tolerance
                for predicted_time in predicted_times[event.column]
            )
            for event in candidate_code.events
        )
        score = (timed_overlap, column_overlap)
        if score > best_score:
            best_score = score
            best_values = [value]
        elif score == best_score:
            best_values.append(value)
    return sum(best_values) / len(best_values), len(best_values)


def segment_count(model: SequentialMemory) -> int:
    return sum(
        len(neuron.segments)
        for column in model.columns
        for neuron in column.neurons
    )


def transient_fingerprint(model: SequentialMemory) -> str:
    payload = repr(
        (
            sorted(model.previous_active_cells.items()),
            sorted(model.previous_winners.items()),
            sorted(
                (
                    column,
                    [
                        (candidate.neuron_index, candidate.time, candidate.score)
                        for candidate in candidates
                    ],
                )
                for column, candidates in model.last_prediction_candidates.items()
            ),
            model._decode_rng.getstate(),
            model._learning_rng.getstate(),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_density_trace(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    write_predictions(path, rows)


def save_strict_checkpoint(
    path: Path,
    *,
    encoder: SSTDCompositeEncoder,
    model: SequentialMemory,
    fingerprint: dict[str, object],
    config: Fig9StrictConfig,
    stream_label: str,
    data_path: Path,
    records: list[TaxiRecord],
    limit: int,
    next_index: int,
    predictions: list[float],
    targets: list[float],
    rows: list[dict[str, object]],
    recent_errors: deque[float],
    missing: int,
    raw_event_counts: list[int],
    raw_column_counts: list[int],
    density_rows: list[dict[str, object]],
) -> None:
    payload = {
        "checkpoint_format": "fig9-strict-v1",
        "git_commit_sha": git_commit_sha(),
        "encoder": encoder,
        "model": model,
        "fingerprint": fingerprint,
        "config": config,
        "stream_label": stream_label,
        "data_path": str(data_path),
        "data_file_sha256": file_sha256(data_path),
        "prefix_end_index": next_index,
        "prefix_end_timestamp": (
            records[next_index - 1].timestamp.isoformat(sep=" ")
            if next_index > 0 and next_index <= len(records)
            else ""
        ),
        "common_prefix_hash": records_prefix_sha256(records, next_index),
        "perturbation_split_timestamp": PAPER_CHANGE_DATE.isoformat(sep=" "),
        "limit": limit,
        "next_index": next_index,
        "predictions": predictions,
        "targets": targets,
        "rows": rows,
        "recent_errors": list(recent_errors),
        "missing": missing,
        "raw_event_counts": raw_event_counts,
        "raw_column_counts": raw_column_counts,
        "density_rows": density_rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_strict_checkpoint(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("checkpoint_format") != "fig9-strict-v1":
        raise ValueError("unsupported Fig.9 strict checkpoint format")
    return payload


def find_timestamp_split(
    records: list[TaxiRecord],
    split_time: datetime = PAPER_CHANGE_DATE,
) -> int:
    for index, record in enumerate(records):
        if record.timestamp >= split_time:
            return index
    raise ValueError(f"no record at or after {split_time.isoformat(sep=' ')}")


def summarize_density(rows: list[dict[str, object]]) -> dict[str, object]:
    by_step: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        by_step.setdefault(int(row["horizon_step"]), []).append(row)

    step_summary: list[dict[str, object]] = []
    for step, step_rows in sorted(by_step.items()):
        raw_columns = [
            float(row["raw_predicted_column_count"])
            for row in step_rows
        ]
        active_cells = [
            float(row["active_source_count"]) for row in step_rows
        ]
        candidates = [
            float(row["candidate_segment_count"]) for row in step_rows
        ]
        accepted = [
            float(row["accepted_candidate_count"]) for row in step_rows
        ]
        prediction_runtimes = [
            float(row["prediction_runtime_seconds"]) for row in step_rows
        ]
        passenger_density = [
            float(row["passenger_predicted_column_count"])
            for row in step_rows
        ]
        absolute_errors = [
            float(row["absolute_error"])
            for row in step_rows
            if row["absolute_error"] != ""
        ]
        targets = [
            abs(float(row["actual_future_passenger"]))
            for row in step_rows
            if row["actual_future_passenger"] != ""
        ]
        no_prediction = sum(
            row["stopped_reason"] in {"no_raw_prediction", "no_prediction_active_cells"}
            for row in step_rows
        )
        step_summary.append(
            {
                "horizon_step": step,
                "rows": len(step_rows),
                "raw_columns_mean": sum(raw_columns) / len(raw_columns) if raw_columns else 0.0,
                "raw_columns_p50": _percentile(raw_columns, 0.50),
                "raw_columns_p90": _percentile(raw_columns, 0.90),
                "raw_columns_p99": _percentile(raw_columns, 0.99),
                "raw_columns_max": max(raw_columns) if raw_columns else 0.0,
                "active_cells_mean": sum(active_cells) / len(active_cells) if active_cells else 0.0,
                "candidate_segments_mean": sum(candidates) / len(candidates) if candidates else 0.0,
                "accepted_candidates_mean": sum(accepted) / len(accepted) if accepted else 0.0,
                "passenger_field_density_mean": sum(passenger_density) / len(passenger_density) if passenger_density else 0.0,
                "prediction_runtime_mean": sum(prediction_runtimes) / len(prediction_runtimes) if prediction_runtimes else 0.0,
                "mape": (
                    sum(absolute_errors) / sum(targets)
                    if absolute_errors and sum(targets)
                    else 0.0
                ),
                "no_prediction_rate": no_prediction / len(step_rows) if step_rows else 0.0,
            }
        )

    final_step = [
        row for row in rows if int(row["horizon_step"]) == max(by_step, default=0)
    ]
    raw = [
        float(row["raw_predicted_column_count"])
        for row in final_step
        if row["absolute_percentage_error"] != ""
    ]
    errors = [
        float(row["absolute_percentage_error"])
        for row in final_step
        if row["absolute_percentage_error"] != ""
    ]
    weekday = [
        float(row["weekday_predicted_column_count"]) for row in final_step
    ]
    time_field = [
        float(row["time_predicted_column_count"]) for row in final_step
    ]
    passenger = [
        float(row["passenger_predicted_column_count"]) for row in final_step
    ]
    return {
        "diagnostic_label": "strict Fig.9 density diagnostic",
        "rows": len(rows),
        "step_summary": step_summary,
        "final_step_raw_density_error_correlation": _correlation(raw, errors),
        "final_step_mean_weekday_columns": sum(weekday) / len(weekday) if weekday else 0.0,
        "final_step_mean_time_columns": sum(time_field) / len(time_field) if time_field else 0.0,
        "final_step_mean_passenger_columns": sum(passenger) / len(passenger) if passenger else 0.0,
        "answers": {
            "step1_already_dense": (
                step_summary[0]["raw_columns_mean"] > 100.0
                if step_summary
                else False
            ),
            "density_monotonic_step1_to_step5": all(
                left["raw_columns_mean"] <= right["raw_columns_mean"]
                for left, right in zip(step_summary, step_summary[1:])
            ),
            "high_mape_mainly_final_step_accumulation": (
                len(step_summary) >= 5
                and step_summary[-1]["raw_columns_mean"]
                > step_summary[0]["raw_columns_mean"]
            ),
        },
    }


def build_fig9_encoder(config: Fig9StrictConfig) -> SSTDCompositeEncoder:
    """Build weekday/time/passenger encoders with fixed Fig.9 column offsets.

    调试时看这里确认三字段列空间没有重叠：
    weekday: 0..29，time: 30..87，passenger: 88..569。
    """

    # STRICT PROTOCOL: Fig.9 使用 30/58/482 三组 mini-column，K=10。
    # 改这里会改变编码容量，不能和论文 strict 结果直接比较。
    day_encoder = SSTDPeriodicEncoder(
        num_columns=config.weekday_columns,
        k=config.k,
        period=7.0,
        column_offset=0,
    )
    time_encoder = SSTDPeriodicEncoder(
        num_columns=config.time_columns,
        k=config.k,
        period=48.0,
        column_offset=config.weekday_columns,
    )
    passenger_encoder = SSTDRealValueEncoder(
        num_columns=config.passenger_columns,
        k=config.k,
        minimum=config.passenger_min,
        maximum=config.passenger_max,
        column_offset=config.weekday_columns + config.time_columns,
    )
    return SSTDCompositeEncoder([day_encoder, time_encoder, passenger_encoder])


def build_strict_model(
    encoder: SSTDCompositeEncoder,
    config: Fig9StrictConfig,
) -> SequentialMemory:
    return SequentialMemory(
        encoder=encoder,  # type: ignore[arg-type]
        num_neurons_per_column=config.neurons_per_column,
        params=MemoryParams(
            l_match=config.l_match,
            forgetting_threshold=config.forgetting_threshold,
            response_scale=config.response_scale,
            burst_context=config.burst_context,
            intracolumn_inhibition=config.intracolumn_inhibition,
            continuous_dynamics=config.continuous_dynamics,
            integration_step=config.integration_step,
            continuous_prediction_impl=config.continuous_prediction_impl,
        ),
        tie_break_seed=config.seed,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_object_sha256(value: object) -> str:
    return hashlib.sha256(
        pickle.dumps(value, protocol=4)
    ).hexdigest()


def model_long_term_fingerprint(model: SequentialMemory) -> str:
    payload = []
    for column_index, column in enumerate(model.columns):
        for neuron_index, neuron in enumerate(column.neurons):
            for segment_index, segment in enumerate(neuron.segments):
                payload.append(
                    (
                        column_index,
                        neuron_index,
                        segment_index,
                        segment.active,
                        segment.target_time,
                        segment.diagnostic_id,
                        segment.creation_sentence_index,
                        segment.creation_transition_index,
                        segment.creation_target_column,
                        segment.creation_target_time,
                        segment.creation_target_neuron,
                        segment.creation_source_cell_ids,
                        segment.creation_source_fingerprint,
                        segment.scenario1_reinforcements,
                        segment.scenario2_reinforcements,
                        tuple(
                            sorted(
                                (
                                    source,
                                    synapse.source,
                                    synapse.delay,
                                    synapse.weight,
                                    synapse.age,
                                )
                                for source, synapse in segment.synapses.items()
                            )
                        ),
                    )
                )
    return stable_object_sha256(payload)


def model_rng_fingerprint(model: SequentialMemory) -> str:
    snapshot = model.snapshot_transient_state()
    return stable_object_sha256(
        {
            "decode_rng_state": snapshot.decode_rng_state,
            "learning_rng_state": snapshot.learning_rng_state,
        }
    )


def continuous_impl_version(impl: str) -> str:
    return CONTINUOUS_IMPL_VERSIONS[impl]


def records_prefix_sha256(records: list[TaxiRecord], stop_index: int) -> str:
    digest = hashlib.sha256()
    for record in records[:stop_index]:
        digest.update(
            f"{record.timestamp.isoformat(sep=' ')},{record.value}\n".encode("utf-8")
        )
    return digest.hexdigest()


def git_commit_sha() -> str:
    candidates = [
        "git",
        str(
            Path.home()
            / ".cache"
            / "codex-runtimes"
            / "codex-primary-runtime"
            / "dependencies"
            / "native"
            / "git"
            / "cmd"
            / "git.exe"
        ),
    ]
    for candidate in candidates:
        try:
            completed = subprocess.run(
                [candidate, "rev-parse", "HEAD"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            return completed.stdout.strip()
        except Exception:
            continue
    return "unknown"


def protocol_fingerprint(
    *,
    data_path: Path,
    records: list[TaxiRecord],
    stream_label: str,
    config: Fig9StrictConfig,
    limit: int,
) -> dict[str, object]:
    """Record the protocol knobs that must stay stable across strict runs.

    DEBUG WATCH: 先看输出的 protocol JSON，再看 summary。若这里记录
    的 strict flags 不是 raw/no-future/no-replay/no-rollout-learning，后面
    的 MAPE 数值就不能当作 strict Fig.9。
    """

    params = MemoryParams(
        response_scale=config.response_scale,
        continuous_dynamics=config.continuous_dynamics,
        integration_step=config.integration_step,
    )
    dynamics = params.dynamics()
    return {
        "git_commit_sha": git_commit_sha(),
        "data_file_path": str(data_path),
        "data_file_rows_used": len(records),
        "data_file_limit": limit,
        "data_file_sha256": file_sha256(data_path),
        "stream_label": stream_label,
        "date_range": [
            records[0].timestamp.isoformat(sep=" ") if records else "",
            records[-1].timestamp.isoformat(sep=" ") if records else "",
        ],
        "encoder_sizes": {
            "weekday": config.weekday_columns,
            "time": config.time_columns,
            "passenger": config.passenger_columns,
            "total": (
                config.weekday_columns
                + config.time_columns
                + config.passenger_columns
            ),
        },
        "K": config.k,
        "neurons_per_column": config.neurons_per_column,
        "L_match": config.l_match,
        "forgetting_threshold": config.forgetting_threshold,
        "response_scale": config.response_scale,
        "V0": dynamics.response_scale,
        "tau_m": dynamics.tau_m,
        "tau_s": dynamics.tau_s,
        "continuous_dynamics": config.continuous_dynamics,
        "continuous_impl": config.continuous_prediction_impl,
        "continuous_impl_version": continuous_impl_version(
            config.continuous_prediction_impl
        ),
        "integration_step": config.integration_step,
        "burst_context": "all-cell" if config.burst_context else "winner-only",
        "intracolumn_inhibition": config.intracolumn_inhibition,
        "propagation_mode": config.propagation_mode,
        "scenario1_rule": config.scenario1_rule,
        "prediction_horizon": config.horizon,
        "warmup_length": config.warmup,
        "rolling_window": config.rolling_window,
        "MAPE_denominator": "sum_abs_targets_reference_58",
        "rolling_MAPE_denominator": "global_mean_abs_target_reference_58",
        "RNG_seeds": {"tie_break_seed": config.seed},
        "uses_future_covariates": config.use_future_covariates,
        "reencodes_decoded_value": config.reencode_decoded_value,
        "learns_during_rollout": config.rollout_learning,
        "restores_transient_state": config.restore_transient_state,
        "strict_label": config.strict_label,
    }


def validate_strict_fingerprint(fingerprint: dict[str, object]) -> None:
    """Fail fast when a historical compensation leaks into the strict runner."""

    forbidden = []
    if fingerprint["burst_context"] != "all-cell":
        forbidden.append("winner-only burst context")
    if fingerprint["propagation_mode"] != "raw":
        forbidden.append(str(fingerprint["propagation_mode"]))
    if fingerprint["uses_future_covariates"]:
        forbidden.append("known-future covariates")
    if fingerprint["reencodes_decoded_value"]:
        forbidden.append("decoded-value replay")
    if fingerprint["learns_during_rollout"]:
        forbidden.append("rollout learning")
    if fingerprint["strict_label"] != "strict":
        forbidden.append("non-strict label")
    if forbidden:
        raise ValueError(f"forbidden strict Fig.9 settings: {', '.join(forbidden)}")


def rollout_raw_autonomous(
    model: SequentialMemory,
    steps: int,
    *,
    config: Fig9StrictConfig | None = None,
    competition_settings: CompetitionSettings | None = None,
    passenger_encoder: SSTDRealValueEncoder | None = None,
    record_index: int | None = None,
    timestamp: datetime | None = None,
    future_records: list[TaxiRecord] | None = None,
) -> RolloutResult:
    """Roll out future SSTD codes using only raw predictive neurons.

    中文调试提示：这里是 5-step rollout 的核心。每一步只调用
    predict_code()，然后把 raw prediction 对应的真实预测细胞放回
    previous_active_cells/previous_winners，绝不把解码值重新 encode 后输入。
    finally 中 restore_transient_state() 会撤销 rollout 对临时状态和 RNG 的影响。
    """

    # DEBUG WATCH: 在这里下断点可观察每个 horizon step 的 raw.events、
    # raw columns 数量和 prediction_active_cells(raw) 是否突然爆炸。
    snapshot = model.snapshot_transient_state()
    competition = competition_settings or CompetitionSettings()
    competition.validate()
    prediction: SymbolCode | None = None
    raw_events: list[int] = []
    raw_columns: list[int] = []
    emitted_events: list[int] = []
    emitted_columns: list[int] = []
    diagnostics: list[dict[str, object]] = []
    try:
        for _step_index in range(steps):
            # DEBUG WATCH: horizon step start。此时 previous_active_cells
            # 来自上一轮 raw predictive neurons，而不是 decoded passenger。
            previous_active_count = len(model.previous_active_cells)
            previous_winner_count = len(model.previous_winners)
            predicted_source_count = len(model.previous_predicted_sources)
            burst_source_count = len(model.previous_burst_only_sources)
            predict_started = time.perf_counter()
            raw = model.predict_code()
            predict_runtime = time.perf_counter() - predict_started
            if raw is None:
                if config is not None:
                    diagnostics.append(
                        {
                            "record_index": record_index if record_index is not None else "",
                            "timestamp": timestamp.isoformat(sep=" ") if timestamp else "",
                            "horizon_step": _step_index + 1,
                            "previous_active_cell_count": previous_active_count,
                            "previous_winner_count": previous_winner_count,
                            "predicted_source_count": predicted_source_count,
                            "burst_source_count": burst_source_count,
                            "active_source_count": model.last_prediction_stats.get("active_source_count", 0),
                            "candidate_segment_count": model.last_prediction_stats.get("candidate_segment_count", 0),
                            "threshold_crossing_segment_count": model.last_prediction_stats.get("threshold_crossing_segment_count", 0),
                            "accepted_candidate_count": 0,
                            "raw_predicted_neuron_count": 0,
                            "raw_predicted_column_count": 0,
                            "weekday_predicted_column_count": 0,
                            "time_predicted_column_count": 0,
                            "passenger_predicted_column_count": 0,
                            "raw_event_count": 0,
                            "competition_mode": competition.mode,
                            "emitted_neuron_count": 0,
                            "emitted_column_count": 0,
                            "inhibited_candidate_count": 0,
                            "mean_original_score": 0.0,
                            "mean_effective_score": 0.0,
                            "mean_accumulated_inhibition": 0.0,
                            "maximum_inhibition": 0.0,
                            "first_emitted_time": "",
                            "last_emitted_time": "",
                            "weekday_emitted_column_count": 0,
                            "time_emitted_column_count": 0,
                            "passenger_emitted_column_count": 0,
                            "competition_runtime_seconds": 0.0,
                            "passenger_candidate_count": 0,
                            "decoded_passenger": "",
                            "actual_future_passenger": (
                                future_records[_step_index].value
                                if future_records is not None
                                and _step_index < len(future_records)
                                else ""
                            ),
                            "target": (
                                future_records[_step_index].value
                                if future_records is not None
                                and _step_index < len(future_records)
                                else ""
                            ),
                            "absolute_error": "",
                            "absolute_percentage_error": "",
                            "prediction_runtime_seconds": predict_runtime,
                            "decode_runtime_seconds": 0.0,
                            "prediction_missing": True,
                            "stopped_reason": "no_raw_prediction",
                        }
                    )
                return RolloutResult(
                    None,
                    tuple(raw_events),
                    tuple(raw_columns),
                    tuple(diagnostics),
                    tuple(emitted_events),
                    tuple(emitted_columns),
                )
            # DEBUG WATCH: after predict_code。raw_event_counts/raw_column_counts
            # 是定位“预测列密度膨胀”的最直接指标。
            raw_events.append(len(raw.events))
            raw_columns.append(len({event.column for event in raw.events}))
            propagated = raw
            raw_active = model.prediction_active_cells(raw)
            competition_result: CompetitionResult | None = None
            competition_runtime = 0.0
            if competition.enabled:
                competition_started = time.perf_counter()
                competition_candidates = candidates_for_prediction(
                    raw,
                    model.last_prediction_candidates,
                    timing_tolerance=model.params.timing_tolerance,
                )
                competition_result = compete_prediction_candidates(
                    competition_candidates,
                    threshold=model.params.dendrite_threshold,
                    inhibition_strength=competition.inhibition_strength,
                    inhibition_tau=competition.inhibition_tau,
                    simultaneous_tolerance=competition.simultaneous_tolerance,
                )
                propagated = emitted_prediction_code(raw, competition_result)
                competition_runtime = time.perf_counter() - competition_started
            if propagated is None:
                emitted_events.append(0)
                emitted_columns.append(0)
                active = {}
            else:
                emitted_events.append(len(propagated.events))
                emitted_columns.append(
                    len({event.column for event in propagated.events})
                )
                active = (
                    model.prediction_active_cells(propagated)
                    if competition.enabled
                    else raw_active
                )
            decode_runtime = 0.0
            decoded_passenger: float | str = ""
            passenger_candidate_count: int | str = ""
            actual_future: float | str = (
                future_records[_step_index].value
                if future_records is not None
                and _step_index < len(future_records)
                else ""
            )
            absolute_error: float | str = ""
            ape: float | str = ""
            if (
                config is not None
                and passenger_encoder is not None
            ):
                if propagated is not None:
                    decode_started = time.perf_counter()
                    try:
                        decoded, candidate_count = passenger_decode_details(
                            passenger_encoder,
                            propagated,
                            model.params.timing_tolerance,
                        )
                    except ValueError:
                        decoded_passenger = ""
                        passenger_candidate_count = 0
                    else:
                        decoded_passenger = decoded
                        passenger_candidate_count = candidate_count
                        if (
                            future_records is not None
                            and _step_index < len(future_records)
                        ):
                            if actual_future:
                                absolute_error = abs(decoded - actual_future)
                                ape = absolute_error / abs(actual_future)
                    decode_runtime = time.perf_counter() - decode_started
                else:
                    passenger_candidate_count = 0
                raw_counts = field_column_counts(raw, config)
                emitted_counts = field_column_counts(propagated, config)
                competition_decisions = (
                    competition_result.decisions
                    if competition_result is not None
                    else ()
                )
                original_scores = [
                    decision.original_score for decision in competition_decisions
                ]
                effective_scores = [
                    decision.effective_score for decision in competition_decisions
                ]
                inhibitions = [
                    decision.accumulated_inhibition
                    for decision in competition_decisions
                ]
                emitted_times = (
                    [
                        item.candidate.time
                        for item in competition_result.emitted_candidates
                    ]
                    if competition_result is not None
                    else [event.time for event in raw.events]
                )
                diagnostics.append(
                    {
                        "record_index": record_index if record_index is not None else "",
                        "timestamp": timestamp.isoformat(sep=" ") if timestamp else "",
                        "horizon_step": _step_index + 1,
                        "previous_active_cell_count": previous_active_count,
                        "previous_winner_count": previous_winner_count,
                        "predicted_source_count": predicted_source_count,
                        "burst_source_count": burst_source_count,
                        "active_source_count": model.last_prediction_stats.get("active_source_count", 0),
                        "candidate_segment_count": model.last_prediction_stats.get("candidate_segment_count", 0),
                        "threshold_crossing_segment_count": model.last_prediction_stats.get("threshold_crossing_segment_count", 0),
                        "accepted_candidate_count": model.last_prediction_stats.get("accepted_candidate_count", 0),
                        "raw_predicted_neuron_count": len(raw_active),
                        "raw_predicted_column_count": len({event.column for event in raw.events}),
                        "weekday_predicted_column_count": raw_counts["weekday"],
                        "time_predicted_column_count": raw_counts["time"],
                        "passenger_predicted_column_count": raw_counts["passenger"],
                        "raw_event_count": len(raw.events),
                        "competition_mode": competition.mode,
                        "emitted_neuron_count": len(active),
                        "emitted_column_count": (
                            len({event.column for event in propagated.events})
                            if propagated is not None
                            else 0
                        ),
                        "inhibited_candidate_count": (
                            len(competition_result.inhibited_candidates)
                            if competition_result is not None
                            else 0
                        ),
                        "mean_original_score": (
                            sum(original_scores) / len(original_scores)
                            if original_scores
                            else 0.0
                        ),
                        "mean_effective_score": (
                            sum(effective_scores) / len(effective_scores)
                            if effective_scores
                            else 0.0
                        ),
                        "mean_accumulated_inhibition": (
                            sum(inhibitions) / len(inhibitions)
                            if inhibitions
                            else 0.0
                        ),
                        "maximum_inhibition": (
                            max(inhibitions) if inhibitions else 0.0
                        ),
                        "first_emitted_time": (
                            min(emitted_times) if emitted_times else ""
                        ),
                        "last_emitted_time": (
                            max(emitted_times) if emitted_times else ""
                        ),
                        "weekday_emitted_column_count": emitted_counts["weekday"],
                        "time_emitted_column_count": emitted_counts["time"],
                        "passenger_emitted_column_count": emitted_counts["passenger"],
                        "competition_runtime_seconds": competition_runtime,
                        "passenger_candidate_count": passenger_candidate_count,
                        "decoded_passenger": decoded_passenger,
                        "actual_future_passenger": actual_future,
                        "target": actual_future,
                        "absolute_error": absolute_error,
                        "absolute_percentage_error": ape,
                        "prediction_runtime_seconds": predict_runtime,
                        "decode_runtime_seconds": decode_runtime,
                        "prediction_missing": propagated is None,
                        "stopped_reason": "",
                    }
                )
            if not active:
                if diagnostics:
                    diagnostics[-1]["stopped_reason"] = "no_prediction_active_cells"
                return RolloutResult(
                    None,
                    tuple(raw_events),
                    tuple(raw_columns),
                    tuple(diagnostics),
                    tuple(emitted_events),
                    tuple(emitted_columns),
                )
            # STATE MUTATION: 下面两行只推进临时检索状态；长期记忆中的
            # segment/synapse/weight/age 不会改变，并会在 finally 中恢复。
            model.previous_active_cells = active
            model.previous_winners = active.copy()
            prediction = propagated
        return RolloutResult(
            prediction,
            tuple(raw_events),
            tuple(raw_columns),
            tuple(diagnostics),
            tuple(emitted_events),
            tuple(emitted_columns),
        )
    finally:
        # DEBUG WATCH: before/after transient restore。比较 restore 前后的
        # previous_active_cells、last_prediction_candidates、RNG state，确认
        # checkpoint/rollout 没污染后续在线学习。
        model.restore_transient_state(snapshot)


def strict_summary_paths(output_dir: Path) -> list[Path]:
    historical = output_dir / "fig9_historical_compensated"
    return [
        path
        for path in (output_dir / "fig9_strict").glob("*summary.json")
        if historical not in path.parents
    ]


def run_strict_stream(
    *,
    records: list[TaxiRecord],
    data_path: Path,
    stream_label: str,
    output_dir: Path,
    config: Fig9StrictConfig,
    limit: int,
    event_hook: Callable[[str, int], None] | None = None,
    print_fingerprint: bool = True,
    density_trace_path: Path | None = None,
    density_summary_path: Path | None = None,
    interval_every: int = 0,
    checkpoint_path: Path | None = None,
    checkpoint_at_index: int | None = None,
    checkpoint_every: int = 0,
    resume_checkpoint: Path | None = None,
    stop_after_index: int | None = None,
    debug_record_index: int | None = None,
    debug_output_json: Path | None = None,
    competition_settings: CompetitionSettings | None = None,
) -> dict[str, object]:
    """Run one original or perturbed stream under the strict Fig.9 protocol."""

    competition = competition_settings or CompetitionSettings()
    competition.validate()
    if len(records) <= config.horizon:
        raise ValueError("Not enough records for the requested horizon.")
    if resume_checkpoint is not None:
        checkpoint = load_strict_checkpoint(resume_checkpoint)
        encoder = checkpoint["encoder"]  # type: ignore[assignment]
        model = checkpoint["model"]  # type: ignore[assignment]
        fingerprint = checkpoint["fingerprint"]  # type: ignore[assignment]
        start_index = int(checkpoint["next_index"])
        predictions = list(checkpoint["predictions"])  # type: ignore[arg-type]
        targets = list(checkpoint["targets"])  # type: ignore[arg-type]
        rows = list(checkpoint["rows"])  # type: ignore[arg-type]
        recent_errors = deque(
            checkpoint["recent_errors"],  # type: ignore[arg-type]
            maxlen=config.rolling_window,
        )
        missing = int(checkpoint["missing"])
        raw_event_counts = list(checkpoint["raw_event_counts"])  # type: ignore[arg-type]
        raw_column_counts = list(checkpoint["raw_column_counts"])  # type: ignore[arg-type]
        density_rows = list(checkpoint["density_rows"])  # type: ignore[arg-type]
        checkpoint_data_hash = checkpoint.get("data_file_sha256")
        current_data_hash = file_sha256(data_path)
        if checkpoint_data_hash != current_data_hash:
            prefix_end = int(checkpoint.get("prefix_end_index", start_index))
            checkpoint_prefix_hash = checkpoint.get("common_prefix_hash")
            current_prefix_hash = records_prefix_sha256(records, prefix_end)
            if checkpoint_prefix_hash != current_prefix_hash:
                raise ValueError(
                    "checkpoint data is incompatible with this stream before resume index"
                )
    else:
        encoder = build_fig9_encoder(config)
        model = build_strict_model(encoder, config)
        fingerprint = protocol_fingerprint(
            data_path=data_path,
            records=records,
            stream_label=stream_label,
            config=config,
            limit=limit,
        )
        start_index = 0
        predictions = []
        targets = []
        rows = []
        recent_errors = deque(maxlen=config.rolling_window)
        missing = 0
        raw_event_counts = []
        raw_column_counts = []
        density_rows = []
    passenger_encoder = encoder.encoders[2]
    validate_strict_fingerprint(fingerprint)
    if print_fingerprint:
        print(json.dumps(fingerprint, indent=2, sort_keys=True))

    start_time = time.perf_counter()
    target_scale = sum(abs(record.value) for record in records) / len(records)
    interval_rows: list[dict[str, object]] = []
    interval_start_time = start_time
    interval_start_index = start_index
    debug_payload: dict[str, object] | None = None

    for index in range(start_index, len(records) - config.horizon):
        record = records[index]
        code = encoder.encode(record_values(record))
        rollout_diagnostics: tuple[dict[str, object], ...] = ()
        if index >= config.warmup:
            target_record = records[index + config.horizon]
            debug_enabled = debug_record_index is not None and index == debug_record_index
            before_rollout_fingerprint = (
                transient_fingerprint(model) if debug_enabled else ""
            )
            observe_segments_before = segment_count(model) if debug_enabled else 0
            observe_synapses_before = synapse_count(model) if debug_enabled else 0
            # STRICT PROTOCOL: 先预测，再 observe 当前真实记录。这样第 index
            # 条记录的真实 passenger_count 不会泄漏到它自己的 horizon 预测里。
            # DEBUG WATCH: before each record prediction。重点看 index、
            # input_timestamp、target_timestamp、previous_active_cells。
            event_hook and event_hook("predict", index)
            rollout = rollout_raw_autonomous(
                model,
                config.horizon,
                competition_settings=competition,
                config=(
                    config
                    if (
                        density_trace_path
                        or density_summary_path
                        or debug_enabled
                        or competition.enabled
                    )
                    else None
                ),
                passenger_encoder=(
                    passenger_encoder
                    if (
                        density_trace_path
                        or density_summary_path
                        or debug_enabled
                        or competition.enabled
                    )
                    else None
                ),  # type: ignore[arg-type]
                record_index=index,
                timestamp=record.timestamp,
                future_records=records[index + 1 : index + config.horizon + 1],
            )
            if debug_enabled:
                debug_payload = {
                    "diagnostic_label": "strict Fig.9 manual debug record",
                    "record_index": index,
                    "input_record": {
                        "timestamp": record.timestamp.isoformat(sep=" "),
                        "weekday": record_values(record)[0],
                        "half_hour_slot": record_values(record)[1],
                        "passenger_count": record.value,
                    },
                    "target_record": {
                        "timestamp": target_record.timestamp.isoformat(sep=" "),
                        "passenger_count": target_record.value,
                    },
                    "prediction_before_observe_state": {
                        "previous_active_cell_count": len(model.previous_active_cells),
                        "previous_winner_count": len(model.previous_winners),
                        "last_prediction_candidate_columns": len(model.last_prediction_candidates),
                        "transient_fingerprint_before_rollout": before_rollout_fingerprint,
                    },
                    "rollout_steps": list(rollout.diagnostics),
                    "transient_fingerprint_after_rollout_restore": transient_fingerprint(model),
                    "observe_before": {
                        "segment_count": observe_segments_before,
                        "synapse_count": observe_synapses_before,
                    },
                }
            raw_event_counts.extend(rollout.raw_event_counts)
            raw_column_counts.extend(rollout.raw_column_counts)
            rollout_diagnostics = rollout.diagnostics
            if density_trace_path or density_summary_path or competition.enabled:
                density_rows.extend(rollout.diagnostics)
            prediction_value: float | str = ""
            absolute_error_value: float | str = ""
            normalized_error_value: float | str = ""
            rolling_value: float | str = ""
            if rollout.code is None:
                missing += 1
            else:
                try:
                    # DEBUG WATCH: before/after passenger decode。decode_likelihood
                    # 只把 raw SSTD 活动转成数值，不能反向污染 model 状态。
                    decode_started = time.perf_counter()
                    prediction = passenger_encoder.decode_likelihood(rollout.code)  # type: ignore[attr-defined]
                    decode_runtime = time.perf_counter() - decode_started
                except ValueError:
                    missing += 1
                else:
                    predictions.append(prediction)
                    targets.append(target_record.value)
                    prediction_value = prediction
                    error = abs(prediction - target_record.value)
                    recent_errors.append(error)
                    absolute_error_value = error
                    normalized_error_value = error / target_scale if target_scale else ""
                    rolling_value = reference_rolling_mape(recent_errors, target_scale)
                    if density_rows:
                        density_rows[-1]["decoded_passenger"] = prediction
                        density_rows[-1]["actual_future_passenger"] = target_record.value
                        density_rows[-1]["absolute_percentage_error"] = (
                            error / abs(target_record.value)
                            if target_record.value
                            else ""
                        )
                        density_rows[-1]["decode_runtime_seconds"] = decode_runtime
            prediction_row: dict[str, object] = {
                "input_index": index + 1,
                "input_timestamp": record.timestamp.isoformat(sep=" "),
                "target_timestamp": target_record.timestamp.isoformat(sep=" "),
                "target": target_record.value,
                "prediction": prediction_value,
                "absolute_error": absolute_error_value,
                "normalized_absolute_error": normalized_error_value,
                "rolling_mape": rolling_value,
                "raw_rollout_event_counts": " ".join(
                    map(str, rollout.raw_event_counts)
                ),
                "raw_rollout_column_counts": " ".join(
                    map(str, rollout.raw_column_counts)
                ),
            }
            if competition.enabled:
                prediction_row["emitted_rollout_event_counts"] = " ".join(
                    map(str, rollout.emitted_event_counts)
                )
                prediction_row["emitted_rollout_column_counts"] = " ".join(
                    map(str, rollout.emitted_column_counts)
                )
            rows.append(prediction_row)
        event_hook and event_hook("observe", index)
        # STATE MUTATION: 这里才把真实当前 record 写入长期记忆，触发三种
        # learning scenario、weight/age/segment 变化。预测阶段不能学习。
        observe_started = time.perf_counter()
        learn_actual_code(model, code)
        observe_runtime = time.perf_counter() - observe_started
        if debug_payload is not None and debug_payload.get("record_index") == index:
            debug_payload["observe_after"] = {
                "segment_count": segment_count(model),
                "synapse_count": synapse_count(model),
                "winner_count": len(model.previous_winners),
                "observe_runtime_seconds": observe_runtime,
                "scenario_counts": {
                    "scenario1": model.last_observe_stats.get("scenario1", 0),
                    "scenario2": model.last_observe_stats.get("scenario2", 0),
                    "scenario3": model.last_observe_stats.get("scenario3", 0),
                },
            }
        if rollout_diagnostics:
            for row in density_rows[-len(rollout_diagnostics) :]:
                row["observe_runtime"] = observe_runtime
        if interval_every > 0 and (index + 1) % interval_every == 0:
            now = time.perf_counter()
            recent_density = [
                row
                for row in density_rows
                if row["record_index"] != ""
                and interval_start_index <= int(row["record_index"]) <= index
            ]
            interval_rows.append(
                {
                    "ending_record_index": index + 1,
                    "mape": mape(predictions, targets),
                    "coverage": (
                        len(predictions) / (len(predictions) + missing)
                        if len(predictions) + missing
                        else 0.0
                    ),
                    "mean_raw_columns": (
                        sum(raw_column_counts) / len(raw_column_counts)
                        if raw_column_counts
                        else 0.0
                    ),
                    "step_density_mean": (
                        sum(
                            float(row["raw_predicted_column_count"])
                            for row in recent_density
                        )
                        / len(recent_density)
                        if recent_density
                        else 0.0
                    ),
                    "segment_count": segment_count(model),
                    "runtime_seconds": now - interval_start_time,
                    "runtime_per_record": (
                        (now - interval_start_time)
                        / max(1, index + 1 - interval_start_index)
                    ),
                }
            )
            interval_start_time = now
            interval_start_index = index + 1
        if (
            checkpoint_path is not None
            and (
                (checkpoint_at_index is not None and index + 1 == checkpoint_at_index)
                or (checkpoint_every > 0 and (index + 1) % checkpoint_every == 0)
            )
        ):
            save_strict_checkpoint(
                checkpoint_path,
                encoder=encoder,
                model=model,
                fingerprint=fingerprint,
                config=config,
                stream_label=stream_label,
                data_path=data_path,
                records=records,
                limit=limit,
                next_index=index + 1,
                predictions=predictions,
                targets=targets,
                rows=rows,
                recent_errors=recent_errors,
                missing=missing,
                raw_event_counts=raw_event_counts,
                raw_column_counts=raw_column_counts,
                density_rows=density_rows,
            )
        if stop_after_index is not None and index + 1 >= stop_after_index:
            break

    attempted = len(rows)
    elapsed = time.perf_counter() - start_time
    summary = {
        "diagnostic_label": "strict Fig.9 reproduction attempt",
        "stream_label": stream_label,
        "records_used": len(records),
        "attempted_predictions": attempted,
        "predictions": len(predictions),
        "missing_predictions": missing,
        "coverage": len(predictions) / attempted if attempted else 0.0,
        "mape": mape(predictions, targets),
        "rolling_window": config.rolling_window,
        "final_rolling_mape": (
            rows[-1]["rolling_mape"] if rows and rows[-1]["rolling_mape"] != "" else ""
        ),
        "mean_raw_event_count": (
            sum(raw_event_counts) / len(raw_event_counts) if raw_event_counts else 0.0
        ),
        "mean_raw_column_count": (
            sum(raw_column_counts) / len(raw_column_counts) if raw_column_counts else 0.0
        ),
        "peak_raw_column_count": max(raw_column_counts) if raw_column_counts else 0,
        "runtime_seconds": elapsed,
        "records_per_second": len(records) / elapsed if elapsed else 0.0,
        "continuous_impl": config.continuous_prediction_impl,
        "continuous_impl_version": continuous_impl_version(
            config.continuous_prediction_impl
        ),
        "final_model_fingerprint": model_long_term_fingerprint(model),
        "final_rng_fingerprint": model_rng_fingerprint(model),
        "prediction_before_observe": True,
        "autonomous_rollout_steps": config.horizon,
        "uses_compensation": False,
    }
    if competition.enabled:
        summary.update(
            {
                "diagnostic_only": True,
                "competition_is_local_choice": True,
                "competition_mode": competition.mode,
                "inhibition_strength": competition.inhibition_strength,
                "inhibition_tau": competition.inhibition_tau,
                "simultaneous_tolerance": competition.simultaneous_tolerance,
            }
        )
    if density_rows:
        density_summary = summarize_density(density_rows)
        summary["density_summary_path"] = str(
            density_summary_path
            or output_dir / f"{stream_label}_density_summary.json"
        )
        summary["density_trace_path"] = str(
            density_trace_path
            or output_dir / f"{stream_label}_density_trace.csv"
        )
    competition_summary: dict[str, object] | None = None
    if competition.enabled:
        competition_summary = summarize_competition(
            density_rows,
            horizon=config.horizon,
            attempted_rollouts=attempted,
            segment_count=segment_count(model),
            runtime_seconds=elapsed,
        )
        summary["competition_trace_path"] = str(
            output_dir / "competition_trace.csv"
        )
        summary["competition_summary_path"] = str(
            output_dir / "competition_summary.json"
        )
    if interval_rows:
        summary["interval_rows"] = interval_rows

    output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(output_dir / f"{stream_label}_predictions.csv", rows)
    (output_dir / f"{stream_label}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / f"{stream_label}_protocol.json").write_text(
        json.dumps(fingerprint, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if density_rows:
        write_density_trace(
            density_trace_path or output_dir / f"{stream_label}_density_trace.csv",
            density_rows,
        )
        write_json(
            density_summary_path
            or output_dir / f"{stream_label}_density_summary.json",
            density_summary,
        )
    if competition.enabled and competition_summary is not None:
        write_predictions(output_dir / "competition_trace.csv", density_rows)
        write_json(
            output_dir / "competition_summary.json",
            competition_summary,
        )
        write_json(
            output_dir / "competition_protocol.json",
            {
                "diagnostic_only": True,
                "competition_is_local_choice": True,
                **asdict(competition),
                "threshold": model.params.dendrite_threshold,
                "score_source": "PredictionCandidate.score",
                "candidate_source": (
                    "same predict_code call via last_prediction_candidates"
                ),
                "strict_protocol_sha256": stable_object_sha256(fingerprint),
            },
        )
    if interval_rows:
        write_predictions(
            output_dir / f"{stream_label}_interval_summary.csv",
            interval_rows,
        )
    if debug_payload is not None:
        write_json(
            debug_output_json or output_dir / f"{stream_label}_debug_record.json",
            debug_payload,
        )
    plot_name = (
        "fig9_b_original_mape.png"
        if stream_label == "original"
        else "fig9_c_perturbed_mape.png"
    )
    plot_adaptation(output_dir / plot_name, rows, datetime(2015, 3, 25))
    if stream_label == "perturbed":
        plot_adaptation(
            output_dir / "fig9_d_adaptation_curve.png",
            rows,
            datetime(2015, 3, 25),
        )
    return summary


def compare_pre_change_predictions(
    original_csv: Path,
    perturbed_csv: Path,
) -> dict[str, object]:
    """Check original/perturbed streams are identical before Apr 1 targets.

    DEBUG WATCH: Apr 1 是论文扰动点。target_timestamp 早于
    2015-04-01 的行若出现 prediction mismatch，说明数据切分或扰动文件
    对齐有问题，而不是模型适应问题。
    """

    with original_csv.open("r", encoding="utf-8", newline="") as handle:
        original = list(csv.DictReader(handle))
    with perturbed_csv.open("r", encoding="utf-8", newline="") as handle:
        perturbed = list(csv.DictReader(handle))
    checked = 0
    mismatches = 0
    for left, right in zip(original, perturbed):
        target_time = datetime.fromisoformat(left["target_timestamp"])
        if target_time >= PAPER_CHANGE_DATE:
            continue
        checked += 1
        fields = ("target", "prediction", "rolling_mape", "raw_rollout_column_counts")
        if any(left[field] != right[field] for field in fields):
            mismatches += 1
    return {
        "pre_change_rows_checked": checked,
        "pre_change_prediction_mismatches": mismatches,
        "pre_change_predictions_identical": mismatches == 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/paper_nyc_taxi.csv")
    parser.add_argument("--perturbed-data", default="data/paper_nyc_taxi_perturb.csv")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5904)
    parser.add_argument("--output-dir", default="results/fig9_strict")
    parser.add_argument("--density-trace", action="store_true")
    parser.add_argument("--density-trace-csv", default="")
    parser.add_argument("--density-summary-json", default="")
    parser.add_argument(
        "--competition-mode",
        choices=("off", "competitive_raw"),
        default="off",
        help="Nonpaper diagnostic competition applied only during rollout.",
    )
    parser.add_argument("--inhibition-strength", type=float, default=0.1)
    parser.add_argument("--inhibition-tau", type=float, default=0.02)
    parser.add_argument("--simultaneous-tolerance", type=float, default=0.0)
    parser.add_argument("--interval-every", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-output", default="")
    parser.add_argument("--profile-dir", default="results/fig9_strict/profiling")
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--checkpoint-at-index", type=int, default=0)
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--stop-after-record", type=int, default=0)
    parser.add_argument("--stop-after-index", type=int, default=0)
    parser.add_argument("--debug-record-index", type=int, default=-1)
    parser.add_argument("--debug-output-json", default="")
    parser.add_argument(
        "--continuous-impl",
        choices=("reference", "optimized_v1", "optimized_v2"),
        default="reference",
        help="Implementation-only continuous integrator switch for strict A/B.",
    )
    parser.add_argument("--april-branch", action="store_true")
    parser.add_argument("--branch-records", type=int, default=2016)
    parser.add_argument(
        "--streams",
        nargs="+",
        choices=("original", "perturbed"),
        default=("original", "perturbed"),
    )
    return parser.parse_args()


def run_april_branch(args: argparse.Namespace, config: Fig9StrictConfig) -> None:
    output_dir = Path(args.output_dir)
    original_records = read_records(Path(args.data), args.limit)
    perturbed_records = read_records(Path(args.perturbed_data), args.limit)
    split_index = find_timestamp_split(original_records)
    checkpoint = output_dir / "april1_shared_prefix.pkl"
    prefix_summary = run_strict_stream(
        records=original_records,
        data_path=Path(args.data),
        stream_label="shared_prefix_original",
        output_dir=output_dir / "april_branch_prefix",
        config=config,
        limit=args.limit,
        print_fingerprint=False,
        checkpoint_path=checkpoint,
        checkpoint_at_index=split_index,
        stop_after_index=split_index,
    )
    stop_after = min(split_index + args.branch_records, len(original_records) - config.horizon)
    original_summary = run_strict_stream(
        records=original_records,
        data_path=Path(args.data),
        stream_label="original_branch",
        output_dir=output_dir / "april_branch_original",
        config=config,
        limit=args.limit,
        print_fingerprint=False,
        resume_checkpoint=checkpoint,
        stop_after_index=stop_after,
        density_trace_path=output_dir / "april_branch_original_density.csv",
        density_summary_path=output_dir / "april_branch_original_density.json",
        interval_every=args.interval_every,
    )
    perturbed_summary = run_strict_stream(
        records=perturbed_records,
        data_path=Path(args.perturbed_data),
        stream_label="perturbed_branch",
        output_dir=output_dir / "april_branch_perturbed",
        config=config,
        limit=args.limit,
        print_fingerprint=False,
        resume_checkpoint=checkpoint,
        stop_after_index=stop_after,
        density_trace_path=output_dir / "april_branch_perturbed_density.csv",
        density_summary_path=output_dir / "april_branch_perturbed_density.json",
        interval_every=args.interval_every,
    )
    write_json(
        output_dir / "april_branch_summary.json",
        {
            "split_index": split_index,
            "split_timestamp": original_records[split_index].timestamp.isoformat(sep=" "),
            "branch_records": stop_after - split_index,
            "prefix_summary": prefix_summary,
            "original_summary": original_summary,
            "perturbed_summary": perturbed_summary,
        },
    )


def run_main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    config = Fig9StrictConfig(
        warmup=args.warmup,
        continuous_prediction_impl=args.continuous_impl,
    )
    competition = CompetitionSettings(
        mode=args.competition_mode,
        inhibition_strength=args.inhibition_strength,
        inhibition_tau=args.inhibition_tau,
        simultaneous_tolerance=args.simultaneous_tolerance,
    )
    competition.validate()
    if args.april_branch:
        if competition.enabled:
            raise ValueError(
                "competitive_raw is not supported by the April branch runner"
            )
        run_april_branch(args, config)
        return
    runtime: dict[str, object] = {
        "started_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "limit": args.limit,
        "warmup": args.warmup,
        "streams": list(args.streams),
        "continuous_impl": config.continuous_prediction_impl,
        "continuous_impl_version": continuous_impl_version(
            config.continuous_prediction_impl
        ),
    }
    if competition.enabled:
        runtime.update(
            {
                "diagnostic_only": True,
                "competition_is_local_choice": True,
                "competition": asdict(competition),
            }
        )
    summaries: dict[str, object] = {}
    for stream in args.streams:
        data_path = Path(args.data if stream == "original" else args.perturbed_data)
        records = read_records(data_path, args.limit)
        summaries[stream] = run_strict_stream(
            records=records,
            data_path=data_path,
            stream_label=stream,
            output_dir=output_dir,
            config=config,
            limit=args.limit,
            density_trace_path=(
                Path(args.density_trace_csv)
                if args.density_trace_csv and len(args.streams) == 1
                else (
                    output_dir / f"{stream}_density_trace.csv"
                    if args.density_trace or args.density_trace_csv
                    else None
                )
            ),
            density_summary_path=(
                Path(args.density_summary_json)
                if args.density_summary_json and len(args.streams) == 1
                else (
                    output_dir / f"{stream}_density_summary.json"
                    if args.density_trace or args.density_summary_json
                    else None
                )
            ),
            interval_every=args.interval_every,
            checkpoint_path=Path(args.checkpoint_path) if args.checkpoint_path else None,
            checkpoint_at_index=(
                args.checkpoint_at_index
                if args.checkpoint_at_index > 0
                else None
            ),
            checkpoint_every=args.checkpoint_every,
            resume_checkpoint=(
                Path(args.resume_from or args.resume_checkpoint)
                if (args.resume_from or args.resume_checkpoint)
                else None
            ),
            stop_after_index=(
                args.stop_after_record
                if args.stop_after_record > 0
                else (
                    args.stop_after_index if args.stop_after_index > 0 else None
                )
            ),
            debug_record_index=(
                args.debug_record_index
                if args.debug_record_index >= 0
                else None
            ),
            debug_output_json=(
                Path(args.debug_output_json) if args.debug_output_json else None
            ),
            competition_settings=competition,
        )
    if {"original", "perturbed"}.issubset(args.streams):
        runtime["pre_change_comparison"] = compare_pre_change_predictions(
            output_dir / "original_predictions.csv",
            output_dir / "perturbed_predictions.csv",
        )
    runtime["summaries"] = summaries
    runtime["finished_at"] = datetime.now().isoformat(sep=" ", timespec="seconds")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "runtime.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if "original" in args.streams:
        source = output_dir / "original_protocol.json"
        if source.exists():
            (output_dir / "protocol.json").write_text(
                source.read_text(encoding="utf-8"),
                encoding="utf-8",
            )


def main() -> None:
    args = parse_args()
    if not args.profile:
        run_main(args)
        return
    profile_path = (
        Path(args.profile_output)
        if args.profile_output
        else Path(args.profile_dir) / "fig9_strict_profile.txt"
    )
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        run_main(args)
    finally:
        profiler.disable()
    profiler.dump_stats(str(profile_path.with_suffix(".pstats")))
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumtime")
    stats.print_stats(40)
    profile_path.write_text(
        stream.getvalue(),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
