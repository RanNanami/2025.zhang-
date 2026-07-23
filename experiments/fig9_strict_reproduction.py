"""Strict Fig.9 taxi adaptation runner.

中文调试提示：这个文件是当前 Fig.9 严格协议入口。主循环按
“读出租车记录 -> 对当前上下文做 5 步自主预测 -> 解码 passenger_count
-> 再把真实当前记录 observe/learn 进去”的顺序执行。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.fig9_paper_snn import (  # noqa: E402
    TaxiRecord,
    learn_actual_code,
    mape,
    plot_adaptation,
    read_records,
    record_values,
    reference_rolling_mape,
    write_predictions,
)
from seqmem.encoding import (  # noqa: E402
    SSTDCompositeEncoder,
    SSTDPeriodicEncoder,
    SSTDRealValueEncoder,
    SymbolCode,
)
from seqmem.model import MemoryParams, SequentialMemory  # noqa: E402


PAPER_CHANGE_DATE = datetime(2015, 4, 1)


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


@dataclass(frozen=True)
class RolloutResult:
    code: SymbolCode | None
    raw_event_counts: tuple[int, ...]
    raw_column_counts: tuple[int, ...]


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
        ),
        tie_break_seed=config.seed,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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
    prediction: SymbolCode | None = None
    raw_events: list[int] = []
    raw_columns: list[int] = []
    try:
        for _step_index in range(steps):
            # DEBUG WATCH: horizon step start。此时 previous_active_cells
            # 来自上一轮 raw predictive neurons，而不是 decoded passenger。
            raw = model.predict_code()
            if raw is None:
                return RolloutResult(None, tuple(raw_events), tuple(raw_columns))
            # DEBUG WATCH: after predict_code。raw_event_counts/raw_column_counts
            # 是定位“预测列密度膨胀”的最直接指标。
            raw_events.append(len(raw.events))
            raw_columns.append(len({event.column for event in raw.events}))
            active = model.prediction_active_cells(raw)
            if not active:
                return RolloutResult(None, tuple(raw_events), tuple(raw_columns))
            # STATE MUTATION: 下面两行只推进临时检索状态；长期记忆中的
            # segment/synapse/weight/age 不会改变，并会在 finally 中恢复。
            model.previous_active_cells = active
            model.previous_winners = active.copy()
            prediction = raw
        return RolloutResult(prediction, tuple(raw_events), tuple(raw_columns))
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
) -> dict[str, object]:
    """Run one original or perturbed stream under the strict Fig.9 protocol."""

    if len(records) <= config.horizon:
        raise ValueError("Not enough records for the requested horizon.")
    encoder = build_fig9_encoder(config)
    passenger_encoder = encoder.encoders[2]
    model = build_strict_model(encoder, config)
    fingerprint = protocol_fingerprint(
        data_path=data_path,
        records=records,
        stream_label=stream_label,
        config=config,
        limit=limit,
    )
    validate_strict_fingerprint(fingerprint)
    if print_fingerprint:
        print(json.dumps(fingerprint, indent=2, sort_keys=True))

    start_time = time.perf_counter()
    predictions: list[float] = []
    targets: list[float] = []
    rows: list[dict[str, object]] = []
    recent_errors: deque[float] = deque(maxlen=config.rolling_window)
    target_scale = sum(abs(record.value) for record in records) / len(records)
    missing = 0
    raw_event_counts: list[int] = []
    raw_column_counts: list[int] = []

    for index, record in enumerate(records[:-config.horizon]):
        code = encoder.encode(record_values(record))
        if index >= config.warmup:
            target_record = records[index + config.horizon]
            # STRICT PROTOCOL: 先预测，再 observe 当前真实记录。这样第 index
            # 条记录的真实 passenger_count 不会泄漏到它自己的 horizon 预测里。
            # DEBUG WATCH: before each record prediction。重点看 index、
            # input_timestamp、target_timestamp、previous_active_cells。
            event_hook and event_hook("predict", index)
            rollout = rollout_raw_autonomous(model, config.horizon)
            raw_event_counts.extend(rollout.raw_event_counts)
            raw_column_counts.extend(rollout.raw_column_counts)
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
                    prediction = passenger_encoder.decode_likelihood(rollout.code)  # type: ignore[attr-defined]
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
            rows.append(
                {
                    "input_index": index + 1,
                    "input_timestamp": record.timestamp.isoformat(sep=" "),
                    "target_timestamp": target_record.timestamp.isoformat(sep=" "),
                    "target": target_record.value,
                    "prediction": prediction_value,
                    "absolute_error": absolute_error_value,
                    "normalized_absolute_error": normalized_error_value,
                    "rolling_mape": rolling_value,
                    "raw_rollout_event_counts": " ".join(map(str, rollout.raw_event_counts)),
                    "raw_rollout_column_counts": " ".join(map(str, rollout.raw_column_counts)),
                }
            )
        event_hook and event_hook("observe", index)
        # STATE MUTATION: 这里才把真实当前 record 写入长期记忆，触发三种
        # learning scenario、weight/age/segment 变化。预测阶段不能学习。
        learn_actual_code(model, code)

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
        "prediction_before_observe": True,
        "autonomous_rollout_steps": config.horizon,
        "uses_compensation": False,
    }

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
    parser.add_argument(
        "--streams",
        nargs="+",
        choices=("original", "perturbed"),
        default=("original", "perturbed"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    config = Fig9StrictConfig(warmup=args.warmup)
    runtime: dict[str, object] = {
        "started_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "limit": args.limit,
        "warmup": args.warmup,
        "streams": list(args.streams),
    }
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


if __name__ == "__main__":
    main()
