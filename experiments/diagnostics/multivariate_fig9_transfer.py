from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "diagnostics"))

from benchmark_real_timeseries import SPECS, DatasetSpec, read_dataset
from etth1_paper_snn import scores, seasonal_source_index
from fig9_paper_snn import (
    learn_actual_code,
    mape,
    rank_sparse_predictions,
    rollout,
    write_predictions,
)
from seqmem.encoding import (
    SSTDCompositeEncoder,
    SSTDPeriodicEncoder,
    SSTDRealValueEncoder,
    SymbolCode,
)
from seqmem.model import MemoryParams, SequentialMemory


DEFAULT_DATA = {
    "etth1": "data/ETTh1.csv",
    "weather": "data/weather/weather.csv",
}


def correlation(xs: list[float], ys: list[float]) -> float:
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    sum_x = sum((x - mean_x) ** 2 for x in xs)
    sum_y = sum((y - mean_y) ** 2 for y in ys)
    denominator = math.sqrt(sum_x * sum_y)
    return numerator / denominator if denominator > 0.0 else 0.0


def select_feature_indices(
    columns: list[str],
    rows: list[list[float]],
    target_column: str,
    warmup: int,
    horizon: int,
    max_features: int,
) -> list[int]:
    if max_features <= 0:
        raise ValueError("max_features must be positive.")
    target_index = columns.index(target_column)
    usable = warmup - horizon
    if usable <= 0:
        raise ValueError("warmup must be greater than horizon.")
    future_targets = [rows[index + horizon][target_index] for index in range(usable)]
    ranked = sorted(
        range(len(columns)),
        key=lambda index: (
            abs(correlation([rows[row][index] for row in range(usable)], future_targets)),
            columns[index],
        ),
        reverse=True,
    )
    selected = ranked[: min(max_features, len(columns))]
    if target_index not in selected:
        selected[-1] = target_index
    selected = [index for index in selected if index != target_index]
    selected.sort()
    selected.append(target_index)
    return selected


def time_phase(timestamp: datetime, spec: DatasetSpec) -> float:
    if spec.seasonal_period == 24:
        return float(timestamp.hour)
    return (timestamp.hour * 60.0 + timestamp.minute) / 10.0


def make_record_values(
    timestamp: datetime,
    row: list[float],
    selected_indices: list[int],
    spec: DatasetSpec,
) -> list[float]:
    return [
        float(timestamp.weekday()),
        time_phase(timestamp, spec),
        *(row[index] for index in selected_indices),
    ]


def build_encoder(
    rows: list[list[float]],
    selected_indices: list[int],
    target_index: int,
    warmup: int,
    seasonal_period: int,
    feature_columns: int,
    target_columns: int,
    active_columns: int,
) -> tuple[SSTDCompositeEncoder, SSTDRealValueEncoder]:
    encoders = [
        SSTDPeriodicEncoder(30, active_columns, period=7.0, column_offset=0),
        SSTDPeriodicEncoder(
            58,
            active_columns,
            period=float(seasonal_period),
            column_offset=30,
        ),
    ]
    offset = 88
    target_encoder: SSTDRealValueEncoder | None = None
    for index in selected_indices:
        training_values = [row[index] for row in rows[:warmup]]
        minimum = min(training_values)
        maximum = max(training_values)
        if maximum <= minimum:
            maximum = minimum + 1.0
        num_columns = target_columns if index == target_index else feature_columns
        encoder = SSTDRealValueEncoder(
            num_columns,
            active_columns,
            minimum=minimum,
            maximum=maximum,
            column_offset=offset,
        )
        encoders.append(encoder)
        offset += num_columns
        if index == target_index:
            target_encoder = encoder
    if target_encoder is None:
        raise ValueError("The target field must be selected.")
    return SSTDCompositeEncoder(encoders), target_encoder


def predict_target_code(
    model: SequentialMemory,
    target_projection_encoder: SSTDCompositeEncoder,
) -> SymbolCode | None:
    """Predict one step while projecting only the target field."""

    if model.predict_code() is None:
        return None
    ranked = rank_sparse_predictions(model, target_projection_encoder, max_predictions=1)
    return ranked[0][1] if ranked else None


def plot_predictions(
    path: Path,
    rows: list[dict[str, object]],
    dataset_name: str,
    target_name: str,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    selected = [row for row in rows if row["prediction"] != ""]
    if not selected:
        return False
    x = [datetime.fromisoformat(str(row["target_timestamp"])) for row in selected]
    figure, axis = plt.subplots(figsize=(11, 4.8))
    axis.plot(x, [float(row["target"]) for row in selected], label="target", linewidth=1.1)
    axis.plot(
        x,
        [float(row["prediction"]) for row in selected],
        label="multivariate Fig.9 DS",
        linewidth=0.9,
        alpha=0.8,
    )
    axis.plot(
        x,
        [float(row["persistence_prediction"]) for row in selected],
        label="persistence",
        linewidth=0.8,
        alpha=0.7,
    )
    axis.set(title=dataset_name, xlabel="Target time", ylabel=target_name)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def run(args: argparse.Namespace) -> None:
    spec = SPECS[args.dataset]
    data_path = Path(args.data or DEFAULT_DATA[args.dataset])
    dates, columns, rows = read_dataset(data_path, spec)
    warmup = args.warmup if args.warmup > 0 else int(len(rows) * args.warmup_ratio)
    evaluation_stop = min(len(rows) - args.horizon, warmup + args.test_size)
    if warmup <= args.horizon or evaluation_stop <= warmup:
        raise ValueError("The requested warmup/test split is not valid.")

    selected_indices = select_feature_indices(
        columns,
        rows,
        spec.target_column,
        warmup,
        args.horizon,
        args.max_features,
    )
    selected_columns = [columns[index] for index in selected_indices]
    target_index = columns.index(spec.target_column)
    encoder, target_encoder = build_encoder(
        rows,
        selected_indices,
        target_index,
        warmup,
        spec.seasonal_period,
        args.feature_columns,
        args.target_columns,
        args.active_columns,
    )
    target_projection_encoder = SSTDCompositeEncoder([target_encoder])
    model = SequentialMemory(
        encoder=encoder,  # type: ignore[arg-type]
        num_neurons_per_column=args.neurons_per_column,
        params=MemoryParams(l_match=4, forgetting_threshold=65.0),
    )

    model_predictions: list[float] = []
    model_targets: list[float] = []
    persistence_predictions: list[float] = []
    seasonal_predictions: list[float] = []
    baseline_targets: list[float] = []
    output_rows: list[dict[str, object]] = []
    missing = 0

    for index in range(evaluation_stop):
        code = encoder.encode(
            make_record_values(dates[index], rows[index], selected_indices, spec)
        )
        learn_actual_code(model, code)
        if index >= warmup:
            target = rows[index + args.horizon][target_index]
            persistence = rows[index][target_index]
            seasonal = rows[
                seasonal_source_index(index, args.horizon, spec.seasonal_period)
            ][target_index]
            persistence_predictions.append(persistence)
            seasonal_predictions.append(seasonal)
            baseline_targets.append(target)
            predicted_code = (
                predict_target_code(model, target_projection_encoder)
                if args.horizon == 1
                else rollout(model, encoder, args.horizon)
            )
            prediction: float | str = ""
            if predicted_code is None:
                missing += 1
            else:
                try:
                    prediction = target_encoder.decode_likelihood(predicted_code)
                except ValueError:
                    missing += 1
                else:
                    model_predictions.append(prediction)
                    model_targets.append(target)
            output_rows.append(
                {
                    "input_index": index,
                    "input_timestamp": dates[index].isoformat(sep=" "),
                    "target_timestamp": dates[index + args.horizon].isoformat(sep=" "),
                    "target": target,
                    "prediction": prediction,
                    "persistence_prediction": persistence,
                    "daily_seasonal_prediction": seasonal,
                }
            )
        if args.report_every > 0 and (index + 1) % args.report_every == 0:
            print(
                f"after {index + 1}: predictions={len(model_predictions)}, "
                f"missing={missing}, wape={mape(model_predictions, model_targets):.3f}",
                flush=True,
            )

    attempted = len(model_predictions) + missing
    print(f"Multivariate Fig. 9 DS transfer: {spec.name}")
    print(f"source: {data_path}")
    print(f"records: {len(rows)}")
    print(f"split: warmup={warmup}, test={evaluation_stop - warmup}")
    print(f"horizon: {args.horizon} step(s)")
    print(f"selected numeric fields: {', '.join(selected_columns)}")
    print(f"all fields: weekday, time phase, {', '.join(selected_columns)}")
    print(f"network columns: {encoder.num_columns}")
    print(f"active columns per field: {args.active_columns}")
    print(f"neurons per column: {args.neurons_per_column}")
    print(f"predictions evaluated: {len(model_predictions)}")
    print(f"missing predictions: {missing}")
    print(f"prediction coverage: {len(model_predictions) / attempted if attempted else 0.0:.3f}")
    print("model,MAE,RMSE,WAPE")
    for name, predictions, targets in (
        ("multivariate-fig9-ds", model_predictions, model_targets),
        ("persistence", persistence_predictions, baseline_targets),
        ("daily-seasonal", seasonal_predictions, baseline_targets),
    ):
        model_mae, model_rmse, model_wape = scores(predictions, targets)
        print(f"{name},{model_mae:.6f},{model_rmse:.6f},{model_wape:.6f}")
    if args.output_csv:
        write_predictions(Path(args.output_csv), output_rows)
        print(f"saved predictions: {args.output_csv}")
    if args.output_plot:
        plotted = plot_predictions(
            Path(args.output_plot), output_rows, spec.name, spec.target_column
        )
        print(f"prediction plot generated: {plotted}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nonpaper multivariate extension of the Fig. 9 DS-memory topology."
    )
    parser.add_argument("--dataset", choices=sorted(SPECS), required=True)
    parser.add_argument("--data", default="")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=11520)
    parser.add_argument("--warmup-ratio", type=float, default=0.7)
    parser.add_argument("--test-size", type=int, default=2880)
    parser.add_argument("--max-features", type=int, default=8)
    parser.add_argument("--feature-columns", type=int, default=58)
    parser.add_argument("--target-columns", type=int, default=482)
    parser.add_argument("--active-columns", type=int, default=10)
    parser.add_argument("--neurons-per-column", type=int, default=32)
    parser.add_argument("--report-every", type=int, default=1000)
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--output-plot", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
