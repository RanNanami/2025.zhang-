from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "diagnostics"))

from etth1_paper_snn import scores, seasonal_source_index
from fig9_paper_snn import mape, rollout, write_predictions
from seqmem.encoding import (
    SSTDCompositeEncoder,
    SSTDPeriodicEncoder,
    SSTDRealValueEncoder,
)
from seqmem.model import MemoryParams, SequentialMemory


@dataclass(frozen=True)
class WeatherRecord:
    timestamp: datetime
    temperature: float


def read_records(path: Path) -> list[WeatherRecord]:
    records: list[WeatherRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            records.append(
                WeatherRecord(
                    timestamp=datetime.strptime(row["date"], "%Y-%m-%d %H:%M:%S"),
                    temperature=float(row["T (degC)"]),
                )
            )
    return records


def record_values(record: WeatherRecord) -> tuple[float, float, float]:
    slot = (record.timestamp.hour * 60 + record.timestamp.minute) / 10.0
    return float(record.timestamp.weekday()), slot, record.temperature


def plot_predictions(path: Path, rows: list[dict[str, object]]) -> bool:
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
        label="Fig.9 DS transfer",
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
    axis.set(xlabel="Target time", ylabel="T (degC)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def run(args: argparse.Namespace) -> None:
    records = read_records(Path(args.data))
    warmup = args.warmup if args.warmup > 0 else int(len(records) * args.warmup_ratio)
    evaluation_stop = min(len(records) - args.horizon, warmup + args.test_size)
    if warmup < 144 or evaluation_stop <= warmup:
        raise ValueError("The requested warmup/test split is not valid.")

    training_values = [record.temperature for record in records[:warmup]]
    value_min = min(training_values) if args.value_min is None else args.value_min
    value_max = max(training_values) if args.value_max is None else args.value_max

    day_encoder = SSTDPeriodicEncoder(30, 10, period=7.0, column_offset=0)
    slot_encoder = SSTDPeriodicEncoder(58, 10, period=144.0, column_offset=30)
    value_encoder = SSTDRealValueEncoder(
        482,
        10,
        minimum=value_min,
        maximum=value_max,
        column_offset=88,
    )
    encoder = SSTDCompositeEncoder([day_encoder, slot_encoder, value_encoder])
    model = SequentialMemory(
        encoder=encoder,  # type: ignore[arg-type]
        num_neurons_per_column=32,
        params=MemoryParams(l_match=4, forgetting_threshold=65.0),
    )

    model_predictions: list[float] = []
    persistence_predictions: list[float] = []
    seasonal_predictions: list[float] = []
    model_targets: list[float] = []
    baseline_targets: list[float] = []
    output_rows: list[dict[str, object]] = []
    missing = 0

    for index in range(evaluation_stop):
        record = records[index]
        model.observe_code(encoder.encode(record_values(record)), learn=True)

        if index >= warmup:
            target_record = records[index + args.horizon]
            persistence_prediction = record.temperature
            seasonal_prediction = records[
                seasonal_source_index(index, args.horizon, period=144)
            ].temperature
            persistence_predictions.append(persistence_prediction)
            seasonal_predictions.append(seasonal_prediction)
            baseline_targets.append(target_record.temperature)
            predicted_code = rollout(model, encoder, args.horizon)
            prediction: float | str = ""
            if predicted_code is None:
                missing += 1
            else:
                try:
                    prediction = value_encoder.decode_likelihood(predicted_code)
                except ValueError:
                    missing += 1
                else:
                    model_predictions.append(prediction)
                    model_targets.append(target_record.temperature)

            output_rows.append(
                {
                    "input_index": index,
                    "input_timestamp": record.timestamp.isoformat(sep=" "),
                    "target_timestamp": target_record.timestamp.isoformat(sep=" "),
                    "target": target_record.temperature,
                    "prediction": prediction,
                    "persistence_prediction": persistence_prediction,
                    "daily_seasonal_prediction": seasonal_prediction,
                }
            )

        if args.report_every > 0 and (index + 1) % args.report_every == 0:
            print(
                f"after {index + 1}: predictions={len(model_predictions)}, "
                f"missing={missing}, wape={mape(model_predictions, model_targets):.3f}",
                flush=True,
            )

    attempted = len(model_predictions) + missing
    print("Weather transfer of the Fig. 9 DS-memory approximation")
    print(f"source: {args.data}")
    print(f"records: {len(records)}")
    print(f"split: warmup={warmup}, test={evaluation_stop - warmup}")
    print(f"horizon: {args.horizon} ten-minute step(s)")
    print("fields: weekday=30, ten-minute-slot=58, T (degC)=482")
    print("active columns per field: 10")
    print("neurons per column: 32")
    print(f"temperature encoder range from warmup only: [{value_min}, {value_max}]")
    print(f"predictions evaluated: {len(model_predictions)}")
    print(f"missing predictions: {missing}")
    print(f"prediction coverage: {len(model_predictions) / attempted if attempted else 0.0:.3f}")
    print("model,MAE,RMSE,WAPE")
    for name, predictions, score_targets in (
        ("fig9-ds-memory", model_predictions, model_targets),
        ("persistence", persistence_predictions, baseline_targets),
        ("daily-seasonal", seasonal_predictions, baseline_targets),
    ):
        model_mae, model_rmse, model_wape = scores(predictions, score_targets)
        print(f"{name},{model_mae:.6f},{model_rmse:.6f},{model_wape:.6f}")

    if args.output_csv:
        write_predictions(Path(args.output_csv), output_rows)
        print(f"saved predictions: {args.output_csv}")
    if args.output_plot:
        plotted = plot_predictions(Path(args.output_plot), output_rows)
        print(f"prediction plot generated: {plotted}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nonpaper Weather transfer using the Fig. 9 DS-memory topology."
    )
    parser.add_argument("--data", default="data/weather/weather.csv")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.7)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--value-min", type=float)
    parser.add_argument("--value-max", type=float)
    parser.add_argument("--report-every", type=int, default=1000)
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--output-plot", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
