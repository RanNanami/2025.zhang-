from __future__ import annotations

import argparse
import csv
import heapq
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from seqmem.encoding import SSTDDiscreteEncoder


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    date_column: str
    target_column: str
    date_formats: tuple[str, ...]
    seasonal_period: int


SPECS = {
    "etth1": DatasetSpec(
        name="ETTh1",
        date_column="date",
        target_column="OT",
        date_formats=("%Y-%m-%d %H:%M:%S",),
        seasonal_period=24,
    ),
    "weather": DatasetSpec(
        name="Weather",
        date_column="date",
        target_column="T (degC)",
        date_formats=("%Y-%m-%d %H:%M:%S", "%d.%m.%Y %H:%M:%S"),
        seasonal_period=144,
    ),
}


def parse_date(value: str, formats: tuple[str, ...]) -> datetime:
    for date_format in formats:
        try:
            return datetime.strptime(value.strip(), date_format)
        except ValueError:
            pass
    raise ValueError(f"Unsupported timestamp: {value!r}")


def read_dataset(path: Path, spec: DatasetSpec) -> tuple[list[datetime], list[str], list[list[float]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"No CSV header in {path}")
        columns = [name for name in reader.fieldnames if name != spec.date_column]
        if spec.target_column not in columns:
            raise ValueError(
                f"Target {spec.target_column!r} not found. Columns: {reader.fieldnames}"
            )
        dates: list[datetime] = []
        values: list[list[float]] = []
        for row in reader:
            try:
                vector = [float(row[name]) for name in columns]
            except (TypeError, ValueError):
                continue
            dates.append(parse_date(row[spec.date_column], spec.date_formats))
            values.append(vector)
    return dates, columns, values


class QuantileCoder:
    def __init__(self, training_values: list[float], bins: int) -> None:
        ordered = sorted(training_values)
        self.boundaries = [
            ordered[min(len(ordered) - 1, int(len(ordered) * i / bins))]
            for i in range(1, bins)
        ]

    def encode(self, value: float) -> int:
        low = 0
        high = len(self.boundaries)
        while low < high:
            middle = (low + high) // 2
            if value <= self.boundaries[middle]:
                high = middle
            else:
                low = middle + 1
        return low


class SSTDVectorMemory:
    """Compact nearest-neighbor retrieval over SSTD code signatures."""

    def __init__(self, num_columns: int, k: int, seed: int) -> None:
        self.encoder = SSTDDiscreteEncoder(num_columns=num_columns, k=k, seed=seed)
        self.samples: list[tuple[tuple[int, ...], float]] = []
        self.inverted: dict[int, list[int]] = defaultdict(list)
        self.token_ids: dict[tuple[int, tuple[int, ...]], int] = {}

    def encode(self, fields: list[str]) -> tuple[int, ...]:
        features: list[int] = []
        for field_index, field in enumerate(fields):
            signature = (field_index, self.encoder.encode(field).columns)
            token_id = self.token_ids.get(signature)
            if token_id is None:
                token_id = len(self.token_ids)
                self.token_ids[signature] = token_id
            features.append(token_id)
        return tuple(features)

    def observe(self, fields: list[str], target: float) -> None:
        features = self.encode(fields)
        index = len(self.samples)
        self.samples.append((features, target))
        for feature in features:
            self.inverted[feature].append(index)

    def predict(self, fields: list[str], neighbors: int, min_overlap: int) -> float | None:
        features = self.encode(fields)
        query = set(features)
        posting_lists = sorted(
            (self.inverted.get(feature, []) for feature in query),
            key=len,
        )
        candidate_indexes: set[int] = set()
        for posting_list in posting_lists[:3]:
            candidate_indexes.update(posting_list)
        candidates = (
            (sum(feature in query for feature in self.samples[index][0]), self.samples[index][1])
            for index in candidate_indexes
        )
        top = heapq.nlargest(
            neighbors,
            (item for item in candidates if item[0] >= min_overlap),
            key=lambda item: item[0],
        )
        if not top:
            return None
        weight = sum(overlap for overlap, _ in top)
        return sum(overlap * target for overlap, target in top) / weight


def make_fields(
    date: datetime,
    row: list[float],
    columns: list[str],
    coders: list[QuantileCoder],
    history: list[list[float]],
    index: int,
    lookback: int,
    target_column: str,
    target_weight: int,
) -> list[str]:
    fields = [
        f"month={date.month}",
        f"dow={date.weekday()}",
        f"hour={date.hour}",
        f"minute={date.minute}",
    ]
    for column, value, coder in zip(columns, row, coders):
        fields.append(f"now:{column}={coder.encode(value)}")
    for lag in (1, lookback):
        lag_row = history[index - lag]
        for column, value, coder in zip(columns, lag_row, coders):
            fields.append(f"lag{lag}:{column}={coder.encode(value)}")
    target_index = columns.index(target_column)
    for copy_index in range(1, target_weight):
        fields.append(
            f"target-copy{copy_index}={coders[target_index].encode(row[target_index])}"
        )
    return fields


def correlation(xs: list[float], ys: list[float]) -> float:
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    sum_x = sum((x - mean_x) ** 2 for x in xs)
    sum_y = sum((y - mean_y) ** 2 for y in ys)
    denominator = math.sqrt(sum_x * sum_y)
    return numerator / denominator if denominator > 0 else 0.0


def metrics(predictions: list[float], targets: list[float], scale: float) -> dict[str, float]:
    errors = [prediction - target for prediction, target in zip(predictions, targets)]
    return {
        "mae": sum(abs(error) for error in errors) / len(errors),
        "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
        "nmae": sum(abs(error) for error in errors) / len(errors) / scale,
        "mape": 100.0
        * sum(abs(error) / max(abs(target), 1.0) for error, target in zip(errors, targets))
        / len(errors),
    }


def evaluate(args: argparse.Namespace) -> None:
    spec = SPECS[args.dataset]
    dates, columns, rows = read_dataset(Path(args.data), spec)
    train_end = int(len(rows) * args.train_ratio)
    test_end = min(len(rows) - args.horizon, train_end + args.test_size)
    if train_end <= args.lookback or test_end <= train_end:
        raise ValueError("Dataset or selected split is too small.")

    original_target_index = columns.index(spec.target_column)
    future_targets = [
        rows[index + args.horizon][original_target_index]
        for index in range(train_end - args.horizon)
    ]
    ranked = sorted(
        range(len(columns)),
        key=lambda column_index: abs(
            correlation(
                [rows[index][column_index] for index in range(train_end - args.horizon)],
                future_targets,
            )
        ),
        reverse=True,
    )
    selected = ranked[: args.max_features]
    if original_target_index not in selected:
        selected[-1] = original_target_index
    selected = sorted(set(selected))
    columns = [columns[index] for index in selected]
    rows = [[row[index] for index in selected] for row in rows]
    target_index = columns.index(spec.target_column)

    coders = [
        QuantileCoder([row[i] for row in rows[:train_end]], args.bins)
        for i in range(len(columns))
    ]
    target_training = [row[target_index] for row in rows[:train_end]]
    target_center = median(target_training)
    scale = sum(abs(value - target_center) for value in target_training) / len(target_training)
    scale = max(scale, 1e-12)

    memory = SSTDVectorMemory(args.num_columns, args.k, args.seed)
    start = max(args.lookback, 0)
    for index in range(start, train_end - args.horizon, args.train_stride):
        fields = make_fields(
            dates[index], rows[index], columns, coders, rows, index, args.lookback,
            spec.target_column, args.target_weight,
        )
        memory.observe(fields, rows[index + args.horizon][target_index])

    outputs: dict[str, list[float]] = {
        "sstd-memory": [],
        "persistence": [],
        "seasonal": [],
        "train-mean": [],
    }
    targets: list[float] = []
    missing = 0
    train_mean = sum(target_training) / len(target_training)
    for index in range(train_end, test_end):
        fields = make_fields(
            dates[index], rows[index], columns, coders, rows, index, args.lookback,
            spec.target_column, args.target_weight,
        )
        prediction = memory.predict(fields, args.neighbors, args.min_overlap)
        if prediction is None:
            prediction = rows[index][target_index]
            missing += 1
        outputs["sstd-memory"].append(prediction)
        outputs["persistence"].append(rows[index][target_index])
        seasonal_index = index + args.horizon - spec.seasonal_period
        outputs["seasonal"].append(
            rows[seasonal_index][target_index]
            if seasonal_index >= 0
            else rows[index][target_index]
        )
        outputs["train-mean"].append(train_mean)
        targets.append(rows[index + args.horizon][target_index])

    print(f"Dataset: {spec.name}")
    print(f"Rows/features: {len(rows)}/{len(columns)}")
    print(f"Target: {spec.target_column}")
    print(f"Selected training-only features: {', '.join(columns)}")
    print(f"Split: train={train_end}, test={len(targets)}, no test-time learning")
    print(f"Horizon: {args.horizon} steps")
    print(f"Memory samples: {len(memory.samples)}, fallback predictions: {missing}")
    print(f"Training memory stride: {args.train_stride}")
    print("model,MAE,RMSE,NMAE,MAPE")
    for name, predictions in outputs.items():
        score = metrics(predictions, targets, scale)
        print(
            f"{name},{score['mae']:.6f},{score['rmse']:.6f},"
            f"{score['nmae']:.6f},{score['mape']:.3f}%"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real multivariate time-series stress test")
    parser.add_argument("--dataset", choices=sorted(SPECS), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--lookback", type=int, default=24)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--test-size", type=int, default=3000)
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--max-features", type=int, default=8)
    parser.add_argument("--target-weight", type=int, default=4)
    parser.add_argument("--num-columns", type=int, default=1024)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--min-overlap", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
