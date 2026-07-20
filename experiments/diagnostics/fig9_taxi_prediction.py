from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from seqmem.encoding import SSTDDiscreteEncoder
from context_memory import (
    SparseContextMemory,
    SparseContextRegressor,
    SparseKNNRegressor,
)
from seqmem.model import MemoryParams, SequentialMemory


@dataclass(frozen=True)
class TaxiRecord:
    timestamp: datetime
    value: float


class EqualWidthValueCoder:
    """Map passenger counts to value bins and back to bin centers."""

    def __init__(self, values: list[float], bins: int) -> None:
        if bins <= 1:
            raise ValueError("bins must be greater than 1.")
        self.bins = bins
        self.min_value = min(values)
        self.max_value = max(values)
        self.width = (self.max_value - self.min_value) / bins
        if self.width == 0:
            self.width = 1.0

    def encode(self, value: float) -> str:
        index = int((value - self.min_value) / self.width)
        index = max(0, min(self.bins - 1, index))
        return f"passenger_bin_{index}"

    def decode(self, symbol: str) -> float:
        index = int(symbol.rsplit("_", 1)[-1])
        return self.min_value + (index + 0.5) * self.width

    def symbols(self) -> set[str]:
        return {f"passenger_bin_{index}" for index in range(self.bins)}


def record_symbol(record: TaxiRecord, value_symbol: str, mode: str) -> str:
    if mode == "value":
        return value_symbol
    slot = record.timestamp.hour * 2 + record.timestamp.minute // 30
    return f"dow_{record.timestamp.weekday()}__slot_{slot}__{value_symbol}"


def read_taxi_csv(path: Path) -> list[TaxiRecord]:
    records: list[TaxiRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                TaxiRecord(
                    timestamp=datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S"),
                    value=float(row["value"]),
                )
            )
    return records


def mape(predictions: list[float], targets: list[float]) -> float:
    errors = [
        abs(prediction - target) / abs(target)
        for prediction, target in zip(predictions, targets)
        if target != 0
    ]
    return 100.0 * sum(errors) / len(errors) if errors else 0.0


def mape_with_floor(
    predictions: list[float], targets: list[float], denominator_floor: float
) -> float:
    if denominator_floor <= 0:
        return mape(predictions, targets)
    errors = [
        abs(prediction - target) / max(abs(target), denominator_floor)
        for prediction, target in zip(predictions, targets)
    ]
    return 100.0 * sum(errors) / len(errors) if errors else 0.0


def predict_steps(
    model: SequentialMemory,
    start_symbol: str,
    steps: int,
    candidate_symbols: set[str],
) -> str | None:
    model.reset_state()
    model.observe(start_symbol, learn=False)
    prediction: str | None = None
    for _ in range(steps):
        prediction = model.predict_symbol(candidate_symbols=candidate_symbols)
        if prediction is None:
            return None
        model.observe(prediction, learn=False)
    return prediction


def run(args: argparse.Namespace) -> None:
    records = read_taxi_csv(Path(args.data))
    if args.limit > 0:
        records = records[: args.limit]

    values = [record.value for record in records]
    if args.baseline == "seasonal":
        run_seasonal_baseline(records, horizon=args.horizon, warmup=args.warmup)
        return
    if args.baseline == "context-memory":
        run_context_memory(records, args)
        return
    if args.baseline == "context-table":
        run_context_table(records, args)
        return
    if args.baseline == "sparse-context":
        run_sparse_context(records, args)
        return
    if args.baseline == "sparse-regression":
        run_sparse_regression(records, args)
        return
    if args.baseline == "sparse-knn":
        run_sparse_knn(records, args)
        return
    coder = EqualWidthValueCoder(values, bins=args.value_bins)
    value_symbols = [coder.encode(value) for value in values]
    symbols = [
        record_symbol(record, value_symbol, args.symbol_mode)
        for record, value_symbol in zip(records, value_symbols)
    ]
    candidate_symbols = set(symbols) if args.symbol_mode == "record" else coder.symbols()

    encoder = SSTDDiscreteEncoder(num_columns=args.num_columns, k=args.k, seed=args.seed)
    params = MemoryParams(
        l_match=args.l_match,
        forgetting_threshold=args.forgetting_threshold,
    )
    model = SequentialMemory(
        encoder=encoder,
        num_neurons_per_column=args.neurons_per_column,
        params=params,
    )

    predictions: list[float] = []
    targets: list[float] = []
    checkpoints: list[tuple[int, float, int]] = []

    for index, symbol in enumerate(symbols):
        if index >= args.warmup and index + args.horizon < len(symbols):
            predicted_symbol = predict_steps(
                model,
                symbol,
                steps=args.horizon,
                candidate_symbols=candidate_symbols,
            )
            if predicted_symbol is not None:
                value_symbol = predicted_symbol.rsplit("__", 1)[-1]
                predictions.append(coder.decode(value_symbol))
                targets.append(values[index + args.horizon])

        model.step(symbol)

        if args.report_every > 0 and (index + 1) % args.report_every == 0:
            checkpoints.append((index + 1, mape(predictions, targets), len(predictions)))

    score = mape(predictions, targets)
    print("Fig. 9-style taxi passenger prediction")
    print(f"records: {len(records)}")
    print(f"time range: {records[0].timestamp} -> {records[-1].timestamp}")
    print(f"horizon steps: {args.horizon}")
    print(f"value bins: {args.value_bins}")
    print(f"symbol mode: {args.symbol_mode}")
    print(f"mini-columns: {args.num_columns}")
    print(f"neurons per mini-column: {args.neurons_per_column}")
    print(f"predictions evaluated: {len(predictions)}")
    print(f"MAPE: {score:.3f}%")
    print()
    print("checkpoints:")
    for index, checkpoint_mape, count in checkpoints[-10:]:
        print(f"  after {index:5d} records: mape={checkpoint_mape:.3f}%, n={count}")


def run_seasonal_baseline(
    records: list[TaxiRecord],
    horizon: int,
    warmup: int,
) -> None:
    history: dict[tuple[int, int], list[float]] = {}
    predictions: list[float] = []
    targets: list[float] = []

    for index, record in enumerate(records):
        slot = record.timestamp.hour * 2 + record.timestamp.minute // 30
        key = (record.timestamp.weekday(), slot)

        if index >= warmup and index + horizon < len(records):
            future = records[index + horizon]
            future_slot = future.timestamp.hour * 2 + future.timestamp.minute // 30
            future_key = (future.timestamp.weekday(), future_slot)
            values = history.get(future_key)
            if values:
                predictions.append(sum(values) / len(values))
                targets.append(future.value)

        history.setdefault(key, []).append(record.value)

    print("Fig. 9 seasonal baseline")
    print(f"records: {len(records)}")
    print(f"time range: {records[0].timestamp} -> {records[-1].timestamp}")
    print(f"horizon steps: {horizon}")
    print(f"predictions evaluated: {len(predictions)}")
    print(f"MAPE: {mape(predictions, targets):.3f}%")


def context_prefix(
    records: list[TaxiRecord],
    index: int,
    horizon: int,
    value_symbol: str,
) -> list[str]:
    current = records[index]
    future = records[index + horizon]
    current_slot = current.timestamp.hour * 2 + current.timestamp.minute // 30
    future_slot = future.timestamp.hour * 2 + future.timestamp.minute // 30
    return [
        f"current_dow_{current.timestamp.weekday()}",
        f"current_slot_{current_slot}",
        f"current_{value_symbol}",
        f"future_dow_{future.timestamp.weekday()}",
        f"future_slot_{future_slot}",
    ]


def run_sparse_context(records: list[TaxiRecord], args: argparse.Namespace) -> None:
    values = [record.value for record in records]
    coder = EqualWidthValueCoder(values, bins=args.value_bins)
    value_symbols = [coder.encode(value) for value in values]
    candidate_symbols = coder.symbols()

    encoder = SSTDDiscreteEncoder(num_columns=args.num_columns, k=args.k, seed=args.seed)
    memory = SparseContextMemory(
        encoder=encoder,
        max_order=args.context_order,
        decay=args.context_decay,
    )

    predictions: list[float] = []
    targets: list[float] = []
    checkpoints: list[tuple[int, float, int]] = []
    fallback_history: dict[tuple[str, int, int], list[float]] = {}
    seasonal_history: dict[tuple[int, int], list[float]] = {}

    max_index = len(records) - args.horizon
    if args.eval_end > 0:
        max_index = min(max_index, args.eval_end)
    for index in range(max_index):
        fields = context_prefix(records, index, args.horizon, value_symbols[index])
        target_symbol = value_symbols[index + args.horizon]
        future = records[index + args.horizon]
        future_slot = future.timestamp.hour * 2 + future.timestamp.minute // 30
        fallback_key = (
            value_symbols[index],
            future.timestamp.weekday(),
            future_slot,
        )
        seasonal_key = (future.timestamp.weekday(), future_slot)

        if index >= args.warmup:
            predicted_symbol, score, margin = memory.predict_with_confidence(
                fields, candidate_symbols
            )
            if predicted_symbol is not None:
                prediction = coder.decode(predicted_symbol)
                fallback_values = fallback_history.get(fallback_key)
                seasonal_values = seasonal_history.get(seasonal_key)
                fallback_prediction = None
                if fallback_values:
                    fallback_prediction = sum(fallback_values) / len(fallback_values)
                elif args.context_backoff and seasonal_values:
                    fallback_prediction = sum(seasonal_values) / len(seasonal_values)
                if (
                    fallback_prediction is not None
                    and args.context_fallback_margin > 0
                    and margin < args.context_fallback_margin
                ):
                    prediction = fallback_prediction
                elif fallback_prediction is not None and args.context_blend > 0:
                    prediction = (
                        (1.0 - args.context_blend) * prediction
                        + args.context_blend * fallback_prediction
                    )
                predictions.append(prediction)
                targets.append(values[index + args.horizon])

        memory.observe(fields, target_symbol)
        fallback_history.setdefault(fallback_key, []).append(values[index + args.horizon])
        seasonal_history.setdefault(seasonal_key, []).append(values[index + args.horizon])

        if args.report_every > 0 and (index + 1) % args.report_every == 0:
            checkpoints.append((index + 1, mape(predictions, targets), len(predictions)))

    print("Fig. 9 sparse-context taxi prediction")
    print(f"records: {len(records)}")
    print(f"time range: {records[0].timestamp} -> {records[-1].timestamp}")
    print(f"horizon steps: {args.horizon}")
    print(f"value bins: {args.value_bins}")
    print(f"mini-columns: {args.num_columns}")
    print(f"k: {args.k}")
    print(f"context order: {args.context_order}")
    print(f"context decay: {args.context_decay}")
    print(f"context fallback margin: {args.context_fallback_margin}")
    print(f"context blend: {args.context_blend}")
    print(f"context backoff: {args.context_backoff}")
    print(f"predictions evaluated: {len(predictions)}")
    print(f"MAPE: {mape(predictions, targets):.3f}%")
    print()
    print("checkpoints:")
    for index, checkpoint_mape, count in checkpoints[-10:]:
        print(f"  after {index:5d} records: mape={checkpoint_mape:.3f}%, n={count}")


def run_sparse_regression(records: list[TaxiRecord], args: argparse.Namespace) -> None:
    values = [record.value for record in records]
    coder = EqualWidthValueCoder(values, bins=args.value_bins)
    value_symbols = [coder.encode(value) for value in values]

    encoder = SSTDDiscreteEncoder(num_columns=args.num_columns, k=args.k, seed=args.seed)
    memory = SparseContextRegressor(
        encoder=encoder,
        max_order=args.context_order,
        decay=args.context_decay,
    )

    predictions: list[float] = []
    targets: list[float] = []
    checkpoints: list[tuple[int, float, int]] = []

    max_index = len(records) - args.horizon
    if args.eval_end > 0:
        max_index = min(max_index, args.eval_end)
    for index in range(max_index):
        fields = context_prefix(records, index, args.horizon, value_symbols[index])
        target = values[index + args.horizon]

        if index >= args.warmup:
            prediction, confidence = memory.predict(fields)
            if prediction is not None and confidence >= args.min_confidence:
                predictions.append(prediction)
                targets.append(target)

        memory.observe(fields, target)

        if args.report_every > 0 and (index + 1) % args.report_every == 0:
            checkpoints.append((index + 1, mape(predictions, targets), len(predictions)))

    print("Fig. 9 sparse-regression taxi prediction")
    print(f"records: {len(records)}")
    print(f"time range: {records[0].timestamp} -> {records[-1].timestamp}")
    print(f"horizon steps: {args.horizon}")
    print(f"value bins: {args.value_bins}")
    print(f"mini-columns: {args.num_columns}")
    print(f"k: {args.k}")
    print(f"context order: {args.context_order}")
    print(f"context decay: {args.context_decay}")
    print(f"min confidence: {args.min_confidence}")
    print(f"predictions evaluated: {len(predictions)}")
    print(f"MAPE: {mape(predictions, targets):.3f}%")
    print()
    print("checkpoints:")
    for index, checkpoint_mape, count in checkpoints[-10:]:
        print(f"  after {index:5d} records: mape={checkpoint_mape:.3f}%, n={count}")


def run_sparse_knn(records: list[TaxiRecord], args: argparse.Namespace) -> None:
    values = [record.value for record in records]
    coder = EqualWidthValueCoder(values, bins=args.value_bins)
    value_symbols = [coder.encode(value) for value in values]

    encoder = SSTDDiscreteEncoder(num_columns=args.num_columns, k=args.k, seed=args.seed)
    memory = SparseKNNRegressor(encoder=encoder)

    predictions: list[float] = []
    targets: list[float] = []
    checkpoints: list[tuple[int, float, int]] = []
    table_history: dict[tuple[str, int, int], list[float]] = {}
    seasonal_history: dict[tuple[int, int], list[float]] = {}

    max_index = len(records) - args.horizon
    if args.eval_end > 0:
        max_index = min(max_index, args.eval_end)
    for index in range(max_index):
        fields = context_prefix(records, index, args.horizon, value_symbols[index])
        target = values[index + args.horizon]
        future = records[index + args.horizon]
        future_slot = future.timestamp.hour * 2 + future.timestamp.minute // 30
        table_key = (value_symbols[index], future.timestamp.weekday(), future_slot)
        seasonal_key = (future.timestamp.weekday(), future_slot)

        if index >= args.warmup:
            prediction, used_neighbors = memory.predict(
                fields,
                neighbors=args.knn_neighbors,
                min_overlap=args.knn_min_overlap,
                max_candidates=args.knn_max_candidates,
            )
            if prediction is None and args.knn_fallback:
                table_values = table_history.get(table_key)
                seasonal_values = seasonal_history.get(seasonal_key)
                if table_values:
                    prediction = sum(table_values) / len(table_values)
                elif seasonal_values:
                    prediction = sum(seasonal_values) / len(seasonal_values)
            if prediction is not None and used_neighbors > 0:
                predictions.append(prediction)
                targets.append(target)
            elif prediction is not None and args.knn_fallback:
                predictions.append(prediction)
                targets.append(target)

        memory.observe(fields, target)
        table_history.setdefault(table_key, []).append(target)
        seasonal_history.setdefault(seasonal_key, []).append(target)

        if args.report_every > 0 and (index + 1) % args.report_every == 0:
            checkpoints.append((index + 1, mape(predictions, targets), len(predictions)))

    print("Fig. 9 sparse-knn taxi prediction")
    print(f"records: {len(records)}")
    print(f"time range: {records[0].timestamp} -> {records[-1].timestamp}")
    print(f"horizon steps: {args.horizon}")
    print(f"value bins: {args.value_bins}")
    print(f"mini-columns: {args.num_columns}")
    print(f"k: {args.k}")
    print(f"knn neighbors: {args.knn_neighbors}")
    print(f"knn min overlap: {args.knn_min_overlap}")
    print(f"knn max candidates: {args.knn_max_candidates}")
    print(f"knn fallback: {args.knn_fallback}")
    print(f"predictions evaluated: {len(predictions)}")
    print(f"MAPE: {mape(predictions, targets):.3f}%")
    if args.mape_floor > 0:
        print(
            f"MAPE@floor={args.mape_floor:g}: "
            f"{mape_with_floor(predictions, targets, args.mape_floor):.3f}%"
        )
    print()
    print("checkpoints:")
    for index, checkpoint_mape, count in checkpoints[-10:]:
        print(f"  after {index:5d} records: mape={checkpoint_mape:.3f}%, n={count}")


def run_context_memory(records: list[TaxiRecord], args: argparse.Namespace) -> None:
    values = [record.value for record in records]
    coder = EqualWidthValueCoder(values, bins=args.value_bins)
    value_symbols = [coder.encode(value) for value in values]
    candidate_symbols = coder.symbols()

    encoder = SSTDDiscreteEncoder(num_columns=args.num_columns, k=args.k, seed=args.seed)
    params = MemoryParams(
        l_match=args.l_match,
        forgetting_threshold=args.forgetting_threshold,
    )
    model = SequentialMemory(
        encoder=encoder,
        num_neurons_per_column=args.neurons_per_column,
        params=params,
    )

    predictions: list[float] = []
    targets: list[float] = []
    checkpoints: list[tuple[int, float, int]] = []

    max_index = len(records) - args.horizon
    for index in range(max_index):
        prefix = context_prefix(records, index, args.horizon, value_symbols[index])

        if index >= args.warmup:
            model.reset_state()
            for token in prefix:
                model.observe(token, learn=False)
            predicted_symbol = model.predict_symbol(candidate_symbols=candidate_symbols)
            if predicted_symbol is not None:
                predictions.append(coder.decode(predicted_symbol))
                targets.append(values[index + args.horizon])

        target_symbol = value_symbols[index + args.horizon]
        model.reset_state()
        for token in prefix + [target_symbol]:
            model.step(token)

        if args.report_every > 0 and (index + 1) % args.report_every == 0:
            checkpoints.append((index + 1, mape(predictions, targets), len(predictions)))

    print("Fig. 9 context-memory taxi prediction")
    print(f"records: {len(records)}")
    print(f"time range: {records[0].timestamp} -> {records[-1].timestamp}")
    print(f"horizon steps: {args.horizon}")
    print(f"value bins: {args.value_bins}")
    print(f"mini-columns: {args.num_columns}")
    print(f"neurons per mini-column: {args.neurons_per_column}")
    print(f"predictions evaluated: {len(predictions)}")
    print(f"MAPE: {mape(predictions, targets):.3f}%")
    print()
    print("checkpoints:")
    for index, checkpoint_mape, count in checkpoints[-10:]:
        print(f"  after {index:5d} records: mape={checkpoint_mape:.3f}%, n={count}")


def run_context_table(records: list[TaxiRecord], args: argparse.Namespace) -> None:
    values = [record.value for record in records]
    coder = EqualWidthValueCoder(values, bins=args.value_bins)
    value_symbols = [coder.encode(value) for value in values]

    history: dict[tuple[str, int, int], list[float]] = {}
    predictions: list[float] = []
    targets: list[float] = []
    checkpoints: list[tuple[int, float, int]] = []

    max_index = len(records) - args.horizon
    for index in range(max_index):
        future = records[index + args.horizon]
        future_slot = future.timestamp.hour * 2 + future.timestamp.minute // 30
        key = (
            value_symbols[index],
            future.timestamp.weekday(),
            future_slot,
        )

        if index >= args.warmup and key in history:
            observed = history[key]
            predictions.append(sum(observed) / len(observed))
            targets.append(future.value)

        history.setdefault(key, []).append(future.value)

        if args.report_every > 0 and (index + 1) % args.report_every == 0:
            checkpoints.append((index + 1, mape(predictions, targets), len(predictions)))

    print("Fig. 9 context-table taxi baseline")
    print(f"records: {len(records)}")
    print(f"time range: {records[0].timestamp} -> {records[-1].timestamp}")
    print(f"horizon steps: {args.horizon}")
    print(f"value bins: {args.value_bins}")
    print(f"predictions evaluated: {len(predictions)}")
    print(f"MAPE: {mape(predictions, targets):.3f}%")
    print()
    print("checkpoints:")
    for index, checkpoint_mape, count in checkpoints[-10:]:
        print(f"  after {index:5d} records: mape={checkpoint_mape:.3f}%, n={count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/nyc_taxi.csv")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--eval-end", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--value-bins", type=int, default=80)
    parser.add_argument("--symbol-mode", choices=["value", "record"], default="record")
    parser.add_argument(
        "--baseline",
        choices=[
            "memory",
            "seasonal",
            "context-memory",
            "context-table",
            "sparse-context",
            "sparse-regression",
            "sparse-knn",
        ],
        default="memory",
    )
    parser.add_argument("--context-order", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--context-decay", type=float, default=1.0)
    parser.add_argument("--context-fallback-margin", type=float, default=0.0)
    parser.add_argument("--context-blend", type=float, default=0.0)
    parser.add_argument("--context-backoff", action="store_true")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--knn-neighbors", type=int, default=25)
    parser.add_argument("--knn-min-overlap", type=int, default=10)
    parser.add_argument("--knn-max-candidates", type=int, default=2000)
    parser.add_argument("--knn-fallback", action="store_true")
    parser.add_argument("--mape-floor", type=float, default=0.0)
    parser.add_argument("--num-columns", type=int, default=570)
    parser.add_argument("--neurons-per-column", type=int, default=32)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--l-match", type=int, default=4)
    parser.add_argument("--forgetting-threshold", type=float, default=65.0)
    parser.add_argument("--report-every", type=int, default=2000)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
