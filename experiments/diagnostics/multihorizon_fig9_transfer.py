from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "diagnostics"))

from benchmark_real_timeseries import SPECS, read_dataset
from etth1_paper_snn import scores, seasonal_source_index
from fig9_paper_snn import learn_actual_code, rank_sparse_predictions, write_predictions
from multivariate_fig9_transfer import (
    DEFAULT_DATA,
    build_encoder,
    make_record_values,
    select_feature_indices,
)
from seqmem.encoding import SSTDCompositeEncoder, SSTDRealValueEncoder, SymbolCode
from seqmem.model import MemoryParams, SequentialMemory


def parse_horizons(value: str) -> tuple[int, ...]:
    horizons = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    if not horizons or horizons[0] <= 0:
        raise argparse.ArgumentTypeError("horizons must be positive integers")
    return horizons


def rollout_checkpoints(
    model: SequentialMemory,
    encoder: SSTDCompositeEncoder,
    horizons: tuple[int, ...],
) -> dict[int, SymbolCode]:
    """Follow one greedy Fig. 9 rollout and retain requested intermediate codes."""

    requested = set(horizons)
    previous_active_cells = model.previous_active_cells.copy()
    previous_winners = model.previous_winners.copy()
    last_prediction_candidates = {
        column: candidates.copy()
        for column, candidates in model.last_prediction_candidates.items()
    }
    learning_rng_state = model._learning_rng.getstate()
    checkpoints: dict[int, SymbolCode] = {}
    active_cells = previous_active_cells
    try:
        for step in range(1, max(horizons) + 1):
            model.previous_active_cells = active_cells
            model.previous_winners = active_cells.copy()
            if model.predict_code() is None:
                break
            ranked = rank_sparse_predictions(model, encoder, max_predictions=1)
            if not ranked:
                break
            code = ranked[0][1]
            active_cells = model.prediction_active_cells(code)
            if not active_cells:
                break
            if step in requested:
                checkpoints[step] = code
        return checkpoints
    finally:
        model.previous_active_cells = previous_active_cells
        model.previous_winners = previous_winners
        model.last_prediction_candidates = last_prediction_candidates
        model._learning_rng.setstate(learning_rng_state)


def run(args: argparse.Namespace) -> None:
    spec = SPECS[args.dataset]
    data_path = Path(args.data or DEFAULT_DATA[args.dataset])
    dates, columns, rows = read_dataset(data_path, spec)
    horizons = args.horizons
    max_horizon = max(horizons)
    evaluation_stop = min(len(rows) - max_horizon, args.warmup + args.test_size)
    if args.warmup <= max_horizon or evaluation_stop <= args.warmup:
        raise ValueError("The requested warmup/test split is not valid.")

    selected_indices = select_feature_indices(
        columns,
        rows,
        spec.target_column,
        args.warmup,
        max_horizon,
        args.max_features,
    )
    target_index = columns.index(spec.target_column)
    encoder, target_encoder = build_encoder(
        rows,
        selected_indices,
        target_index,
        args.warmup,
        spec.seasonal_period,
        args.feature_columns,
        args.target_columns,
        args.active_columns,
    )
    model = SequentialMemory(
        encoder=encoder,  # type: ignore[arg-type]
        num_neurons_per_column=args.neurons_per_column,
        params=MemoryParams(l_match=4, forgetting_threshold=65.0),
    )

    predictions = {horizon: [] for horizon in horizons}
    targets = {horizon: [] for horizon in horizons}
    persistence = {horizon: [] for horizon in horizons}
    seasonal = {horizon: [] for horizon in horizons}
    baseline_targets = {horizon: [] for horizon in horizons}
    missing = {horizon: 0 for horizon in horizons}
    output_rows: list[dict[str, object]] = []

    for index in range(evaluation_stop):
        actual_code = encoder.encode(
            make_record_values(dates[index], rows[index], selected_indices, spec)
        )
        learn_actual_code(model, actual_code)
        if index >= args.warmup:
            codes = rollout_checkpoints(model, encoder, horizons)
            for horizon in horizons:
                target = rows[index + horizon][target_index]
                persistence_value = rows[index][target_index]
                seasonal_value = rows[
                    seasonal_source_index(index, horizon, spec.seasonal_period)
                ][target_index]
                persistence[horizon].append(persistence_value)
                seasonal[horizon].append(seasonal_value)
                baseline_targets[horizon].append(target)
                prediction: float | str = ""
                code = codes.get(horizon)
                if code is None:
                    missing[horizon] += 1
                else:
                    try:
                        prediction = target_encoder.decode_likelihood(code)
                    except ValueError:
                        missing[horizon] += 1
                    else:
                        predictions[horizon].append(prediction)
                        targets[horizon].append(target)
                output_rows.append(
                    {
                        "horizon": horizon,
                        "input_index": index,
                        "input_timestamp": dates[index].isoformat(sep=" "),
                        "target_timestamp": dates[index + horizon].isoformat(sep=" "),
                        "target": target,
                        "prediction": prediction,
                        "persistence_prediction": persistence_value,
                        "daily_seasonal_prediction": seasonal_value,
                    }
                )
        if args.report_every > 0 and (index + 1) % args.report_every == 0:
            print(f"after {index + 1} records", flush=True)

    print(f"Multihorizon Fig. 9 DS transfer: {spec.name}")
    print(f"source: {data_path}")
    print(f"split: warmup={args.warmup}, test={evaluation_stop - args.warmup}")
    print(f"horizons: {', '.join(map(str, horizons))}")
    print(
        "selected numeric fields: "
        + ", ".join(columns[index] for index in selected_indices)
    )
    print("horizon,model_MAE,model_RMSE,persistence_MAE,seasonal_MAE,coverage")
    summary_rows: list[dict[str, object]] = []
    for horizon in horizons:
        model_mae, model_rmse, _ = scores(predictions[horizon], targets[horizon])
        persistence_mae, _, _ = scores(
            persistence[horizon], baseline_targets[horizon]
        )
        seasonal_mae, _, _ = scores(seasonal[horizon], baseline_targets[horizon])
        attempted = len(predictions[horizon]) + missing[horizon]
        coverage = len(predictions[horizon]) / attempted if attempted else 0.0
        print(
            f"{horizon},{model_mae:.6f},{model_rmse:.6f},"
            f"{persistence_mae:.6f},{seasonal_mae:.6f},{coverage:.3f}"
        )
        summary_rows.append(
            {
                "horizon": horizon,
                "model_mae": model_mae,
                "model_rmse": model_rmse,
                "persistence_mae": persistence_mae,
                "seasonal_mae": seasonal_mae,
                "coverage": coverage,
                "predictions": len(predictions[horizon]),
            }
        )

    if args.output_csv:
        write_predictions(Path(args.output_csv), output_rows)
        print(f"saved predictions: {args.output_csv}")
    if args.output_summary:
        path = Path(args.output_summary)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"saved summary: {args.output_summary}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one multivariate Fig. 9 DS model at several horizons."
    )
    parser.add_argument("--dataset", choices=sorted(SPECS), required=True)
    parser.add_argument("--data", default="")
    parser.add_argument("--horizons", type=parse_horizons, default=(12, 24, 48, 96))
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=20)
    parser.add_argument("--max-features", type=int, default=8)
    parser.add_argument("--feature-columns", type=int, default=58)
    parser.add_argument("--target-columns", type=int, default=482)
    parser.add_argument("--active-columns", type=int, default=10)
    parser.add_argument("--neurons-per-column", type=int, default=32)
    parser.add_argument("--report-every", type=int, default=250)
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--output-summary", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
