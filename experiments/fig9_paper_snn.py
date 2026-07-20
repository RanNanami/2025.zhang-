from __future__ import annotations

import argparse
import csv
import sys
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seqmem.encoding import (
    SSTDCompositeEncoder,
    SSTDPeriodicEncoder,
    SSTDRealValueEncoder,
    SpikeEvent,
    SymbolCode,
)
from seqmem.model import MemoryParams, SequentialMemory


@dataclass(frozen=True)
class TaxiRecord:
    timestamp: datetime
    value: float


def read_records(path: Path, limit: int) -> list[TaxiRecord]:
    records: list[TaxiRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            value_text = row.get("passenger_count") or row.get("value")
            try:
                timestamp = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                value = float(value_text)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                # The [58] file begins with two NuPIC type/flag metadata rows.
                continue
            records.append(TaxiRecord(timestamp=timestamp, value=value))
            if limit > 0 and len(records) >= limit:
                break
    return records


def error_ratio(errors: Iterable[float], targets: Iterable[float]) -> float:
    absolute_error = sum(errors)
    target_scale = sum(abs(target) for target in targets)
    return absolute_error / target_scale if target_scale else 0.0


def mape(predictions: list[float], targets: list[float]) -> float:
    """Return the mean-error/mean-target normalization used by reference [58]."""

    absolute_error = sum(
        abs(prediction - target)
        for prediction, target in zip(predictions, targets)
    )
    return error_ratio((absolute_error,), targets)


def reference_rolling_mape(errors: Iterable[float], target_scale: float) -> float:
    """Return the rolling metric used by the plotting code in reference [58]."""

    values = list(errors)
    if not values or target_scale == 0.0:
        return 0.0
    return sum(values) / len(values) / target_scale


def write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_adaptation(
    path: Path,
    rows: list[dict[str, object]],
    plot_start: datetime,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    selected = [
        row
        for row in rows
        if row["rolling_mape"] != ""
        and datetime.fromisoformat(str(row["target_timestamp"])) >= plot_start
    ]
    if not selected:
        return False
    x = [datetime.fromisoformat(str(row["target_timestamp"])) for row in selected]
    y = [float(row["rolling_mape"]) for row in selected]
    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.plot(x, y, linewidth=1.2, label="DS memory")
    axis.axvline(datetime(2015, 4, 1), color="tab:orange", linestyle="--", label="perturbation")
    axis.set(xlabel="Target time", ylabel="MAPE over last 400 predictions")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def record_values(record: TaxiRecord) -> tuple[float, float, float]:
    slot = record.timestamp.hour * 2 + record.timestamp.minute // 30
    return float(record.timestamp.weekday()), float(slot), record.value


def project_sparse_prediction(
    model: SequentialMemory,
    encoder: SSTDCompositeEncoder,
) -> SymbolCode | None:
    ranked = rank_sparse_predictions(model, encoder, max_predictions=1)
    return ranked[0][1] if ranked else None


def project_eventwise_prediction(
    model: SequentialMemory,
    encoder: SSTDCompositeEncoder,
) -> SymbolCode | None:
    """Reproduce the old per-event projection for controlled diagnostics."""

    projected_events: list[SpikeEvent] = []
    for field in encoder.encoders:
        start = field.column_offset  # type: ignore[attr-defined]
        stop = start + field.num_columns
        used_columns: set[int] = set()
        for event_time in field.event_times:
            choices = [
                (candidate.score, column)
                for column, candidates in model.last_prediction_candidates.items()
                if start <= column < stop and column not in used_columns
                for candidate in candidates
                if abs(candidate.time - event_time)
                <= model.params.timing_tolerance
            ]
            if not choices:
                continue
            _score, column = max(choices)
            used_columns.add(column)
            projected_events.append(SpikeEvent(column, event_time))
    return SymbolCode(tuple(projected_events)) if projected_events else None


def rank_sparse_predictions(
    model: SequentialMemory,
    encoder: SSTDCompositeEncoder,
    max_predictions: int,
) -> list[tuple[tuple[int, float, int, float], SymbolCode]]:
    if max_predictions <= 0:
        return []
    field_rankings: list[
        list[tuple[tuple[int, float, int, float], SymbolCode]]
    ] = []
    for field in encoder.encoders:
        start = field.column_offset  # type: ignore[attr-defined]
        stop = start + field.num_columns
        timed_evidence: dict[tuple[int, float], float] = {}
        column_evidence: dict[int, float] = {}
        for column, candidates in model.last_prediction_candidates.items():
            if not start <= column < stop:
                continue
            for candidate in candidates:
                event = (column, round(candidate.time, 12))
                timed_evidence[event] = max(
                    timed_evidence.get(event, 0.0), candidate.score
                )
                column_evidence[column] = max(
                    column_evidence.get(column, 0.0), candidate.score
                )
        if not timed_evidence:
            return []
        legal_codes = field.likelihood_codes_for_events(  # type: ignore[attr-defined]
            set(timed_evidence)
        )
        if not legal_codes:
            return []

        def evidence(code: SymbolCode) -> tuple[int, float, int, float]:
            timed = [
                timed_evidence.get((event.column, round(event.time, 12)), 0.0)
                for event in code.events
            ]
            columns = [column_evidence.get(event.column, 0.0) for event in code.events]
            return (
                sum(score > 0.0 for score in timed),
                sum(timed),
                sum(score > 0.0 for score in columns),
                sum(columns),
            )

        ranked_field = sorted(
            ((evidence(code), code) for code in legal_codes),
            key=lambda item: (item[0], item[1].columns),
            reverse=True,
        )[:max_predictions]
        field_rankings.append(ranked_field)

    combined: list[tuple[tuple[int, float, int, float], SymbolCode]] = []
    for choices in product(*field_rankings):
        score = tuple(
            sum(choice[0][index] for choice in choices) for index in range(4)
        )
        events = tuple(event for _field_score, code in choices for event in code.events)
        if len(events) == encoder.k:
            combined.append((score, SymbolCode(events)))
    combined.sort(key=lambda item: (item[0], item[1].columns), reverse=True)
    return combined[:max_predictions]


def rollout(
    model: SequentialMemory,
    encoder: SSTDCompositeEncoder,
    steps: int,
    beam_width: int = 1,
    future_context_codes: list[SymbolCode] | None = None,
    projection_mode: str = "coherent",
    retrieval_mode: str = "neural",
) -> SymbolCode | None:
    previous_active_cells = model.previous_active_cells.copy()
    previous_winners = model.previous_winners.copy()
    last_prediction_candidates = {
        column: candidates.copy()
        for column, candidates in model.last_prediction_candidates.items()
    }
    learning_rng_state = model._learning_rng.getstate()
    prediction: SymbolCode | None = None
    try:
        beam: list[
            tuple[tuple[int, float, int, float], dict[int, float], SymbolCode | None]
        ] = [((0, 0.0, 0, 0.0), previous_active_cells, None)]
        for step_index in range(steps):
            expanded: dict[
                tuple[tuple[int, float], ...],
                tuple[
                    tuple[int, float, int, float],
                    dict[int, float],
                    SymbolCode,
                ],
            ] = {}
            for cumulative, active_cells, _previous_code in beam:
                model.previous_active_cells = active_cells
                model.previous_winners = active_cells.copy()
                raw_prediction = model.predict_code()
                if raw_prediction is None:
                    continue
                if projection_mode == "raw":
                    choices = [((0, 0.0, 0, 0.0), raw_prediction)]
                elif projection_mode == "eventwise":
                    eventwise = project_eventwise_prediction(model, encoder)
                    choices = (
                        [((0, 0.0, 0, 0.0), eventwise)] if eventwise is not None else []
                    )
                else:
                    choices = rank_sparse_predictions(
                        model, encoder, max_predictions=beam_width
                    )
                for step_score, code in choices:
                    output_code = code
                    next_active = model.prediction_active_cells(code)
                    if future_context_codes is not None:
                        context_code = future_context_codes[step_index]
                        passenger_events = tuple(
                            event for event in code.events if event.column >= 88
                        )
                        output_code = SymbolCode(context_code.events + passenger_events)
                        next_active = model.prediction_active_cells(
                            SymbolCode(passenger_events)
                        )
                        next_active.update(
                            model.external_input_active_cells(context_code)
                        )
                    if retrieval_mode == "proximal-replay":
                        model.observe_code(output_code, learn=False)
                        next_active = model.previous_active_cells.copy()
                    if not next_active:
                        continue
                    total = tuple(
                        cumulative[index] + step_score[index] for index in range(4)
                    )
                    state_key = tuple(sorted(next_active.items()))
                    previous = expanded.get(state_key)
                    if previous is None or total > previous[0]:
                        expanded[state_key] = (total, next_active, output_code)
            if not expanded:
                return None
            beam = sorted(
                expanded.values(), key=lambda item: item[0], reverse=True
            )[:beam_width]
        prediction = beam[0][2]
        return prediction
    finally:
        model.previous_active_cells = previous_active_cells
        model.previous_winners = previous_winners
        model.last_prediction_candidates = last_prediction_candidates
        model._learning_rng.setstate(learning_rng_state)


def learn_actual_code(model: SequentialMemory, code: SymbolCode) -> None:
    """Advance predictions before applying the paper's three learning cases."""

    model.predict_code()
    model.observe_code(code, learn=True)


def run(args: argparse.Namespace) -> None:
    records = read_records(Path(args.data), args.limit)
    if len(records) <= args.horizon:
        raise ValueError("Not enough records for the requested horizon.")

    day_encoder = SSTDPeriodicEncoder(
        num_columns=30,
        k=10,
        period=7.0,
        column_offset=0,
    )
    time_encoder = SSTDPeriodicEncoder(
        num_columns=58,
        k=10,
        period=48.0,
        column_offset=30,
    )
    passenger_encoder = SSTDRealValueEncoder(
        num_columns=482,
        k=10,
        minimum=args.passenger_min,
        maximum=args.passenger_max,
        column_offset=88,
    )
    encoder = SSTDCompositeEncoder([day_encoder, time_encoder, passenger_encoder])
    model = SequentialMemory(
        encoder=encoder,  # type: ignore[arg-type]
        num_neurons_per_column=32,
        params=MemoryParams(
            l_match=4,
            forgetting_threshold=65.0,
            response_scale=args.response_scale,
            burst_context=args.burst_context,
            intracolumn_inhibition=args.intracolumn_inhibition,
            continuous_dynamics=args.continuous_dynamics,
            integration_step=args.integration_step,
        ),
    )

    predictions: list[float] = []
    targets: list[float] = []
    rows: list[dict[str, object]] = []
    recent_errors: deque[float] = deque(maxlen=args.rolling_window)
    target_scale = sum(abs(record.value) for record in records) / len(records)
    missing = 0
    phase_states = [({}, {}) for _ in range(args.horizon)]
    for index, record in enumerate(records[:-args.horizon]):
        if args.forecast_mode == "direct-lag":
            phase = index % args.horizon
            active_cells, winners = phase_states[phase]
            model.previous_active_cells = active_cells.copy()
            model.previous_winners = winners.copy()
            model.last_prediction_candidates = {}
        code = encoder.encode(record_values(record))
        if args.learning_mode == "paper":
            learn_actual_code(model, code)
        else:
            model.last_prediction_candidates = {}
            model.observe_code(code, learn=True)
        if args.forecast_mode == "direct-lag":
            phase_states[index % args.horizon] = (
                model.previous_active_cells.copy(),
                model.previous_winners.copy(),
            )

        if index >= args.warmup:
            target_record = records[index + args.horizon]
            prediction_value: float | str = ""
            predicted_day_value: float | str = ""
            predicted_slot_value: float | str = ""
            absolute_error_value: float | str = ""
            normalized_error_value: float | str = ""
            rolling_value: float | str = ""
            future_context_codes = None
            if args.known_future_time:
                future_context_codes = [
                    SymbolCode(
                        day_encoder.encode(record_values(future)[0]).events
                        + time_encoder.encode(record_values(future)[1]).events
                    )
                    for future in records[index + 1 : index + args.horizon + 1]
                ]
            if args.forecast_mode == "direct-lag":
                predicted_code = (
                    project_sparse_prediction(model, encoder)
                    if model.predict_code() is not None
                    else None
                )
            else:
                predicted_code = rollout(
                    model,
                    encoder,
                    args.horizon,
                    beam_width=args.rollout_beam_width,
                    future_context_codes=future_context_codes,
                    projection_mode=args.projection_mode,
                    retrieval_mode=args.retrieval_mode,
                )
            if predicted_code is None:
                missing += 1
            else:
                try:
                    prediction = passenger_encoder.decode_likelihood(predicted_code)
                except ValueError:
                    missing += 1
                else:
                    predictions.append(prediction)
                    targets.append(target_record.value)
                    prediction_value = prediction
                    try:
                        predicted_day_value = day_encoder.decode(predicted_code)
                    except ValueError:
                        pass
                    try:
                        predicted_slot_value = time_encoder.decode(predicted_code)
                    except ValueError:
                        pass
                    error = abs(prediction - target_record.value)
                    recent_errors.append(error)
                    absolute_error_value = error
                    normalized_error_value = error / target_scale
                    rolling_value = reference_rolling_mape(
                        recent_errors, target_scale
                    )
            rows.append(
                {
                    "input_index": index + 1,
                    "input_timestamp": record.timestamp.isoformat(sep=" "),
                    "target_timestamp": target_record.timestamp.isoformat(sep=" "),
                    "target": target_record.value,
                    "prediction": prediction_value,
                    "predicted_day": predicted_day_value,
                    "predicted_slot": predicted_slot_value,
                    "absolute_error": absolute_error_value,
                    "normalized_absolute_error": normalized_error_value,
                    "rolling_mape": rolling_value,
                }
            )

        if args.report_every > 0 and (index + 1) % args.report_every == 0:
            print(
                f"after {index + 1}: predictions={len(predictions)}, "
                f"missing={missing}, mape={mape(predictions, targets):.3f}"
            )

    print("Fig. 9 paper-aligned DS-memory approximation")
    print(f"source: {args.data}")
    print(f"records: {len(records)}")
    print(f"date range: {records[0].timestamp} to {records[-1].timestamp}")
    print(f"horizon: {args.horizon} steps")
    print(f"forecast mode: {args.forecast_mode}")
    print(f"learning mode: {args.learning_mode}")
    print(f"projection mode: {args.projection_mode}")
    print(f"retrieval mode: {args.retrieval_mode}")
    print(f"burst cells drive context: {args.burst_context}")
    print(f"intracolumn inhibition: {args.intracolumn_inhibition}")
    print(f"continuous dynamics integration: {args.continuous_dynamics}")
    print(f"known future time clamped: {args.known_future_time}")
    print("column groups: day=30, time=58, passenger=482")
    print("active columns per field: 10")
    print("neurons per column: 32")
    print(f"predictions evaluated: {len(predictions)}")
    print(f"missing predictions: {missing}")
    attempted = len(predictions) + missing
    coverage = len(predictions) / attempted if attempted else 0.0
    print(f"prediction coverage: {coverage:.3f}")
    print(f"MAPE: {mape(predictions, targets):.3f}")
    if args.output_csv:
        write_predictions(Path(args.output_csv), rows)
        print(f"saved predictions: {args.output_csv}")
    if args.output_plot:
        plotted = plot_adaptation(
            Path(args.output_plot),
            rows,
            datetime.fromisoformat(args.plot_start),
        )
        print(f"adaptation plot generated: {plotted}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/paper_nyc_taxi.csv")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--warmup",
        type=int,
        default=5904,
        help="Evaluation starts after record 5904, matching reference [58].",
    )
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument(
        "--forecast-mode",
        choices=("recursive", "direct-lag"),
        default="recursive",
        help="Direct-lag is a diagnostic because the paper does not specify horizon training.",
    )
    parser.add_argument(
        "--learning-mode",
        choices=("paper", "growth-only"),
        default="paper",
        help="Growth-only reproduces the old runner omission for diagnosis only.",
    )
    parser.add_argument("--rollout-beam-width", type=int, default=1)
    parser.add_argument(
        "--projection-mode",
        choices=("raw", "coherent", "eventwise"),
        default="raw",
        help="Raw follows predictive neuron spikes; other modes are diagnostics.",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=("neural", "proximal-replay"),
        default="neural",
    )
    parser.add_argument(
        "--known-future-time",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Clamp only known future weekday/time covariates during rollout.",
    )
    parser.add_argument("--passenger-min", type=float, default=0.0)
    parser.add_argument("--passenger-max", type=float, default=40000.0)
    parser.add_argument(
        "--response-scale",
        type=float,
        default=None,
        help="Optional V0 override; default normalizes the kernel peak to one.",
    )
    parser.add_argument(
        "--burst-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="All cells fire when an active mini-column has no predictive cell.",
    )
    parser.add_argument(
        "--intracolumn-inhibition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only the earliest predictive spike per mini-column.",
    )
    parser.add_argument(
        "--continuous-dynamics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Numerically integrate dendritic threshold crossing and soma firing.",
    )
    parser.add_argument("--integration-step", type=float, default=0.005)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--rolling-window", type=int, default=400)
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--output-plot", default="")
    parser.add_argument("--plot-start", default="2015-03-25")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
