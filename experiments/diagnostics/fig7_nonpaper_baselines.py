from __future__ import annotations

"""Non-paper diagnostic models.

Zhang et al. copy their comparison curves from reference [58]. The local
PyTorch/NumPy models below are useful stress tests, but are not reproductions
of those reported curves and are intentionally excluded from strict runners.
"""

import argparse
import csv
import random
import statistics
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

from fig7_sequence_prediction import (
    build_sequence_set,
    ending_count,
    execute,
    make_stream,
)
from seqmem.encoding import SSTDDiscreteEncoder


@dataclass
class BaselineResult:
    model: str
    backend: str
    evaluated: int
    cumulative_accuracy: float
    final_moving_accuracy: float
    checkpoints: list[tuple[int, float]]
    note: str = ""


class NeuralOnlineModel:
    def __init__(
        self,
        kind: str,
        vocabulary_size: int,
        class_count: int,
        context_length: int,
        seed: int,
    ) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as error:
            raise RuntimeError("PyTorch is required for TDNN and LSTM baselines") from error

        torch.manual_seed(seed)
        self.torch = torch
        self.context_length = context_length
        embedding_size = 24
        if kind == "tdnn":
            self.network = nn.Sequential(
                nn.Embedding(vocabulary_size, embedding_size),
                _PermuteForConv(),
                nn.Conv1d(embedding_size, 48, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1),
                nn.Flatten(),
                nn.Linear(48, class_count),
            )
        elif kind == "lstm":
            self.network = _LSTMNetwork(
                vocabulary_size, embedding_size, hidden_size=48, class_count=class_count
            )
        else:
            raise ValueError(kind)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.01)
        self.loss = nn.CrossEntropyLoss()

    def scores(self, context: list[int]):
        torch = self.torch
        self.network.eval()
        with torch.no_grad():
            return self.network(torch.tensor([context], dtype=torch.long))[0]

    def learn(self, context: list[int], target: int) -> None:
        torch = self.torch
        self.network.train()
        self.optimizer.zero_grad()
        logits = self.network(torch.tensor([context], dtype=torch.long))
        loss = self.loss(logits, torch.tensor([target], dtype=torch.long))
        loss.backward()
        self.optimizer.step()


def _torch_modules():
    from torch import nn

    class PermuteForConv(nn.Module):
        def forward(self, values):
            return values.transpose(1, 2)

    class LSTMNetwork(nn.Module):
        def __init__(self, vocabulary_size, embedding_size, hidden_size, class_count):
            super().__init__()
            self.embedding = nn.Embedding(vocabulary_size, embedding_size)
            self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
            self.output = nn.Linear(hidden_size, class_count)

        def forward(self, values):
            embedded = self.embedding(values)
            sequence, _state = self.lstm(embedded)
            return self.output(sequence[:, -1, :])

    return PermuteForConv, LSTMNetwork


try:
    _PermuteForConv, _LSTMNetwork = _torch_modules()
except ImportError:
    _PermuteForConv = None  # type: ignore[assignment]
    _LSTMNetwork = None  # type: ignore[assignment]


class ELMOnlineModel:
    """Fixed random hidden layer with an online-trained output layer."""

    def __init__(
        self,
        vocabulary_size: int,
        class_count: int,
        context_length: int,
        seed: int,
    ) -> None:
        import numpy as np

        self.np = np
        rng = np.random.default_rng(seed)
        embedding_size = 16
        hidden_size = 128
        self.embedding = rng.normal(0, 0.3, (vocabulary_size, embedding_size))
        self.hidden_weight = rng.normal(
            0, 0.15, (context_length * embedding_size, hidden_size)
        )
        self.hidden_bias = rng.normal(0, 0.1, hidden_size)
        self.output_weight = np.zeros((hidden_size, class_count))
        self.learning_rate = 0.08

    def _hidden(self, context: list[int]):
        values = self.embedding[context].reshape(-1)
        return self.np.tanh(values @ self.hidden_weight + self.hidden_bias)

    def scores(self, context: list[int]):
        return self._hidden(context) @ self.output_weight

    def learn(self, context: list[int], target: int) -> None:
        hidden = self._hidden(context)
        logits = hidden @ self.output_weight
        shifted = logits - logits.max()
        probabilities = self.np.exp(shifted)
        probabilities /= probabilities.sum()
        probabilities[target] -= 1.0
        self.output_weight -= self.learning_rate * self.np.outer(hidden, probabilities)


def top_predictions(
    scores,
    class_symbols: list[str],
    count: int,
    eligible_symbols: set[str],
) -> list[str]:
    try:
        values = scores.detach().cpu().tolist()
    except AttributeError:
        values = scores.tolist()
    ranked = sorted(
        (
            index
            for index, symbol in enumerate(class_symbols)
            if symbol in eligible_symbols
        ),
        key=lambda index: (-values[index], index),
    )
    return [class_symbols[index] for index in ranked[:count]]


def run_supervised_baseline(
    name: str,
    stream,
    sequences: list[list[str]],
    symbol_to_id: dict[str, int],
    class_symbols: list[str],
    max_predictions: int,
    context_length: int,
    seed: int,
    report_every: int,
) -> BaselineResult:
    class_to_id = {symbol: index for index, symbol in enumerate(class_symbols)}
    ending_symbols = set(stream.expected_by_index.values())
    valid_transitions = {
        (sequence[index], sequence[index + 1])
        for sequence in sequences
        for index in range(len(sequence) - 1)
    }
    if name == "elm":
        model = ELMOnlineModel(
            len(symbol_to_id) + 1, len(class_symbols), context_length, seed
        )
        backend = "numpy-online-elm"
    else:
        model = NeuralOnlineModel(
            name,
            len(symbol_to_id) + 1,
            len(class_symbols),
            context_length,
            seed,
        )
        backend = f"pytorch-{name}"

    history: deque[int] = deque([0] * context_length, maxlen=context_length)
    moving: deque[bool] = deque(maxlen=100)
    checkpoints: list[tuple[int, float]] = []
    correct = 0
    evaluated = 0
    for element, symbol in enumerate(stream.symbols, start=1):
        history.append(symbol_to_id[symbol])
        context = list(history)
        expected = stream.expected_by_index.get(element)
        if expected is not None:
            predicted = top_predictions(
                model.scores(context),
                class_symbols,
                max_predictions,
                ending_symbols,
            )
            hit = expected in predicted
            correct += hit
            evaluated += 1
            moving.append(hit)
        target = None
        if element < len(stream.symbols):
            transition = (symbol, stream.symbols[element])
            if transition in valid_transitions:
                target = transition[1]
        if target is not None:
            model.learn(context, class_to_id[target])
        if report_every > 0 and element % report_every == 0 and moving:
            checkpoints.append((element, sum(moving) / len(moving)))

    return BaselineResult(
        model=name.upper(),
        backend=backend,
        evaluated=evaluated,
        cumulative_accuracy=correct / evaluated if evaluated else 0.0,
        final_moving_accuracy=sum(moving) / len(moving) if moving else 0.0,
        checkpoints=checkpoints,
        note="All symbols enter context; supervised update occurs on sequence transitions.",
    )


def run_htm_baseline(
    stream,
    encoder: SSTDDiscreteEncoder,
    candidate_symbols: set[str],
    max_predictions: int,
    report_every: int,
) -> BaselineResult:
    try:
        import numpy as np
        from htm.bindings.algorithms import TemporalMemory
        from htm.bindings.sdr import SDR
    except ImportError:
        return BaselineResult(
            model="HTM",
            backend="unavailable",
            evaluated=0,
            cumulative_accuracy=float("nan"),
            final_moving_accuracy=float("nan"),
            checkpoints=[],
            note="htm.core is not installed; this row is intentionally not replaced by a mock.",
        )

    memory = TemporalMemory(
        columnDimensions=(encoder.num_columns,),
        cellsPerColumn=10,
        activationThreshold=8,
        initialPermanence=0.5,
        connectedPermanence=0.5,
        minThreshold=3,
        maxNewSynapseCount=10,
        permanenceIncrement=0.1,
        permanenceDecrement=0.01,
        predictedSegmentDecrement=0.01,
    )
    moving: deque[bool] = deque(maxlen=100)
    checkpoints: list[tuple[int, float]] = []
    correct = 0
    evaluated = 0
    codes = {symbol: encoder.encode(symbol) for symbol in candidate_symbols}
    for element, symbol in enumerate(stream.symbols, start=1):
        active = SDR((encoder.num_columns,))
        active.sparse = np.asarray(encoder.encode(symbol).columns, dtype=np.uint32)
        memory.compute(active, learn=True)
        memory.activateDendrites(learn=True)
        expected = stream.expected_by_index.get(element)
        if expected is not None:
            predicted_columns = {
                memory.columnForCell(int(cell))
                for cell in memory.getPredictiveCells().sparse
            }
            ranked = sorted(
                candidate_symbols,
                key=lambda candidate: (
                    -len(predicted_columns & set(codes[candidate].columns)), candidate
                ),
            )[:max_predictions]
            hit = expected in ranked and bool(predicted_columns)
            correct += hit
            evaluated += 1
            moving.append(hit)
        if report_every > 0 and element % report_every == 0 and moving:
            checkpoints.append((element, sum(moving) / len(moving)))

    return BaselineResult(
        model="HTM",
        backend="htm.core TemporalMemory",
        evaluated=evaluated,
        cumulative_accuracy=correct / evaluated if evaluated else 0.0,
        final_moving_accuracy=sum(moving) / len(moving) if moving else 0.0,
        checkpoints=checkpoints,
    )


def snn_args(args: argparse.Namespace, seed: int) -> argparse.Namespace:
    return argparse.Namespace(
        seed=seed,
        task=args.task,
        max_elements=args.max_elements,
        noise_symbols=50000,
        modify_after_elements=args.modify_after_elements,
        forgetting_threshold=25.0 if args.task == "single" else 50.0,
        recovery_accuracy=1.0,
        report_every=args.report_every,
        output_csv="",
    )


def run_trial(args: argparse.Namespace, seed: int) -> list[BaselineResult]:
    sequences = build_sequence_set(args.task)
    stream = make_stream(
        sequences,
        args.max_elements,
        50000,
        random.Random(seed),
        args.modify_after_elements,
    )
    all_symbols = sorted(set(stream.symbols))
    symbol_to_id = {symbol: index + 1 for index, symbol in enumerate(all_symbols)}
    class_symbols = sorted({symbol for sequence in sequences for symbol in sequence})
    maximum = ending_count(args.task)
    results: list[BaselineResult] = []
    for model in args.models:
        print(f"  running {model}", flush=True)
        if model == "snn":
            result = execute(snn_args(args, seed))
            results.append(
                BaselineResult(
                    "SNN",
                    "paper-aligned SequentialMemory",
                    result.evaluated,
                    result.cumulative_accuracy,
                    result.final_moving_accuracy,
                    [(element, moving) for element, moving, _ in result.checkpoints],
                )
            )
        elif model == "htm":
            encoder = SSTDDiscreteEncoder(num_columns=200, k=10, seed=seed)
            results.append(
                run_htm_baseline(
                    stream,
                    encoder,
                    set(stream.expected_by_index.values()),
                    maximum,
                    args.report_every,
                )
            )
        else:
            results.append(
                run_supervised_baseline(
                    model,
                    stream,
                    sequences,
                    symbol_to_id,
                    class_symbols,
                    maximum,
                    args.context_length,
                    seed,
                    args.report_every,
                )
            )
    return results


def write_results(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(
    path: Path,
    rows: list[dict[str, object]],
    modification_element: int,
) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 5.2))
    for model in sorted({str(row["model"]) for row in rows}):
        selected = [row for row in rows if row["model"] == model]
        x = [int(row["elements"]) for row in selected]
        mean = [float(row["mean_moving_accuracy"]) for row in selected]
        std = [float(row["std_moving_accuracy"]) for row in selected]
        axis.plot(x, mean, label=model)
        axis.fill_between(
            x,
            [max(0.0, value - spread) for value, spread in zip(mean, std)],
            [min(1.0, value + spread) for value, spread in zip(mean, std)],
            alpha=0.16,
        )
    if modification_element >= 0:
        axis.axvline(
            modification_element,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="modification",
        )
    axis.set(
        xlabel="Processed elements",
        ylabel="Moving accuracy (last 100)",
        ylim=(0.0, 1.02),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["single", "multiple-2", "multiple-4"], default="single")
    parser.add_argument("--models", nargs="+", choices=["snn", "htm", "tdnn", "lstm", "elm"], default=["snn", "htm", "tdnn", "lstm", "elm"])
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--max-elements", type=int, default=20000)
    parser.add_argument("--context-length", type=int, default=7)
    parser.add_argument("--modify-after-elements", type=int, default=10000)
    parser.add_argument("--report-every", type=int, default=250)
    parser.add_argument("--output-dir", default="results/diagnostics/fig7_nonpaper_baselines")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trial_results: list[tuple[int, BaselineResult]] = []
    for trial in range(args.trials):
        seed = args.seed + trial
        print(f"trial {trial + 1}/{args.trials}, seed={seed}", flush=True)
        trial_results.extend((trial + 1, result) for result in run_trial(args, seed))

    detail_rows: list[dict[str, object]] = []
    for trial, result in trial_results:
        detail_rows.append(
            {
                "trial": trial,
                "model": result.model,
                "backend": result.backend,
                "evaluated": result.evaluated,
                "cumulative_accuracy": result.cumulative_accuracy,
                "final_moving_accuracy": result.final_moving_accuracy,
                "note": result.note,
            }
        )
    summary_rows: list[dict[str, object]] = []
    for model in sorted({result.model for _trial, result in trial_results}):
        selected = [
            result.final_moving_accuracy
            for _trial, result in trial_results
            if result.model == model and result.evaluated
        ]
        summary_rows.append(
            {
                "model": model,
                "completed_trials": len(selected),
                "mean_final_moving_accuracy": statistics.fmean(selected) if selected else "",
                "std_final_moving_accuracy": statistics.pstdev(selected) if selected else "",
            }
        )
    curve_values: dict[tuple[str, int], list[float]] = {}
    for _trial, result in trial_results:
        for elements, moving_accuracy in result.checkpoints:
            curve_values.setdefault((result.model, elements), []).append(
                moving_accuracy
            )
    curve_rows: list[dict[str, object]] = []
    for (model, elements), values in sorted(curve_values.items()):
        curve_rows.append(
            {
                "model": model,
                "elements": elements,
                "mean_moving_accuracy": statistics.fmean(values),
                "std_moving_accuracy": statistics.pstdev(values),
                "completed_trials": len(values),
            }
        )
    output = Path(args.output_dir)
    write_results(output / "trials.csv", detail_rows)
    write_results(output / "summary.csv", summary_rows)
    write_results(output / "curves.csv", curve_rows)
    plot_curves(output / "curves.png", curve_rows, args.modify_after_elements)
    for row in summary_rows:
        print(row)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
