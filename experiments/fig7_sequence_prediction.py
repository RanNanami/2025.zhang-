from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seqmem.encoding import SSTDDiscreteEncoder
from seqmem.model import MemoryParams, SequentialMemory


PAPER_SEQUENCE_SETS = {
    "single": [
        [6, 8, 7, 4, 2, 3, 0],
        [1, 8, 7, 4, 2, 3, 5],
        [6, 3, 4, 2, 7, 8, 5],
        [1, 3, 4, 2, 7, 8, 0],
        [0, 9, 7, 8, 5, 3, 4, 1],
        [2, 9, 7, 8, 5, 3, 4, 6],
        [0, 4, 3, 5, 8, 7, 9, 6],
        [2, 4, 3, 5, 8, 7, 9, 1],
    ],
    "multiple-2": [
        [4, 8, 3, 10, 9, 6, 1], [4, 8, 3, 10, 9, 6, 2],
        [5, 8, 3, 10, 9, 6, 0], [5, 8, 3, 10, 9, 6, 7],
        [4, 6, 9, 10, 3, 8, 7], [4, 6, 9, 10, 3, 8, 0],
        [5, 6, 9, 10, 3, 8, 2], [5, 6, 9, 10, 3, 8, 1],
        [4, 3, 8, 6, 1, 10, 11, 9], [4, 3, 8, 6, 1, 10, 11, 2],
        [5, 3, 8, 6, 1, 10, 11, 0], [5, 3, 8, 6, 1, 10, 11, 7],
        [4, 11, 10, 1, 6, 8, 3, 7], [4, 11, 10, 1, 6, 8, 3, 0],
        [5, 11, 10, 1, 6, 8, 3, 2], [5, 11, 10, 1, 6, 8, 3, 9],
    ],
    "multiple-4": [
        [7, 4, 12, 5, 14, 1, 2], [7, 4, 12, 5, 14, 1, 3],
        [7, 4, 12, 5, 14, 1, 0], [7, 4, 12, 5, 14, 1, 9],
        [11, 4, 12, 5, 14, 1, 13], [11, 4, 12, 5, 14, 1, 10],
        [11, 4, 12, 5, 14, 1, 6], [11, 4, 12, 5, 14, 1, 8],
        [7, 1, 14, 5, 12, 4, 8], [7, 1, 14, 5, 12, 4, 6],
        [7, 1, 14, 5, 12, 4, 10], [7, 1, 14, 5, 12, 4, 13],
        [11, 1, 14, 5, 12, 4, 9], [11, 1, 14, 5, 12, 4, 0],
        [11, 1, 14, 5, 12, 4, 3], [11, 1, 14, 5, 12, 4, 2],
        [9, 4, 5, 15, 6, 1, 12, 2], [9, 4, 5, 15, 6, 1, 12, 3],
        [9, 4, 5, 15, 6, 1, 12, 0], [9, 4, 5, 15, 6, 1, 12, 10],
        [13, 4, 5, 15, 6, 1, 12, 14], [13, 4, 5, 15, 6, 1, 12, 11],
        [13, 4, 5, 15, 6, 1, 12, 7], [13, 4, 5, 15, 6, 1, 12, 8],
        [9, 1, 12, 6, 15, 4, 5, 8], [9, 1, 12, 6, 15, 4, 5, 7],
        [9, 1, 12, 6, 15, 4, 5, 11], [9, 1, 12, 6, 15, 4, 5, 14],
        [13, 1, 12, 6, 15, 4, 5, 10], [13, 1, 12, 6, 15, 4, 5, 0],
        [13, 1, 12, 6, 15, 4, 5, 3], [13, 1, 12, 6, 15, 4, 5, 2],
    ],
}


@dataclass(frozen=True)
class StreamData:
    symbols: list[str]
    expected_by_index: dict[int, str]
    modification_element: int | None


@dataclass
class Fig7Result:
    task: str
    stream_length: int
    evaluated: int
    final_moving_accuracy: float
    checkpoints: list[tuple[int, float]]
    modification_element: int | None
    recovery_elements: int | None
    prediction_count_histogram: dict[int, int]
    ending_accuracy: dict[str, tuple[int, int]]


def build_sequence_set(task: str) -> list[list[str]]:
    return [[str(symbol) for symbol in sequence] for sequence in PAPER_SEQUENCE_SETS[task]]


def swap_pair_endings(sequences: list[list[str]]) -> list[list[str]]:
    modified = [sequence.copy() for sequence in sequences]
    for index in range(0, len(modified), 2):
        if index + 1 < len(modified):
            modified[index][-1], modified[index + 1][-1] = (
                modified[index + 1][-1],
                modified[index][-1],
            )
    return modified


def make_stream(
    sequences: list[list[str]],
    max_elements: int,
    noise_symbols: int,
    rng: random.Random,
    modify_after_elements: int,
) -> StreamData:
    stream: list[str] = []
    expected_by_index: dict[int, str] = {}
    noise_pool = [f"noise_{index}" for index in range(noise_symbols)]
    modified_sequences = swap_pair_endings(sequences)
    modification_element: int | None = None

    while len(stream) < max_elements:
        modified = modify_after_elements >= 0 and len(stream) >= modify_after_elements
        if modified and modification_element is None:
            modification_element = len(stream) + 1
        active_sequences = modified_sequences if modified else sequences
        sequence = rng.choice(active_sequences)
        for symbol_index, symbol in enumerate(sequence):
            if len(stream) >= max_elements:
                break
            stream.append(symbol)
            if symbol_index == len(sequence) - 2:
                expected_by_index[len(stream)] = sequence[-1]

        if noise_pool and len(stream) < max_elements:
            stream.append(rng.choice(noise_pool))

    return StreamData(
        stream,
        expected_by_index,
        modification_element,
    )


def ending_count(task: str) -> int:
    return {"single": 1, "multiple-2": 2, "multiple-4": 4}[task]


def paper_forgetting_threshold(task: str) -> float:
    return 25.0 if task == "single" else 50.0


def execute(args: argparse.Namespace) -> Fig7Result:
    rng = random.Random(args.seed)
    sequences = build_sequence_set(args.task)
    stream = make_stream(
        sequences=sequences,
        max_elements=(
            args.max_elements
            if args.max_elements > 0
            else (20000 if args.task == "single" else 10000)
        ),
        noise_symbols=args.noise_symbols,
        rng=rng,
        modify_after_elements=args.modify_after_elements,
    )
    final_sequences = (
        swap_pair_endings(sequences)
        if stream.modification_element is not None
        else sequences
    )

    encoder = SSTDDiscreteEncoder(num_columns=200, k=10, seed=args.seed)
    sequence_vocabulary = {
        symbol for sequence in sequences for symbol in sequence
    }
    for symbol in sorted(sequence_vocabulary, key=int):
        encoder.encode(symbol)
    model = SequentialMemory(
        encoder=encoder,
        num_neurons_per_column=10,
        tie_break_seed=args.seed,
        params=MemoryParams(
            l_match=3,
            forgetting_threshold=(
                args.forgetting_threshold
                if args.forgetting_threshold is not None
                else paper_forgetting_threshold(args.task)
            ),
        ),
    )

    max_predictions = ending_count(args.task)
    total = 0
    moving: deque[bool] = deque(maxlen=100)
    checkpoints: list[tuple[int, float]] = []
    recovery_elements: int | None = None
    prediction_count_histogram: dict[int, int] = {}
    ending_counts: dict[str, list[int]] = {}

    for index, symbol in enumerate(stream.symbols, start=1):
        # Advance the network without decoding against every known symbol.
        # With the paper's 50,000-item noise vocabulary, decoding on every
        # input would add an unrelated O(vocabulary) cost to online learning.
        model.predict_code()
        model.observe(symbol)
        expected = stream.expected_by_index.get(index)
        if expected is not None:
            predictions = model.predict_symbols(
                max_predictions=max_predictions,
                candidate_symbols=sequence_vocabulary,
            )
            is_correct = expected in predictions
            prediction_count_histogram[len(predictions)] = (
                prediction_count_histogram.get(len(predictions), 0) + 1
            )
            ending = ending_counts.setdefault(expected, [0, 0])
            ending[1] += 1
            if is_correct:
                ending[0] += 1
            total += 1
            moving.append(is_correct)
            if (
                stream.modification_element is not None
                and index >= stream.modification_element
                and recovery_elements is None
                and len(moving) == moving.maxlen
                and sum(moving) / len(moving) >= args.recovery_accuracy
            ):
                recovery_elements = index - stream.modification_element

        if args.report_every > 0 and index % args.report_every == 0 and moving:
            checkpoints.append((index, sum(moving) / len(moving)))

    return Fig7Result(
        task=args.task,
        stream_length=len(stream.symbols),
        evaluated=total,
        final_moving_accuracy=sum(moving) / len(moving) if moving else 0.0,
        checkpoints=checkpoints,
        modification_element=stream.modification_element,
        recovery_elements=recovery_elements,
        prediction_count_histogram=prediction_count_histogram,
        ending_accuracy={
            ending: (counts[0], counts[1])
            for ending, counts in ending_counts.items()
        },
    )


def write_csv(path: Path, result: Fig7Result) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["elements", "moving_accuracy_100"])
        writer.writerows(result.checkpoints)


def print_result(result: Fig7Result, args: argparse.Namespace) -> None:
    print("Fig. 7 strict online sequence prediction")
    print(f"task: {result.task}")
    print(f"stream length: {result.stream_length} symbols")
    print(f"evaluated sequence endings: {result.evaluated}")
    print("spike-response dynamics: enabled (paper protocol)")
    print(f"final moving accuracy (100): {result.final_moving_accuracy:.3f}")
    print(f"modification element: {result.modification_element}")
    print(f"recovery elements: {result.recovery_elements}")
    print(f"prediction count histogram: {result.prediction_count_histogram}")
    worst_endings = sorted(
        result.ending_accuracy.items(),
        key=lambda item: item[1][0] / item[1][1],
    )[:5]
    print(
        "lowest ending accuracies: "
        + ", ".join(
            f"{ending}={correct / total:.3f} ({correct}/{total})"
            for ending, (correct, total) in worst_endings
        )
    )
    print("accuracy checkpoints:")
    for elements, moving_accuracy in result.checkpoints[-10:]:
        print(f"  after {elements:5d}: moving100={moving_accuracy:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--task", choices=["single", "multiple-2", "multiple-4"], default="single")
    parser.add_argument(
        "--max-elements",
        type=int,
        default=0,
        help="Paper default: 20000 for single-ending, 10000 for multiple-ending.",
    )
    parser.add_argument("--noise-symbols", type=int, default=50000)
    parser.add_argument(
        "--modify-after-elements",
        type=int,
        default=10000,
        help="Swap paired endings after this many stream elements; -1 disables it.",
    )
    parser.add_argument(
        "--forgetting-threshold",
        type=float,
        default=None,
        help="Override the paper default (25 for single, 50 for multiple endings).",
    )
    parser.add_argument("--recovery-accuracy", type=float, default=1.0)
    parser.add_argument("--report-every", type=int, default=500)
    parser.add_argument("--output-csv", default="")
    return parser.parse_args()


def run(args: argparse.Namespace) -> Fig7Result:
    result = execute(args)
    print_result(result, args)
    if args.output_csv:
        write_csv(Path(args.output_csv), result)
    return result


if __name__ == "__main__":
    run(parse_args())
