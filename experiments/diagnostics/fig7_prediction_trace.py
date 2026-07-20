from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

from fig7_sequence_prediction import build_sequence_set, make_stream
from seqmem.encoding import SSTDDiscreteEncoder
from seqmem.model import MemoryParams, SequentialMemory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    sequences = build_sequence_set("single")
    stream = make_stream(
        sequences,
        args.elements,
        50000,
        random.Random(args.seed),
        -1,
    )
    encoder = SSTDDiscreteEncoder(200, 10, seed=args.seed)
    model = SequentialMemory(
        encoder,
        10,
        MemoryParams(
            l_match=3,
            forgetting_threshold=25.0,
        ),
    )

    predicted_event_counts: Counter[int] = Counter()
    expected_timed_overlaps: Counter[int] = Counter()
    expected_column_overlaps: Counter[int] = Counter()
    expected_ranks: Counter[int | str] = Counter()

    for index, symbol in enumerate(stream.symbols, start=1):
        model.predict_code()
        model.observe(symbol)
        expected = stream.expected_by_index.get(index)
        if expected is None:
            continue
        predicted = model.predict_code()
        if predicted is None:
            predicted_event_counts[0] += 1
            expected_timed_overlaps[0] += 1
            expected_column_overlaps[0] += 1
            expected_ranks["missing"] += 1
            continue

        predicted_times: dict[int, list[float]] = {}
        for event in predicted.events:
            predicted_times.setdefault(event.column, []).append(event.time)
        expected_code = encoder.encode(expected)
        column_overlap = sum(event.column in predicted_times for event in expected_code.events)
        timed_overlap = sum(
            event.column in predicted_times
            and any(
                abs(predicted_time - event.time)
                <= model.params.timing_tolerance
                for predicted_time in predicted_times[event.column]
            )
            for event in expected_code.events
        )
        predicted_event_counts[len(predicted.events)] += 1
        expected_timed_overlaps[timed_overlap] += 1
        expected_column_overlaps[column_overlap] += 1

        candidates = model.predict_symbols(max_predictions=100000)
        try:
            rank = candidates.index(expected) + 1
        except ValueError:
            expected_ranks["absent"] += 1
        else:
            expected_ranks[rank] += 1

    print("predicted event counts:", sorted(predicted_event_counts.items()))
    print("expected timed overlaps:", sorted(expected_timed_overlaps.items()))
    print("expected column overlaps:", sorted(expected_column_overlaps.items()))
    print("expected ranks:", sorted(expected_ranks.items(), key=lambda item: str(item[0])))


if __name__ == "__main__":
    main()
