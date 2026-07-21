from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seqmem.encoding import SSTDDiscreteEncoder
from seqmem.model import MemoryParams, SequentialMemory


DETAIL_FIELDS = [
    "checkpoint_sentences",
    "sentence_index",
    "retrieval_mode",
    "recall_step",
    "expected_word",
    "decoded_word",
    "raw_event_count",
    "raw_predicted_column_count",
    "prediction_active_cell_count",
    "advance_success",
    "timed_overlap_of_selected_word",
    "column_overlap_of_selected_word",
    "timed_overlap_of_expected_word",
    "column_overlap_of_expected_word",
    "expected_word_rank",
    "number_of_ranked_candidates",
    "top_5_candidates",
    "stopped_reason",
    "previous_active_cell_count",
    "previous_winner_count",
    "burst_cell_count",
]


def levenshtein(a: list[str], b: list[str]) -> int:
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, start=1):
        current = [i]
        for j, right in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            substitute = previous[j - 1] + (left != right)
            current.append(min(insert, delete, substitute))
        previous = current
    return previous[-1]


def read_cbt_sentences(path: Path, num_sentences: int, seed: int) -> list[list[str]]:
    """Select CBT sentences exactly as required by the Fig. 8 protocol."""

    eligible: list[list[str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            words = line.strip().split()
            if len(words) >= 10 and not words[0].startswith("_BOOK_"):
                eligible.append(words[:10])
    if len(eligible) < num_sentences:
        raise ValueError(
            f"CBT source has only {len(eligible)} eligible sentences; "
            f"requested {num_sentences}."
        )
    return random.Random(seed).sample(eligible, num_sentences)


def train_sentence(model: SequentialMemory, sentence: list[str]) -> None:
    model.reset_state()
    for word in sentence:
        model.predict_code()
        model.observe(word)


def recall_suffix(
    model: SequentialMemory,
    prefix: list[str],
    suffix_length: int,
    retrieval_mode: str = "neural",
    *,
    expected_suffix: list[str] | None = None,
    details_rows: list[dict[str, object]] | None = None,
    checkpoint_sentences: int = 0,
    sentence_index: int = 0,
) -> list[str]:
    if retrieval_mode not in {"neural", "proximal-replay"}:
        raise ValueError(f"unsupported retrieval mode: {retrieval_mode}")

    model.reset_state()
    for word in prefix:
        model.predict_code()
        model.observe(word, learn=False)

    recalled: list[str] = []
    for step in range(suffix_length):
        expected_word = (
            expected_suffix[step]
            if expected_suffix is not None and step < len(expected_suffix)
            else ""
        )
        previous_active_count = len(model.previous_active_cells)
        previous_winner_count = len(model.previous_winners)
        raw_code = model.predict_code()
        if raw_code is None:
            _append_detail(
                details_rows,
                checkpoint_sentences,
                sentence_index,
                retrieval_mode,
                step,
                expected_word,
                stopped_reason="no_raw_prediction",
                previous_active_cell_count=previous_active_count,
                previous_winner_count=previous_winner_count,
            )
            break

        active_cells = model.prediction_active_cells(raw_code)
        decoded = model.decode_symbol_from_prediction(raw_code)
        ranking = model.last_symbol_ranking
        selected = next(
            (item for item in ranking if item[2] == decoded),
            None,
        )
        expected_ranked = next(
            (
                (rank, item)
                for rank, item in enumerate(ranking, start=1)
                if item[2] == expected_word
            ),
            None,
        )
        common = {
            "raw_event_count": len(raw_code.events),
            "raw_predicted_column_count": len(
                {event.column for event in raw_code.events}
            ),
            "prediction_active_cell_count": len(active_cells),
            "timed_overlap_of_selected_word": selected[0] if selected else 0,
            "column_overlap_of_selected_word": selected[1] if selected else 0,
            "timed_overlap_of_expected_word": (
                expected_ranked[1][0] if expected_ranked else 0
            ),
            "column_overlap_of_expected_word": (
                expected_ranked[1][1] if expected_ranked else 0
            ),
            "expected_word_rank": expected_ranked[0] if expected_ranked else "",
            "number_of_ranked_candidates": len(ranking),
            "top_5_candidates": json.dumps(
                [
                    {
                        "word": symbol,
                        "timed_overlap": timed,
                        "column_overlap": columns,
                    }
                    for timed, columns, symbol in ranking[:5]
                ],
                ensure_ascii=False,
            ),
            "previous_active_cell_count": previous_active_count,
            "previous_winner_count": previous_winner_count,
        }
        if decoded is None:
            _append_detail(
                details_rows,
                checkpoint_sentences,
                sentence_index,
                retrieval_mode,
                step,
                expected_word,
                stopped_reason="decode_failed",
                **common,
            )
            break

        recalled.append(decoded)
        if retrieval_mode == "neural":
            advanced = model.advance_prediction(raw_code)
            if not advanced:
                _append_detail(
                    details_rows,
                    checkpoint_sentences,
                    sentence_index,
                    retrieval_mode,
                    step,
                    expected_word,
                    decoded_word=decoded,
                    advance_success=False,
                    stopped_reason="no_prediction_active_cells",
                    **common,
                )
                break
            burst_count = len(set(model.previous_active_cells) - set(active_cells))
            advance_success: bool | str = True
        else:
            model.observe(decoded, learn=False)
            burst_count = len(set(model.previous_active_cells) - set(active_cells))
            advance_success = ""

        _append_detail(
            details_rows,
            checkpoint_sentences,
            sentence_index,
            retrieval_mode,
            step,
            expected_word,
            decoded_word=decoded,
            advance_success=advance_success,
            stopped_reason="completed",
            burst_cell_count=burst_count,
            **common,
        )
    return recalled


def _append_detail(
    rows: list[dict[str, object]] | None,
    checkpoint_sentences: int,
    sentence_index: int,
    retrieval_mode: str,
    step: int,
    expected_word: str,
    **values: object,
) -> None:
    if rows is None:
        return
    row: dict[str, object] = {field: "" for field in DETAIL_FIELDS}
    row.update(
        {
            "checkpoint_sentences": checkpoint_sentences,
            "sentence_index": sentence_index,
            "retrieval_mode": retrieval_mode,
            "recall_step": step + 1,
            "expected_word": expected_word,
        }
    )
    row.update(values)
    rows.append(row)


def evaluate(
    model: SequentialMemory,
    sentences: list[list[str]],
    prefix_length: int,
    eval_samples: int,
    seed: int,
    retrieval_mode: str = "neural",
    details_rows: list[dict[str, object]] | None = None,
    checkpoint_sentences: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> float:
    snapshot = model.snapshot_transient_state()
    try:
        indexed_sentences = list(enumerate(sentences, start=1))
        if eval_samples > 0 and len(indexed_sentences) > eval_samples:
            rng = random.Random(seed)
            indexed_sentences = rng.sample(indexed_sentences, eval_samples)

        distances: list[int] = []
        checkpoint = (
            len(sentences) if checkpoint_sentences is None else checkpoint_sentences
        )
        for sentence_index, sentence in indexed_sentences:
            prefix = sentence[:prefix_length]
            expected = sentence[prefix_length:]
            recalled = recall_suffix(
                model,
                prefix,
                suffix_length=len(expected),
                retrieval_mode=retrieval_mode,
                expected_suffix=expected,
                details_rows=details_rows,
                checkpoint_sentences=checkpoint,
                sentence_index=sentence_index,
            )
            distances.append(levenshtein(recalled, expected))
            if progress_callback is not None:
                progress_callback(len(distances), len(indexed_sentences))
        return sum(distances) / len(distances) if distances else 0.0
    finally:
        model.restore_transient_state(snapshot)


def plot_capacity(path: Path, checkpoints: list[tuple[int, float]]) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.plot(
        [sentences * 10 for sentences, _distance in checkpoints],
        [distance for _sentences, distance in checkpoints],
        marker="o",
        label="DS memory",
    )
    axis.set(
        xlabel="Number of symbols",
        ylabel="Mean Levenshtein distance",
        ylim=(0, 4.05),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def run(args: argparse.Namespace) -> None:
    sentences = read_cbt_sentences(Path(args.data), args.num_sentences, args.seed)

    encoder = SSTDDiscreteEncoder(num_columns=args.num_columns, k=args.k, seed=args.seed)
    params = MemoryParams(
        l_match=args.l_match,
        forgetting_threshold=args.forgetting_threshold,
    )
    model = SequentialMemory(
        encoder=encoder,
        num_neurons_per_column=args.neurons_per_column,
        params=params,
        tie_break_seed=args.seed,
    )

    checkpoints: list[tuple[int, float]] = []
    details_rows: list[dict[str, object]] = []
    for index, sentence in enumerate(sentences, start=1):
        train_sentence(model, sentence)
        if index % args.report_every == 0 or index == len(sentences):
            subset = sentences[:index]
            mean_distance = evaluate(
                model,
                subset,
                args.prefix_length,
                args.eval_samples,
                args.seed + index,
                retrieval_mode=args.retrieval_mode,
                details_rows=details_rows if args.details_csv else None,
                checkpoint_sentences=index,
            )
            checkpoints.append((index, mean_distance))
            print(
                f"after {index:5d} sentences: "
                f"levenshtein={mean_distance:.3f}",
                flush=True,
            )
            if args.output_csv:
                output = Path(args.output_csv)
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(
                        ["sentences", "mean_levenshtein"]
                    )
                    writer.writerows(checkpoints)
            if args.details_csv:
                details_output = Path(args.details_csv)
                details_output.parent.mkdir(parents=True, exist_ok=True)
                with details_output.open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=DETAIL_FIELDS)
                    writer.writeheader()
                    writer.writerows(details_rows)

    _final_count, final_distance = checkpoints[-1]
    print("Fig. 8 CBT sentence memory")
    print(f"source: {args.data}")
    print(f"sentences: {len(sentences)}")
    print("sentence length: 10")
    print(f"prefix length: {args.prefix_length}")
    print(f"retrieval mode: {args.retrieval_mode}")
    print(f"encoded vocabulary size: {len(encoder.known_symbols())}")
    print(f"mini-columns: {args.num_columns}")
    print(f"neurons per mini-column: {args.neurons_per_column}")
    print(f"mean Levenshtein distance: {final_distance:.3f}")
    if args.eval_samples > 0:
        print(f"evaluation samples: {min(args.eval_samples, len(sentences))}")
    print()
    print("checkpoints:")
    for index, mean_distance in checkpoints[-10:]:
        print(f"  after {index:5d} sentences: levenshtein={mean_distance:.3f}")
    if args.output_plot:
        generated = plot_capacity(Path(args.output_plot), checkpoints)
        print(f"capacity plot generated: {generated}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--data", default="data/CBTest/data/cbt_train.txt")
    parser.add_argument("--num-sentences", type=int, default=1000)
    parser.add_argument("--prefix-length", type=int, default=6)
    parser.add_argument("--num-columns", type=int, default=100)
    parser.add_argument("--neurons-per-column", type=int, default=10)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--l-match", type=int, default=3)
    parser.add_argument("--forgetting-threshold", type=float, default=500.0)
    parser.add_argument("--report-every", type=int, default=50)
    parser.add_argument(
        "--retrieval-mode",
        choices=("neural", "proximal-replay"),
        default="neural",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=0,
        help="Number of sentences sampled for a diagnostic evaluation; 0 evaluates the full paper set.",
    )
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--output-plot", default="")
    parser.add_argument("--details-csv", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
