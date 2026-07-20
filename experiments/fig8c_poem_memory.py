from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fig8_sentence_memory import levenshtein, recall_suffix, train_sentence
from seqmem.encoding import SSTDDiscreteEncoder, SpikeEvent, SymbolCode
from seqmem.model import MemoryParams, SequentialMemory


@dataclass(frozen=True)
class Poem:
    title: str
    author: str
    lines: tuple[str, str, str, str]


@dataclass(frozen=True)
class RetrievalDistances:
    goal_based: int
    contextual: int
    contextual_single: int
    goal_line: int
    contextual_line: int
    feedback_character: int

    def as_dict(self) -> dict[str, int]:
        return {
            "goal_based": self.goal_based,
            "contextual": self.contextual,
            "contextual_single": self.contextual_single,
            "goal_line": self.goal_line,
            "contextual_line": self.contextual_line,
            "feedback_character": self.feedback_character,
        }


def read_poems(path: Path, count: int) -> list[Poem]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    poems = [
        Poem(
            title=str(item["title"]),
            author=str(item["author"]),
            lines=tuple(item["lines"]),  # type: ignore[arg-type]
        )
        for item in payload["poems"][:count]
    ]
    if len(poems) < count:
        raise ValueError(f"Dataset has {len(poems)} poems; requested {count}.")
    if any(len(poem.lines) != 4 or any(len(line) != 5 for line in poem.lines) for poem in poems):
        raise ValueError("Every poem must contain four five-character lines.")
    return poems


class InterLayerSequenceMemory:
    """Explicit SSTD feedback synapses from a concept to an item sequence."""

    def __init__(
        self,
        source_encoder: SSTDDiscreteEncoder,
        target_encoder: SSTDDiscreteEncoder,
        initial_weight: float = 0.5,
        source_neurons: int = 10,
    ) -> None:
        if source_neurons <= 0:
            raise ValueError("source_neurons must be positive.")
        self.source_encoder = source_encoder
        self.target_encoder = target_encoder
        self.initial_weight = initial_weight
        self.source_neurons = source_neurons
        self.feedback: dict[
            tuple[int, float, int, int, float], dict[int, float]
        ] = defaultdict(dict)
        self._source_cells: dict[tuple[str, int, float], int] = {}
        self._source_cell_use: dict[tuple[int, float], list[int]] = {}

    def _source_events(
        self, concept: str, allocate: bool
    ) -> list[tuple[SpikeEvent, int]]:
        events: list[tuple[SpikeEvent, int]] = []
        for event in self.source_encoder.encode(concept).events:
            concept_key = (concept, event.column, event.time)
            neuron = self._source_cells.get(concept_key)
            if neuron is None and allocate:
                use = self._source_cell_use.setdefault(
                    (event.column, event.time), [0] * self.source_neurons
                )
                neuron = min(range(self.source_neurons), key=lambda index: (use[index], index))
                use[neuron] += 1
                self._source_cells[concept_key] = neuron
            if neuron is not None:
                events.append((event, neuron))
        return events

    def learn(self, concept: str, items: tuple[str, ...]) -> None:
        source_events = self._source_events(concept, allocate=True)
        for position, item in enumerate(items):
            target_code = self.target_encoder.encode(item)
            for source, source_neuron in source_events:
                for target in target_code.events:
                    key = (
                        source.column,
                        source.time,
                        source_neuron,
                        position,
                        target.time,
                    )
                    weights = self.feedback[key]
                    weights[target.column] = weights.get(target.column, 0.0) + self.initial_weight

    def retrieve(self, concept: str, item_count: int) -> list[str]:
        source_events = self._source_events(concept, allocate=False)
        if not source_events:
            return []
        recalled: list[str] = []
        for position in range(item_count):
            used_columns: set[int] = set()
            events: list[SpikeEvent] = []
            for target_time in self.target_encoder.event_times:
                scores: dict[int, float] = defaultdict(float)
                for source, source_neuron in source_events:
                    weights = self.feedback.get(
                        (
                            source.column,
                            source.time,
                            source_neuron,
                            position,
                            target_time,
                        ),
                        {},
                    )
                    for column, weight in weights.items():
                        if column not in used_columns:
                            scores[column] += weight
                if not scores:
                    break
                column = max(scores, key=lambda item: (scores[item], -item))
                used_columns.add(column)
                events.append(SpikeEvent(column=column, time=target_time))
            if len(events) != self.target_encoder.k:
                break
            prediction = self._decode(SymbolCode(tuple(events)))
            if prediction is None:
                break
            recalled.append(prediction)
        return recalled

    def _decode(self, code: SymbolCode) -> str | None:
        predicted = {(event.column, event.time) for event in code.events}
        candidates: set[str] = set()
        for event in code.events:
            candidates.update(
                self.target_encoder.symbols_for_event(event.column, event.time)
            )
        if not candidates:
            candidates.update(self.target_encoder.known_symbols())
        ranked: list[tuple[int, int, str]] = []
        predicted_columns = {event.column for event in code.events}
        for symbol in candidates:
            candidate = self.target_encoder.encode(symbol)
            timed = sum((event.column, event.time) in predicted for event in candidate.events)
            columns = sum(event.column in predicted_columns for event in candidate.events)
            ranked.append((timed, columns, symbol))
        if not ranked:
            return None
        ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return ranked[0][2]


def make_layer(columns: int, neurons: int, k: int, seed: int) -> SequentialMemory:
    return SequentialMemory(
        encoder=SSTDDiscreteEncoder(columns, k, seed=seed),
        num_neurons_per_column=neurons,
        params=MemoryParams(l_match=3, forgetting_threshold=500.0),
        tie_break_seed=seed,
    )


def evaluate_poem(
    poem: Poem,
    character_layer: SequentialMemory,
    line_layer: SequentialMemory,
    goal_memory: InterLayerSequenceMemory,
    character_feedback: InterLayerSequenceMemory,
    evaluate_single: bool = True,
) -> RetrievalDistances:
    complete = list("".join(poem.lines))

    if evaluate_single:
        single_recalled = recall_suffix(character_layer, complete[:5], 15)
        single_distance = levenshtein(single_recalled, complete[5:])
    else:
        single_distance = 0

    recalled_lines = recall_suffix(line_layer, [poem.lines[0]], 3)
    contextual_line_distance = levenshtein(recalled_lines, list(poem.lines[1:]))
    contextual_recalled = [
        character
        for line in recalled_lines
        for character in character_feedback.retrieve(line, 5)
    ]
    contextual_distance = levenshtein(contextual_recalled, complete[5:])

    goal_lines = goal_memory.retrieve(poem.title, 4)
    goal_line_distance = levenshtein(goal_lines, list(poem.lines))
    goal_recalled = [
        character
        for line in goal_lines
        for character in character_feedback.retrieve(line, 5)
    ]
    goal_distance = levenshtein(goal_recalled, complete)
    feedback_distance = sum(
        levenshtein(character_feedback.retrieve(line, 5), list(line))
        for line in poem.lines
    )
    return RetrievalDistances(
        goal_based=goal_distance,
        contextual=contextual_distance,
        contextual_single=single_distance,
        goal_line=goal_line_distance,
        contextual_line=contextual_line_distance,
        feedback_character=feedback_distance,
    )


def plot_results(path: Path, rows: list[dict[str, float | int]]) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    figure, axis = plt.subplots(figsize=(7.4, 4.8))
    x = [int(row["poems"]) for row in rows]
    axis.plot(x, [float(row["goal_based"]) for row in rows], color="tab:red", marker="o", label="Goal-based retrieval")
    axis.plot(x, [float(row["contextual"]) for row in rows], color="tab:blue", marker="o", label="Contextual retrieval")
    axis.plot(x, [float(row["contextual_single"]) for row in rows], color="green", marker="o", label="Contextual retrieval-S")
    axis.set(xlabel="Number of poems", ylabel="Mean Levenshtein distance", xticks=x)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def plot_diagnostics(path: Path, rows: list[dict[str, float | int]]) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    x = [int(row["poems"]) for row in rows]
    axes[0].plot(
        x,
        [float(row["goal_line"]) for row in rows],
        color="tab:red",
        marker="o",
        label="Title to lines (max 4)",
    )
    axes[0].plot(
        x,
        [float(row["contextual_line"]) for row in rows],
        color="tab:blue",
        marker="o",
        label="First line to suffix (max 3)",
    )
    axes[0].set(xlabel="Number of poems", ylabel="Mean line-token distance", xticks=x)
    axes[0].legend()
    axes[1].plot(
        x,
        [float(row["feedback_character"]) for row in rows],
        color="tab:purple",
        marker="o",
        label="True lines to characters (max 20)",
    )
    axes[1].set(
        xlabel="Number of poems",
        ylabel="Mean character distance",
        xticks=x,
    )
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_single_cache(
    summary_path: str | None, details_path: str | None
) -> tuple[dict[int, float], dict[str, int]]:
    summary: dict[int, float] = {}
    details: dict[str, int] = {}
    if summary_path:
        with Path(summary_path).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                summary[int(row["poems"])] = float(row["contextual_single"])
    if details_path:
        with Path(details_path).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                details[row["title"]] = int(row["contextual_single"])
    return summary, details


def run(args: argparse.Namespace) -> None:
    checkpoints = sorted(set(args.checkpoints))
    poems = read_poems(Path(args.data), max(checkpoints))
    single_summary, single_details = read_single_cache(
        args.reuse_single_summary, args.reuse_single_details
    )
    missing_cache_points = set(checkpoints) - set(single_summary)
    if single_summary and missing_cache_points:
        raise ValueError(
            f"Single-layer cache is missing checkpoints: {sorted(missing_cache_points)}"
        )
    character_layer = make_layer(args.columns, args.neurons, args.k, args.seed)
    line_layer = make_layer(args.columns, args.neurons, args.k, args.seed + 1)
    title_encoder = SSTDDiscreteEncoder(args.columns, args.k, seed=args.seed + 2)
    goal_memory = InterLayerSequenceMemory(
        title_encoder,
        line_layer.encoder,  # type: ignore[arg-type]
        source_neurons=args.interlayer_source_neurons,
    )
    character_feedback = InterLayerSequenceMemory(
        line_layer.encoder,  # type: ignore[arg-type]
        character_layer.encoder,  # type: ignore[arg-type]
        source_neurons=args.interlayer_source_neurons,
    )

    summary: list[dict[str, float | int]] = []
    final_details: list[dict[str, object]] = []
    for index, poem in enumerate(poems, start=1):
        train_sentence(character_layer, list("".join(poem.lines)))
        train_sentence(line_layer, list(poem.lines))
        goal_memory.learn(poem.title, poem.lines)
        for line in poem.lines:
            character_feedback.learn(line, tuple(line))

        if index not in checkpoints:
            continue
        details: list[dict[str, object]] = []
        totals = {field: 0 for field in RetrievalDistances.__dataclass_fields__}
        for stored in poems[:index]:
            distances = evaluate_poem(
                stored,
                character_layer,
                line_layer,
                goal_memory,
                character_feedback,
                evaluate_single=not single_summary,
            )
            distance_values = distances.as_dict()
            if single_summary:
                distance_values["contextual_single"] = single_details.get(
                    stored.title, 0
                )
            totals = {
                field: totals[field] + distance_values[field]
                for field in totals
            }
            details.append(
                {
                    "poems": index,
                    "title": stored.title,
                    "author": stored.author,
                    **distance_values,
                }
            )
        row = {
            "poems": index,
            **{field: total / index for field, total in totals.items()},
        }
        if single_summary:
            row["contextual_single"] = single_summary[index]
        summary.append(row)
        final_details = details
        print(
            f"after {index:3d} poems: goal={row['goal_based']:.3f}, "
            f"context={row['contextual']:.3f}, single={row['contextual_single']:.3f}",
            flush=True,
        )
        if args.output_csv:
            write_csv(Path(args.output_csv), summary)

    if args.output_details:
        write_csv(Path(args.output_details), final_details)
    plotted = plot_results(Path(args.output_plot), summary) if args.output_plot else False
    diagnostics_plotted = (
        plot_diagnostics(Path(args.output_diagnostic_plot), summary)
        if args.output_diagnostic_plot
        else False
    )
    print("Fig. 8(c) three-layer poem-memory approximation")
    print(f"source: {args.data}")
    print(f"poems: {len(poems)}")
    print(f"layer size: {args.columns} columns x {args.neurons} neurons")
    print(f"active columns per symbol: {args.k}")
    print(f"inter-layer source neurons per event: {args.interlayer_source_neurons}")
    print("author poem list/network layer sizes published: no")
    print(f"plot generated: {plotted}")
    print(f"diagnostic plot generated: {diagnostics_plotted}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/fig8c_poems.json")
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[40, 80, 120, 160, 200])
    parser.add_argument("--columns", type=int, default=100)
    parser.add_argument("--neurons", type=int, default=10)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--interlayer-source-neurons", type=int, default=10)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--output-csv", default="results/fig8c_poem_memory.csv")
    parser.add_argument("--output-details", default="results/fig8c_poem_details.csv")
    parser.add_argument("--output-plot", default="results/fig8c_poem_memory.png")
    parser.add_argument(
        "--output-diagnostic-plot",
        default="results/fig8c_poem_diagnostics.png",
    )
    parser.add_argument("--reuse-single-summary")
    parser.add_argument("--reuse-single-details")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
