"""Capture deterministic pre-refactor Fig.8/Fig.9 regression baselines.

The fixtures exercise the existing public runners and model APIs. They do not
change model code, parameters, ordering, or defaults. Fig.9 uses an explicitly
labeled reduced-warmup regression fixture rather than a paper experiment.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.common.hashing import sha256_file as file_sha256
from experiments.fig8_sentence_memory import evaluate, read_cbt_sentences
from experiments.fig9 import read_records, record_values
from experiments.fig9_strict_reproduction import load_strict_checkpoint
from seqmem.encoding import SSTDDiscreteEncoder
from seqmem.model import MemoryParams, PredictionCandidate, SequentialMemory


FROZEN_SCIENTIFIC_COMMIT = "1864f46aee5cc0aee814fd4e6154b06f8bfeb08f"
FIXTURE_SIZES = (10, 20, 50)


def canonicalize(value: Any, seen: set[int] | None = None) -> Any:
    """Convert supported state into stable JSON-compatible structures."""

    if seen is None:
        seen = set()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"float": "nan"}
        if math.isinf(value):
            return {"float": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, random.Random):
        return {"random_state": canonicalize(value.getstate(), seen)}
    if dataclasses.is_dataclass(value):
        return {
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": [
                [field.name, canonicalize(getattr(value, field.name), seen)]
                for field in dataclasses.fields(value)
            ],
        }
    if isinstance(value, dict):
        pairs = [
            [canonicalize(key, seen), canonicalize(item, seen)]
            for key, item in value.items()
        ]
        pairs.sort(key=lambda pair: canonical_json(pair[0]))
        return {"dict": pairs}
    if isinstance(value, (list, tuple)):
        return [canonicalize(item, seen) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [canonicalize(item, seen) for item in value]
        items.sort(key=canonical_json)
        return {"set": items}
    object_id = id(value)
    if object_id in seen:
        return {"cycle": f"{type(value).__module__}.{type(value).__qualname__}"}
    if hasattr(value, "__dict__"):
        seen.add(object_id)
        try:
            fields = [
                [name, canonicalize(item, seen)]
                for name, item in sorted(vars(value).items())
                if not callable(item)
            ]
        finally:
            seen.remove(object_id)
        return {
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": fields,
        }
    raise TypeError(f"unsupported canonical value: {type(value)!r}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        canonicalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def encoder_payload(encoder: object, *, include_rng: bool) -> dict[str, object]:
    fields = []
    for name, value in sorted(vars(encoder).items()):
        if not include_rng and isinstance(value, random.Random):
            continue
        fields.append([name, canonicalize(value)])
    return {
        "class": f"{type(encoder).__module__}.{type(encoder).__qualname__}",
        "fields": fields,
    }


def graph_payload(model: SequentialMemory, *, include_diagnostics: bool) -> list[object]:
    rows: list[object] = []
    for column_index, column in enumerate(model.columns):
        for neuron_index, neuron in enumerate(column.neurons):
            for segment_index, segment in enumerate(neuron.segments):
                segment_row: dict[str, object] = {
                    "column": column_index,
                    "neuron": neuron_index,
                    "segment": segment_index,
                    "active": segment.active,
                    "target_time": segment.target_time,
                    "synapses": [
                        {
                            "source": source,
                            "weight": synapse.weight,
                            "delay": synapse.delay,
                            "age": synapse.age,
                        }
                        for source, synapse in sorted(segment.synapses.items())
                    ],
                }
                if include_diagnostics:
                    segment_row["provenance"] = {
                        "diagnostic_id": segment.diagnostic_id,
                        "creation_sentence_index": segment.creation_sentence_index,
                        "creation_transition_index": segment.creation_transition_index,
                        "creation_target_column": segment.creation_target_column,
                        "creation_target_time": segment.creation_target_time,
                        "creation_target_neuron": segment.creation_target_neuron,
                        "creation_source_cell_ids": segment.creation_source_cell_ids,
                        "creation_source_fingerprint": segment.creation_source_fingerprint,
                        "scenario1_reinforcements": segment.scenario1_reinforcements,
                        "scenario2_reinforcements": segment.scenario2_reinforcements,
                    }
                rows.append(segment_row)
    return rows


def segment_keys(model: SequentialMemory) -> dict[int, tuple[int, int, int]]:
    keys: dict[int, tuple[int, int, int]] = {}
    for column_index, column in enumerate(model.columns):
        for neuron_index, neuron in enumerate(column.neurons):
            for segment_index, segment in enumerate(neuron.segments):
                keys[id(segment)] = (column_index, neuron_index, segment_index)
    return keys


def candidate_payload(model: SequentialMemory) -> list[object]:
    keys = segment_keys(model)
    rows: list[object] = []
    for column, candidates in sorted(model.last_prediction_candidates.items()):
        for order, candidate in enumerate(candidates):
            rows.append(
                {
                    "column": column,
                    "order": order,
                    "neuron": candidate.neuron_index,
                    "score": candidate.score,
                    "time": candidate.time,
                    "segment_key": keys.get(id(candidate.segment)),
                    "dendritic_crossing_time": candidate.dendritic_crossing_time,
                    "crossing_synapse_contributions": candidate.crossing_synapse_contributions,
                    "peak_dendritic_potential": candidate.peak_dendritic_potential,
                    "threshold_margin": candidate.threshold_margin,
                    "predicted_soma_firing_time": candidate.predicted_soma_firing_time,
                }
            )
    return rows


def rng_payload(model: SequentialMemory) -> dict[str, object]:
    encoder_rngs: list[dict[str, object]] = []
    queue = [("encoder", model.encoder)]
    seen: set[int] = set()
    while queue:
        path, value = queue.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, random.Random):
            encoder_rngs.append({"path": path, "state": value.getstate()})
            continue
        if hasattr(value, "__dict__"):
            for name, child in sorted(vars(value).items(), reverse=True):
                if isinstance(child, random.Random) or hasattr(child, "__dict__"):
                    queue.append((f"{path}.{name}", child))
                elif isinstance(child, (list, tuple)):
                    for index, item in reversed(list(enumerate(child))):
                        if isinstance(item, random.Random) or hasattr(item, "__dict__"):
                            queue.append((f"{path}.{name}[{index}]", item))
    encoder_rngs.sort(key=lambda row: str(row["path"]))
    return {
        "learning_rng": model._learning_rng.getstate(),
        "decode_rng": model._decode_rng.getstate(),
        "encoder_rngs": encoder_rngs,
    }


def science_payload(model: SequentialMemory) -> dict[str, object]:
    return {
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "params": dataclasses.asdict(model.params),
        "encoder": encoder_payload(model.encoder, include_rng=False),
        "columns": graph_payload(model, include_diagnostics=False),
    }


def full_state_payload(model: SequentialMemory) -> dict[str, object]:
    incoming = []
    keys = segment_keys(model)
    for source, targets in sorted(model._incoming_index.items()):
        incoming.append(
            [
                source,
                [
                    [column, neuron, keys.get(id(segment))]
                    for column, neuron, segment in targets
                ],
            ]
        )
    return {
        "science": science_payload(model),
        "graph_with_diagnostic_metadata": graph_payload(model, include_diagnostics=True),
        "encoder_with_rng": encoder_payload(model.encoder, include_rng=True),
        "previous_active_cells": sorted(model.previous_active_cells.items()),
        "previous_winners": sorted(model.previous_winners.items()),
        "last_prediction_candidates": candidate_payload(model),
        "last_prediction_stats": model.last_prediction_stats,
        "last_observe_stats": model.last_observe_stats,
        "last_symbol_ranking": model.last_symbol_ranking,
        "previous_predicted_sources": sorted(model.previous_predicted_sources),
        "previous_burst_only_sources": sorted(model.previous_burst_only_sources),
        "incoming_index": incoming,
        "dirty_incoming_sources": sorted(model._dirty_incoming_sources),
        "next_segment_diagnostic_id": model._next_segment_diagnostic_id,
        "rng": rng_payload(model),
    }


def structural_summary(model: SequentialMemory) -> dict[str, int]:
    segments = [
        segment
        for column in model.columns
        for neuron in column.neurons
        for segment in neuron.segments
    ]
    return {
        "column_count": len(model.columns),
        "neuron_count": sum(len(column.neurons) for column in model.columns),
        "segment_count": len(segments),
        "active_segment_count": sum(segment.active for segment in segments),
        "synapse_count": sum(len(segment.synapses) for segment in segments),
        "scenario1_segment_reinforcement_count": sum(
            segment.scenario1_reinforcements for segment in segments
        ),
        "scenario2_segment_reinforcement_count": sum(
            segment.scenario2_reinforcements for segment in segments
        ),
    }


def model_fingerprints(model: SequentialMemory) -> dict[str, object]:
    candidates = candidate_payload(model)
    active = sorted(model.previous_active_cells.items())
    winners = sorted(model.previous_winners.items())
    rng = rng_payload(model)
    return {
        "SCIENCE_MODEL_SHA256": stable_sha256(science_payload(model)),
        "FULL_STATE_SHA256": stable_sha256(full_state_payload(model)),
        "RNG_SHA256": stable_sha256(rng),
        "LEARNING_RNG_SHA256": stable_sha256(rng["learning_rng"]),
        "DECODE_RNG_SHA256": stable_sha256(rng["decode_rng"]),
        "PREVIOUS_ACTIVE_SHA256": stable_sha256(active),
        "PREVIOUS_WINNER_SHA256": stable_sha256(winners),
        "CANDIDATE_SHA256": stable_sha256(candidates),
        "structure": structural_summary(model),
    }


def prediction_s3_count(
    candidates: list[tuple[int, float]],
    actual_events: list[tuple[int, float]],
    tolerance: float,
) -> int:
    actual = {column: event_time for column, event_time in actual_events}
    return sum(
        1
        for column, predicted_time in candidates
        if column not in actual or abs(actual[column] - predicted_time) > tolerance
    )


def write_prediction_csv(path: Path, rows: list[dict[str, object]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return file_sha256(path)


def capture_fig8(size: int, output_dir: Path) -> dict[str, object]:
    seed = 11
    data_path = ROOT / "data/CBTest/data/cbt_train.txt"
    sentences = read_cbt_sentences(data_path, size, seed)
    encoder = SSTDDiscreteEncoder(num_columns=100, k=10, seed=seed)
    model = SequentialMemory(
        encoder=encoder,
        num_neurons_per_column=10,
        params=MemoryParams(l_match=3, forgetting_threshold=500.0),
        tie_break_seed=seed,
    )
    scenarios = {"Scenario1": 0, "Scenario2A": 0, "Scenario2B": 0, "Scenario3": 0}
    for sentence in sentences:
        model.reset_state()
        for word in sentence:
            model.predict_code()
            saved_candidates = [
                (column, candidate.time)
                for column, candidates in model.last_prediction_candidates.items()
                for candidate in candidates
            ]
            model.observe(word)
            actual_code = encoder.encode(word)
            actual_events = [(event.column, event.time) for event in actual_code.events]
            scenarios["Scenario1"] += int(model.last_observe_stats.get("scenario1", 0))
            scenarios["Scenario2A"] += int(model.last_observe_stats.get("scenario2", 0))
            scenarios["Scenario2B"] += int(model.last_observe_stats.get("scenario3", 0))
            scenarios["Scenario3"] += prediction_s3_count(
                saved_candidates, actual_events, model.params.timing_tolerance
            )

    before_evaluation = model_fingerprints(model)
    details: list[dict[str, object]] = []
    metric = evaluate(
        model,
        sentences,
        prefix_length=6,
        eval_samples=0,
        seed=seed + size,
        retrieval_mode="neural",
        neural_propagation="raw",
        details_rows=details,
        checkpoint_sentences=size,
    )
    after_evaluation = model_fingerprints(model)
    if before_evaluation != after_evaluation:
        raise AssertionError(f"Fig.8 evaluation polluted state for size {size}")

    prediction_rows = [
        {
            "sentence_index": row.get("sentence_index", ""),
            "recall_step": row.get("recall_step", ""),
            "expected_word": row.get("expected_word", ""),
            "decoded_word": row.get("decoded_word", ""),
            "raw_event_count": row.get("raw_event_count", ""),
            "raw_predicted_column_count": row.get("raw_predicted_column_count", ""),
            "prediction_active_cell_count": row.get("prediction_active_cell_count", ""),
            "stopped_reason": row.get("stopped_reason", ""),
        }
        for row in details
    ]
    prediction_path = output_dir / "predictions.csv"
    prediction_sha = write_prediction_csv(prediction_path, prediction_rows)
    protocol = {
        "classification": "REGRESSION_FIXTURE",
        "paper_result": False,
        "family": "Fig8",
        "frozen_scientific_commit": FROZEN_SCIENTIFIC_COMMIT,
        "capture_parent_commit": git_head(),
        "python": sys.version,
        "dataset": data_path.relative_to(ROOT).as_posix(),
        "dataset_sha256": file_sha256(data_path),
        "size": size,
        "seed": seed,
        "sentence_length": 10,
        "prefix_length": 6,
        "suffix_length": 4,
        "num_columns": 100,
        "neurons_per_column": 10,
        "k": 10,
        "l_match": 3,
        "forgetting_threshold": 500.0,
        "retrieval_mode": "neural",
        "neural_propagation": "raw",
        "input_rows_sha256": stable_sha256(sentences),
    }
    result = {
        "name": f"fig8_{size}",
        "protocol": protocol,
        "prediction_sha256": prediction_sha,
        "metric": {"mean_levenshtein": metric, "evaluated_sentences": size},
        "scenario_summary": scenarios,
        **before_evaluation,
        "evaluation_read_only_verified": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "state_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def fig9_warmup(size: int) -> int:
    return {10: 2, 20: 5, 50: 10}[size]


def capture_fig9(size: int, output_dir: Path, temp_root: Path) -> dict[str, object]:
    warmup = fig9_warmup(size)
    run_dir = temp_root / f"fig9_{size}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "checkpoint.pkl"
    command = [
        sys.executable,
        "experiments/fig9_strict_reproduction.py",
        "--data", "data/paper_nyc_taxi.csv",
        "--perturbed-data", "data/paper_nyc_taxi_perturb.csv",
        "--limit", str(size),
        "--warmup", str(warmup),
        "--prediction-horizon", "5",
        "--tie-break-seed", "0",
        "--l-match", "4",
        "--continuous-impl", "reference",
        "--streams", "original",
        "--output-dir", str(run_dir),
        "--checkpoint-path", str(checkpoint),
        "--checkpoint-every", "1",
        "--long-sequence-ledger",
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(SRC), str(ROOT)])
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    (run_dir / "baseline_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (run_dir / "baseline_stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Fig.9 baseline {size} failed with {completed.returncode}:\n"
            + completed.stderr[-4000:]
        )

    predictions_path = run_dir / "original_predictions.csv"
    summary_path = run_dir / "original_summary.json"
    runner_protocol_path = run_dir / "original_protocol.json"
    payload = load_strict_checkpoint(checkpoint)
    model = payload["model"]
    if not isinstance(model, SequentialMemory):
        raise TypeError(f"unexpected checkpoint model type: {type(model)!r}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    runner_protocol = json.loads(runner_protocol_path.read_text(encoding="utf-8"))

    records = read_records(ROOT / "data/paper_nyc_taxi.csv", size)
    input_rows = [
        [record.timestamp.isoformat(sep=" "), *record_values(record)] for record in records
    ]
    ledger_path = run_dir / "long_sequence_activity_trace.csv"
    ledger_rows = []
    if ledger_path.exists():
        with ledger_path.open("r", encoding="utf-8", newline="") as handle:
            ledger_rows = list(csv.DictReader(handle))
    ledger_scenarios = {
        "Scenario1_evaluated_observations": sum(int(row["scenario1_count"]) for row in ledger_rows),
        "Scenario2A_evaluated_observations": sum(int(row["scenario2_count"]) for row in ledger_rows),
        "Scenario2B_evaluated_observations": sum(int(row["scenario3_count"]) for row in ledger_rows),
        "Scenario3_wrong_prediction_count": None,
        "Scenario3_note": "Not persistently counted by the strict runner; null is intentional.",
    }
    fingerprints = model_fingerprints(model)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(predictions_path, output_dir / "predictions.csv")
    shutil.copyfile(runner_protocol_path, output_dir / "runner_protocol.json")
    prediction_sha = file_sha256(output_dir / "predictions.csv")
    protocol = {
        "classification": "REGRESSION_FIXTURE",
        "paper_result": False,
        "reduced_warmup": True,
        "family": "Fig9",
        "frozen_scientific_commit": FROZEN_SCIENTIFIC_COMMIT,
        "capture_parent_commit": git_head(),
        "python": sys.version,
        "dataset": "data/paper_nyc_taxi.csv",
        "dataset_sha256": file_sha256(ROOT / "data/paper_nyc_taxi.csv"),
        "size": size,
        "warmup": warmup,
        "prediction_horizon": 5,
        "seed": 0,
        "l_match": 4,
        "continuous_impl": "reference",
        "streams": ["original"],
        "diagnostics": {"long_sequence_ledger": True, "all_model_diagnostics": False},
        "input_rows_sha256": stable_sha256(input_rows),
        "runner_protocol_sha256": stable_sha256(runner_protocol),
    }
    result = {
        "name": f"fig9_{size}",
        "protocol": protocol,
        "prediction_sha256": prediction_sha,
        "metric": {
            "mape": summary["mape"],
            "coverage": summary["coverage"],
            "predictions": summary["predictions"],
            "mean_raw_column_count": summary["mean_raw_column_count"],
            "peak_raw_column_count": summary["peak_raw_column_count"],
        },
        "scenario_summary": ledger_scenarios,
        **fingerprints,
        "checkpoint_next_index": payload["next_index"],
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "state_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    shutil.rmtree(run_dir)
    return result


def comparison_projection(result: dict[str, object]) -> dict[str, object]:
    keys = (
        "prediction_sha256", "metric", "scenario_summary", "SCIENCE_MODEL_SHA256",
        "FULL_STATE_SHA256", "RNG_SHA256", "LEARNING_RNG_SHA256",
        "DECODE_RNG_SHA256", "PREVIOUS_ACTIVE_SHA256", "PREVIOUS_WINNER_SHA256",
        "CANDIDATE_SHA256", "structure",
    )
    return {key: result[key] for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/refactor/phase01_20260814_233129/baselines/pre_refactor",
    )
    parser.add_argument("--families", nargs="+", choices=("fig8", "fig9"), default=("fig8", "fig9"))
    parser.add_argument("--sizes", nargs="+", type=int, default=FIXTURE_SIZES)
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    temp_root = ROOT / "tmp/refactor_phase01_baselines"
    temp_root.mkdir(parents=True, exist_ok=True)

    manifest_runs: list[dict[str, object]] = []
    reproducibility: list[dict[str, object]] = []
    for family in args.families:
        for size in args.sizes:
            if size not in FIXTURE_SIZES:
                raise ValueError(f"unsupported Phase 0/1 fixture size: {size}")
            run_dir = output / f"{family}_{size}"
            if run_dir.exists():
                shutil.rmtree(run_dir)
            first = capture_fig8(size, run_dir) if family == "fig8" else capture_fig9(size, run_dir, temp_root)
            reference = comparison_projection(first)
            attempts = 1
            for repeat_index in range(1, args.repeat):
                repeat_dir = output / "_repeat_check" / f"{family}_{size}_{repeat_index}"
                if repeat_dir.exists():
                    shutil.rmtree(repeat_dir)
                repeated = (
                    capture_fig8(size, repeat_dir)
                    if family == "fig8"
                    else capture_fig9(size, repeat_dir, temp_root)
                )
                attempts += 1
                if comparison_projection(repeated) != reference:
                    raise AssertionError(f"non-reproducible baseline: {family}_{size}")
                shutil.rmtree(repeat_dir)
            manifest_runs.append(first)
            reproducibility.append(
                {
                    "name": first["name"],
                    "attempts": attempts,
                    "all_behavior_fingerprints_equal": True,
                }
            )
            print(f"captured {family}_{size}", flush=True)

    repeat_root = output / "_repeat_check"
    if repeat_root.exists():
        shutil.rmtree(repeat_root)
    if temp_root.exists() and not any(temp_root.iterdir()):
        temp_root.rmdir()
    manifest = {
        "classification": "PRE_REFACTOR_GOLDEN_REGRESSION_BASELINE",
        "paper_result": False,
        "frozen_scientific_commit": FROZEN_SCIENTIFIC_COMMIT,
        "capture_parent_commit": git_head(),
        "python": sys.version,
        "runs": manifest_runs,
        "reproducibility": reproducibility,
    }
    (output / "baseline_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "reproducibility_check.json").write_text(
        json.dumps(reproducibility, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
