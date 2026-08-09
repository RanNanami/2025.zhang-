"""Strict Fig.9 taxi adaptation runner.

中文调试提示：这个文件是当前 Fig.9 严格协议入口。主循环按
“读出租车记录 -> 对当前上下文做 5 步自主预测 -> 解码 passenger_count
-> 再把真实当前记录 observe/learn 进去”的顺序执行。
"""

from __future__ import annotations

import argparse
import atexit
import cProfile
import csv
import faulthandler
import gzip
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import pickle
import pstats
import subprocess
import sys
import threading
import time
import zlib
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.fig9 import (  # noqa: E402
    TaxiRecord,
    learn_actual_code,
    mape,
    plot_adaptation,
    read_records,
    record_values,
    reference_rolling_mape,
    write_predictions,
)
from experiments.diagnostics.fig9_competitive_inhibition import (  # noqa: E402
    CompetitionResult,
    CompetitionSettings,
    candidates_for_prediction,
    compete_prediction_candidates,
    emitted_prediction_code,
    summarize_competition,
)
from experiments.diagnostics.fig9_oracle_candidate import (  # noqa: E402
    FieldColumnRanges,
    analyze_oracle_step,
    summarize_oracle_rows,
)
from experiments.diagnostics.fig9_candidate_score_trace import (  # noqa: E402
    candidate_score_trace_rows,
    field_for_column,
)
from experiments.diagnostics.fig9_candidate_context_oracle import (  # noqa: E402
    ContextCompositionIndex,
    project_actual_composition_rows,
    write_join_outputs,
)
from experiments.diagnostics.fig9_branch_provenance import (  # noqa: E402
    BRANCH_LEVELS,
    DIAGNOSTIC_MARKERS as BRANCH_DIAGNOSTIC_MARKERS,
    BranchProvenanceRegistry,
    branch_trace_rows,
    summarize_branch_steps,
)
from experiments.diagnostics.fig9_preselection_segments import (  # noqa: E402
    DIAGNOSTIC_MARKERS as PRESELECTION_DIAGNOSTIC_MARKERS,
    PRESELECTION_LEVELS,
    build_preselection_rows,
)
from experiments.diagnostics.fig9_teacher_forced_identity import (  # noqa: E402
    DIAGNOSTIC_MARKERS as TEACHER_FORCED_DIAGNOSTIC_MARKERS,
    OBSERVE_SCENARIO_LEVELS,
    REFERENCE_NEURON_SELECTION_LEVELS,
    TEACHER_FORCED_LEVELS,
    build_teacher_forced_observation_rows,
    legacy_teacher_forced_rows,
    observe_scenario_rows,
)
from experiments.diagnostics.fig9_intracolumn_selector import (  # noqa: E402
    DIAGNOSTIC_MARKERS as INTRACOLUMN_DIAGNOSTIC_MARKERS,
    SELECTOR_VERSION as INTRACOLUMN_SELECTOR_VERSION,
    mark_competition_outcomes,
    selection_trace_rows,
    summarize_selection_rows,
    write_selection_trace,
)
from experiments.diagnostics.fig9_match_overlap import (  # noqa: E402
    DIAGNOSTIC_MARKERS as MATCH_OVERLAP_DIAGNOSTIC_MARKERS,
    MATCH_OVERLAP_LEVELS,
    SourceTraceFilter,
    capture_match_overlap,
    filter_source_trace_rows,
    finalize_match_overlap,
)
from experiments.diagnostics.fig9_segment_context_composition import (  # noqa: E402
    LEVELS as SEGMENT_CONTEXT_COMPOSITION_LEVELS,
    MARKERS as SEGMENT_CONTEXT_COMPOSITION_MARKERS,
    SegmentContextCompositionTracker,
    protocol as segment_context_composition_protocol,
)
from experiments.diagnostics.fig9_context_trajectory import (  # noqa: E402
    CONTEXT_TRAJECTORY_LEVELS,
    CONTEXT_TRAJECTORY_MARKERS,
    ContextTrajectoryTracker,
)
from experiments.diagnostics.fig9_temporal_context import (  # noqa: E402
    TemporalContextTracker,
    protocol as temporal_context_protocol,
)
from experiments.diagnostics.fig9_autonomous_context_provenance import (  # noqa: E402
    AutonomousContextProvenanceTracker,
    protocol as autonomous_context_provenance_protocol,
)
from experiments.diagnostics.fig9_segment_reinforcement import (  # noqa: E402
    DIAGNOSTIC_MARKERS as SEGMENT_REINFORCEMENT_MARKERS,
    SEGMENT_REINFORCEMENT_LEVELS,
    build_segment_reinforcement_rows,
    enrich_segment_reinforcement_rows,
)
from experiments.diagnostics.fig9_independent_reference import (  # noqa: E402
    IndependentReferenceTracker,
    capture_before_matching,
    join_after_observation,
    protocol as independent_reference_protocol,
    record_actual_history,
    write_trace as write_independent_reference_trace,
)
from experiments.diagnostics.fig9_actual_branch_provenance import (  # noqa: E402
    ActualBranchProvenanceTracker,
    capture_prematch as capture_actual_branch_prematch,
    join_after_observation as join_actual_branch_after_observation,
    record_actual_transition,
    write_lineage_registry,
    write_registry as write_actual_branch_registry,
    write_rows as write_actual_branch_rows,
    MARKERS as ACTUAL_BRANCH_MARKERS,
)
from seqmem.encoding import (  # noqa: E402
    SSTDCompositeEncoder,
    SSTDPeriodicEncoder,
    SSTDRealValueEncoder,
    SymbolCode,
)
from seqmem.model import (  # noqa: E402
    INTRACOLUMN_SELECTION_POLICIES,
    IntracolumnSelectionTrace,
    MemoryParams,
    ObservationTrace,
    PreselectionTrace,
    PredictionTrace,
    SequentialMemory,
)


PAPER_CHANGE_DATE = datetime(2015, 4, 1)
CONTINUOUS_IMPL_VERSIONS = {
    "reference": "reference-v1",
    "optimized_v1": "optimized-v1-local-bindings-local-response-memo",
    "optimized_v2": "optimized-v2-exact-arrivals-memo",
}

_NATIVE_CRASH_HANDLE: io.TextIOWrapper | None = None
_NATIVE_CRASH_PATH: Path | None = None
_NATIVE_TRACEBACK_STOP: threading.Event | None = None
_NATIVE_TRACEBACK_THREAD: threading.Thread | None = None
_NATIVE_LOG_LOCK = threading.Lock()
_LAST_PROCESS_MEMORY_MB = 0.0


@dataclass(frozen=True)
class Fig9StrictConfig:
    horizon: int = 5
    warmup: int = 5904
    rolling_window: int = 400
    seed: int = 0
    weekday_columns: int = 30
    time_columns: int = 58
    passenger_columns: int = 482
    k: int = 10
    neurons_per_column: int = 32
    l_match: int = 4
    forgetting_threshold: float = 65.0
    passenger_min: float = 0.0
    passenger_max: float = 40000.0
    response_scale: float | None = None
    continuous_dynamics: bool = True
    integration_step: float = 0.005
    burst_context: bool = True
    intracolumn_inhibition: bool = True
    propagation_mode: str = "raw"
    scenario1_rule: str = "direct-reinforce-predicted-segment"
    use_future_covariates: bool = False
    reencode_decoded_value: bool = False
    rollout_learning: bool = False
    restore_transient_state: bool = True
    strict_label: str = "strict"
    continuous_prediction_impl: str = "reference"


@dataclass(frozen=True)
class RolloutResult:
    code: SymbolCode | None
    raw_event_counts: tuple[int, ...]
    raw_column_counts: tuple[int, ...]
    diagnostics: tuple[dict[str, object], ...] = ()
    emitted_event_counts: tuple[int, ...] = ()
    emitted_column_counts: tuple[int, ...] = ()
    oracle_diagnostics: tuple[dict[str, object], ...] = ()
    candidate_score_diagnostics: tuple[dict[str, object], ...] = ()
    branch_candidate_diagnostics: tuple[dict[str, object], ...] = ()
    branch_segment_diagnostics: tuple[dict[str, object], ...] = ()
    branch_source_diagnostics: tuple[dict[str, object], ...] = ()
    preselection_funnel_diagnostics: tuple[dict[str, object], ...] = ()
    preselection_segment_diagnostics: tuple[dict[str, object], ...] = ()
    preselection_group_diagnostics: tuple[dict[str, object], ...] = ()
    preselection_replacement_diagnostics: tuple[dict[str, object], ...] = ()
    intracolumn_selection_diagnostics: tuple[dict[str, object], ...] = ()


@dataclass
class StrictRunState:
    encoder: SSTDCompositeEncoder
    model: SequentialMemory
    fingerprint: dict[str, object]
    stream_label: str
    data_path: str
    limit: int
    next_index: int
    predictions: list[float] = field(default_factory=list)
    targets: list[float] = field(default_factory=list)
    rows: list[dict[str, object]] = field(default_factory=list)
    recent_errors: deque[float] = field(default_factory=deque)
    missing: int = 0
    raw_event_counts: list[int] = field(default_factory=list)
    raw_column_counts: list[int] = field(default_factory=list)
    density_rows: list[dict[str, object]] = field(default_factory=list)
    interval_rows: list[dict[str, object]] = field(default_factory=list)
    started_at: float = 0.0


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return ordered[index]


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left, right)
    )
    left_var = sum((a - left_mean) ** 2 for a in left)
    right_var = sum((b - right_mean) ** 2 for b in right)
    if left_var == 0.0 or right_var == 0.0:
        return 0.0
    return numerator / (left_var * right_var) ** 0.5


def field_column_counts(code: SymbolCode | None, config: Fig9StrictConfig) -> dict[str, int]:
    if code is None:
        return {"weekday": 0, "time": 0, "passenger": 0}
    weekday_stop = config.weekday_columns
    time_stop = config.weekday_columns + config.time_columns
    columns = {event.column for event in code.events}
    return {
        "weekday": sum(0 <= column < weekday_stop for column in columns),
        "time": sum(weekday_stop <= column < time_stop for column in columns),
        "passenger": sum(time_stop <= column < config.weekday_columns + config.time_columns + config.passenger_columns for column in columns),
    }


def synapse_count(model: SequentialMemory) -> int:
    return sum(
        len(segment.synapses)
        for column in model.columns
        for neuron in column.neurons
        for segment in neuron.segments
    )


def passenger_decode_details(
    passenger_encoder: SSTDRealValueEncoder,
    code: SymbolCode,
    timing_tolerance: float,
) -> tuple[float, int]:
    predicted_times: dict[int, list[float]] = {}
    for event in code.events:
        if (
            passenger_encoder.column_offset
            <= event.column
            < passenger_encoder.column_offset + passenger_encoder.num_columns
        ):
            predicted_times.setdefault(event.column, []).append(event.time)
    if not predicted_times:
        raise ValueError("code contains no columns from this encoder.")
    best_score = (-1, -1)
    best_values: list[float] = []
    for value, candidate_code in passenger_encoder.likelihood_grid():
        column_overlap = sum(
            event.column in predicted_times for event in candidate_code.events
        )
        timed_overlap = sum(
            event.column in predicted_times
            and any(
                abs(predicted_time - event.time) <= timing_tolerance
                for predicted_time in predicted_times[event.column]
            )
            for event in candidate_code.events
        )
        score = (timed_overlap, column_overlap)
        if score > best_score:
            best_score = score
            best_values = [value]
        elif score == best_score:
            best_values.append(value)
    return sum(best_values) / len(best_values), len(best_values)


def segment_count(model: SequentialMemory) -> int:
    return sum(
        len(neuron.segments)
        for column in model.columns
        for neuron in column.neurons
    )


def transient_fingerprint(model: SequentialMemory) -> str:
    payload = repr(
        (
            sorted(model.previous_active_cells.items()),
            sorted(model.previous_winners.items()),
            sorted(
                (
                    column,
                    [
                        (candidate.neuron_index, candidate.time, candidate.score)
                        for candidate in candidates
                    ],
                )
                for column, candidates in model.last_prediction_candidates.items()
            ),
            model._decode_rng.getstate(),
            model._learning_rng.getstate(),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_json_atomic(path: Path, payload: object) -> None:
    """Write JSON beside a checkpoint without exposing a partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_diagnostic_csv(
    path: Path,
    rows: list[dict[str, object]],
    *,
    compress: bool = False,
) -> Path | None:
    """Write a diagnostic CSV or CSV.GZ without changing row contents."""

    if not rows:
        return None
    actual_path = (
        path.with_suffix(path.suffix + ".gz") if compress else path
    )
    actual_path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        handle_context = gzip.open(
            actual_path,
            "wt",
            encoding="utf-8",
            newline="",
        )
    else:
        handle_context = actual_path.open(
            "w",
            encoding="utf-8",
            newline="",
        )
    with handle_context as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return actual_path


def append_diagnostic_csv(
    path: Path,
    rows: list[dict[str, object]],
    *,
    compress: bool = False,
) -> Path | None:
    """Append a homogeneous trace batch and flush it out of process memory."""

    if not rows:
        return None
    actual_path = path.with_suffix(path.suffix + ".gz") if compress else path
    actual_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not actual_path.exists() or actual_path.stat().st_size == 0
    opener = gzip.open if compress else Path.open
    if compress:
        handle_context = opener(
            actual_path, "at", encoding="utf-8", newline=""
        )
    else:
        handle_context = opener(
            actual_path, "a", encoding="utf-8", newline=""
        )
    with handle_context as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()
    return actual_path


def working_set_memory_mb() -> float:
    """Read working set out-of-process, avoiding an unsafe ctypes boundary."""

    if os.name != "nt":
        try:
            import resource

            value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return float(value) / (1024.0 if value > 1024 * 1024 else 1.0)
        except (ImportError, OSError):
            return 0.0

    process = subprocess.Popen(
        [
            "tasklist.exe",
            "/FI",
            f"PID eq {os.getpid()}",
            "/FO",
            "CSV",
            "/NH",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    # Read the tiny tasklist response on the calling thread. subprocess.run
    # with capture_output creates transient Windows reader threads; periodic
    # faulthandler enumeration previously raced with those threads exiting.
    assert process.stdout is not None
    output = process.stdout.read()
    process.stdout.close()
    returncode = process.wait()
    if returncode != 0 or not output.strip():
        return 0.0
    try:
        row = next(csv.reader([output.strip().splitlines()[0]]))
        kilobytes = int("".join(character for character in row[-1] if character.isdigit()))
    except (IndexError, StopIteration, ValueError):
        return 0.0
    return kilobytes / 1024.0


def native_environment() -> dict[str, object]:
    """Collect versions without importing optional native libraries."""

    packages = {}
    for name in (
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "numexpr",
        "threadpoolctl",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    native_modules = sorted(
        {
            str(getattr(module, "__file__", ""))
            for module in sys.modules.values()
            if str(getattr(module, "__file__", "")).lower().endswith(
                (".pyd", ".dll")
            )
        }
    )
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "architecture": platform.architecture(),
        "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
        "zlib_compile": zlib.ZLIB_VERSION,
        "packages": packages,
        "loaded_native_modules_at_start": native_modules,
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


def close_native_crash_logging() -> None:
    global _NATIVE_CRASH_HANDLE, _NATIVE_TRACEBACK_STOP, _NATIVE_TRACEBACK_THREAD
    if _NATIVE_CRASH_HANDLE is None:
        return
    if _NATIVE_TRACEBACK_STOP is not None:
        _NATIVE_TRACEBACK_STOP.set()
    if _NATIVE_TRACEBACK_THREAD is not None:
        _NATIVE_TRACEBACK_THREAD.join(timeout=2.0)
    _NATIVE_TRACEBACK_STOP = None
    _NATIVE_TRACEBACK_THREAD = None
    faulthandler.disable()
    with _NATIVE_LOG_LOCK:
        _NATIVE_CRASH_HANDLE.flush()
        _NATIVE_CRASH_HANDLE.close()
    _NATIVE_CRASH_HANDLE = None


def _periodic_native_traceback(stop: threading.Event) -> None:
    """Write synchronous stack snapshots without the Windows watchdog thread."""

    while not stop.wait(60.0):
        handle = _NATIVE_CRASH_HANDLE
        if handle is None:
            return
        with _NATIVE_LOG_LOCK:
            handle.write("PERIODIC_TRACEBACK interval_seconds=60\n")
            handle.flush()
            # Unlike dump_traceback_later, this call runs while a normal Python
            # thread owns the GIL. That avoids asynchronously walking frames as
            # the model mutates Python dictionaries on Windows.
            faulthandler.dump_traceback(file=handle, all_threads=True)
            handle.flush()


def install_native_crash_logging(
    output_dir: Path,
    *,
    periodic_traceback_seconds: float = 0.0,
) -> Path:
    """Keep a dedicated faulthandler descriptor alive for the whole process."""

    global _NATIVE_CRASH_HANDLE, _NATIVE_CRASH_PATH
    global _NATIVE_TRACEBACK_STOP, _NATIVE_TRACEBACK_THREAD
    close_native_crash_logging()
    output_dir.mkdir(parents=True, exist_ok=True)
    _NATIVE_CRASH_PATH = output_dir / "native_crash_faulthandler.log"
    _NATIVE_CRASH_HANDLE = _NATIVE_CRASH_PATH.open(
        "a", encoding="utf-8", buffering=1
    )
    faulthandler.enable(file=_NATIVE_CRASH_HANDLE, all_threads=True)
    if periodic_traceback_seconds > 0.0:
        if periodic_traceback_seconds != 60.0:
            raise ValueError("only the audited 60-second traceback interval is supported")
        _NATIVE_TRACEBACK_STOP = threading.Event()
        _NATIVE_TRACEBACK_THREAD = threading.Thread(
            target=_periodic_native_traceback,
            args=(_NATIVE_TRACEBACK_STOP,),
            name="fig9-periodic-traceback",
            daemon=True,
        )
        _NATIVE_TRACEBACK_THREAD.start()
    environment = native_environment()
    write_json_atomic(output_dir / "native_environment.json", environment)
    with _NATIVE_LOG_LOCK:
        _NATIVE_CRASH_HANDLE.write(
            "NATIVE_ENVIRONMENT " + json.dumps(environment, sort_keys=True) + "\n"
        )
        _NATIVE_CRASH_HANDLE.write(
            "PERIODIC_TRACEBACK "
            + ("interval_seconds=60\n" if periodic_traceback_seconds else "disabled\n")
        )
        _NATIVE_CRASH_HANDLE.flush()
    return _NATIVE_CRASH_PATH


def write_native_crash_breadcrumb(
    *,
    current_index: int,
    rollout_step: int | None,
    phase: str,
    model: SequentialMemory,
) -> None:
    global _LAST_PROCESS_MEMORY_MB
    if _NATIVE_CRASH_HANDLE is None:
        return
    if phase == "observation_enter":
        _LAST_PROCESS_MEMORY_MB = working_set_memory_mb()
    payload = {
        "timestamp": datetime.now().isoformat(sep=" ", timespec="milliseconds"),
        "current_index": current_index,
        "rollout_step": rollout_step,
        "phase": phase,
        "segment_count": segment_count(model),
        "candidate_count": int(
            model.last_prediction_stats.get("candidate_segment_count", 0)
        ),
        "process_memory_mb": _LAST_PROCESS_MEMORY_MB,
    }
    with _NATIVE_LOG_LOCK:
        _NATIVE_CRASH_HANDLE.write(
            "FIG9_NATIVE_PHASE " + json.dumps(payload, sort_keys=True) + "\n"
        )
        _NATIVE_CRASH_HANDLE.flush()


class PruneBreadcrumbRecorder:
    """Write scalar-only prune breadcrumbs without retaining model objects."""

    def __init__(self, config: Fig9StrictConfig) -> None:
        self.current_index = -1
        self.config = config
        self.sequence = 0

    def set_index(self, current_index: int) -> None:
        self.current_index = current_index

    def __call__(self, payload: dict[str, object]) -> None:
        if _NATIVE_CRASH_HANDLE is None:
            return
        self.sequence += 1
        column = payload.get("encoded_column")
        field = (
            field_for_column(
                int(column),
                FieldColumnRanges.from_sizes(
                    self.config.weekday_columns,
                    self.config.time_columns,
                    self.config.passenger_columns,
                ),
            )
            if isinstance(column, int)
            else ""
        )
        record = {
            "timestamp": datetime.now().isoformat(
                sep=" ", timespec="milliseconds"
            ),
            "current_index": self.current_index,
            "rollout_step": None,
            "phase": payload.get("phase", ""),
            "field": field,
            "encoded_column": column,
            "stable_neuron_id": payload.get("stable_neuron_id", ""),
            "stable_segment_id": payload.get("stable_segment_id"),
            "target_column": payload.get("target_column"),
            "neuron_index": payload.get("neuron_index"),
            "segment_count": payload.get("segment_count", 0),
            "container_length": payload.get("container_length", 0),
            "segment_id_hash": payload.get("segment_id_hash", ""),
            "actual_branch_provenance_diagnostic": True,
            "callback_scalar_only": True,
            "phase_sequence": self.sequence,
            # Refreshing Windows process memory for every prune callback is
            # disproportionately expensive.  The observation breadcrumb
            # updates this scalar once per record; all prune phases reuse it.
            "process_memory_mb": _LAST_PROCESS_MEMORY_MB,
        }
        with _NATIVE_LOG_LOCK:
            _NATIVE_CRASH_HANDLE.write(
                "FIG9_NATIVE_PRUNE " + json.dumps(record, sort_keys=True) + "\n"
            )
            _NATIVE_CRASH_HANDLE.flush()


atexit.register(close_native_crash_logging)


def write_density_trace(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    write_predictions(path, rows)


def save_strict_checkpoint(
    path: Path,
    *,
    encoder: SSTDCompositeEncoder,
    model: SequentialMemory,
    fingerprint: dict[str, object],
    config: Fig9StrictConfig,
    stream_label: str,
    data_path: Path,
    records: list[TaxiRecord],
    limit: int,
    next_index: int,
    predictions: list[float],
    targets: list[float],
    rows: list[dict[str, object]],
    recent_errors: deque[float],
    missing: int,
    raw_event_counts: list[int],
    raw_column_counts: list[int],
    density_rows: list[dict[str, object]],
    branch_provenance_payload: dict[str, object] | None = None,
    diagnostic_state: dict[str, object] | None = None,
) -> None:
    payload = {
        "checkpoint_format": "fig9-strict-v1",
        "git_commit_sha": git_commit_sha(),
        "encoder": encoder,
        "model": model,
        "fingerprint": fingerprint,
        "config": config,
        "L_match": config.l_match,
        "stream_label": stream_label,
        "data_path": str(data_path),
        "data_file_sha256": file_sha256(data_path),
        "prefix_end_index": next_index,
        "prefix_end_timestamp": (
            records[next_index - 1].timestamp.isoformat(sep=" ")
            if next_index > 0 and next_index <= len(records)
            else ""
        ),
        "common_prefix_hash": records_prefix_sha256(records, next_index),
        "perturbation_split_timestamp": PAPER_CHANGE_DATE.isoformat(sep=" "),
        "limit": limit,
        "next_index": next_index,
        "predictions": predictions,
        "targets": targets,
        "rows": rows,
        "recent_errors": list(recent_errors),
        "missing": missing,
        "raw_event_counts": raw_event_counts,
        "raw_column_counts": raw_column_counts,
        "density_rows": density_rows,
        "branch_provenance": branch_provenance_payload,
        "diagnostic_state": diagnostic_state,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def checkpoint_metadata_path(checkpoint_path: Path) -> Path:
    """Return the final metadata record for a strict checkpoint."""

    return checkpoint_path.with_name(
        checkpoint_path.stem + ".metadata.json"
    )


def write_checkpoint_metadata(
    checkpoint_path: Path,
    *,
    next_index: int,
    fingerprint: dict[str, object],
    model: SequentialMemory,
    branch_provenance_payload: dict[str, object] | None,
) -> None:
    """Publish a durable checkpoint boundary after pickle and sidecar writes."""

    sidecar = branch_checkpoint_sidecar_path(checkpoint_path)
    sidecar_enabled = (
        branch_provenance_payload is not None and sidecar.exists()
    )
    payload = {
        "checkpoint_format": "fig9-strict-v1",
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "next_index": next_index,
        "last_fully_completed_index": next_index,
        "saved_at": datetime.now().isoformat(sep=" ", timespec="milliseconds"),
        "model_fingerprint": model_long_term_fingerprint(model),
        "rng_fingerprint": model_rng_fingerprint(model),
        "branch_provenance_version": (
            branch_provenance_payload.get("version")
            if branch_provenance_payload is not None
            else None
        ),
        "branch_provenance_sidecar_path": (
            str(sidecar.resolve()) if sidecar_enabled else None
        ),
        "branch_provenance_sidecar_sha256": (
            file_sha256(sidecar) if sidecar_enabled else None
        ),
        "trace_boundary": {"next_index": next_index},
        "protocol_sha256": stable_object_sha256(fingerprint),
        "git_commit_sha": git_commit_sha(),
        "atomic_complete": True,
    }
    write_json_atomic(checkpoint_metadata_path(checkpoint_path), payload)


class _StrictCheckpointUnpickler(pickle.Unpickler):
    """Load checkpoints written by either module or direct-script entrypoints."""

    def find_class(self, module: str, name: str) -> type:
        if module == "__main__" and name == "Fig9StrictConfig":
            return Fig9StrictConfig
        return super().find_class(module, name)


def load_strict_checkpoint(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        payload = _StrictCheckpointUnpickler(handle).load()
    if payload.get("checkpoint_format") != "fig9-strict-v1":
        raise ValueError("unsupported Fig.9 strict checkpoint format")
    return payload


def validate_checkpoint_l_match(
    checkpoint: dict[str, object],
    config: Fig9StrictConfig,
) -> None:
    """Reject resumes that would silently change the real matching threshold."""

    checkpoint_config = checkpoint.get("config")
    checkpoint_l_match = checkpoint.get("L_match")
    if checkpoint_l_match is None and isinstance(
        checkpoint_config, Fig9StrictConfig
    ):
        checkpoint_l_match = checkpoint_config.l_match
    if checkpoint_l_match is None:
        fingerprint = checkpoint.get("fingerprint")
        if isinstance(fingerprint, dict):
            checkpoint_l_match = fingerprint.get("L_match")
    if checkpoint_l_match is None:
        model = checkpoint.get("model")
        params = getattr(model, "params", None)
        checkpoint_l_match = getattr(params, "l_match", None)
    if checkpoint_l_match is None:
        raise ValueError("checkpoint does not record L_match")
    if int(checkpoint_l_match) != config.l_match:
        raise ValueError(
            "checkpoint L_match mismatch: "
            f"checkpoint={checkpoint_l_match}, requested={config.l_match}"
        )


def branch_checkpoint_sidecar_path(checkpoint_path: Path) -> Path:
    """Return the external provenance sidecar without changing checkpoint v1."""

    return checkpoint_path.with_suffix(
        checkpoint_path.suffix + ".branch_provenance.json"
    )


def find_timestamp_split(
    records: list[TaxiRecord],
    split_time: datetime = PAPER_CHANGE_DATE,
) -> int:
    for index, record in enumerate(records):
        if record.timestamp >= split_time:
            return index
    raise ValueError(f"no record at or after {split_time.isoformat(sep=' ')}")


def summarize_density(rows: list[dict[str, object]]) -> dict[str, object]:
    by_step: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        by_step.setdefault(int(row["horizon_step"]), []).append(row)

    step_summary: list[dict[str, object]] = []
    for step, step_rows in sorted(by_step.items()):
        raw_columns = [
            float(row["raw_predicted_column_count"])
            for row in step_rows
        ]
        active_cells = [
            float(row["active_source_count"]) for row in step_rows
        ]
        candidates = [
            float(row["candidate_segment_count"]) for row in step_rows
        ]
        accepted = [
            float(row["accepted_candidate_count"]) for row in step_rows
        ]
        prediction_runtimes = [
            float(row["prediction_runtime_seconds"]) for row in step_rows
        ]
        passenger_density = [
            float(row["passenger_predicted_column_count"])
            for row in step_rows
        ]
        absolute_errors = [
            float(row["absolute_error"])
            for row in step_rows
            if row["absolute_error"] != ""
        ]
        targets = [
            abs(float(row["actual_future_passenger"]))
            for row in step_rows
            if row["actual_future_passenger"] != ""
        ]
        no_prediction = sum(
            row["stopped_reason"] in {"no_raw_prediction", "no_prediction_active_cells"}
            for row in step_rows
        )
        step_summary.append(
            {
                "horizon_step": step,
                "rows": len(step_rows),
                "raw_columns_mean": sum(raw_columns) / len(raw_columns) if raw_columns else 0.0,
                "raw_columns_p50": _percentile(raw_columns, 0.50),
                "raw_columns_p90": _percentile(raw_columns, 0.90),
                "raw_columns_p99": _percentile(raw_columns, 0.99),
                "raw_columns_max": max(raw_columns) if raw_columns else 0.0,
                "active_cells_mean": sum(active_cells) / len(active_cells) if active_cells else 0.0,
                "candidate_segments_mean": sum(candidates) / len(candidates) if candidates else 0.0,
                "accepted_candidates_mean": sum(accepted) / len(accepted) if accepted else 0.0,
                "passenger_field_density_mean": sum(passenger_density) / len(passenger_density) if passenger_density else 0.0,
                "prediction_runtime_mean": sum(prediction_runtimes) / len(prediction_runtimes) if prediction_runtimes else 0.0,
                "mape": (
                    sum(absolute_errors) / sum(targets)
                    if absolute_errors and sum(targets)
                    else 0.0
                ),
                "no_prediction_rate": no_prediction / len(step_rows) if step_rows else 0.0,
            }
        )

    final_step = [
        row for row in rows if int(row["horizon_step"]) == max(by_step, default=0)
    ]
    raw = [
        float(row["raw_predicted_column_count"])
        for row in final_step
        if row["absolute_percentage_error"] != ""
    ]
    errors = [
        float(row["absolute_percentage_error"])
        for row in final_step
        if row["absolute_percentage_error"] != ""
    ]
    weekday = [
        float(row["weekday_predicted_column_count"]) for row in final_step
    ]
    time_field = [
        float(row["time_predicted_column_count"]) for row in final_step
    ]
    passenger = [
        float(row["passenger_predicted_column_count"]) for row in final_step
    ]
    return {
        "diagnostic_label": "strict Fig.9 density diagnostic",
        "rows": len(rows),
        "step_summary": step_summary,
        "final_step_raw_density_error_correlation": _correlation(raw, errors),
        "final_step_mean_weekday_columns": sum(weekday) / len(weekday) if weekday else 0.0,
        "final_step_mean_time_columns": sum(time_field) / len(time_field) if time_field else 0.0,
        "final_step_mean_passenger_columns": sum(passenger) / len(passenger) if passenger else 0.0,
        "answers": {
            "step1_already_dense": (
                step_summary[0]["raw_columns_mean"] > 100.0
                if step_summary
                else False
            ),
            "density_monotonic_step1_to_step5": all(
                left["raw_columns_mean"] <= right["raw_columns_mean"]
                for left, right in zip(step_summary, step_summary[1:])
            ),
            "high_mape_mainly_final_step_accumulation": (
                len(step_summary) >= 5
                and step_summary[-1]["raw_columns_mean"]
                > step_summary[0]["raw_columns_mean"]
            ),
        },
    }


def build_fig9_encoder(config: Fig9StrictConfig) -> SSTDCompositeEncoder:
    """Build weekday/time/passenger encoders with fixed Fig.9 column offsets.

    调试时看这里确认三字段列空间没有重叠：
    weekday: 0..29，time: 30..87，passenger: 88..569。
    """

    # STRICT PROTOCOL: Fig.9 使用 30/58/482 三组 mini-column，K=10。
    # 改这里会改变编码容量，不能和论文 strict 结果直接比较。
    day_encoder = SSTDPeriodicEncoder(
        num_columns=config.weekday_columns,
        k=config.k,
        period=7.0,
        column_offset=0,
    )
    time_encoder = SSTDPeriodicEncoder(
        num_columns=config.time_columns,
        k=config.k,
        period=48.0,
        column_offset=config.weekday_columns,
    )
    passenger_encoder = SSTDRealValueEncoder(
        num_columns=config.passenger_columns,
        k=config.k,
        minimum=config.passenger_min,
        maximum=config.passenger_max,
        column_offset=config.weekday_columns + config.time_columns,
    )
    return SSTDCompositeEncoder([day_encoder, time_encoder, passenger_encoder])


def build_strict_model(
    encoder: SSTDCompositeEncoder,
    config: Fig9StrictConfig,
) -> SequentialMemory:
    return SequentialMemory(
        encoder=encoder,  # type: ignore[arg-type]
        num_neurons_per_column=config.neurons_per_column,
        params=MemoryParams(
            l_match=config.l_match,
            forgetting_threshold=config.forgetting_threshold,
            response_scale=config.response_scale,
            burst_context=config.burst_context,
            intracolumn_inhibition=config.intracolumn_inhibition,
            continuous_dynamics=config.continuous_dynamics,
            integration_step=config.integration_step,
            continuous_prediction_impl=config.continuous_prediction_impl,
        ),
        tie_break_seed=config.seed,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_object_sha256(value: object) -> str:
    return hashlib.sha256(
        pickle.dumps(value, protocol=4)
    ).hexdigest()


def model_long_term_fingerprint(model: SequentialMemory) -> str:
    payload = []
    for column_index, column in enumerate(model.columns):
        for neuron_index, neuron in enumerate(column.neurons):
            for segment_index, segment in enumerate(neuron.segments):
                payload.append(
                    (
                        column_index,
                        neuron_index,
                        segment_index,
                        segment.active,
                        segment.target_time,
                        segment.diagnostic_id,
                        segment.creation_sentence_index,
                        segment.creation_transition_index,
                        segment.creation_target_column,
                        segment.creation_target_time,
                        segment.creation_target_neuron,
                        segment.creation_source_cell_ids,
                        segment.creation_source_fingerprint,
                        segment.scenario1_reinforcements,
                        segment.scenario2_reinforcements,
                        tuple(
                            sorted(
                                (
                                    source,
                                    synapse.source,
                                    synapse.delay,
                                    synapse.weight,
                                    synapse.age,
                                )
                                for source, synapse in segment.synapses.items()
                            )
                        ),
                    )
                )
    return stable_object_sha256(payload)


def model_rng_fingerprint(model: SequentialMemory) -> str:
    snapshot = model.snapshot_transient_state()
    return stable_object_sha256(
        {
            "decode_rng_state": snapshot.decode_rng_state,
            "learning_rng_state": snapshot.learning_rng_state,
        }
    )


def continuous_impl_version(impl: str) -> str:
    return CONTINUOUS_IMPL_VERSIONS[impl]


def records_prefix_sha256(records: list[TaxiRecord], stop_index: int) -> str:
    digest = hashlib.sha256()
    for record in records[:stop_index]:
        digest.update(
            f"{record.timestamp.isoformat(sep=' ')},{record.value}\n".encode("utf-8")
        )
    return digest.hexdigest()


def git_commit_sha() -> str:
    candidates = [
        "git",
        str(
            Path.home()
            / ".cache"
            / "codex-runtimes"
            / "codex-primary-runtime"
            / "dependencies"
            / "native"
            / "git"
            / "cmd"
            / "git.exe"
        ),
    ]
    for candidate in candidates:
        try:
            completed = subprocess.run(
                [candidate, "rev-parse", "HEAD"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            return completed.stdout.strip()
        except Exception:
            continue
    return "unknown"


def protocol_fingerprint(
    *,
    data_path: Path,
    records: list[TaxiRecord],
    stream_label: str,
    config: Fig9StrictConfig,
    limit: int,
) -> dict[str, object]:
    """Record the protocol knobs that must stay stable across strict runs.

    DEBUG WATCH: 先看输出的 protocol JSON，再看 summary。若这里记录
    的 strict flags 不是 raw/no-future/no-replay/no-rollout-learning，后面
    的 MAPE 数值就不能当作 strict Fig.9。
    """

    params = MemoryParams(
        response_scale=config.response_scale,
        continuous_dynamics=config.continuous_dynamics,
        integration_step=config.integration_step,
    )
    dynamics = params.dynamics()
    return {
        "git_commit_sha": git_commit_sha(),
        "data_file_path": str(data_path),
        "data_file_rows_used": len(records),
        "data_file_limit": limit,
        "data_file_sha256": file_sha256(data_path),
        "stream_label": stream_label,
        "date_range": [
            records[0].timestamp.isoformat(sep=" ") if records else "",
            records[-1].timestamp.isoformat(sep=" ") if records else "",
        ],
        "encoder_sizes": {
            "weekday": config.weekday_columns,
            "time": config.time_columns,
            "passenger": config.passenger_columns,
            "total": (
                config.weekday_columns
                + config.time_columns
                + config.passenger_columns
            ),
        },
        "K": config.k,
        "neurons_per_column": config.neurons_per_column,
        "L_match": config.l_match,
        "forgetting_threshold": config.forgetting_threshold,
        "response_scale": config.response_scale,
        "V0": dynamics.response_scale,
        "tau_m": dynamics.tau_m,
        "tau_s": dynamics.tau_s,
        "continuous_dynamics": config.continuous_dynamics,
        "continuous_impl": config.continuous_prediction_impl,
        "continuous_impl_version": continuous_impl_version(
            config.continuous_prediction_impl
        ),
        "integration_step": config.integration_step,
        "burst_context": "all-cell" if config.burst_context else "winner-only",
        "intracolumn_inhibition": config.intracolumn_inhibition,
        "propagation_mode": config.propagation_mode,
        "scenario1_rule": config.scenario1_rule,
        "prediction_horizon": config.horizon,
        "warmup_length": config.warmup,
        "rolling_window": config.rolling_window,
        "MAPE_denominator": "sum_abs_targets_reference_58",
        "rolling_MAPE_denominator": "global_mean_abs_target_reference_58",
        "RNG_seeds": {"tie_break_seed": config.seed},
        "uses_future_covariates": config.use_future_covariates,
        "reencodes_decoded_value": config.reencode_decoded_value,
        "learns_during_rollout": config.rollout_learning,
        "restores_transient_state": config.restore_transient_state,
        "strict_label": config.strict_label,
    }


def validate_strict_fingerprint(fingerprint: dict[str, object]) -> None:
    """Fail fast when a historical compensation leaks into the strict runner."""

    forbidden = []
    if fingerprint["burst_context"] != "all-cell":
        forbidden.append("winner-only burst context")
    if fingerprint["propagation_mode"] != "raw":
        forbidden.append(str(fingerprint["propagation_mode"]))
    if fingerprint["uses_future_covariates"]:
        forbidden.append("known-future covariates")
    if fingerprint["reencodes_decoded_value"]:
        forbidden.append("decoded-value replay")
    if fingerprint["learns_during_rollout"]:
        forbidden.append("rollout learning")
    if fingerprint["strict_label"] != "strict":
        forbidden.append("non-strict label")
    if forbidden:
        raise ValueError(f"forbidden strict Fig.9 settings: {', '.join(forbidden)}")


def lmatch_ablation_markers(l_match: int) -> dict[str, object]:
    """Return auditable labels for a nonpaper real-L_match experiment."""

    return {
        "diagnostic_only": True,
        "lmatch_real_ablation": True,
        "strict_default_unchanged": Fig9StrictConfig().l_match == 4,
        "uses_ground_truth_for_selection": False,
        "uses_future_covariates": False,
        "uses_compensation": False,
        "L_match": l_match,
        "recurrent_trajectory_divergence": True,
    }


def rollout_raw_autonomous(
    model: SequentialMemory,
    steps: int,
    *,
    config: Fig9StrictConfig | None = None,
    competition_settings: CompetitionSettings | None = None,
    oracle_candidate_diagnostic: bool = False,
    candidate_separability_trace: bool = False,
    branch_provenance_diagnostic: bool = False,
    branch_provenance_level: str = "candidate",
    branch_source_top_n: int | None = None,
    branch_registry: BranchProvenanceRegistry | None = None,
    preselection_segment_diagnostic: bool = False,
    preselection_segment_level: str = "crossing",
    oracle_encoder: SSTDCompositeEncoder | None = None,
    passenger_encoder: SSTDRealValueEncoder | None = None,
    stream_label: str = "original",
    record_index: int | None = None,
    timestamp: datetime | None = None,
    future_records: list[TaxiRecord] | None = None,
    intracolumn_selection_policy: str = "existing",
    intracolumn_selection_diagnostic: bool = False,
    context_trajectory_tracker: ContextTrajectoryTracker | None = None,
    context_oracle_unified_trace: bool = False,
    temporal_context_tracker: TemporalContextTracker | None = None,
    autonomous_context_provenance_tracker: AutonomousContextProvenanceTracker | None = None,
    native_phase_hook: Callable[[int, str], None] | None = None,
) -> RolloutResult:
    """Roll out future SSTD codes using only raw predictive neurons.

    中文调试提示：这里是 5-step rollout 的核心。每一步只调用
    predict_code()，然后把 raw prediction 对应的真实预测细胞放回
    previous_active_cells/previous_winners，绝不把解码值重新 encode 后输入。
    finally 中 restore_transient_state() 会撤销 rollout 对临时状态和 RNG 的影响。
    """

    # DEBUG WATCH: 在这里下断点可观察每个 horizon step 的 raw.events、
    # raw columns 数量和 prediction_active_cells(raw) 是否突然爆炸。
    snapshot = model.snapshot_transient_state()
    competition = competition_settings or CompetitionSettings()
    competition.validate()
    if oracle_candidate_diagnostic and not competition.enabled:
        raise ValueError(
            "oracle candidate diagnostics require competitive_raw mode"
        )
    if candidate_separability_trace and not oracle_candidate_diagnostic:
        raise ValueError(
            "candidate separability trace requires oracle candidate diagnostics"
        )
    if branch_provenance_level not in BRANCH_LEVELS:
        raise ValueError(
            "branch provenance level must be summary, candidate, or full"
        )
    if branch_provenance_diagnostic and not oracle_candidate_diagnostic:
        raise ValueError(
            "branch provenance diagnostics require oracle candidate diagnostics"
        )
    if branch_provenance_diagnostic and not competition.enabled:
        raise ValueError(
            "branch provenance diagnostics require competitive_raw mode"
        )
    if branch_provenance_diagnostic and branch_registry is None:
        raise ValueError("branch provenance diagnostics require a registry")
    if preselection_segment_level not in PRESELECTION_LEVELS:
        raise ValueError(
            "preselection segment level must be summary, crossing, or full"
        )
    if preselection_segment_diagnostic and not oracle_candidate_diagnostic:
        raise ValueError(
            "preselection diagnostics require oracle candidate diagnostics"
        )
    if preselection_segment_diagnostic and not competition.enabled:
        raise ValueError(
            "preselection diagnostics require competitive_raw mode"
        )
    if preselection_segment_diagnostic and branch_registry is None:
        raise ValueError(
            "preselection diagnostics require a provenance registry"
        )
    if oracle_candidate_diagnostic and (
        config is None or oracle_encoder is None or future_records is None
    ):
        raise ValueError(
            "oracle diagnostics require config, oracle_encoder, and future_records"
        )
    if intracolumn_selection_policy not in INTRACOLUMN_SELECTION_POLICIES:
        raise ValueError("unsupported intracolumn selection policy")
    if intracolumn_selection_diagnostic and config is None:
        raise ValueError(
            "intracolumn selection diagnostics require the strict config"
        )
    oracle_ranges = (
        FieldColumnRanges.from_sizes(
            config.weekday_columns,
            config.time_columns,
            config.passenger_columns,
        )
        if (oracle_candidate_diagnostic or autonomous_context_provenance_tracker is not None) and config is not None
        else None
    )
    prediction: SymbolCode | None = None
    raw_events: list[int] = []
    raw_columns: list[int] = []
    emitted_events: list[int] = []
    emitted_columns: list[int] = []
    diagnostics: list[dict[str, object]] = []
    oracle_diagnostics: list[dict[str, object]] = []
    candidate_score_diagnostics: list[dict[str, object]] = []
    branch_candidate_diagnostics: list[dict[str, object]] = []
    branch_segment_diagnostics: list[dict[str, object]] = []
    branch_source_diagnostics: list[dict[str, object]] = []
    preselection_funnel_diagnostics: list[dict[str, object]] = []
    preselection_segment_diagnostics: list[dict[str, object]] = []
    preselection_group_diagnostics: list[dict[str, object]] = []
    preselection_replacement_diagnostics: list[dict[str, object]] = []
    intracolumn_selection_diagnostics: list[dict[str, object]] = []
    registry_predicted_before = (
        branch_registry.current_predicted_sources.copy()
        if branch_registry is not None
        else set()
    )
    registry_burst_before = (
        branch_registry.current_burst_sources.copy()
        if branch_registry is not None
        else set()
    )
    try:
        if temporal_context_tracker is not None and record_index is not None:
            temporal_context_tracker.begin_record(record_index)
        provenance_rollout_started = False
        for _step_index in range(steps):
            if native_phase_hook is not None:
                native_phase_hook(_step_index + 1, "rollout_step_enter")
            step_intracolumn_rows: list[dict[str, object]] = []
            # DEBUG WATCH: horizon step start。此时 previous_active_cells
            # 来自上一轮 raw predictive neurons，而不是 decoded passenger。
            previous_active_count = len(model.previous_active_cells)
            previous_winner_count = len(model.previous_winners)
            predicted_source_count = len(model.previous_predicted_sources)
            burst_source_count = len(model.previous_burst_only_sources)
            branch_active_sources = (
                model._active_sources().copy()
                if (
                    branch_provenance_diagnostic
                    or preselection_segment_diagnostic
                    or context_oracle_unified_trace
                    or autonomous_context_provenance_tracker is not None
                )
                else {}
            )
            if (
                autonomous_context_provenance_tracker is not None
                and record_index is not None
                and not provenance_rollout_started
            ):
                autonomous_context_provenance_tracker.begin_rollout(
                    anchor_record_index=record_index,
                    anchor_timestamp=(
                        timestamp.isoformat(sep=" ")
                        if timestamp is not None
                        else ""
                    ),
                    pre_rollout_active=branch_active_sources,
                    pre_rollout_predicted=(
                        branch_registry.current_predicted_sources
                        if branch_registry is not None
                        else set()
                    ),
                    pre_rollout_burst=(
                        branch_registry.current_burst_sources
                        if branch_registry is not None
                        else set()
                    ),
                )
                provenance_rollout_started = True
            predict_started = time.perf_counter()
            prediction_trace = (
                PredictionTrace() if branch_provenance_diagnostic else None
            )
            preselection_trace = (
                PreselectionTrace()
                if (
                    preselection_segment_diagnostic
                    or context_trajectory_tracker is not None
                )
                else None
            )
            intracolumn_trace = (
                IntracolumnSelectionTrace()
                if intracolumn_selection_diagnostic
                else None
            )
            raw = model.predict_code(
                trace=prediction_trace,
                preselection_trace=preselection_trace,
                intracolumn_selection_policy=intracolumn_selection_policy,
                intracolumn_selection_trace=intracolumn_trace,
            )
            if native_phase_hook is not None:
                native_phase_hook(_step_index + 1, "rollout_predict_complete")
            provenance_target_columns: tuple[int, ...] = ()
            provenance_target_by_field: dict[str, int] = {}
            provenance_target_timestamp = ""
            if (
                autonomous_context_provenance_tracker is not None
                and oracle_encoder is not None
                and future_records is not None
                and _step_index < len(future_records)
            ):
                provenance_target = oracle_encoder.encode(
                    record_values(future_records[_step_index])
                )
                provenance_target_columns = tuple(
                    int(event.column) for event in provenance_target.events
                )
                provenance_target_by_field = {
                    field_for_column(event.column, oracle_ranges): int(event.column)
                    for event in provenance_target.events
                }
                provenance_target_timestamp = future_records[
                    _step_index
                ].timestamp.isoformat(sep=" ")
            autonomous_context_index = None
            if (
                context_oracle_unified_trace
                and branch_registry is not None
                and oracle_ranges is not None
            ):
                autonomous_context_index = ContextCompositionIndex.from_model(
                    model=model,
                    registry=branch_registry,
                    ranges=oracle_ranges,
                    actual_record_index=(
                        record_index if record_index is not None else 0
                    ),
                    active_sources=branch_active_sources,
                )
            predict_runtime = time.perf_counter() - predict_started
            if (
                intracolumn_trace is not None
                and config is not None
                and oracle_ranges is not None
            ):
                step_intracolumn_rows = selection_trace_rows(
                    trace=intracolumn_trace,
                    policy=intracolumn_selection_policy,
                    input_index=(
                        record_index + 1
                        if record_index is not None
                        else 0
                    ),
                    input_timestamp=(
                        timestamp.isoformat(sep=" ")
                        if timestamp is not None
                        else ""
                    ),
                    horizon_step=_step_index + 1,
                    ranges=oracle_ranges,
                )
            if raw is None:
                if autonomous_context_provenance_tracker is not None:
                    autonomous_context_provenance_tracker.record_step(
                        model=model,
                        raw=None,
                        propagated=None,
                        next_active_cells={},
                        competition_result=None,
                        actual_record_index=(
                            record_index if record_index is not None else 0
                        ),
                        horizon_step=_step_index + 1,
                        field_for_column=(
                            lambda column: field_for_column(column, oracle_ranges)
                        ),
                        target_columns=provenance_target_columns,
                        target_column_by_field=provenance_target_by_field,
                        input_index=(
                            record_index + 1 if record_index is not None else 0
                        ),
                        input_timestamp=(
                            timestamp.isoformat(sep=" ")
                            if timestamp is not None
                            else ""
                        ),
                        target_timestamp=provenance_target_timestamp,
                        registry=branch_registry,
                    )
                if context_trajectory_tracker is not None:
                    assert preselection_trace is not None
                    context_trajectory_tracker.record_transition(
                        model=model,
                        trajectory_kind="autonomous_rollout",
                        actual_record_index=(
                            record_index if record_index is not None else 0
                        ),
                        horizon_step=_step_index + 1,
                        preselection_trace=preselection_trace,
                        raw_code=None,
                        competition_result=None,
                        next_active_cells={},
                        next_winners={},
                    )
                intracolumn_selection_diagnostics.extend(
                    step_intracolumn_rows
                )
                if (
                    oracle_candidate_diagnostic
                    and oracle_encoder is not None
                    and oracle_ranges is not None
                    and future_records is not None
                    and _step_index < len(future_records)
                ):
                    target_record = future_records[_step_index]
                    oracle_diagnostics.append(
                        analyze_oracle_step(
                            prediction_input_index=(
                                record_index + 1
                                if record_index is not None
                                else 0
                            ),
                            input_timestamp=(
                                timestamp.isoformat(sep=" ")
                                if timestamp is not None
                                else ""
                            ),
                            horizon_step=_step_index + 1,
                            target_timestamp=target_record.timestamp.isoformat(
                                sep=" "
                            ),
                            target_passenger=target_record.value,
                            decoded_prediction="",
                            absolute_error="",
                            raw_prediction=None,
                            emitted_prediction=None,
                            competition_result=None,
                            target_code=oracle_encoder.encode(
                                record_values(target_record)
                            ),
                            ranges=oracle_ranges,
                        )
                    )
                if (
                    preselection_segment_diagnostic
                    and preselection_trace is not None
                    and branch_registry is not None
                    and oracle_encoder is not None
                    and oracle_ranges is not None
                    and future_records is not None
                    and _step_index < len(future_records)
                ):
                    target_record = future_records[_step_index]
                    (
                        funnel_row,
                        segment_rows,
                        group_rows,
                        replacement_rows,
                    ) = build_preselection_rows(
                        model=model,
                        registry=branch_registry,
                        trace=preselection_trace,
                        stream_label=stream_label,
                        policy=competition.simultaneous_policy,
                        input_index=(
                            record_index + 1
                            if record_index is not None
                            else 0
                        ),
                        input_timestamp=(
                            timestamp.isoformat(sep=" ")
                            if timestamp is not None
                            else ""
                        ),
                        target_timestamp=target_record.timestamp.isoformat(
                            sep=" "
                        ),
                        horizon_step=_step_index + 1,
                        target_code=oracle_encoder.encode(
                            record_values(target_record)
                        ),
                        ranges=oracle_ranges,
                        active_sources=branch_active_sources,
                        competition_result=None,
                        raw_code=None,
                        runtime_seconds=predict_runtime,
                        level=preselection_segment_level,
                    )
                    preselection_funnel_diagnostics.append(funnel_row)
                    preselection_segment_diagnostics.extend(segment_rows)
                    preselection_group_diagnostics.extend(group_rows)
                    preselection_replacement_diagnostics.extend(
                        replacement_rows
                    )
                if config is not None:
                    diagnostics.append(
                        {
                            "record_index": record_index if record_index is not None else "",
                            "timestamp": timestamp.isoformat(sep=" ") if timestamp else "",
                            "horizon_step": _step_index + 1,
                            "previous_active_cell_count": previous_active_count,
                            "previous_winner_count": previous_winner_count,
                            "predicted_source_count": predicted_source_count,
                            "burst_source_count": burst_source_count,
                            "active_source_count": model.last_prediction_stats.get("active_source_count", 0),
                            "candidate_segment_count": model.last_prediction_stats.get("candidate_segment_count", 0),
                            "threshold_crossing_segment_count": model.last_prediction_stats.get("threshold_crossing_segment_count", 0),
                            "accepted_candidate_count": 0,
                            "raw_predicted_neuron_count": 0,
                            "raw_predicted_column_count": 0,
                            "weekday_predicted_column_count": 0,
                            "time_predicted_column_count": 0,
                            "passenger_predicted_column_count": 0,
                            "raw_event_count": 0,
                            "competition_mode": competition.mode,
                            **(
                                {
                                    "diagnostic_only": True,
                                    "competition_is_local_choice": True,
                                    "simultaneous_policy": competition.simultaneous_policy,
                                    "simultaneous_bin_width": competition.simultaneous_bin_width,
                                    "batch_index": "",
                                    "batch_start_time": "",
                                    "batch_end_time": "",
                                    "batch_candidate_count": "",
                                    "batch_emitted_count": "",
                                    "earlier_batch_inhibition": 0.0,
                                    "same_batch_inhibition": 0.0,
                                }
                                if competition.enabled
                                else {}
                            ),
                            "emitted_neuron_count": 0,
                            "emitted_column_count": 0,
                            "inhibited_candidate_count": 0,
                            "mean_original_score": 0.0,
                            "mean_effective_score": 0.0,
                            "mean_accumulated_inhibition": 0.0,
                            "maximum_inhibition": 0.0,
                            "first_emitted_time": "",
                            "last_emitted_time": "",
                            "weekday_emitted_column_count": 0,
                            "time_emitted_column_count": 0,
                            "passenger_emitted_column_count": 0,
                            "competition_runtime_seconds": 0.0,
                            "passenger_candidate_count": 0,
                            "decoded_passenger": "",
                            "actual_future_passenger": (
                                future_records[_step_index].value
                                if future_records is not None
                                and _step_index < len(future_records)
                                else ""
                            ),
                            "target": (
                                future_records[_step_index].value
                                if future_records is not None
                                and _step_index < len(future_records)
                                else ""
                            ),
                            "absolute_error": "",
                            "absolute_percentage_error": "",
                            "prediction_runtime_seconds": predict_runtime,
                            "decode_runtime_seconds": 0.0,
                            "prediction_missing": True,
                            "stopped_reason": "no_raw_prediction",
                        }
                    )
                return RolloutResult(
                    None,
                    tuple(raw_events),
                    tuple(raw_columns),
                    tuple(diagnostics),
                    tuple(emitted_events),
                    tuple(emitted_columns),
                    tuple(oracle_diagnostics),
                    tuple(candidate_score_diagnostics),
                    tuple(branch_candidate_diagnostics),
                    tuple(branch_segment_diagnostics),
                    tuple(branch_source_diagnostics),
                    tuple(preselection_funnel_diagnostics),
                    tuple(preselection_segment_diagnostics),
                    tuple(preselection_group_diagnostics),
                    tuple(preselection_replacement_diagnostics),
                    tuple(intracolumn_selection_diagnostics),
                )
            # DEBUG WATCH: after predict_code。raw_event_counts/raw_column_counts
            # 是定位“预测列密度膨胀”的最直接指标。
            raw_events.append(len(raw.events))
            raw_columns.append(len({event.column for event in raw.events}))
            propagated = raw
            raw_active = model.prediction_active_cells(raw)
            competition_result: CompetitionResult | None = None
            competition_runtime = 0.0
            if competition.enabled:
                if native_phase_hook is not None:
                    native_phase_hook(_step_index + 1, "competition_enter")
                competition_started = time.perf_counter()
                competition_candidates = candidates_for_prediction(
                    raw,
                    model.last_prediction_candidates,
                    timing_tolerance=model.params.timing_tolerance,
                )
                competition_result = compete_prediction_candidates(
                    competition_candidates,
                    threshold=model.params.dendrite_threshold,
                    inhibition_strength=competition.inhibition_strength,
                    inhibition_tau=competition.inhibition_tau,
                    simultaneous_tolerance=competition.simultaneous_tolerance,
                    simultaneous_policy=competition.simultaneous_policy,
                    simultaneous_bin_width=competition.simultaneous_bin_width,
                )
                propagated = emitted_prediction_code(raw, competition_result)
                competition_runtime = time.perf_counter() - competition_started
            emitted_column_ids = (
                {
                    decision.candidate.column_index
                    for decision in competition_result.decisions
                    if decision.emitted
                }
                if competition_result is not None
                else {event.column for event in raw.events}
            )
            mark_competition_outcomes(
                step_intracolumn_rows,
                emitted_columns=emitted_column_ids,
            )
            intracolumn_selection_diagnostics.extend(
                step_intracolumn_rows
            )
            if propagated is None:
                emitted_events.append(0)
                emitted_columns.append(0)
                active = {}
            else:
                emitted_events.append(len(propagated.events))
                emitted_columns.append(
                    len({event.column for event in propagated.events})
                )
                active = (
                    model.prediction_active_cells(propagated)
                    if competition.enabled
                    else raw_active
                )
            if autonomous_context_provenance_tracker is not None:
                autonomous_context_provenance_tracker.record_step(
                    model=model,
                    raw=raw,
                    propagated=propagated,
                    next_active_cells=active,
                    competition_result=competition_result,
                    actual_record_index=(
                        record_index if record_index is not None else 0
                    ),
                    horizon_step=_step_index + 1,
                    field_for_column=(
                        lambda column: field_for_column(column, oracle_ranges)
                    ),
                    target_columns=provenance_target_columns,
                    target_column_by_field=provenance_target_by_field,
                    input_index=(
                        record_index + 1 if record_index is not None else 0
                    ),
                    input_timestamp=(
                        timestamp.isoformat(sep=" ") if timestamp is not None else ""
                    ),
                    target_timestamp=provenance_target_timestamp,
                    registry=branch_registry,
                )
            if context_trajectory_tracker is not None:
                if native_phase_hook is not None:
                    native_phase_hook(
                        _step_index + 1, "context_trajectory_rollout_enter"
                    )
                assert preselection_trace is not None
                context_trajectory_tracker.record_transition(
                    model=model,
                    trajectory_kind="autonomous_rollout",
                    actual_record_index=(
                        record_index if record_index is not None else 0
                    ),
                    horizon_step=_step_index + 1,
                    preselection_trace=preselection_trace,
                    raw_code=raw,
                    competition_result=competition_result,
                    next_active_cells=active,
                    next_winners=active,
                )
            decode_runtime = 0.0
            decoded_passenger: float | str = ""
            passenger_candidate_count: int | str = ""
            actual_future: float | str = (
                future_records[_step_index].value
                if future_records is not None
                and _step_index < len(future_records)
                else ""
            )
            absolute_error: float | str = ""
            ape: float | str = ""
            if (
                config is not None
                and passenger_encoder is not None
            ):
                if propagated is not None:
                    decode_started = time.perf_counter()
                    try:
                        decoded, candidate_count = passenger_decode_details(
                            passenger_encoder,
                            propagated,
                            model.params.timing_tolerance,
                        )
                    except ValueError:
                        decoded_passenger = ""
                        passenger_candidate_count = 0
                    else:
                        decoded_passenger = decoded
                        passenger_candidate_count = candidate_count
                        if (
                            future_records is not None
                            and _step_index < len(future_records)
                        ):
                            if actual_future:
                                absolute_error = abs(decoded - actual_future)
                                ape = absolute_error / abs(actual_future)
                    decode_runtime = time.perf_counter() - decode_started
                else:
                    passenger_candidate_count = 0
                raw_counts = field_column_counts(raw, config)
                emitted_counts = field_column_counts(propagated, config)
                competition_decisions = (
                    competition_result.decisions
                    if competition_result is not None
                    else ()
                )
                original_scores = [
                    decision.original_score for decision in competition_decisions
                ]
                effective_scores = [
                    decision.effective_score for decision in competition_decisions
                ]
                inhibitions = [
                    decision.accumulated_inhibition
                    for decision in competition_decisions
                ]
                competition_batches = (
                    competition_result.batches
                    if competition_result is not None
                    else ()
                )
                emitted_times = (
                    [
                        item.candidate.time
                        for item in competition_result.emitted_candidates
                    ]
                    if competition_result is not None
                    else [event.time for event in raw.events]
                )
                diagnostics.append(
                    {
                        "record_index": record_index if record_index is not None else "",
                        "timestamp": timestamp.isoformat(sep=" ") if timestamp else "",
                        "horizon_step": _step_index + 1,
                        "previous_active_cell_count": previous_active_count,
                        "previous_winner_count": previous_winner_count,
                        "predicted_source_count": predicted_source_count,
                        "burst_source_count": burst_source_count,
                        "active_source_count": model.last_prediction_stats.get("active_source_count", 0),
                        "candidate_segment_count": model.last_prediction_stats.get("candidate_segment_count", 0),
                        "threshold_crossing_segment_count": model.last_prediction_stats.get("threshold_crossing_segment_count", 0),
                        "accepted_candidate_count": model.last_prediction_stats.get("accepted_candidate_count", 0),
                        "raw_predicted_neuron_count": len(raw_active),
                        "raw_predicted_column_count": len({event.column for event in raw.events}),
                        "weekday_predicted_column_count": raw_counts["weekday"],
                        "time_predicted_column_count": raw_counts["time"],
                        "passenger_predicted_column_count": raw_counts["passenger"],
                        "raw_event_count": len(raw.events),
                        "competition_mode": competition.mode,
                        **(
                            {
                                "diagnostic_only": True,
                                "competition_is_local_choice": True,
                                "simultaneous_policy": competition.simultaneous_policy,
                                "simultaneous_bin_width": competition.simultaneous_bin_width,
                                "batch_index": " ".join(
                                    str(batch.batch_index)
                                    for batch in competition_batches
                                ),
                                "batch_start_time": " ".join(
                                    f"{batch.batch_start_time:.12g}"
                                    for batch in competition_batches
                                ),
                                "batch_end_time": " ".join(
                                    f"{batch.batch_end_time:.12g}"
                                    for batch in competition_batches
                                ),
                                "batch_candidate_count": " ".join(
                                    str(batch.candidate_count)
                                    for batch in competition_batches
                                ),
                                "batch_emitted_count": " ".join(
                                    str(batch.emitted_count)
                                    for batch in competition_batches
                                ),
                                "earlier_batch_inhibition": (
                                    sum(
                                        decision.earlier_batch_inhibition
                                        for decision in competition_decisions
                                    )
                                    / len(competition_decisions)
                                    if competition_decisions
                                    else 0.0
                                ),
                                "same_batch_inhibition": (
                                    sum(
                                        decision.same_batch_inhibition
                                        for decision in competition_decisions
                                    )
                                    / len(competition_decisions)
                                    if competition_decisions
                                    else 0.0
                                ),
                            }
                            if competition.enabled
                            else {}
                        ),
                        "emitted_neuron_count": len(active),
                        "emitted_column_count": (
                            len({event.column for event in propagated.events})
                            if propagated is not None
                            else 0
                        ),
                        "inhibited_candidate_count": (
                            len(competition_result.inhibited_candidates)
                            if competition_result is not None
                            else 0
                        ),
                        "mean_original_score": (
                            sum(original_scores) / len(original_scores)
                            if original_scores
                            else 0.0
                        ),
                        "mean_effective_score": (
                            sum(effective_scores) / len(effective_scores)
                            if effective_scores
                            else 0.0
                        ),
                        "mean_accumulated_inhibition": (
                            sum(inhibitions) / len(inhibitions)
                            if inhibitions
                            else 0.0
                        ),
                        "maximum_inhibition": (
                            max(inhibitions) if inhibitions else 0.0
                        ),
                        "first_emitted_time": (
                            min(emitted_times) if emitted_times else ""
                        ),
                        "last_emitted_time": (
                            max(emitted_times) if emitted_times else ""
                        ),
                        "weekday_emitted_column_count": emitted_counts["weekday"],
                        "time_emitted_column_count": emitted_counts["time"],
                        "passenger_emitted_column_count": emitted_counts["passenger"],
                        "competition_runtime_seconds": competition_runtime,
                        "passenger_candidate_count": passenger_candidate_count,
                        "decoded_passenger": decoded_passenger,
                        "actual_future_passenger": actual_future,
                        "target": actual_future,
                        "absolute_error": absolute_error,
                        "absolute_percentage_error": ape,
                        "prediction_runtime_seconds": predict_runtime,
                        "decode_runtime_seconds": decode_runtime,
                        "prediction_missing": propagated is None,
                        "stopped_reason": "",
                    }
                )
            if (
                oracle_candidate_diagnostic
                and oracle_encoder is not None
                and oracle_ranges is not None
                and future_records is not None
                and _step_index < len(future_records)
            ):
                if native_phase_hook is not None:
                    native_phase_hook(_step_index + 1, "oracle_diagnostic_enter")
                target_record = future_records[_step_index]
                target_code = oracle_encoder.encode(
                    record_values(target_record)
                )
                if temporal_context_tracker is not None:
                    target_columns_by_field: dict[str, int] = {}
                    for target_column in sorted(target_code.columns):
                        target_columns_by_field.setdefault(
                            field_for_column(target_column, oracle_ranges),
                            int(target_column),
                        )
                    temporal_context_tracker.record_autonomous_candidates(
                        model=model,
                        registry=branch_registry,
                        actual_record_index=(
                            record_index if record_index is not None else 0
                        ),
                        horizon_step=_step_index + 1,
                        field_for_column=lambda column: field_for_column(
                            int(column), oracle_ranges
                        ),
                        target_columns=target_code.columns,
                        target_column_by_field=target_columns_by_field,
                        input_index=(
                            record_index + 1 if record_index is not None else 0
                        ),
                        input_timestamp=(
                            timestamp.isoformat(sep=" ")
                            if timestamp is not None
                            else ""
                        ),
                        candidates=model.last_prediction_candidates,
                        emitted_candidate_ids=(
                            id(item.candidate)
                            for item in (
                                competition_result.emitted_candidates
                                if competition_result is not None
                                else ()
                            )
                        ),
                    )
                oracle_diagnostics.append(
                    analyze_oracle_step(
                        prediction_input_index=(
                            record_index + 1
                            if record_index is not None
                            else 0
                        ),
                        input_timestamp=(
                            timestamp.isoformat(sep=" ")
                            if timestamp is not None
                            else ""
                        ),
                        horizon_step=_step_index + 1,
                        target_timestamp=target_record.timestamp.isoformat(
                            sep=" "
                        ),
                        target_passenger=target_record.value,
                        decoded_prediction=decoded_passenger,
                        absolute_error=absolute_error,
                        raw_prediction=raw,
                        emitted_prediction=propagated,
                        competition_result=competition_result,
                        target_code=target_code,
                        ranges=oracle_ranges,
                    )
                )
                if (
                    candidate_separability_trace
                    and competition_result is not None
                ):
                    candidate_score_diagnostics.extend(
                        candidate_score_trace_rows(
                            policy=competition.simultaneous_policy,
                            input_index=(
                                record_index + 1
                                if record_index is not None
                                else 0
                            ),
                            target_timestamp=target_record.timestamp.isoformat(
                                sep=" "
                            ),
                            horizon_step=_step_index + 1,
                            competition_result=competition_result,
                            target_code=target_code,
                            ranges=oracle_ranges,
                        )
                    )
                if (
                    branch_provenance_diagnostic
                    and competition_result is not None
                    and branch_registry is not None
                ):
                    if native_phase_hook is not None:
                        native_phase_hook(
                            _step_index + 1, "branch_provenance_enter"
                        )
                    candidate_rows, segment_rows, source_rows = (
                        branch_trace_rows(
                            model=model,
                            registry=branch_registry,
                            stream_label=stream_label,
                            policy=competition.simultaneous_policy,
                            input_index=(
                                record_index + 1
                                if record_index is not None
                                else 0
                            ),
                            input_timestamp=(
                                timestamp.isoformat(sep=" ")
                                if timestamp is not None
                                else ""
                            ),
                            target_timestamp=(
                                target_record.timestamp.isoformat(sep=" ")
                            ),
                            horizon_step=_step_index + 1,
                            target_code=target_code,
                            ranges=oracle_ranges,
                            competition_result=competition_result,
                            active_sources=branch_active_sources,
                            level=branch_provenance_level,
                            source_top_n=branch_source_top_n,
                            context_index=autonomous_context_index,
                        )
                    )
                    branch_candidate_diagnostics.extend(candidate_rows)
                    branch_segment_diagnostics.extend(segment_rows)
                    branch_source_diagnostics.extend(source_rows)
                if (
                    preselection_segment_diagnostic
                    and preselection_trace is not None
                    and branch_registry is not None
                ):
                    if native_phase_hook is not None:
                        native_phase_hook(_step_index + 1, "preselection_enter")
                    (
                        funnel_row,
                        segment_rows,
                        group_rows,
                        replacement_rows,
                    ) = build_preselection_rows(
                        model=model,
                        registry=branch_registry,
                        trace=preselection_trace,
                        stream_label=stream_label,
                        policy=competition.simultaneous_policy,
                        input_index=(
                            record_index + 1
                            if record_index is not None
                            else 0
                        ),
                        input_timestamp=(
                            timestamp.isoformat(sep=" ")
                            if timestamp is not None
                            else ""
                        ),
                        target_timestamp=target_record.timestamp.isoformat(
                            sep=" "
                        ),
                        horizon_step=_step_index + 1,
                        target_code=target_code,
                        ranges=oracle_ranges,
                        active_sources=branch_active_sources,
                        competition_result=competition_result,
                        raw_code=raw,
                        runtime_seconds=predict_runtime,
                        level=preselection_segment_level,
                    )
                    preselection_funnel_diagnostics.append(funnel_row)
                    preselection_segment_diagnostics.extend(segment_rows)
                    preselection_group_diagnostics.extend(group_rows)
                    preselection_replacement_diagnostics.extend(
                        replacement_rows
                    )
            if not active:
                if diagnostics:
                    diagnostics[-1]["stopped_reason"] = "no_prediction_active_cells"
                return RolloutResult(
                    None,
                    tuple(raw_events),
                    tuple(raw_columns),
                    tuple(diagnostics),
                    tuple(emitted_events),
                    tuple(emitted_columns),
                    tuple(oracle_diagnostics),
                    tuple(candidate_score_diagnostics),
                    tuple(branch_candidate_diagnostics),
                    tuple(branch_segment_diagnostics),
                    tuple(branch_source_diagnostics),
                    tuple(preselection_funnel_diagnostics),
                    tuple(preselection_segment_diagnostics),
                    tuple(preselection_group_diagnostics),
                    tuple(preselection_replacement_diagnostics),
                    tuple(intracolumn_selection_diagnostics),
                )
            # STATE MUTATION: 下面两行只推进临时检索状态；长期记忆中的
            # segment/synapse/weight/age 不会改变，并会在 finally 中恢复。
            if native_phase_hook is not None:
                native_phase_hook(_step_index + 1, "state_propagation_enter")
            model.previous_active_cells = active
            model.previous_winners = active.copy()
            if temporal_context_tracker is not None:
                temporal_context_tracker.record_autonomous_state(
                    active,
                    _step_index + 1,
                )
            if branch_registry is not None:
                branch_registry.set_autonomous_sources(active)
            prediction = propagated
        return RolloutResult(
            prediction,
            tuple(raw_events),
            tuple(raw_columns),
            tuple(diagnostics),
            tuple(emitted_events),
            tuple(emitted_columns),
            tuple(oracle_diagnostics),
            tuple(candidate_score_diagnostics),
            tuple(branch_candidate_diagnostics),
            tuple(branch_segment_diagnostics),
            tuple(branch_source_diagnostics),
            tuple(preselection_funnel_diagnostics),
            tuple(preselection_segment_diagnostics),
            tuple(preselection_group_diagnostics),
            tuple(preselection_replacement_diagnostics),
            tuple(intracolumn_selection_diagnostics),
        )
    finally:
        # DEBUG WATCH: before/after transient restore。比较 restore 前后的
        # previous_active_cells、last_prediction_candidates、RNG state，确认
        # checkpoint/rollout 没污染后续在线学习。
        model.restore_transient_state(snapshot)
        if branch_registry is not None:
            branch_registry.current_predicted_sources = (
                registry_predicted_before
            )
            branch_registry.current_burst_sources = registry_burst_before


def strict_summary_paths(output_dir: Path) -> list[Path]:
    historical = output_dir / "fig9_historical_compensated"
    return [
        path
        for path in (output_dir / "fig9_strict").glob("*summary.json")
        if historical not in path.parents
    ]


def run_strict_stream(
    *,
    records: list[TaxiRecord],
    data_path: Path,
    stream_label: str,
    output_dir: Path,
    config: Fig9StrictConfig,
    limit: int,
    event_hook: Callable[[str, int], None] | None = None,
    print_fingerprint: bool = True,
    density_trace_path: Path | None = None,
    density_summary_path: Path | None = None,
    interval_every: int = 0,
    checkpoint_path: Path | None = None,
    checkpoint_at_index: int | None = None,
    checkpoint_every: int = 0,
    resume_checkpoint: Path | None = None,
    stop_after_index: int | None = None,
    debug_record_index: int | None = None,
    debug_output_json: Path | None = None,
    competition_settings: CompetitionSettings | None = None,
    oracle_candidate_diagnostic: bool = False,
    candidate_separability_trace: bool = False,
    branch_provenance_diagnostic: bool = False,
    branch_provenance_level: str = "candidate",
    branch_source_top_n: int | None = None,
    preselection_segment_diagnostic: bool = False,
    preselection_segment_level: str = "crossing",
    preselection_segment_compress: bool = False,
    preselection_max_rows: int | None = None,
    teacher_forced_winner_diagnostic: bool = False,
    teacher_forced_winner_level: str = "summary",
    observe_scenario_diagnostic: bool = False,
    observe_scenario_level: str = "summary",
    reference_neuron_selection_diagnostic: bool = False,
    reference_neuron_selection_level: str = "summary",
    intracolumn_selection_policy: str = "existing",
    intracolumn_selection_diagnostic: bool = False,
    match_overlap_diagnostic: bool = False,
    match_overlap_level: str = "summary",
    match_overlap_compress: bool = False,
    timing_eligibility_decomposition: bool = False,
    source_trace_filter: SourceTraceFilter | None = None,
    context_trajectory_diagnostic: bool = False,
    context_trajectory_level: str = "summary",
    temporal_context_diagnostic: bool = False,
    temporal_context_level: str = "summary",
    autonomous_context_provenance_diagnostic: bool = False,
    autonomous_context_provenance_level: str = "candidate",
    lmatch_real_ablation: bool = False,
    progress_every: int = 0,
    stream_diagnostic_traces: bool = False,
    debug_end_index: int | None = None,
    context_trajectory_compress: bool = True,
    segment_reinforcement_diagnostic: bool = False,
    segment_reinforcement_level: str = "summary",
    independent_reference_diagnostic: bool = False,
    independent_reference_level: str = "summary",
    independent_reference_compress: bool = False,
    actual_branch_provenance_diagnostic: bool = False,
    actual_branch_provenance_level: str = "summary",
    actual_branch_provenance_compress: bool = False,
    segment_context_composition_diagnostic: bool = False,
    segment_context_composition_level: str = "match",
    segment_context_composition_compress: bool = False,
    context_oracle_unified_trace: bool = False,
) -> dict[str, object]:
    """Run one original or perturbed stream under the strict Fig.9 protocol."""

    competition = competition_settings or CompetitionSettings()
    competition.validate()
    if intracolumn_selection_policy not in INTRACOLUMN_SELECTION_POLICIES:
        raise ValueError("unsupported intracolumn selection policy")
    if intracolumn_selection_diagnostic and not oracle_candidate_diagnostic:
        raise ValueError(
            "intracolumn selection diagnostics require oracle diagnostics"
        )
    if match_overlap_level not in MATCH_OVERLAP_LEVELS:
        raise ValueError(
            "match overlap level must be summary, segment, or source"
        )
    if context_trajectory_level not in CONTEXT_TRAJECTORY_LEVELS:
        raise ValueError(
            "context trajectory level must be summary, column, or cell"
        )
    if temporal_context_level not in {"summary", "candidate"}:
        raise ValueError("temporal context level must be summary or candidate")
    if autonomous_context_provenance_level not in {"summary", "candidate", "source"}:
        raise ValueError(
            "autonomous context provenance level must be summary, candidate, or source"
        )
    if segment_reinforcement_level not in SEGMENT_REINFORCEMENT_LEVELS:
        raise ValueError(
            "segment reinforcement level must be summary, event, or segment"
        )
    if independent_reference_level not in {"summary", "event", "segment"}:
        raise ValueError("independent reference level must be summary, event, or segment")
    if actual_branch_provenance_level not in {"summary", "event", "segment"}:
        raise ValueError("actual branch provenance level must be summary, event, or segment")
    if segment_context_composition_level not in SEGMENT_CONTEXT_COMPOSITION_LEVELS:
        raise ValueError(
            "segment context composition level must be summary, match, or source"
        )
    if context_oracle_unified_trace and not (
        oracle_candidate_diagnostic
        and branch_provenance_diagnostic
        and segment_context_composition_diagnostic
    ):
        raise ValueError(
            "context-oracle unified trace requires oracle, branch provenance, "
            "and segment context composition diagnostics"
        )
    if timing_eligibility_decomposition and (
        not match_overlap_diagnostic or match_overlap_level != "source"
    ):
        raise ValueError(
            "timing eligibility decomposition requires source-level "
            "match-overlap diagnostics"
        )
    if oracle_candidate_diagnostic and not competition.enabled:
        raise ValueError(
            "oracle candidate diagnostics require competitive_raw mode"
        )
    if candidate_separability_trace and not oracle_candidate_diagnostic:
        raise ValueError(
            "candidate separability trace requires oracle candidate diagnostics"
        )
    if branch_provenance_diagnostic and not oracle_candidate_diagnostic:
        raise ValueError(
            "branch provenance diagnostics require oracle candidate diagnostics"
        )
    if branch_provenance_level not in BRANCH_LEVELS:
        raise ValueError(
            "branch provenance level must be summary, candidate, or full"
        )
    if preselection_segment_diagnostic and not oracle_candidate_diagnostic:
        raise ValueError(
            "preselection diagnostics require oracle candidate diagnostics"
        )
    if preselection_segment_level not in PRESELECTION_LEVELS:
        raise ValueError(
            "preselection segment level must be summary, crossing, or full"
        )
    if preselection_max_rows is not None and preselection_max_rows <= 0:
        raise ValueError("preselection max rows must be positive")
    if teacher_forced_winner_level not in TEACHER_FORCED_LEVELS:
        raise ValueError(
            "teacher-forced winner level must be summary, cell, or segment"
        )
    if teacher_forced_winner_diagnostic and not oracle_candidate_diagnostic:
        raise ValueError(
            "teacher-forced winner diagnostics require oracle diagnostics"
        )
    if teacher_forced_winner_diagnostic and not branch_provenance_diagnostic:
        raise ValueError(
            "teacher-forced winner diagnostics require branch provenance"
        )
    if teacher_forced_winner_diagnostic and not preselection_segment_diagnostic:
        raise ValueError(
            "teacher-forced winner diagnostics require preselection diagnostics"
        )
    if observe_scenario_diagnostic and not teacher_forced_winner_diagnostic:
        raise ValueError(
            "observe-scenario diagnostics require teacher-forced winners"
        )
    if (
        reference_neuron_selection_diagnostic
        and not teacher_forced_winner_diagnostic
    ):
        raise ValueError(
            "reference-neuron selection diagnostics require "
            "teacher-forced winners"
        )
    if (
        reference_neuron_selection_diagnostic
        and not preselection_segment_diagnostic
    ):
        raise ValueError(
            "reference-neuron selection diagnostics require preselection trace"
        )
    if observe_scenario_level not in OBSERVE_SCENARIO_LEVELS:
        raise ValueError(
            "observe scenario level must be summary, column, or full"
        )
    if (
        reference_neuron_selection_level
        not in REFERENCE_NEURON_SELECTION_LEVELS
    ):
        raise ValueError(
            "reference-neuron selection level must be summary, crossing, or full"
        )
    if len(records) <= config.horizon:
        raise ValueError("Not enough records for the requested horizon.")
    resume_provenance_audit: dict[str, object] | None = None
    if resume_checkpoint is not None:
        checkpoint = load_strict_checkpoint(resume_checkpoint)
        validate_checkpoint_l_match(checkpoint, config)
        encoder = checkpoint["encoder"]  # type: ignore[assignment]
        model = checkpoint["model"]  # type: ignore[assignment]
        fingerprint = checkpoint["fingerprint"]  # type: ignore[assignment]
        start_index = int(checkpoint["next_index"])
        predictions = list(checkpoint["predictions"])  # type: ignore[arg-type]
        targets = list(checkpoint["targets"])  # type: ignore[arg-type]
        rows = list(checkpoint["rows"])  # type: ignore[arg-type]
        recent_errors = deque(
            checkpoint["recent_errors"],  # type: ignore[arg-type]
            maxlen=config.rolling_window,
        )
        missing = int(checkpoint["missing"])
        raw_event_counts = list(checkpoint["raw_event_counts"])  # type: ignore[arg-type]
        raw_column_counts = list(checkpoint["raw_column_counts"])  # type: ignore[arg-type]
        density_rows = list(checkpoint["density_rows"])  # type: ignore[arg-type]
        checkpoint_diagnostic_state = checkpoint.get("diagnostic_state") or {}
        embedded_provenance = checkpoint.get("branch_provenance")
        sidecar = branch_checkpoint_sidecar_path(resume_checkpoint)
        provenance_payload = (
            embedded_provenance
            if isinstance(embedded_provenance, dict)
            else (
                json.loads(sidecar.read_text(encoding="utf-8"))
                if sidecar.exists()
                else None
            )
        )
        branch_registry = (
            BranchProvenanceRegistry.from_checkpoint_payload(
                model, provenance_payload
            )
            if (
                branch_provenance_diagnostic
                or preselection_segment_diagnostic
                or teacher_forced_winner_diagnostic
                or match_overlap_diagnostic
                or context_trajectory_diagnostic
                or temporal_context_diagnostic
                or autonomous_context_provenance_diagnostic
                or segment_reinforcement_diagnostic
                or independent_reference_diagnostic
                or actual_branch_provenance_diagnostic
                or segment_context_composition_diagnostic
            )
            else None
        )
        if branch_registry is not None:
            binding = branch_registry.binding_summary(model)
            if provenance_payload is None:
                raise ValueError(
                    "checkpoint resume requires branch provenance payload"
                )
            expected = len(provenance_payload.get("segments", ()))
            if binding["bound_segment_count"] != expected:
                raise ValueError(
                    "checkpoint branch provenance rebind mismatch: "
                    f"expected={expected}, bound={binding['bound_segment_count']}"
                )
            resume_provenance_audit = {
                **binding,
                "expected_provenance_segments": expected,
                "checkpoint_next_index": start_index,
                "source": (
                    "embedded_checkpoint"
                    if isinstance(embedded_provenance, dict)
                    else "sidecar"
                ),
            }
        checkpoint_data_hash = checkpoint.get("data_file_sha256")
        current_data_hash = file_sha256(data_path)
        if checkpoint_data_hash != current_data_hash:
            prefix_end = int(checkpoint.get("prefix_end_index", start_index))
            checkpoint_prefix_hash = checkpoint.get("common_prefix_hash")
            current_prefix_hash = records_prefix_sha256(records, prefix_end)
            if checkpoint_prefix_hash != current_prefix_hash:
                raise ValueError(
                    "checkpoint data is incompatible with this stream before resume index"
                )
    else:
        encoder = build_fig9_encoder(config)
        model = build_strict_model(encoder, config)
        fingerprint = protocol_fingerprint(
            data_path=data_path,
            records=records,
            stream_label=stream_label,
            config=config,
            limit=limit,
        )
        start_index = 0
        predictions = []
        targets = []
        rows = []
        recent_errors = deque(maxlen=config.rolling_window)
        missing = 0
        raw_event_counts = []
        raw_column_counts = []
        density_rows = []
        checkpoint_diagnostic_state = {}
        branch_registry = (
            BranchProvenanceRegistry()
            if (
                branch_provenance_diagnostic
                or preselection_segment_diagnostic
                or teacher_forced_winner_diagnostic
                or match_overlap_diagnostic
                or context_trajectory_diagnostic
                or temporal_context_diagnostic
                or autonomous_context_provenance_diagnostic
                or segment_reinforcement_diagnostic
                or independent_reference_diagnostic
                or actual_branch_provenance_diagnostic
                or segment_context_composition_diagnostic
            )
            else None
        )
    passenger_encoder = encoder.encoders[2]
    if debug_end_index is not None:
        if debug_end_index <= start_index:
            raise ValueError("debug end index must be after checkpoint next_index")
        if debug_end_index - start_index > 10:
            raise ValueError("debug isolation runs are limited to 10 records")
        stop_after_index = debug_end_index
    oracle_encoder = (
        build_fig9_encoder(config)
        if (oracle_candidate_diagnostic or autonomous_context_provenance_diagnostic)
        else None
    )
    oracle_rows: list[dict[str, object]] = []
    candidate_score_rows: list[dict[str, object]] = []
    branch_candidate_rows: list[dict[str, object]] = []
    branch_segment_rows: list[dict[str, object]] = []
    branch_source_rows: list[dict[str, object]] = []
    preselection_funnel_rows: list[dict[str, object]] = []
    preselection_segment_rows: list[dict[str, object]] = []
    preselection_group_rows: list[dict[str, object]] = []
    preselection_replacement_rows: list[dict[str, object]] = []
    intracolumn_selection_rows: list[dict[str, object]] = []
    teacher_forced_observation_rows: list[dict[str, object]] = list(
        checkpoint_diagnostic_state.get(
            "teacher_forced_observation_rows",
            [],
        )
    )
    match_overlap_column_rows: list[dict[str, object]] = []
    match_overlap_segment_rows: list[dict[str, object]] = []
    match_overlap_source_rows: list[dict[str, object]] = []
    context_trajectory_rows: list[dict[str, object]] = []
    segment_reinforcement_rows: list[dict[str, object]] = list(
        checkpoint_diagnostic_state.get("segment_reinforcement_rows", [])
    )
    independent_reference_event_rows: list[dict[str, object]] = list(
        checkpoint_diagnostic_state.get("independent_reference_event_rows", [])
    )
    independent_reference_segment_rows: list[dict[str, object]] = list(
        checkpoint_diagnostic_state.get("independent_reference_segment_rows", [])
    )
    independent_reference_tracker = IndependentReferenceTracker.from_checkpoint_payload(
        checkpoint_diagnostic_state.get("independent_reference_tracker")
    )
    actual_branch_tracker = ActualBranchProvenanceTracker.from_checkpoint_payload(
        checkpoint_diagnostic_state.get("actual_branch_tracker")
    )
    actual_branch_event_rows: list[dict[str, object]] = list(
        checkpoint_diagnostic_state.get("actual_branch_event_rows", [])
    )
    actual_branch_segment_rows: list[dict[str, object]] = list(
        checkpoint_diagnostic_state.get("actual_branch_segment_rows", [])
    )
    unified_actual_candidate_rows: list[dict[str, object]] = []
    context_trajectory_row_count = 0
    context_trajectory_path = output_dir / "context_trajectory_column_trace.csv"
    streamed_trace_counts = {
        "branch_segment": 0,
        "preselection_segment": 0,
        "preselection_group": 0,
        "preselection_replacement": 0,
        "match_overlap_segment": 0,
    }

    def flush_large_diagnostic_batches() -> None:
        if not stream_diagnostic_traces:
            return
        batches = (
            (
                output_dir / "branch_segment_trace.csv",
                branch_segment_rows,
                False,
            ),
            (
                output_dir / "preselection_segment_trace.csv",
                preselection_segment_rows,
                preselection_segment_compress,
            ),
            (
                output_dir / "preselection_group_trace.csv",
                preselection_group_rows,
                preselection_segment_compress,
            ),
            (
                output_dir / "preselection_replacement_trace.csv",
                preselection_replacement_rows,
                preselection_segment_compress,
            ),
            (
                output_dir / "match_overlap_segment_trace.csv",
                match_overlap_segment_rows,
                match_overlap_compress,
            ),
        )
        for path, batch, compress in batches:
            append_diagnostic_csv(path, batch, compress=compress)
            batch.clear()
    context_trajectory_tracker = (
        ContextTrajectoryTracker()
        if context_trajectory_diagnostic
        else None
    )
    temporal_context_tracker = (
        TemporalContextTracker.from_checkpoint_payload(
            checkpoint_diagnostic_state.get("temporal_context_tracker")
        )
        if temporal_context_diagnostic and checkpoint_diagnostic_state
        else TemporalContextTracker()
        if temporal_context_diagnostic
        else None
    )
    autonomous_context_provenance_tracker = (
        AutonomousContextProvenanceTracker.from_checkpoint_payload(
            checkpoint_diagnostic_state.get(
                "autonomous_context_provenance_tracker"
            ),
            stream_label=stream_label,
            level=autonomous_context_provenance_level,
            seed=config.seed,
        )
        if autonomous_context_provenance_diagnostic
        else None
    )
    if autonomous_context_provenance_tracker is not None:
        autonomous_context_provenance_tracker.enable_streaming(output_dir)
    if temporal_context_tracker is not None:
        temporal_context_tracker.enable_streaming(output_dir, compress=True)
        for pending_row in temporal_context_tracker.pending_rows():
            temporal_context_tracker._emit(pending_row)
    segment_context_composition_tracker = (
        SegmentContextCompositionTracker(
            registry=branch_registry,
            level=segment_context_composition_level,
        )
        if segment_context_composition_diagnostic and branch_registry is not None
        else None
    )
    if segment_context_composition_tracker is not None:
        segment_context_composition_tracker.enable_streaming(
            output_dir,
            compress=segment_context_composition_compress,
        )
    teacher_forced_ranges = (
        FieldColumnRanges.from_sizes(
            config.weekday_columns,
            config.time_columns,
            config.passenger_columns,
        )
        if teacher_forced_winner_diagnostic
        or segment_reinforcement_diagnostic
        or independent_reference_diagnostic
        or actual_branch_provenance_diagnostic
        or segment_context_composition_diagnostic
        or temporal_context_diagnostic
        else None
    )
    match_overlap_ranges = (
        FieldColumnRanges.from_sizes(
            config.weekday_columns,
            config.time_columns,
            config.passenger_columns,
        )
        if match_overlap_diagnostic
        or context_trajectory_diagnostic
        or segment_reinforcement_diagnostic
        or independent_reference_diagnostic
        or actual_branch_provenance_diagnostic
        or segment_context_composition_diagnostic
        or temporal_context_diagnostic
        else None
    )
    validate_strict_fingerprint(fingerprint)
    if lmatch_real_ablation:
        fingerprint.update(lmatch_ablation_markers(config.l_match))
    if print_fingerprint:
        print(json.dumps(fingerprint, indent=2, sort_keys=True))

    start_time = time.perf_counter()
    progress_started = start_time
    progress_path = output_dir / "progress.jsonl"
    target_scale = sum(abs(record.value) for record in records) / len(records)
    interval_rows: list[dict[str, object]] = []
    interval_start_time = start_time
    interval_start_index = start_index
    debug_payload: dict[str, object] | None = None
    last_completed_index = start_index
    prune_breadcrumb_recorder = (
        PruneBreadcrumbRecorder(config)
        if actual_branch_provenance_diagnostic
        else None
    )
    # This hook is diagnostic-only.  It receives no Segment/Neuron reference
    # and is removed by SequentialMemory.__getstate__ during checkpointing.
    model.prune_diagnostic_callback = prune_breadcrumb_recorder

    for index in range(start_index, len(records) - config.horizon):
        if prune_breadcrumb_recorder is not None:
            prune_breadcrumb_recorder.set_index(index)
        write_native_crash_breadcrumb(
            current_index=index,
            rollout_step=None,
            phase="observation_enter",
            model=model,
        )
        record = records[index]
        code = encoder.encode(record_values(record))
        temporal_actual_capture = (
            temporal_context_diagnostic and index >= config.warmup
        )
        rollout_diagnostics: tuple[dict[str, object], ...] = ()
        if index >= config.warmup:
            target_record = records[index + config.horizon]
            debug_enabled = debug_record_index is not None and index == debug_record_index
            before_rollout_fingerprint = (
                transient_fingerprint(model) if debug_enabled else ""
            )
            observe_segments_before = segment_count(model) if debug_enabled else 0
            observe_synapses_before = synapse_count(model) if debug_enabled else 0
            # STRICT PROTOCOL: 先预测，再 observe 当前真实记录。这样第 index
            # 条记录的真实 passenger_count 不会泄漏到它自己的 horizon 预测里。
            # DEBUG WATCH: before each record prediction。重点看 index、
            # input_timestamp、target_timestamp、previous_active_cells。
            event_hook and event_hook("predict", index)
            rollout = rollout_raw_autonomous(
                model,
                config.horizon,
                competition_settings=competition,
                oracle_candidate_diagnostic=oracle_candidate_diagnostic,
                candidate_separability_trace=candidate_separability_trace,
                branch_provenance_diagnostic=branch_provenance_diagnostic,
                branch_provenance_level=branch_provenance_level,
                branch_source_top_n=branch_source_top_n,
                branch_registry=branch_registry,
                preselection_segment_diagnostic=(
                    preselection_segment_diagnostic
                ),
                preselection_segment_level=preselection_segment_level,
                oracle_encoder=oracle_encoder,
                config=(
                    config
                    if (
                        density_trace_path
                        or density_summary_path
                        or debug_enabled
                        or competition.enabled
                    )
                    else None
                ),
                passenger_encoder=(
                    passenger_encoder
                    if (
                        density_trace_path
                        or density_summary_path
                        or debug_enabled
                        or competition.enabled
                    )
                    else None
                ),  # type: ignore[arg-type]
                stream_label=stream_label,
                record_index=index,
                timestamp=record.timestamp,
                future_records=records[index + 1 : index + config.horizon + 1],
                intracolumn_selection_policy=intracolumn_selection_policy,
                intracolumn_selection_diagnostic=(
                    intracolumn_selection_diagnostic
                ),
                context_trajectory_tracker=context_trajectory_tracker,
                context_oracle_unified_trace=context_oracle_unified_trace,
                temporal_context_tracker=temporal_context_tracker,
                autonomous_context_provenance_tracker=(
                    autonomous_context_provenance_tracker
                ),
                native_phase_hook=lambda step, phase: write_native_crash_breadcrumb(
                    current_index=index,
                    rollout_step=step,
                    phase=phase,
                    model=model,
                ),
            )
            write_native_crash_breadcrumb(
                current_index=index,
                rollout_step=config.horizon,
                phase="rollout_complete",
                model=model,
            )
            if debug_enabled:
                debug_payload = {
                    "diagnostic_label": "strict Fig.9 manual debug record",
                    "record_index": index,
                    "input_record": {
                        "timestamp": record.timestamp.isoformat(sep=" "),
                        "weekday": record_values(record)[0],
                        "half_hour_slot": record_values(record)[1],
                        "passenger_count": record.value,
                    },
                    "target_record": {
                        "timestamp": target_record.timestamp.isoformat(sep=" "),
                        "passenger_count": target_record.value,
                    },
                    "prediction_before_observe_state": {
                        "previous_active_cell_count": len(model.previous_active_cells),
                        "previous_winner_count": len(model.previous_winners),
                        "last_prediction_candidate_columns": len(model.last_prediction_candidates),
                        "transient_fingerprint_before_rollout": before_rollout_fingerprint,
                    },
                    "rollout_steps": list(rollout.diagnostics),
                    "transient_fingerprint_after_rollout_restore": transient_fingerprint(model),
                    "observe_before": {
                        "segment_count": observe_segments_before,
                        "synapse_count": observe_synapses_before,
                    },
                }
            raw_event_counts.extend(rollout.raw_event_counts)
            raw_column_counts.extend(rollout.raw_column_counts)
            rollout_diagnostics = rollout.diagnostics
            if oracle_candidate_diagnostic:
                oracle_rows.extend(rollout.oracle_diagnostics)
            if candidate_separability_trace:
                candidate_score_rows.extend(
                    rollout.candidate_score_diagnostics
                )
            if branch_provenance_diagnostic:
                branch_candidate_rows.extend(
                    rollout.branch_candidate_diagnostics
                )
                branch_segment_rows.extend(
                    rollout.branch_segment_diagnostics
                )
                streamed_trace_counts["branch_segment"] += len(
                    rollout.branch_segment_diagnostics
                )
                branch_source_rows.extend(
                    rollout.branch_source_diagnostics
                )
            if preselection_segment_diagnostic:
                preselection_funnel_rows.extend(
                    rollout.preselection_funnel_diagnostics
                )
                preselection_segment_rows.extend(
                    rollout.preselection_segment_diagnostics
                )
                streamed_trace_counts["preselection_segment"] += len(
                    rollout.preselection_segment_diagnostics
                )
                preselection_group_rows.extend(
                    rollout.preselection_group_diagnostics
                )
                streamed_trace_counts["preselection_group"] += len(
                    rollout.preselection_group_diagnostics
                )
                preselection_replacement_rows.extend(
                    rollout.preselection_replacement_diagnostics
                )
                streamed_trace_counts["preselection_replacement"] += len(
                    rollout.preselection_replacement_diagnostics
                )
            if intracolumn_selection_diagnostic:
                intracolumn_selection_rows.extend(
                    rollout.intracolumn_selection_diagnostics
                )
            if density_trace_path or density_summary_path or competition.enabled:
                density_rows.extend(rollout.diagnostics)
            prediction_value: float | str = ""
            absolute_error_value: float | str = ""
            normalized_error_value: float | str = ""
            rolling_value: float | str = ""
            if rollout.code is None:
                missing += 1
            else:
                try:
                    # DEBUG WATCH: before/after passenger decode。decode_likelihood
                    # 只把 raw SSTD 活动转成数值，不能反向污染 model 状态。
                    decode_started = time.perf_counter()
                    prediction = passenger_encoder.decode_likelihood(rollout.code)  # type: ignore[attr-defined]
                    decode_runtime = time.perf_counter() - decode_started
                except ValueError:
                    missing += 1
                else:
                    predictions.append(prediction)
                    targets.append(target_record.value)
                    prediction_value = prediction
                    error = abs(prediction - target_record.value)
                    recent_errors.append(error)
                    absolute_error_value = error
                    normalized_error_value = error / target_scale if target_scale else ""
                    rolling_value = reference_rolling_mape(recent_errors, target_scale)
                    if density_rows:
                        density_rows[-1]["decoded_passenger"] = prediction
                        density_rows[-1]["actual_future_passenger"] = target_record.value
                        density_rows[-1]["absolute_percentage_error"] = (
                            error / abs(target_record.value)
                            if target_record.value
                            else ""
                        )
                        density_rows[-1]["decode_runtime_seconds"] = decode_runtime
            prediction_row: dict[str, object] = {
                "input_index": index + 1,
                "input_timestamp": record.timestamp.isoformat(sep=" "),
                "target_timestamp": target_record.timestamp.isoformat(sep=" "),
                "target": target_record.value,
                "prediction": prediction_value,
                "absolute_error": absolute_error_value,
                "normalized_absolute_error": normalized_error_value,
                "rolling_mape": rolling_value,
                "raw_rollout_event_counts": " ".join(
                    map(str, rollout.raw_event_counts)
                ),
                "raw_rollout_column_counts": " ".join(
                    map(str, rollout.raw_column_counts)
                ),
            }
            rows.append(prediction_row)
        event_hook and event_hook("observe", index)
        write_native_crash_breadcrumb(
            current_index=index,
            rollout_step=None,
            phase="actual_observation_predict_enter",
            model=model,
        )
        # STATE MUTATION: 这里才把真实当前 record 写入长期记忆，触发三种
        # learning scenario、weight/age/segment 变化。预测阶段不能学习。
        observe_started = time.perf_counter()
        previous_segment_ids = (
            branch_registry.segment_object_ids(model)
            if branch_registry is not None
            and not (
                teacher_forced_winner_diagnostic
                or match_overlap_diagnostic
                or context_trajectory_diagnostic
                or independent_reference_diagnostic
                or actual_branch_provenance_diagnostic
            )
            else set()
        )
        independent_preexisting_ids = set(previous_segment_ids)
        pre_observe_segment_count = (
            segment_count(model)
            if teacher_forced_winner_diagnostic
            else 0
        )
        creation_sources = (
            # _grow_segment and Scenario-2 growth both use previous_winners.
            # Provenance must record that exact creation context, not the
            # broader all-cell context used for segment matching.
            model.previous_winners.copy()
            if branch_registry is not None
            else {}
        )
        teacher_forced_trace = (
            ObservationTrace(
                capture_scenario_details=(
                    observe_scenario_diagnostic
                    or match_overlap_diagnostic
                    or context_trajectory_diagnostic
                    or temporal_actual_capture
                    or segment_reinforcement_diagnostic
                    or independent_reference_diagnostic
                    or actual_branch_provenance_diagnostic
                    or segment_context_composition_diagnostic
                ),
            )
            if (
                teacher_forced_winner_diagnostic
                or match_overlap_diagnostic
                or context_trajectory_diagnostic
                or temporal_actual_capture
                or segment_reinforcement_diagnostic
                or independent_reference_diagnostic
                or actual_branch_provenance_diagnostic
                or segment_context_composition_diagnostic
            )
            else None
        )
        reinforcement_traces = []
        if segment_reinforcement_diagnostic:
            model.reinforcement_trace_callback = reinforcement_traces.append
        match_overlap_capture = None
        actual_prediction_trace = (
            PreselectionTrace()
            if context_trajectory_diagnostic
            else None
        )
        actual_raw: SymbolCode | None = None
        independent_event_batch: list[dict[str, object]] = []
        independent_segment_batch: list[dict[str, object]] = []
        independent_private: dict[int, list[dict[str, object]]] = {}
        actual_branch_event_batch: list[dict[str, object]] = []
        actual_branch_segment_batch: list[dict[str, object]] = []
        actual_branch_preexisting_ids: set[int] = set()
        if (
            match_overlap_diagnostic
            or context_trajectory_diagnostic
            or independent_reference_diagnostic
            or actual_branch_provenance_diagnostic
            or segment_context_composition_diagnostic
            or temporal_actual_capture
        ):
            if (
                branch_registry is None
                or teacher_forced_trace is None
            ):
                raise RuntimeError("match-overlap diagnostic state unavailable")
            # Preserve learn_actual_code's strict predict-then-observe order.
            actual_raw = model.predict_code(
                preselection_trace=actual_prediction_trace,
            )
            if segment_context_composition_tracker is not None:
                if match_overlap_ranges is None:
                    raise RuntimeError("segment context composition ranges unavailable")
                segment_context_composition_tracker.capture_pre_observation(
                    model=model,
                    ranges=match_overlap_ranges,
                    actual_record_index=index,
                )
            if (
                match_overlap_diagnostic
                or context_trajectory_diagnostic
                or temporal_actual_capture
                or segment_context_composition_diagnostic
            ):
                if match_overlap_ranges is None:
                    raise RuntimeError("match-overlap ranges unavailable")
                match_overlap_capture = capture_match_overlap(
                    model=model,
                    code=code,
                    registry=branch_registry,
                    stream_label=stream_label,
                    actual_record_index=index,
                    actual_record=record,
                    ranges=match_overlap_ranges,
                    level=(
                        "source"
                        if context_trajectory_diagnostic
                        or temporal_actual_capture
                        or segment_context_composition_diagnostic
                        else match_overlap_level
                    ),
                    timing_eligibility_decomposition=(
                        timing_eligibility_decomposition
                    ),
                )
            if independent_reference_diagnostic:
                if teacher_forced_ranges is None:
                    raise RuntimeError("independent reference ranges unavailable")
                (
                    independent_event_batch,
                    independent_segment_batch,
                    independent_private,
                ) = capture_before_matching(
                    model=model,
                    registry=branch_registry,
                    tracker=independent_reference_tracker,
                    code=code,
                    actual_record_index=index,
                    ranges=teacher_forced_ranges,
                )
            if actual_branch_provenance_diagnostic:
                if teacher_forced_ranges is None:
                    raise RuntimeError("actual branch provenance ranges unavailable")
                (
                    actual_branch_event_batch,
                    actual_branch_segment_batch,
                    actual_branch_preexisting_ids,
                ) = capture_actual_branch_prematch(
                    model=model,
                    registry=branch_registry,
                    tracker=actual_branch_tracker,
                    code=code,
                    actual_record_index=index,
                    timestamp=record.timestamp.isoformat(sep=" "),
                    ranges=teacher_forced_ranges,
                    l_match=config.l_match,
                )
            model.observe_code(
                code,
                learn=True,
                observation_trace=teacher_forced_trace,
            )
        elif teacher_forced_trace is None:
            learn_actual_code(model, code)
        else:
            learn_actual_code(
                model,
                code,
                observation_trace=teacher_forced_trace,
            )
        if segment_reinforcement_diagnostic:
            model.reinforcement_trace_callback = None
        if branch_registry is not None:
            reference_predicted_sources: set[int] | None = None
            reference_burst_sources: set[int] | None = None
            if teacher_forced_winner_diagnostic:
                reference_predicted_sources = (
                    branch_registry.current_predicted_sources.copy()
                )
                reference_burst_sources = (
                    branch_registry.current_burst_sources.copy()
                )
            if teacher_forced_trace is not None:
                branch_registry.capture_created_segments(
                    model,
                    created=(
                        (
                            event.target_column,
                            event.winner_neuron,
                            event.created_segment,
                        )
                        for event in teacher_forced_trace.events
                        if event.created_segment is not None
                    ),
                    creation_transition_index=index,
                    creation_sources=creation_sources,
                )
            else:
                branch_registry.capture_new_segments(
                    model,
                    previous_segment_ids=previous_segment_ids,
                    creation_transition_index=index,
                    creation_sources=creation_sources,
                )
            if (
                match_overlap_capture is not None
                and teacher_forced_trace is not None
            ):
                completed_overlap = finalize_match_overlap(
                    match_overlap_capture,
                    teacher_forced_trace,
                    branch_registry,
                )
                if segment_context_composition_tracker is not None:
                    composition_transition_rows = segment_context_composition_tracker.consume_transition(
                        capture=match_overlap_capture,
                        completed=completed_overlap,
                        actual_record_index=index,
                        stream_label=stream_label,
                    )
                    if context_oracle_unified_trace:
                        unified_actual_candidate_rows.extend(
                            project_actual_composition_rows(
                                composition_transition_rows
                            )
                        )
                source_rows = completed_overlap.source_rows
                if temporal_context_tracker is not None:
                    temporal_context_tracker.record_actual_source_rows(
                        source_rows=completed_overlap.source_rows,
                        actual_record_index=index,
                        target_field_by_column=lambda column: field_for_column(
                            int(column), match_overlap_ranges
                        ),
                        l_match=config.l_match,
                    )
                if source_trace_filter is not None:
                    source_rows = filter_source_trace_rows(
                        source_rows,
                        source_trace_filter,
                    )
                if context_trajectory_tracker is not None:
                    dependency_rows = context_trajectory_tracker.dependency_rows(
                        source_rows=source_rows,
                        ranges=match_overlap_ranges,
                        level=context_trajectory_level,
                    )
                    context_trajectory_row_count += len(dependency_rows)
                    if stream_diagnostic_traces:
                        append_diagnostic_csv(
                            context_trajectory_path,
                            dependency_rows,
                            compress=context_trajectory_compress,
                        )
                    else:
                        context_trajectory_rows.extend(dependency_rows)
                if match_overlap_diagnostic:
                    match_overlap_column_rows.extend(
                        completed_overlap.column_rows
                    )
                    if match_overlap_level in {"segment", "source"}:
                        match_overlap_segment_rows.extend(
                            completed_overlap.segment_rows
                        )
                        streamed_trace_counts["match_overlap_segment"] += len(
                            completed_overlap.segment_rows
                        )
                    if match_overlap_level == "source":
                        match_overlap_source_rows.extend(source_rows)
            if context_trajectory_tracker is not None:
                assert actual_prediction_trace is not None
                context_trajectory_tracker.record_transition(
                    model=model,
                    trajectory_kind="actual_observation",
                    actual_record_index=index,
                    horizon_step=0,
                    preselection_trace=actual_prediction_trace,
                    raw_code=actual_raw,
                    competition_result=None,
                    next_active_cells=model.previous_active_cells,
                    next_winners=model.previous_winners,
                )
            if (
                (
                    teacher_forced_winner_diagnostic
                    or segment_reinforcement_diagnostic
                )
                and teacher_forced_trace is not None
                and teacher_forced_ranges is not None
            ):
                teacher_forced_observation_rows.extend(
                    build_teacher_forced_observation_rows(
                        model=model,
                        registry=branch_registry,
                        trace=teacher_forced_trace,
                        stream_label=stream_label,
                        actual_record_index=index,
                        actual_record=record,
                        ranges=teacher_forced_ranges,
                        pre_observe_segment_count=(
                            pre_observe_segment_count
                        ),
                        level=(
                            "segment"
                            if segment_reinforcement_diagnostic
                            else teacher_forced_winner_level
                        ),
                        predicted_sources=reference_predicted_sources,
                        burst_sources=reference_burst_sources,
                    )
                )
            if segment_reinforcement_diagnostic:
                if (
                    teacher_forced_trace is None
                    or teacher_forced_ranges is None
                ):
                    raise RuntimeError(
                        "segment reinforcement diagnostic state unavailable"
                    )
                segment_reinforcement_rows.extend(
                    build_segment_reinforcement_rows(
                        model=model,
                        registry=branch_registry,
                        observation_trace=teacher_forced_trace,
                        reinforcement_traces=reinforcement_traces,
                        actual_record_index=index,
                        actual_record=record,
                        ranges=teacher_forced_ranges,
                        level=segment_reinforcement_level,
                    )
                )
            if independent_reference_diagnostic:
                if teacher_forced_trace is None:
                    raise RuntimeError("independent reference trace unavailable")
                join_after_observation(
                    independent_event_batch,
                    independent_segment_batch,
                    trace=teacher_forced_trace,
                    registry=branch_registry,
                    private=independent_private,
                    actual_record_index=index,
                )
                independent_reference_event_rows.extend(independent_event_batch)
                independent_reference_segment_rows.extend(independent_segment_batch)
                record_actual_history(
                    independent_reference_tracker,
                    trace=teacher_forced_trace,
                    registry=branch_registry,
                    preexisting_ids=independent_preexisting_ids,
                    actual_record_index=index,
                )
            if actual_branch_provenance_diagnostic:
                if teacher_forced_trace is None:
                    raise RuntimeError("actual branch provenance trace unavailable")
                join_actual_branch_after_observation(
                    event_rows=actual_branch_event_batch,
                    segment_rows=actual_branch_segment_batch,
                    trace=teacher_forced_trace,
                    registry=branch_registry,
                    tracker=actual_branch_tracker,
                    preexisting_ids=actual_branch_preexisting_ids,
                    actual_record_index=index,
                    source_cells=creation_sources,
                )
                actual_branch_event_rows.extend(actual_branch_event_batch)
                actual_branch_segment_rows.extend(actual_branch_segment_batch)
                record_actual_transition(
                    actual_branch_tracker,
                    trace=teacher_forced_trace,
                    registry=branch_registry,
                    preexisting_ids=actual_branch_preexisting_ids,
                    actual_record_index=index,
                    ranges=teacher_forced_ranges,
                    source_cells=creation_sources,
                )
            branch_registry.update_source_labels(model, code)
            if segment_context_composition_tracker is not None:
                segment_context_composition_tracker.record_observation(
                    model=model,
                    actual_record_index=index,
                )
            if temporal_context_tracker is not None:
                temporal_context_tracker.record_actual_observation(
                    model.previous_winners,
                    index,
                )
            if autonomous_context_provenance_tracker is not None:
                autonomous_context_provenance_tracker.record_actual_observation(
                    model._active_sources(),
                    model.previous_winners,
                    index,
                )
        observe_runtime = time.perf_counter() - observe_started
        write_native_crash_breadcrumb(
            current_index=index,
            rollout_step=None,
            phase="actual_observation_complete",
            model=model,
        )
        if debug_payload is not None and debug_payload.get("record_index") == index:
            debug_payload["observe_after"] = {
                "segment_count": segment_count(model),
                "synapse_count": synapse_count(model),
                "winner_count": len(model.previous_winners),
                "observe_runtime_seconds": observe_runtime,
                "scenario_counts": {
                    "scenario1": model.last_observe_stats.get("scenario1", 0),
                    "scenario2": model.last_observe_stats.get("scenario2", 0),
                    "scenario3": model.last_observe_stats.get("scenario3", 0),
                },
            }
        if rollout_diagnostics:
            for row in density_rows[-len(rollout_diagnostics) :]:
                row["observe_runtime"] = observe_runtime
        if interval_every > 0 and (index + 1) % interval_every == 0:
            now = time.perf_counter()
            recent_density = [
                row
                for row in density_rows
                if row["record_index"] != ""
                and interval_start_index <= int(row["record_index"]) <= index
            ]
            interval_rows.append(
                {
                    "ending_record_index": index + 1,
                    "mape": mape(predictions, targets),
                    "coverage": (
                        len(predictions) / (len(predictions) + missing)
                        if len(predictions) + missing
                        else 0.0
                    ),
                    "mean_raw_columns": (
                        sum(raw_column_counts) / len(raw_column_counts)
                        if raw_column_counts
                        else 0.0
                    ),
                    "step_density_mean": (
                        sum(
                            float(row["raw_predicted_column_count"])
                            for row in recent_density
                        )
                        / len(recent_density)
                        if recent_density
                        else 0.0
                    ),
                    "segment_count": segment_count(model),
                    "runtime_seconds": now - interval_start_time,
                    "runtime_per_record": (
                        (now - interval_start_time)
                        / max(1, index + 1 - interval_start_index)
                    ),
                }
            )
            interval_start_time = now
            interval_start_index = index + 1
        if progress_every > 0 and (index + 1) % progress_every == 0:
            write_native_crash_breadcrumb(
                current_index=index,
                rollout_step=None,
                phase="diagnostic_flush_enter",
                model=model,
            )
            flush_large_diagnostic_batches()
            progress_now = time.perf_counter()
            progress = {
                "current_index": index + 1,
                "elapsed_seconds": progress_now - start_time,
                "segment_count": segment_count(model),
                "synapse_count": synapse_count(model),
                "active_source_count": int(
                    model.last_prediction_stats.get("active_source_count", 0)
                ),
                "candidate_segment_count": int(
                    model.last_prediction_stats.get("candidate_segment_count", 0)
                ),
                "working_set_memory_mb": working_set_memory_mb(),
                "time_for_last_10_rows": progress_now - progress_started,
            }
            output_dir.mkdir(parents=True, exist_ok=True)
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(progress, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            print("FIG9_PROGRESS " + json.dumps(progress, sort_keys=True), flush=True)
            progress_started = progress_now
            write_native_crash_breadcrumb(
                current_index=index,
                rollout_step=None,
                phase="observation_complete",
                model=model,
            )
        if (
            checkpoint_path is not None
            and (
                (checkpoint_at_index is not None and index + 1 == checkpoint_at_index)
                or (checkpoint_every > 0 and (index + 1) % checkpoint_every == 0)
            )
        ):
            provenance_payload = (
                branch_registry.checkpoint_payload(model)
                if branch_registry is not None
                else None
            )
            save_strict_checkpoint(
                checkpoint_path,
                encoder=encoder,
                model=model,
                fingerprint=fingerprint,
                config=config,
                stream_label=stream_label,
                data_path=data_path,
                records=records,
                limit=limit,
                next_index=index + 1,
                predictions=predictions,
                targets=targets,
                rows=rows,
                recent_errors=recent_errors,
                missing=missing,
                raw_event_counts=raw_event_counts,
                raw_column_counts=raw_column_counts,
                density_rows=density_rows,
                branch_provenance_payload=provenance_payload,
                diagnostic_state=(
                    {
                        "teacher_forced_observation_rows": (
                            teacher_forced_observation_rows
                        ),
                        "segment_reinforcement_rows": (
                            segment_reinforcement_rows
                        ),
                        "independent_reference_event_rows": independent_reference_event_rows,
                        "independent_reference_segment_rows": independent_reference_segment_rows,
                        "independent_reference_tracker": independent_reference_tracker.checkpoint_payload(),
                        "actual_branch_tracker": actual_branch_tracker.checkpoint_payload(),
                        "actual_branch_event_rows": actual_branch_event_rows,
                        "actual_branch_segment_rows": actual_branch_segment_rows,
                        "temporal_context_tracker": (
                            temporal_context_tracker.checkpoint_payload()
                            if temporal_context_tracker is not None
                            else None
                        ),
                        "autonomous_context_provenance_tracker": (
                            autonomous_context_provenance_tracker.checkpoint_payload()
                            if autonomous_context_provenance_tracker is not None
                            else None
                        ),
                    }
                    if segment_reinforcement_diagnostic or independent_reference_diagnostic or actual_branch_provenance_diagnostic or temporal_context_diagnostic or autonomous_context_provenance_diagnostic
                    else None
                ),
            )
            if provenance_payload is not None and checkpoint_path is not None:
                write_json_atomic(
                    branch_checkpoint_sidecar_path(checkpoint_path),
                    provenance_payload,
                )
            write_checkpoint_metadata(
                checkpoint_path,
                next_index=index + 1,
                fingerprint=fingerprint,
                model=model,
                branch_provenance_payload=provenance_payload,
            )
        last_completed_index = index + 1
        if stop_after_index is not None and index + 1 >= stop_after_index:
            break

    flush_large_diagnostic_batches()
    if temporal_context_tracker is not None:
        temporal_context_tracker.close()
    if autonomous_context_provenance_tracker is not None:
        autonomous_context_provenance_tracker.close()
        autonomous_context_provenance_tracker.write_run_outputs(output_dir)
    if (
        checkpoint_path is not None
        and checkpoint_every > 0
        and last_completed_index > start_index
        and last_completed_index % checkpoint_every != 0
    ):
        provenance_payload = (
            branch_registry.checkpoint_payload(model)
            if branch_registry is not None
            else None
        )
        save_strict_checkpoint(
            checkpoint_path,
            encoder=encoder,
            model=model,
            fingerprint=fingerprint,
            config=config,
            stream_label=stream_label,
            data_path=data_path,
            records=records,
            limit=limit,
            next_index=last_completed_index,
            predictions=predictions,
            targets=targets,
            rows=rows,
            recent_errors=recent_errors,
            missing=missing,
            raw_event_counts=raw_event_counts,
            raw_column_counts=raw_column_counts,
            density_rows=density_rows,
            branch_provenance_payload=provenance_payload,
            diagnostic_state=(
                {
                    "teacher_forced_observation_rows": (
                        teacher_forced_observation_rows
                    ),
                    "segment_reinforcement_rows": segment_reinforcement_rows,
                    "independent_reference_event_rows": independent_reference_event_rows,
                    "independent_reference_segment_rows": independent_reference_segment_rows,
                    "independent_reference_tracker": independent_reference_tracker.checkpoint_payload(),
                    "actual_branch_tracker": actual_branch_tracker.checkpoint_payload(),
                    "actual_branch_event_rows": actual_branch_event_rows,
                    "actual_branch_segment_rows": actual_branch_segment_rows,
                    "temporal_context_tracker": (
                        temporal_context_tracker.checkpoint_payload()
                        if temporal_context_tracker is not None
                        else None
                    ),
                    "autonomous_context_provenance_tracker": (
                        autonomous_context_provenance_tracker.checkpoint_payload()
                        if autonomous_context_provenance_tracker is not None
                        else None
                    ),
                }
                if segment_reinforcement_diagnostic or independent_reference_diagnostic or actual_branch_provenance_diagnostic or temporal_context_diagnostic or autonomous_context_provenance_diagnostic
                else None
            ),
        )
        if provenance_payload is not None:
            write_json_atomic(
                branch_checkpoint_sidecar_path(checkpoint_path),
                provenance_payload,
            )
        write_checkpoint_metadata(
            checkpoint_path,
            next_index=last_completed_index,
            fingerprint=fingerprint,
            model=model,
            branch_provenance_payload=provenance_payload,
        )
    attempted = len(rows)
    elapsed = time.perf_counter() - start_time
    summary = {
        "diagnostic_label": "strict Fig.9 reproduction attempt",
        "stream_label": stream_label,
        "records_used": len(records),
        "next_observation_index": last_completed_index,
        "attempted_predictions": attempted,
        "predictions": len(predictions),
        "missing_predictions": missing,
        "coverage": len(predictions) / attempted if attempted else 0.0,
        "mape": mape(predictions, targets),
        "rolling_window": config.rolling_window,
        "final_rolling_mape": (
            rows[-1]["rolling_mape"] if rows and rows[-1]["rolling_mape"] != "" else ""
        ),
        "mean_raw_event_count": (
            sum(raw_event_counts) / len(raw_event_counts) if raw_event_counts else 0.0
        ),
        "mean_raw_column_count": (
            sum(raw_column_counts) / len(raw_column_counts) if raw_column_counts else 0.0
        ),
        "peak_raw_column_count": max(raw_column_counts) if raw_column_counts else 0,
        "runtime_seconds": elapsed,
        "records_per_second": len(records) / elapsed if elapsed else 0.0,
        "continuous_impl": config.continuous_prediction_impl,
        "continuous_impl_version": continuous_impl_version(
            config.continuous_prediction_impl
        ),
        "final_model_fingerprint": model_long_term_fingerprint(model),
        "final_rng_fingerprint": model_rng_fingerprint(model),
        "final_segment_count": segment_count(model),
        "final_synapse_count": synapse_count(model),
        "prediction_before_observe": True,
        "autonomous_rollout_steps": config.horizon,
        "uses_compensation": False,
        "L_match": config.l_match,
    }
    if resume_provenance_audit is not None:
        summary["resume_provenance_audit"] = resume_provenance_audit
    if lmatch_real_ablation:
        summary.update(lmatch_ablation_markers(config.l_match))
    if competition.enabled:
        summary.update(
            {
                "diagnostic_only": True,
                "competition_is_local_choice": True,
                "competition_mode": competition.mode,
                "inhibition_strength": competition.inhibition_strength,
                "inhibition_tau": competition.inhibition_tau,
                "simultaneous_tolerance": competition.simultaneous_tolerance,
                "simultaneous_policy": competition.simultaneous_policy,
                "simultaneous_bin_width": competition.simultaneous_bin_width,
            }
        )
    if segment_context_composition_tracker is not None:
        composition_paths = segment_context_composition_tracker.write_outputs(
            output_dir,
            compress=segment_context_composition_compress,
        )
        write_json(
            output_dir / "segment_context_composition_protocol.json",
            {
                **segment_context_composition_protocol(
                    level=segment_context_composition_level,
                    compressed=segment_context_composition_compress,
                ),
                "output_paths": composition_paths,
                "strict_protocol_sha256": stable_object_sha256(fingerprint),
            },
        )
    if context_trajectory_diagnostic:
        summary["context_trajectory_trace_path"] = str(
            context_trajectory_path.with_suffix(
                context_trajectory_path.suffix + ".gz"
            )
            if context_trajectory_compress
            else context_trajectory_path
        )
        summary["context_trajectory_row_count"] = context_trajectory_row_count
        summary["context_trajectory_trace_streamed"] = stream_diagnostic_traces
    if temporal_context_tracker is not None:
        summary["temporal_context_diagnostic"] = {
            "enabled": True,
            "level": temporal_context_level,
            "candidate_rows": temporal_context_tracker.candidate_row_count,
            "actual_rows": temporal_context_tracker.actual_row_count,
            "trace_path": str(output_dir / "temporal_candidate_trace.csv.gz"),
            "protocol_version": temporal_context_protocol()["version"],
            "strict_default_unchanged": True,
        }
    if autonomous_context_provenance_tracker is not None:
        summary["autonomous_context_provenance_diagnostic"] = {
            "enabled": True,
            "level": autonomous_context_provenance_level,
            "candidate_rows": autonomous_context_provenance_tracker.candidate_row_count,
            "source_rows": autonomous_context_provenance_tracker.source_row_count,
            "candidate_trace_path": str(
                output_dir / "autonomous_candidate_provenance_trace.csv.gz"
            ),
            "protocol_version": autonomous_context_provenance_protocol(
                autonomous_context_provenance_level
            )["version"],
            "strict_default_unchanged": True,
            "read_only": True,
        }
    if segment_context_composition_tracker is not None:
        summary["segment_context_composition"] = (
            segment_context_composition_tracker.summary()
        )
    if density_rows:
        density_summary = summarize_density(density_rows)
        summary["density_summary_path"] = str(
            density_summary_path
            or output_dir / f"{stream_label}_density_summary.json"
        )
        summary["density_trace_path"] = str(
            density_trace_path
            or output_dir / f"{stream_label}_density_trace.csv"
        )
    competition_summary: dict[str, object] | None = None
    if competition.enabled:
        competition_summary = summarize_competition(
            density_rows,
            horizon=config.horizon,
            attempted_rollouts=attempted,
            segment_count=segment_count(model),
            runtime_seconds=elapsed,
        )
        competition_summary.update(
            {
                "simultaneous_policy": competition.simultaneous_policy,
                "simultaneous_bin_width": competition.simultaneous_bin_width,
                "same_batch_candidates_inhibit_each_other": (
                    competition.simultaneous_policy == "sequential"
                ),
            }
        )
        if oracle_candidate_diagnostic:
            for step_payload in competition_summary["step_summary"]:
                step = int(step_payload["horizon_step"])
                oracle_step = [
                    row
                    for row in oracle_rows
                    if int(row["horizon_step"]) == step
                ]
                step_payload[
                    "target_candidates_sharing_batch_with_earlier_false"
                ] = sum(
                    int(
                        row[
                            "target_candidates_sharing_batch_with_earlier_false"
                        ]
                    )
                    for row in oracle_step
                )
        summary["competition_trace_path"] = str(
            output_dir / "competition_trace.csv"
        )
        summary["competition_summary_path"] = str(
            output_dir / "competition_summary.json"
        )
    if interval_rows:
        summary["interval_rows"] = interval_rows

    output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(output_dir / f"{stream_label}_predictions.csv", rows)
    (output_dir / f"{stream_label}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / f"{stream_label}_protocol.json").write_text(
        json.dumps(fingerprint, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if density_rows:
        write_density_trace(
            density_trace_path or output_dir / f"{stream_label}_density_trace.csv",
            density_rows,
        )
        write_json(
            density_summary_path
            or output_dir / f"{stream_label}_density_summary.json",
            density_summary,
        )
    if competition.enabled and competition_summary is not None:
        write_predictions(output_dir / "competition_trace.csv", density_rows)
        write_json(
            output_dir / "competition_summary.json",
            competition_summary,
        )
        write_json(
            output_dir / "competition_protocol.json",
            {
                "diagnostic_only": True,
                "competition_is_local_choice": True,
                **asdict(competition),
                "same_batch_candidates_inhibit_each_other": (
                    competition.simultaneous_policy == "sequential"
                ),
                "threshold": model.params.dendrite_threshold,
                "score_source": "PredictionCandidate.score",
                "candidate_source": (
                    "same predict_code call via last_prediction_candidates"
                ),
                "strict_protocol_sha256": stable_object_sha256(fingerprint),
            },
        )
    if intracolumn_selection_diagnostic:
        write_selection_trace(
            output_dir / "intracolumn_selection_trace.csv",
            intracolumn_selection_rows,
        )
        selection_summary = summarize_selection_rows(
            intracolumn_selection_rows,
            policy=intracolumn_selection_policy,
        )
        write_json(
            output_dir / "intracolumn_selection_summary.json",
            selection_summary,
        )
        write_json(
            output_dir / "intracolumn_selection_protocol.json",
            {
                **INTRACOLUMN_DIAGNOSTIC_MARKERS,
                "policy": intracolumn_selection_policy,
                "selector_version": INTRACOLUMN_SELECTOR_VERSION,
                "contributor_metric": "contributor_count",
                "missing_metric_order": "after finite values",
                "stable_tie_break": "original event-winner insertion order",
                "candidate_source": (
                    "same predict_code call, after event selection and before "
                    "intercolumn competition"
                ),
                "strict_protocol_sha256": stable_object_sha256(fingerprint),
            },
        )
    if oracle_candidate_diagnostic:
        write_predictions(
            output_dir / "oracle_candidate_trace.csv",
            oracle_rows,
        )
        write_json(
            output_dir / "oracle_candidate_summary.json",
            summarize_oracle_rows(oracle_rows, horizon=config.horizon),
        )
    if candidate_separability_trace:
        write_predictions(
            output_dir / "candidate_separability_trace.csv",
            candidate_score_rows,
        )
    if branch_provenance_diagnostic:
        write_predictions(
            output_dir / "branch_candidate_trace.csv",
            branch_candidate_rows,
        )
        if branch_provenance_level in {"candidate", "full"}:
            if not stream_diagnostic_traces:
                write_predictions(
                    output_dir / "branch_segment_trace.csv",
                    branch_segment_rows,
                )
        if branch_provenance_level == "full":
            write_predictions(
                output_dir / "branch_source_trace.csv",
                branch_source_rows,
            )
        write_json(
            output_dir / "branch_step_summary.json",
            summarize_branch_steps(branch_candidate_rows),
        )
        write_json(
            output_dir / "branch_provenance_protocol.json",
            {
                **BRANCH_DIAGNOSTIC_MARKERS,
                "version": "fig9-branch-provenance-v1",
                "level": branch_provenance_level,
                "source_top_n": branch_source_top_n,
                "source_rows_truncated": branch_source_top_n is not None,
                "candidate_source": (
                    "same predict_code call via last_prediction_candidates"
                ),
                "candidate_score_semantics": (
                    "single winning segment response returned by continuous "
                    "prediction; not a multi-segment sum"
                ),
                "branch_id_definition": (
                    "SHA256(creation_transition_index,target_column,"
                    "target_neuron,creation_source_fingerprint,"
                    "stable_creation_ordinal)"
                ),
                "branch_similarity_definition": (
                    "Jaccard over source-cell sets; not a branch ID"
                ),
                "chimera_thresholds": {
                    "coherent_single_branch": 0.8,
                    "mostly_coherent": 0.5,
                    "high_score_percentile": 0.8,
                },
                "unavailable_fields": [
                    "historical winner sets beyond exact creation source set",
                    "multiple supporting segments per PredictionCandidate",
                ],
                "strict_protocol_sha256": stable_object_sha256(fingerprint),
            },
        )
    if context_oracle_unified_trace:
        unified_rows = [
            *unified_actual_candidate_rows,
            *branch_candidate_rows,
        ]
        join_dir = output_dir / "context_oracle_join"
        join_consistency = write_join_outputs(join_dir, rows=unified_rows)
        summary["context_oracle_unified_trace"] = {
            "version": "fig9-candidate-context-oracle-v1",
            "trajectory_kinds": [
                "actual_observation",
                "autonomous_rollout",
            ],
            "row_unit": "CANDIDATE_SEGMENT",
            "candidate_rows": len(unified_rows),
            "actual_observation_rows": len(unified_actual_candidate_rows),
            "autonomous_rollout_rows": len(branch_candidate_rows),
            "join_consistency": join_consistency,
            "oracle_label_posthoc_only": True,
            "ground_truth_does_not_affect_model": True,
            "step2_plus_recurrent_trajectory_divergence": True,
            "path": str(join_dir),
        }
        write_json(
            output_dir / "context_oracle_unified_protocol.json",
            {
                "version": "fig9-candidate-context-oracle-v1",
                "diagnostic_only": True,
                "offline_analysis_only": True,
                "row_unit": "CANDIDATE_SEGMENT",
                "trajectory_kinds": [
                    "actual_observation",
                    "autonomous_rollout",
                ],
                "candidate_event_id_fields": [
                    "actual_record_index",
                    "horizon_step",
                    "field",
                    "observed_encoded_column",
                    "candidate_target_column",
                    "candidate_target_neuron",
                    "stable_segment_provenance_id",
                    "candidate_generation_ordinal",
                    "observed_event_time",
                ],
                "oracle_target_column_posthoc_only": True,
                "ground_truth_does_not_affect_matching": True,
                "ground_truth_does_not_affect_selection": True,
                "ground_truth_does_not_affect_competition": True,
                "ground_truth_does_not_affect_learning": True,
                "ground_truth_does_not_affect_rng": True,
                "join_consistency": join_consistency,
            },
        )
    if preselection_segment_diagnostic:
        traced_segments = (
            preselection_segment_rows[:preselection_max_rows]
            if preselection_max_rows is not None
            else preselection_segment_rows
        )
        traced_groups = (
            preselection_group_rows[:preselection_max_rows]
            if preselection_max_rows is not None
            else preselection_group_rows
        )
        traced_replacements = (
            preselection_replacement_rows[:preselection_max_rows]
            if preselection_max_rows is not None
            else preselection_replacement_rows
        )
        write_diagnostic_csv(
            output_dir / "preselection_funnel_trace.csv",
            preselection_funnel_rows,
        )
        if not stream_diagnostic_traces:
            write_diagnostic_csv(
                output_dir / "preselection_segment_trace.csv",
                traced_segments,
                compress=preselection_segment_compress,
            )
            write_diagnostic_csv(
                output_dir / "preselection_group_trace.csv",
                traced_groups,
                compress=preselection_segment_compress,
            )
            write_diagnostic_csv(
                output_dir / "preselection_replacement_trace.csv",
                traced_replacements,
                compress=preselection_segment_compress,
            )
        total_inspected = sum(
            int(row["inspected_segment_count"])
            for row in preselection_funnel_rows
        )
        total_crossed = sum(
            int(row["threshold_crossing_count"])
            for row in preselection_funnel_rows
        )
        total_candidates = sum(
            int(row["saved_candidate_count"])
            for row in preselection_funnel_rows
        )
        write_json(
            output_dir / "preselection_summary.json",
            {
                **PRESELECTION_DIAGNOSTIC_MARKERS,
                "steps": len(preselection_funnel_rows),
                "inspected_segment_count": total_inspected,
                "threshold_crossing_count": total_crossed,
                "saved_candidate_count": total_candidates,
                "crossing_to_candidate_ratio": (
                    total_candidates / total_crossed
                    if total_crossed
                    else 0.0
                ),
                "unknown_elimination_count": sum(
                    int(row["unknown_elimination_count"])
                    for row in preselection_funnel_rows
                ),
                "actual_segment_rows": (
                    streamed_trace_counts["preselection_segment"]
                    if stream_diagnostic_traces else len(traced_segments)
                ),
                "total_segment_rows_before_truncation": (
                    streamed_trace_counts["preselection_segment"]
                    if stream_diagnostic_traces else len(preselection_segment_rows)
                ),
                "actual_group_rows": (
                    streamed_trace_counts["preselection_group"]
                    if stream_diagnostic_traces else len(traced_groups)
                ),
                "total_group_rows_before_truncation": (
                    streamed_trace_counts["preselection_group"]
                    if stream_diagnostic_traces else len(preselection_group_rows)
                ),
                "actual_replacement_rows": (
                    streamed_trace_counts["preselection_replacement"]
                    if stream_diagnostic_traces else len(traced_replacements)
                ),
                "total_replacement_rows_before_truncation": (
                    streamed_trace_counts["preselection_replacement"]
                    if stream_diagnostic_traces else len(preselection_replacement_rows)
                ),
            },
        )
        write_json(
            output_dir / "preselection_protocol.json",
            {
                **PRESELECTION_DIAGNOSTIC_MARKERS,
                "version": "fig9-preselection-segments-v1",
                "level": preselection_segment_level,
                "compressed": preselection_segment_compress,
                "trace_truncated": (
                    preselection_max_rows is not None
                    and (
                        len(traced_segments)
                        < len(preselection_segment_rows)
                        or len(traced_groups) < len(preselection_group_rows)
                        or len(traced_replacements)
                        < len(preselection_replacement_rows)
                    )
                ),
                "requested_max_rows": preselection_max_rows,
                "actual_rows": {
                    "segments": streamed_trace_counts["preselection_segment"] if stream_diagnostic_traces else len(traced_segments),
                    "groups": streamed_trace_counts["preselection_group"] if stream_diagnostic_traces else len(traced_groups),
                    "replacements": streamed_trace_counts["preselection_replacement"] if stream_diagnostic_traces else len(traced_replacements),
                },
                "funnel_counts_are_untruncated": True,
                "selection_stages": [
                    "INSPECTED",
                    "UPPER_BOUND_ELIGIBILITY",
                    "RESPONSE_COMPUTED",
                    "THRESHOLD_CROSSED",
                    "VALID_FIRING_WINDOW",
                    "EVENT_SCORE_WINNER",
                    "COLUMN_EARLIEST_SELECTION",
                    "PREDICTION_CANDIDATE",
                    "COMPETITION_EMITTED",
                ],
                "nonexistent_stage": (
                    "there is no independent per-neuron winner group"
                ),
                "event_group_key": (
                    "(target_column, round(predicted_soma_time, 12))"
                ),
                "event_selection_rule": (
                    "replace only when score > previous score"
                ),
                "column_selection_rule": (
                    "replace only when predicted time < previous time"
                ),
                "strict_protocol_sha256": stable_object_sha256(fingerprint),
            },
        )
    if segment_reinforcement_diagnostic:
        assert branch_registry is not None
        final_segment_ids = {
            origin.segment_provenance_id
            for column in model.columns
            for neuron in column.neurons
            for segment in neuron.segments
            if (origin := branch_registry.provenance_for(segment)) is not None
        }
        completed_reinforcement_rows = enrich_segment_reinforcement_rows(
            segment_reinforcement_rows,
            observation_rows=teacher_forced_observation_rows,
            density_rows=density_rows,
            final_segment_ids=final_segment_ids,
        )
        write_diagnostic_csv(
            output_dir / "segment_reinforcement_event_trace.csv",
            completed_reinforcement_rows,
            compress=True,
        )
        write_json(
            output_dir / "segment_reinforcement_protocol.json",
            {
                **SEGMENT_REINFORCEMENT_MARKERS,
                "version": "fig9-segment-reinforcement-v1",
                "level": segment_reinforcement_level,
                "rows": len(completed_reinforcement_rows),
                "selection_behavior_changed": False,
                "reference_rule": (
                    "Scenario-1 pre-observation timed predictive identity only; "
                    "Scenario-2 operational winner is circular and invalid"
                ),
                "strict_protocol_sha256": stable_object_sha256(fingerprint),
            },
        )
    if independent_reference_diagnostic:
        event_path = write_independent_reference_trace(
            output_dir / "independent_reference_event_trace.csv",
            independent_reference_event_rows,
            compress=independent_reference_compress,
        )
        segment_path = write_independent_reference_trace(
            output_dir / "independent_reference_segment_trace.csv",
            independent_reference_segment_rows,
            compress=independent_reference_compress,
        )
        write_json(
            output_dir / "independent_reference_protocol.json",
            {
                **independent_reference_protocol(fingerprint, independent_reference_level, independent_reference_compress),
                "event_trace_path": str(event_path),
                "segment_trace_path": str(segment_path),
                "event_rows": len(independent_reference_event_rows),
                "segment_rows": len(independent_reference_segment_rows),
                "actual_history_is_separate_from_autonomous_rollout": True,
            },
        )
    if actual_branch_provenance_diagnostic:
        branch_event_path = write_actual_branch_rows(
            output_dir / "actual_branch_event_trace.csv",
            actual_branch_event_rows,
            compress=actual_branch_provenance_compress,
        )
        branch_segment_path = write_actual_branch_rows(
            output_dir / "actual_branch_segment_trace.csv",
            actual_branch_segment_rows,
            compress=actual_branch_provenance_compress,
        )
        registry_path = write_actual_branch_registry(
            output_dir / "actual_branch_registry.csv",
            actual_branch_tracker,
            compress=actual_branch_provenance_compress,
        )
        lineage_event_path = write_actual_branch_rows(
            output_dir / "actual_lineage_event_trace.csv",
            actual_branch_tracker.lineage_events,
            compress=actual_branch_provenance_compress,
        )
        lineage_registry_path = write_lineage_registry(
            output_dir / "actual_lineage_registry.csv",
            actual_branch_tracker,
            compress=actual_branch_provenance_compress,
        )
        write_json(
            output_dir / "actual_branch_provenance_protocol.json",
            {
                **ACTUAL_BRANCH_MARKERS,
                "version": "fig9-actual-branch-provenance-v2",
                "level": actual_branch_provenance_level,
                "compressed": actual_branch_provenance_compress,
                "event_trace_path": str(branch_event_path),
                "segment_trace_path": str(branch_segment_path),
                "registry_path": str(registry_path),
                "lineage_event_trace_path": str(lineage_event_path),
                "lineage_registry_path": str(lineage_registry_path),
                "event_rows": len(actual_branch_event_rows),
                "segment_rows": len(actual_branch_segment_rows),
                "anchor_count": len(actual_branch_tracker.anchors),
                "branch_count": len(actual_branch_tracker.branches),
                "lineage_count": len(actual_branch_tracker.lineages),
                "lineage_event_rows": len(actual_branch_tracker.lineage_events),
                "actual_history_is_separate_from_autonomous_rollout": True,
                "current_matching_is_post_observation_join_only": True,
            },
        )
    if teacher_forced_winner_diagnostic:
        write_diagnostic_csv(
            output_dir / "teacher_forced_observation_trace.csv",
            legacy_teacher_forced_rows(teacher_forced_observation_rows),
        )
        available = sum(
            bool(row["observed_winner_available"])
            for row in teacher_forced_observation_rows
        )
        write_json(
            output_dir / "teacher_forced_winner_protocol.json",
            {
                **TEACHER_FORCED_DIAGNOSTIC_MARKERS,
                "version": "fig9-teacher-forced-winner-v1",
                "level": teacher_forced_winner_level,
                "reference_definition": (
                    "winner identity produced when the target record is later "
                    "observed in the normal online stream"
                ),
                "prediction_before_observe": True,
                "rows": len(teacher_forced_observation_rows),
                "winner_reference_available_rate": (
                    available / len(teacher_forced_observation_rows)
                    if teacher_forced_observation_rows
                    else 0.0
                ),
                "strict_protocol_sha256": stable_object_sha256(fingerprint),
            },
        )
        if observe_scenario_diagnostic:
            write_diagnostic_csv(
                output_dir / "observe_scenario_trace.csv",
                observe_scenario_rows(
                    teacher_forced_observation_rows,
                    level=observe_scenario_level,
                ),
            )
            write_json(
                output_dir / "observe_scenario_protocol.json",
                {
                    **TEACHER_FORCED_DIAGNOSTIC_MARKERS,
                    "observe_scenario_diagnostic": True,
                    "reference_neuron_selection_diagnostic": True,
                    "level": observe_scenario_level,
                    "forgetting_evidence_available": False,
                    "selection_behavior_changed": False,
                    "strict_protocol_sha256": stable_object_sha256(
                        fingerprint
                    ),
                },
            )
    if teacher_forced_winner_diagnostic and reference_neuron_selection_diagnostic:
        write_json(
            output_dir / "reference_neuron_selection_protocol.json",
            {
                **TEACHER_FORCED_DIAGNOSTIC_MARKERS,
                "reference_neuron_selection_diagnostic": True,
                "target_column_is_oracle_conditioned_for_analysis": True,
                "not_a_deployable_prediction_result": True,
                "level": reference_neuron_selection_level,
                "selection_behavior_changed": False,
            },
        )
    if match_overlap_diagnostic:
        write_diagnostic_csv(
            output_dir / "match_overlap_column_trace.csv",
            match_overlap_column_rows,
        )
        if match_overlap_level in {"segment", "source"}:
            if not stream_diagnostic_traces:
                write_diagnostic_csv(
                    output_dir / "match_overlap_segment_trace.csv",
                    match_overlap_segment_rows,
                    compress=match_overlap_compress,
                )
        if match_overlap_level == "source":
            write_diagnostic_csv(
                output_dir / "match_overlap_source_trace.csv",
                match_overlap_source_rows,
                compress=match_overlap_compress,
            )
        write_json(
            output_dir / "match_overlap_protocol.json",
            {
                **MATCH_OVERLAP_DIAGNOSTIC_MARKERS,
                "version": "fig9-match-overlap-v1",
                "level": match_overlap_level,
                "compressed": match_overlap_compress,
                "matching_context": (
                    "previous_active_cells with previous_winners fallback "
                    "when burst_context=true; previous_winners otherwise"
                ),
                "segment_creation_context": "previous_winners",
                "actual_overlap": (
                    "count of unique active source cell IDs whose synapse "
                    "arrival is within timing_tolerance of dendritic time"
                ),
                "L_match": model.params.l_match,
                "selection_behavior_changed": False,
                "timing_eligibility_decomposition": (
                    timing_eligibility_decomposition
                ),
                "diagnostic_does_not_affect_matching": True,
                "diagnostic_does_not_affect_prediction": True,
                "diagnostic_does_not_affect_learning": True,
                "ground_truth_does_not_affect_model": True,
                "filtered_trace_does_not_filter_model_execution": True,
                "counterfactual_does_not_affect_model": True,
                "trace_filter_only": source_trace_filter is not None,
                "model_execution_unfiltered": True,
                "source_trace_filter": (
                    asdict(source_trace_filter)
                    if source_trace_filter is not None
                    else None
                ),
                "unavailable_fields": [
                    "future segment deletion after each captured observation",
                    "predicted-versus-burst labels at historical segment creation",
                ],
                "strict_protocol_sha256": stable_object_sha256(
                    fingerprint
                ),
            },
        )
    if context_trajectory_diagnostic:
        trace_path = (
            (
                context_trajectory_path.with_suffix(
                    context_trajectory_path.suffix + ".gz"
                )
                if context_trajectory_compress
                else context_trajectory_path
            )
            if stream_diagnostic_traces
            and (
                context_trajectory_path.with_suffix(
                    context_trajectory_path.suffix + ".gz"
                )
                if context_trajectory_compress
                else context_trajectory_path
            ).exists()
            else write_diagnostic_csv(
                context_trajectory_path,
                context_trajectory_rows,
                compress=context_trajectory_compress,
            )
        )
        write_json(
            output_dir / "context_trajectory_protocol.json",
            {
                **CONTEXT_TRAJECTORY_MARKERS,
                "version": "fig9-context-trajectory-v1",
                "level": context_trajectory_level,
                "compressed": context_trajectory_compress,
                "trace_path": str(trace_path) if trace_path else "",
                "rows": context_trajectory_row_count,
                "streamed_incrementally": stream_diagnostic_traces,
                "source_dependency_filter": "SOURCE_COLUMN_NOT_ACTIVE",
                "source_trace_filter": (
                    asdict(source_trace_filter)
                    if source_trace_filter is not None
                    else None
                ),
                "trace_filter_only": source_trace_filter is not None,
                "model_execution_unfiltered": True,
                "actual_observation_context_update": (
                    "observe_code proximal input updates previous_active_cells "
                    "and learning winners"
                ),
                "autonomous_context_update": (
                    "competition-emitted prediction cells are copied into "
                    "previous_active_cells and previous_winners"
                ),
                "strict_protocol_sha256": stable_object_sha256(fingerprint),
            },
        )
    if temporal_context_tracker is not None:
        write_json(
            output_dir / "temporal_context_protocol.json",
            {
                **temporal_context_protocol(),
                "level": temporal_context_level,
                "trace_path": str(output_dir / "temporal_candidate_trace.csv.gz"),
                "candidate_rows": temporal_context_tracker.candidate_row_count,
                "actual_rows": temporal_context_tracker.actual_row_count,
                "strict_protocol_sha256": stable_object_sha256(fingerprint),
            },
        )
    if autonomous_context_provenance_tracker is not None:
        write_json(
            output_dir / "autonomous_context_provenance_protocol.json",
            {
                **autonomous_context_provenance_protocol(
                    autonomous_context_provenance_level
                ),
                "candidate_trace_path": str(
                    output_dir / "autonomous_candidate_provenance_trace.csv.gz"
                ),
                "candidate_rows": autonomous_context_provenance_tracker.candidate_row_count,
                "source_rows": autonomous_context_provenance_tracker.source_row_count,
                "strict_protocol_sha256": stable_object_sha256(fingerprint),
            },
        )
    if interval_rows:
        write_predictions(
            output_dir / f"{stream_label}_interval_summary.csv",
            interval_rows,
        )
    if debug_payload is not None:
        write_json(
            debug_output_json or output_dir / f"{stream_label}_debug_record.json",
            debug_payload,
        )
    plot_name = (
        "fig9_b_original_mape.png"
        if stream_label == "original"
        else "fig9_c_perturbed_mape.png"
    )
    plot_adaptation(output_dir / plot_name, rows, datetime(2015, 3, 25))
    if stream_label == "perturbed":
        plot_adaptation(
            output_dir / "fig9_d_adaptation_curve.png",
            rows,
            datetime(2015, 3, 25),
        )
    return summary


def compare_pre_change_predictions(
    original_csv: Path,
    perturbed_csv: Path,
) -> dict[str, object]:
    """Check original/perturbed streams are identical before Apr 1 targets.

    DEBUG WATCH: Apr 1 是论文扰动点。target_timestamp 早于
    2015-04-01 的行若出现 prediction mismatch，说明数据切分或扰动文件
    对齐有问题，而不是模型适应问题。
    """

    with original_csv.open("r", encoding="utf-8", newline="") as handle:
        original = list(csv.DictReader(handle))
    with perturbed_csv.open("r", encoding="utf-8", newline="") as handle:
        perturbed = list(csv.DictReader(handle))
    checked = 0
    mismatches = 0
    for left, right in zip(original, perturbed):
        target_time = datetime.fromisoformat(left["target_timestamp"])
        if target_time >= PAPER_CHANGE_DATE:
            continue
        checked += 1
        fields = ("target", "prediction", "rolling_mape", "raw_rollout_column_counts")
        if any(left[field] != right[field] for field in fields):
            mismatches += 1
    return {
        "pre_change_rows_checked": checked,
        "pre_change_prediction_mismatches": mismatches,
        "pre_change_predictions_identical": mismatches == 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/paper_nyc_taxi.csv")
    parser.add_argument("--perturbed-data", default="data/paper_nyc_taxi_perturb.csv")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5904)
    parser.add_argument("--prediction-horizon", type=int, default=5)
    parser.add_argument("--tie-break-seed", type=int, default=0)
    parser.add_argument(
        "--l-match",
        type=int,
        choices=(2, 3, 4),
        default=4,
        help="Real matching threshold; 2/3 require --lmatch-real-ablation.",
    )
    parser.add_argument(
        "--lmatch-real-ablation",
        action="store_true",
        help="Label a controlled nonpaper real-L_match experiment.",
    )
    parser.add_argument("--output-dir", default="results/fig9_strict")
    parser.add_argument("--density-trace", action="store_true")
    parser.add_argument("--density-trace-csv", default="")
    parser.add_argument("--density-summary-json", default="")
    parser.add_argument(
        "--competition-mode",
        choices=("off", "competitive_raw"),
        default="off",
        help="Nonpaper diagnostic competition applied only during rollout.",
    )
    parser.add_argument("--inhibition-strength", type=float, default=0.1)
    parser.add_argument("--inhibition-tau", type=float, default=0.02)
    parser.add_argument("--simultaneous-tolerance", type=float, default=0.0)
    parser.add_argument(
        "--simultaneous-policy",
        choices=("sequential", "batched"),
        default="sequential",
    )
    parser.add_argument(
        "--simultaneous-bin-width",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--oracle-candidate-diagnostic",
        action="store_true",
        help="Analyze true target candidates after competition; never affects prediction.",
    )
    parser.add_argument(
        "--candidate-separability-trace",
        action="store_true",
        help="Write candidate labels for offline analysis; requires oracle diagnostics.",
    )
    parser.add_argument(
        "--branch-provenance-diagnostic",
        action="store_true",
        help="Capture read-only candidate/segment provenance for offline analysis.",
    )
    parser.add_argument(
        "--branch-provenance-level",
        choices=tuple(sorted(BRANCH_LEVELS)),
        default="candidate",
    )
    parser.add_argument(
        "--branch-source-top-n",
        type=int,
        default=0,
        help="Full mode source cap; zero records every positive contributor.",
    )
    parser.add_argument(
        "--preselection-segment-diagnostic",
        action="store_true",
        help="Capture the read-only segment funnel before PredictionCandidate.",
    )
    parser.add_argument(
        "--preselection-segment-level",
        choices=tuple(sorted(PRESELECTION_LEVELS)),
        default="crossing",
    )
    parser.add_argument(
        "--preselection-segment-compress",
        action="store_true",
        help="Write segment/group/replacement detail as CSV.GZ.",
    )
    parser.add_argument(
        "--preselection-max-rows",
        type=int,
        default=0,
        help="Explicit debug truncation; zero keeps every detail row.",
    )
    parser.add_argument(
        "--teacher-forced-winner-diagnostic",
        action="store_true",
        help="Capture future-observed winner identity for offline analysis.",
    )
    parser.add_argument(
        "--teacher-forced-winner-level",
        choices=tuple(sorted(TEACHER_FORCED_LEVELS)),
        default="summary",
    )
    parser.add_argument(
        "--observe-scenario-diagnostic",
        action="store_true",
        help="Capture read-only Scenario 1/2/3 assignment locals.",
    )
    parser.add_argument(
        "--observe-scenario-level",
        choices=tuple(sorted(OBSERVE_SCENARIO_LEVELS)),
        default="summary",
    )
    parser.add_argument(
        "--reference-neuron-selection-diagnostic",
        action="store_true",
        help="Enable offline within-column reference-neuron analysis inputs.",
    )
    parser.add_argument(
        "--reference-neuron-selection-level",
        choices=tuple(sorted(REFERENCE_NEURON_SELECTION_LEVELS)),
        default="summary",
    )
    parser.add_argument(
        "--intracolumn-selection-policy",
        choices=tuple(sorted(INTRACOLUMN_SELECTION_POLICIES)),
        default="existing",
        help="Nonpaper local-choice ablation; existing preserves strict behavior.",
    )
    parser.add_argument(
        "--intracolumn-selection-diagnostic",
        action="store_true",
        help="Write the same-call per-column selector trace.",
    )
    parser.add_argument(
        "--match-overlap-diagnostic",
        action="store_true",
        help="Capture read-only Scenario-3 overlap decomposition inputs.",
    )
    parser.add_argument(
        "--match-overlap-level",
        choices=tuple(sorted(MATCH_OVERLAP_LEVELS)),
        default="summary",
    )
    parser.add_argument(
        "--match-overlap-compress",
        action="store_true",
        help="Write segment/source match-overlap traces as CSV.GZ.",
    )
    parser.add_argument(
        "--segment-context-composition-diagnostic",
        action="store_true",
        help="Capture read-only source composition and segment-context quality.",
    )
    parser.add_argument(
        "--segment-context-composition-level",
        choices=tuple(sorted(SEGMENT_CONTEXT_COMPOSITION_LEVELS)),
        default="match",
    )
    parser.add_argument(
        "--segment-context-composition-compress",
        action="store_true",
        help="Write segment-context composition traces as CSV.GZ.",
    )
    parser.add_argument(
        "--context-oracle-unified-trace",
        action="store_true",
        help=(
            "Write stable candidate/context/oracle projections for offline "
            "lossless-join analysis."
        ),
    )
    parser.add_argument(
        "--pairwise-context-diagnostic",
        action="store_true",
        help=(
            "Capture the read-only candidate/context inputs required by the "
            "offline source-pair coherence analyzer."
        ),
    )
    parser.add_argument(
        "--temporal-context-diagnostic",
        action="store_true",
        help="Capture bounded causal candidate temporal summaries for offline analysis.",
    )
    parser.add_argument(
        "--temporal-context-level",
        choices=("summary", "candidate"),
        default="summary",
        help="Temporal summary granularity; source-debug rows are intentionally unsupported.",
    )
    parser.add_argument(
        "--autonomous-context-provenance-diagnostic",
        action="store_true",
        help="Capture read-only source identity and activation provenance during autonomous rollout.",
    )
    parser.add_argument(
        "--autonomous-context-provenance-level",
        choices=("summary", "candidate", "source"),
        default="candidate",
        help="Provenance output level; source rows are bounded smoke-test detail.",
    )
    parser.add_argument(
        "--timing-eligibility-decomposition",
        action="store_true",
        help="Add read-only timing/eligibility evidence to source rows.",
    )
    parser.add_argument("--source-trace-filter-field", default="")
    parser.add_argument("--source-trace-filter-scenario", default="")
    parser.add_argument(
        "--source-trace-filter-record-start", type=int, default=None
    )
    parser.add_argument(
        "--source-trace-filter-record-end", type=int, default=None
    )
    parser.add_argument(
        "--source-trace-filter-loss-only", action="store_true"
    )
    parser.add_argument(
        "--source-trace-filter-segments-existing-only",
        action="store_true",
    )
    parser.add_argument(
        "--context-trajectory-diagnostic",
        action="store_true",
        help="Trace read-only source-column progress across context stages.",
    )
    parser.add_argument(
        "--context-trajectory-level",
        choices=tuple(sorted(CONTEXT_TRAJECTORY_LEVELS)),
        default="summary",
    )
    parser.add_argument(
        "--segment-reinforcement-diagnostic",
        action="store_true",
        help="Trace selected/reinforced segments for offline analysis only.",
    )
    parser.add_argument(
        "--segment-reinforcement-level",
        choices=tuple(sorted(SEGMENT_REINFORCEMENT_LEVELS)),
        default="summary",
    )
    parser.add_argument(
        "--independent-reference-diagnostic",
        action="store_true",
        help="Capture pre-matching independent segment references for offline analysis.",
    )
    parser.add_argument(
        "--independent-reference-level",
        choices=("summary", "event", "segment"),
        default="summary",
    )
    parser.add_argument(
        "--independent-reference-compress",
        action="store_true",
        help="Write independent reference traces as CSV.GZ.",
    )
    parser.add_argument(
        "--actual-branch-provenance-diagnostic",
        action="store_true",
        help="Capture actual-history branch continuity for offline analysis.",
    )
    parser.add_argument(
        "--actual-branch-provenance-level",
        choices=("summary", "event", "segment"),
        default="summary",
    )
    parser.add_argument(
        "--actual-branch-provenance-compress",
        action="store_true",
        help="Write actual branch traces as CSV.GZ.",
    )
    parser.add_argument("--interval-every", type=int, default=0)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Flush resource and model-size progress every N records.",
    )
    parser.add_argument(
        "--stream-diagnostic-traces",
        action="store_true",
        help="Append the largest read-only traces instead of retaining them in RAM.",
    )
    parser.add_argument(
        "--context-trajectory-uncompressed",
        action="store_true",
        help="Debug-only output switch; does not alter model execution.",
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-output", default="")
    parser.add_argument("--profile-dir", default="results/fig9_strict/profiling")
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--checkpoint-at-index", type=int, default=0)
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--stop-after-record", type=int, default=0)
    parser.add_argument("--stop-after-index", type=int, default=0)
    parser.add_argument("--debug-record-index", type=int, default=-1)
    parser.add_argument("--debug-output-json", default="")
    parser.add_argument(
        "--debug-end-index",
        type=int,
        default=0,
        help="Isolation-only stop index; may advance at most 10 records.",
    )
    parser.add_argument(
        "--continuous-impl",
        choices=("reference", "optimized_v1", "optimized_v2"),
        default="reference",
        help="Implementation-only continuous integrator switch for strict A/B.",
    )
    parser.add_argument("--april-branch", action="store_true")
    parser.add_argument("--branch-records", type=int, default=2016)
    parser.add_argument(
        "--streams",
        nargs="+",
        choices=("original", "perturbed"),
        default=("original", "perturbed"),
    )
    return parser.parse_args()


def run_april_branch(args: argparse.Namespace, config: Fig9StrictConfig) -> None:
    output_dir = Path(args.output_dir)
    original_records = read_records(Path(args.data), args.limit)
    perturbed_records = read_records(Path(args.perturbed_data), args.limit)
    split_index = find_timestamp_split(original_records)
    checkpoint = output_dir / "april1_shared_prefix.pkl"
    prefix_summary = run_strict_stream(
        records=original_records,
        data_path=Path(args.data),
        stream_label="shared_prefix_original",
        output_dir=output_dir / "april_branch_prefix",
        config=config,
        limit=args.limit,
        print_fingerprint=False,
        checkpoint_path=checkpoint,
        checkpoint_at_index=split_index,
        stop_after_index=split_index,
    )
    stop_after = min(split_index + args.branch_records, len(original_records) - config.horizon)
    original_summary = run_strict_stream(
        records=original_records,
        data_path=Path(args.data),
        stream_label="original_branch",
        output_dir=output_dir / "april_branch_original",
        config=config,
        limit=args.limit,
        print_fingerprint=False,
        resume_checkpoint=checkpoint,
        stop_after_index=stop_after,
        density_trace_path=output_dir / "april_branch_original_density.csv",
        density_summary_path=output_dir / "april_branch_original_density.json",
        interval_every=args.interval_every,
    )
    perturbed_summary = run_strict_stream(
        records=perturbed_records,
        data_path=Path(args.perturbed_data),
        stream_label="perturbed_branch",
        output_dir=output_dir / "april_branch_perturbed",
        config=config,
        limit=args.limit,
        print_fingerprint=False,
        resume_checkpoint=checkpoint,
        stop_after_index=stop_after,
        density_trace_path=output_dir / "april_branch_perturbed_density.csv",
        density_summary_path=output_dir / "april_branch_perturbed_density.json",
        interval_every=args.interval_every,
    )
    write_json(
        output_dir / "april_branch_summary.json",
        {
            "split_index": split_index,
            "split_timestamp": original_records[split_index].timestamp.isoformat(sep=" "),
            "branch_records": stop_after - split_index,
            "prefix_summary": prefix_summary,
            "original_summary": original_summary,
            "perturbed_summary": perturbed_summary,
        },
    )


def run_main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    install_native_crash_logging(output_dir)
    config = Fig9StrictConfig(
        horizon=args.prediction_horizon,
        warmup=args.warmup,
        seed=args.tie_break_seed,
        l_match=args.l_match,
        continuous_prediction_impl=args.continuous_impl,
    )
    if config.l_match != Fig9StrictConfig().l_match and not args.lmatch_real_ablation:
        raise ValueError(
            "non-default L_match requires --lmatch-real-ablation"
        )
    competition = CompetitionSettings(
        mode=args.competition_mode,
        inhibition_strength=args.inhibition_strength,
        inhibition_tau=args.inhibition_tau,
        simultaneous_tolerance=args.simultaneous_tolerance,
        simultaneous_policy=args.simultaneous_policy,
        simultaneous_bin_width=args.simultaneous_bin_width,
    )
    competition.validate()
    if args.april_branch:
        if competition.enabled:
            raise ValueError(
                "competitive_raw is not supported by the April branch runner"
            )
        run_april_branch(args, config)
        return
    runtime: dict[str, object] = {
        "started_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "limit": args.limit,
        "warmup": args.warmup,
        "prediction_horizon": config.horizon,
        "tie_break_seed": config.seed,
        "streams": list(args.streams),
        "continuous_impl": config.continuous_prediction_impl,
        "continuous_impl_version": continuous_impl_version(
            config.continuous_prediction_impl
        ),
        "L_match": config.l_match,
    }
    if args.lmatch_real_ablation:
        runtime.update(lmatch_ablation_markers(config.l_match))
    if competition.enabled:
        runtime.update(
            {
                "diagnostic_only": True,
                "competition_is_local_choice": True,
                "competition": asdict(competition),
            }
        )
    summaries: dict[str, object] = {}
    pairwise_diagnostic = args.pairwise_context_diagnostic
    temporal_diagnostic = args.temporal_context_diagnostic
    autonomous_context_provenance_diagnostic = (
        args.autonomous_context_provenance_diagnostic
    )
    oracle_diagnostic = (
        args.oracle_candidate_diagnostic
        or pairwise_diagnostic
        or temporal_diagnostic
        or autonomous_context_provenance_diagnostic
    )
    branch_diagnostic = (
        args.branch_provenance_diagnostic
        or pairwise_diagnostic
    )
    composition_diagnostic = (
        args.segment_context_composition_diagnostic or pairwise_diagnostic
    )
    composition_level = (
        "source" if pairwise_diagnostic else args.segment_context_composition_level
    )
    unified_context_oracle = (
        args.context_oracle_unified_trace or pairwise_diagnostic
    )
    runtime["pairwise_context_diagnostic"] = pairwise_diagnostic
    runtime["temporal_context_diagnostic"] = temporal_diagnostic
    runtime["autonomous_context_provenance_diagnostic"] = (
        autonomous_context_provenance_diagnostic
    )
    for stream in args.streams:
        data_path = Path(args.data if stream == "original" else args.perturbed_data)
        records = read_records(data_path, args.limit)
        summaries[stream] = run_strict_stream(
            records=records,
            data_path=data_path,
            stream_label=stream,
            output_dir=output_dir,
            config=config,
            limit=args.limit,
            density_trace_path=(
                Path(args.density_trace_csv)
                if args.density_trace_csv and len(args.streams) == 1
                else (
                    output_dir / f"{stream}_density_trace.csv"
                    if args.density_trace or args.density_trace_csv
                    else None
                )
            ),
            density_summary_path=(
                Path(args.density_summary_json)
                if args.density_summary_json and len(args.streams) == 1
                else (
                    output_dir / f"{stream}_density_summary.json"
                    if args.density_trace or args.density_summary_json
                    else None
                )
            ),
            interval_every=args.interval_every,
            checkpoint_path=Path(args.checkpoint_path) if args.checkpoint_path else None,
            checkpoint_at_index=(
                args.checkpoint_at_index
                if args.checkpoint_at_index > 0
                else None
            ),
            checkpoint_every=args.checkpoint_every,
            resume_checkpoint=(
                Path(args.resume_from or args.resume_checkpoint)
                if (args.resume_from or args.resume_checkpoint)
                else None
            ),
            stop_after_index=(
                args.stop_after_record
                if args.stop_after_record > 0
                else (
                    args.stop_after_index if args.stop_after_index > 0 else None
                )
            ),
            debug_record_index=(
                args.debug_record_index
                if args.debug_record_index >= 0
                else None
            ),
            debug_output_json=(
                Path(args.debug_output_json) if args.debug_output_json else None
            ),
            competition_settings=competition,
            oracle_candidate_diagnostic=oracle_diagnostic,
            candidate_separability_trace=args.candidate_separability_trace,
            branch_provenance_diagnostic=branch_diagnostic,
            branch_provenance_level=args.branch_provenance_level,
            branch_source_top_n=(
                args.branch_source_top_n
                if args.branch_source_top_n > 0
                else None
            ),
            preselection_segment_diagnostic=(
                args.preselection_segment_diagnostic
            ),
            preselection_segment_level=args.preselection_segment_level,
            preselection_segment_compress=(
                args.preselection_segment_compress
            ),
            preselection_max_rows=(
                args.preselection_max_rows
                if args.preselection_max_rows > 0
                else None
            ),
            teacher_forced_winner_diagnostic=(
                args.teacher_forced_winner_diagnostic
            ),
            teacher_forced_winner_level=args.teacher_forced_winner_level,
            observe_scenario_diagnostic=args.observe_scenario_diagnostic,
            observe_scenario_level=args.observe_scenario_level,
            reference_neuron_selection_diagnostic=(
                args.reference_neuron_selection_diagnostic
            ),
            reference_neuron_selection_level=(
                args.reference_neuron_selection_level
            ),
            intracolumn_selection_policy=args.intracolumn_selection_policy,
            intracolumn_selection_diagnostic=(
                args.intracolumn_selection_diagnostic
            ),
            match_overlap_diagnostic=args.match_overlap_diagnostic,
            match_overlap_level=args.match_overlap_level,
            match_overlap_compress=args.match_overlap_compress,
            timing_eligibility_decomposition=(
                args.timing_eligibility_decomposition
            ),
            source_trace_filter=(
                SourceTraceFilter(
                    field=args.source_trace_filter_field or None,
                    scenario=args.source_trace_filter_scenario or None,
                    record_start=args.source_trace_filter_record_start,
                    record_end=args.source_trace_filter_record_end,
                    loss_only=args.source_trace_filter_loss_only,
                    existing_segments_only=(
                        args.source_trace_filter_segments_existing_only
                    ),
                )
                if (
                    args.source_trace_filter_field
                    or args.source_trace_filter_scenario
                    or args.source_trace_filter_record_start is not None
                    or args.source_trace_filter_record_end is not None
                    or args.source_trace_filter_loss_only
                    or args.source_trace_filter_segments_existing_only
                )
                else None
            ),
            context_trajectory_diagnostic=(
                args.context_trajectory_diagnostic
            ),
            context_trajectory_level=args.context_trajectory_level,
            temporal_context_diagnostic=temporal_diagnostic,
            temporal_context_level=args.temporal_context_level,
            autonomous_context_provenance_diagnostic=(
                autonomous_context_provenance_diagnostic
            ),
            autonomous_context_provenance_level=(
                args.autonomous_context_provenance_level
            ),
            lmatch_real_ablation=args.lmatch_real_ablation,
            progress_every=args.progress_every,
            stream_diagnostic_traces=args.stream_diagnostic_traces,
            debug_end_index=(
                args.debug_end_index if args.debug_end_index > 0 else None
            ),
            context_trajectory_compress=(
                not args.context_trajectory_uncompressed
            ),
            segment_reinforcement_diagnostic=(
                args.segment_reinforcement_diagnostic
            ),
            segment_reinforcement_level=args.segment_reinforcement_level,
            independent_reference_diagnostic=args.independent_reference_diagnostic,
            independent_reference_level=args.independent_reference_level,
            independent_reference_compress=args.independent_reference_compress,
            actual_branch_provenance_diagnostic=args.actual_branch_provenance_diagnostic,
            actual_branch_provenance_level=args.actual_branch_provenance_level,
            actual_branch_provenance_compress=args.actual_branch_provenance_compress,
            segment_context_composition_diagnostic=composition_diagnostic,
            segment_context_composition_level=composition_level,
            segment_context_composition_compress=(
                args.segment_context_composition_compress
            ),
            context_oracle_unified_trace=unified_context_oracle,
        )
    if {"original", "perturbed"}.issubset(args.streams):
        runtime["pre_change_comparison"] = compare_pre_change_predictions(
            output_dir / "original_predictions.csv",
            output_dir / "perturbed_predictions.csv",
        )
    runtime["summaries"] = summaries
    runtime["finished_at"] = datetime.now().isoformat(sep=" ", timespec="seconds")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "runtime.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if "original" in args.streams:
        source = output_dir / "original_protocol.json"
        if source.exists():
            (output_dir / "protocol.json").write_text(
                source.read_text(encoding="utf-8"),
                encoding="utf-8",
            )


def main() -> None:
    args = parse_args()
    if not args.profile:
        run_main(args)
        return
    profile_path = (
        Path(args.profile_output)
        if args.profile_output
        else Path(args.profile_dir) / "fig9_strict_profile.txt"
    )
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        run_main(args)
    finally:
        profiler.disable()
    profiler.dump_stats(str(profile_path.with_suffix(".pstats")))
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumtime")
    stats.print_stats(40)
    profile_path.write_text(
        stream.getvalue(),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
