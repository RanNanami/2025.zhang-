"""Read-only Fig.9 paper-to-code parity audit primitives.

This module deliberately does not run or mutate ``SequentialMemory``.  It
turns paper evidence, repository data, and the current strict configuration
into deterministic records that can be tested and rendered by the diagnostic
entry point.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from experiments.common.hashing import sha256_file as file_sha256
from experiments.fig9.data import TaxiRecord, read_records, record_values
from experiments.fig9.metrics import mape
from experiments.fig9_strict_reproduction import (
    Fig9StrictConfig,
    build_fig9_encoder,
)
from seqmem.dynamics import DSDynamicsParams
from seqmem.encoding import SpikeEvent, SymbolCode
from seqmem.model import MemoryParams


PAPER_PDF_NAME = (
    "Zhang et al. - 2025 - Toward Building Human-Like Sequential Memory "
    "Using Brain-Inspired Spiking Neural Models.pdf"
)
PAPER_FIG9_APPROX_MAPE = 0.10
PAPER_FIG9_DIGITIZATION_UNCERTAINTY = 0.01
PAPER_CHANGE_DATE = datetime(2015, 4, 1)
CURRENT_DATA_RELATIVE = Path("data/paper_nyc_taxi.csv")
CURRENT_PERTURBED_RELATIVE = Path("data/paper_nyc_taxi_perturb.csv")


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def stable_sha256(value: object) -> str:
    text = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def paper_protocol_fingerprint() -> dict[str, object]:
    """Return only paper-supported values; unspecified values stay ``None``."""

    protocol = {
        "data_source": "NYC Taxi & Limousine Commission trip-record source [66]",
        "author_processed_dataset_publicly_identified": False,
        "records": 17520,
        "duration_days": 365,
        "sampling_interval_minutes": 30,
        "exact_start_timestamp": None,
        "exact_end_timestamp": None,
        "timezone": None,
        "passenger_aggregation_rule": None,
        "missing_bin_treatment": None,
        "fields": ["day_of_week", "time_of_day", "number_of_passengers"],
        "weekday_encoding": "periodic SSTD Gaussian ring",
        "time_encoding": "periodic SSTD Gaussian ring",
        "passenger_encoding": "real-value SSTD Gaussian population code",
        "weekday_columns": 30,
        "time_columns": 58,
        "passenger_columns": 482,
        "total_columns": 570,
        "active_columns_per_field": 10,
        "neurons_per_column": 32,
        "passenger_min": None,
        "passenger_max": None,
        "gaussian_interval_l": None,
        "gaussian_sigma": None,
        "spike_times": "evenly distributed in first half oscillation cycle",
        "decoder": "most likely prediction; numerical rule not specified",
        "horizon_records": 5,
        "horizon_hours": 2.5,
        "rollout_cycles_for_horizon": None,
        "metric_name": "MAPE",
        "metric_formula": None,
        "warmup_records": None,
        "evaluation_range": None,
        "rolling_window": None,
        "l_match": 4,
        "forgetting_threshold": 65,
        "initial_weight": 0.5,
        "correct_weight_delta": 0.1,
        "incorrect_weight_delta": 0.01,
        "l_weight": 10,
        "l_age": 1,
        "v_dep": 0.5,
        "oscillation_amplitude": 0.5,
        "dendrite_threshold": 1.0,
        "soma_threshold": 1.0,
        "tau_m": None,
        "tau_s": None,
        "integration_step": None,
        "timing_tolerance": None,
        "random_seed": None,
        "modified_after": "2015-04-01",
        "modified_weekday_morning": "07:00-11:00, -20%",
        "modified_weekday_evening": "21:00-23:00, +20%",
    }
    evidence = {
        "data_source": {
            "level": "EXPLICIT_IN_PAPER",
            "page": 10,
            "location": "Section III-D and reference [66] on page 12",
        },
        "records": {
            "level": "EXPLICIT_IN_PAPER",
            "page": 10,
            "location": "Section III-D",
        },
        "encoding_topology": {
            "level": "EXPLICIT_IN_PAPER",
            "page": 10,
            "location": "Section III-D",
        },
        "encoding_algorithm": {
            "level": "EXPLICIT_IN_PAPER",
            "page": 5,
            "location": "Section II-C and Fig. 3",
        },
        "network_parameters": {
            "level": "EXPLICIT_IN_PAPER",
            "page": 8,
            "location": "Table I and Section III-A",
        },
        "l_match_and_forgetting": {
            "level": "EXPLICIT_IN_PAPER",
            "page": 8,
            "location": "Section III-A paragraph below Table I",
        },
        "prediction_horizon": {
            "level": "EXPLICIT_IN_PAPER",
            "page": 10,
            "location": "Section III-D",
        },
        "autonomous_retrieval": {
            "level": "IMPLIED_BY_METHOD",
            "page": 7,
            "location": "Section II-E/F and Fig. 5",
        },
        "decoder": {
            "level": "NOT_SPECIFIED_IN_PAPER",
            "page": 10,
            "location": "Only 'most likely prediction results' is stated",
        },
        "evaluation": {
            "level": "NOT_SPECIFIED_IN_PAPER",
            "page": 10,
            "location": "MAPE named; warmup/range/formula omitted",
        },
    }
    return {
        "label": "paper_explicit_and_method_implied_fig9_protocol",
        "protocol": protocol,
        "evidence": evidence,
        "fingerprint_sha256": stable_sha256(protocol),
    }


def current_protocol_fingerprint(root: Path) -> dict[str, object]:
    config = Fig9StrictConfig()
    encoder = build_fig9_encoder(config)
    passenger = encoder.encoders[2]
    params = MemoryParams(
        l_match=config.l_match,
        forgetting_threshold=config.forgetting_threshold,
        response_scale=config.response_scale,
        continuous_dynamics=config.continuous_dynamics,
        integration_step=config.integration_step,
        burst_context=config.burst_context,
        intracolumn_inhibition=config.intracolumn_inhibition,
        continuous_prediction_impl=config.continuous_prediction_impl,
    )
    dynamics = params.dynamics()
    data_path = root / CURRENT_DATA_RELATIVE
    records = read_records(data_path, 0)
    protocol = {
        "data_path": str(CURRENT_DATA_RELATIVE).replace("\\", "/"),
        "data_sha256": file_sha256(data_path),
        "records": len(records),
        "first_timestamp": records[0].timestamp.isoformat(sep=" "),
        "last_timestamp": records[-1].timestamp.isoformat(sep=" "),
        "sampling_interval_minutes": 30,
        "data_provenance": "byte-identical to bundled Numenta [58] reference CSV",
        "fields_used": ["python_weekday", "half_hour_slot", "passenger_count"],
        "weekday_columns": config.weekday_columns,
        "time_columns": config.time_columns,
        "passenger_columns": config.passenger_columns,
        "total_columns": encoder.num_columns,
        "active_columns_per_field": config.k,
        "neurons_per_column": config.neurons_per_column,
        "passenger_min": config.passenger_min,
        "passenger_max": config.passenger_max,
        "passenger_spacing": passenger.spacing,
        "passenger_sigma": passenger.sigma,
        "passenger_likelihood_grid_size": len(passenger.likelihood_grid()),
        "spike_times": list(passenger.event_times),
        "decoder": "maximum (timed overlap, column overlap), average tied values",
        "timing_tolerance": params.timing_tolerance,
        "horizon_records": config.horizon,
        "prediction_before_current_observation": True,
        "rollout": "five raw autonomous predict/advance cycles",
        "metric": "sum absolute error / sum absolute target (WAPE-like [58] MAPE)",
        "warmup_records_default": config.warmup,
        "diagnostic_250_500_warmup": 200,
        "rolling_window": config.rolling_window,
        "l_match": config.l_match,
        "forgetting_threshold": config.forgetting_threshold,
        "initial_weight": params.w0,
        "correct_weight_delta": params.delta_w,
        "incorrect_weight_delta": params.delta_w_bad,
        "l_weight": params.l_weight,
        "l_age": params.l_age,
        "v_dep": dynamics.v_dep,
        "oscillation_amplitude": dynamics.oscillation_amplitude,
        "dendrite_threshold": params.dendrite_threshold,
        "soma_threshold": dynamics.soma_threshold,
        "tau_m": params.tau_m,
        "tau_s": params.tau_s,
        "response_scale": params.response_scale,
        "integration_step": params.integration_step,
        "burst_context": "all-cell",
        "intracolumn_inhibition": params.intracolumn_inhibition,
        "intercolumn_competition": "off by strict default",
        "propagation_mode": config.propagation_mode,
        "use_future_covariates": config.use_future_covariates,
        "reencode_decoded_value": config.reencode_decoded_value,
        "rollout_learning": config.rollout_learning,
        "tie_break_seed": config.seed,
        "continuous_impl": config.continuous_prediction_impl,
    }
    locations = {
        "data_path": "experiments/fig9_strict_reproduction.py:5349",
        "configuration": "experiments/fig9_strict_reproduction.py:240",
        "encoder": "experiments/fig9_strict_reproduction.py:1454",
        "prediction_loop": "experiments/fig9_strict_reproduction.py:3394",
        "decoder": "src/seqmem/encoding.py:142",
        "metric": "experiments/fig9/metrics.py:16",
        "learning": "src/seqmem/model.py:2031",
    }
    return {
        "label": "current_default_fig9_strict_interpretation",
        "config": asdict(config),
        "protocol": protocol,
        "code_locations": locations,
        "fingerprint_sha256": stable_sha256(protocol),
    }


def dataset_profile(
    path: Path,
    *,
    display_path: str,
    source: str,
    generation_script: str,
    raw_or_derived: str,
) -> dict[str, object]:
    records = read_records(path, 0)
    intervals = [
        int((right.timestamp - left.timestamp).total_seconds() // 60)
        for left, right in zip(records, records[1:])
    ]
    timestamps = [record.timestamp for record in records]
    duplicate_count = len(timestamps) - len(set(timestamps))
    missing_intervals = sum(max(0, interval // 30 - 1) for interval in intervals)
    values = [record.value for record in records]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    with path.open("r", encoding="utf-8", newline="") as handle:
        columns = next(csv.reader(handle))
    return {
        "file_path": display_path,
        "sha256": file_sha256(path),
        "row_count": len(records),
        "first_timestamp": records[0].timestamp.isoformat(sep=" "),
        "last_timestamp": records[-1].timestamp.isoformat(sep=" "),
        "interval_minutes": Counter(intervals).most_common(1)[0][0],
        "columns": "|".join(columns),
        "missing_timestamps": missing_intervals,
        "duplicates": duplicate_count,
        "value_min": min(values),
        "value_max": max(values),
        "value_mean": mean,
        "value_std_population": math.sqrt(variance),
        "source": source,
        "generation_script": generation_script,
        "raw_or_derived": raw_or_derived,
        "continuous_30_minute_grid": set(intervals) == {30},
    }


def dataset_inventory(root: Path) -> list[dict[str, object]]:
    specs = (
        (
            CURRENT_DATA_RELATIVE,
            "Bundled Numenta [58] continuous-sequence reference asset; underlying TLC relationship plausible",
            "No repository generator; introduced in initial commit c195603",
            "processed half-hour aggregate",
        ),
        (
            CURRENT_PERTURBED_RELATIVE,
            "Numenta [58] perturbation asset copied into repository",
            ".deps/generatePerturbedNYCtaxiData.py",
            "derived from paper_nyc_taxi.csv",
        ),
        (
            Path("data/nyc_taxi.csv"),
            "NAB-style seven-month prefix of the bundled one-year aggregate",
            "No repository generator",
            "processed half-hour aggregate subset",
        ),
        (
            Path(".deps/reference_nyc_taxi.csv"),
            "Bundled Numenta [58] reference asset",
            "External reference artifact",
            "processed half-hour aggregate",
        ),
        (
            Path(".deps/reference_nyc_taxi_perturb.csv"),
            "Bundled Numenta [58] perturbed reference asset",
            ".deps/generatePerturbedNYCtaxiData.py",
            "derived processed aggregate",
        ),
    )
    return [
        dataset_profile(
            root / relative,
            display_path=str(relative).replace("\\", "/"),
            source=source,
            generation_script=generator,
            raw_or_derived=kind,
        )
        for relative, source, generator, kind in specs
        if (root / relative).exists()
    ]


def _event_text(code: SymbolCode) -> tuple[str, str]:
    return (
        " ".join(str(event.column) for event in code.events),
        " ".join(f"{event.time:.12g}" for event in code.events),
    )


def encoding_fixture_rows(root: Path) -> list[dict[str, object]]:
    records = read_records(root / CURRENT_DATA_RELATIVE, 0)
    values = sorted(record.value for record in records)
    config = Fig9StrictConfig()
    encoder = build_fig9_encoder(config)
    fixtures: list[tuple[str, str, float]] = [
        *(('weekday', f'weekday_{value}', float(value)) for value in (0, 1, 6)),
        *(('time', f'time_{label}', float(value)) for label, value in (
            ('0000', 0), ('0030', 1), ('0700', 14), ('1200', 24), ('2330', 47)
        )),
        *(('passenger', label, float(value)) for label, value in (
            ('dataset_min', values[0]),
            ('dataset_q25', values[len(values) // 4]),
            ('dataset_median', values[len(values) // 2]),
            ('dataset_q75', values[3 * len(values) // 4]),
            ('dataset_max', values[-1]),
            ('neighbor_left', 15000.0),
            ('neighbor_right', 15001.0),
        )),
    ]
    field_index = {'weekday': 0, 'time': 1, 'passenger': 2}
    rows = []
    for field, label, value in fixtures:
        code = encoder.encoders[field_index[field]].encode(value)  # type: ignore[attr-defined]
        columns, times = _event_text(code)
        rows.append(
            {
                'field': field,
                'fixture': label,
                'input_value': value,
                'event_count': len(code.events),
                'active_columns_in_firing_order': columns,
                'spike_times': times,
                'code_sha256': stable_sha256([(event.column, event.time) for event in code.events]),
            }
        )
    return rows


def ideal_roundtrip_rows(root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    records = read_records(root / CURRENT_DATA_RELATIVE, 0)
    passenger = build_fig9_encoder(Fig9StrictConfig()).encoders[2]
    decode_cache: dict[tuple[SpikeEvent, ...], float] = {}
    rows: list[dict[str, object]] = []
    for index, record in enumerate(records):
        code = passenger.encode(record.value)  # type: ignore[attr-defined]
        decoded = decode_cache.get(code.events)
        if decoded is None:
            decoded = passenger.decode_likelihood(code)  # type: ignore[attr-defined]
            decode_cache[code.events] = decoded
        error = decoded - record.value
        rows.append(
            {
                'record_index': index,
                'timestamp': record.timestamp.isoformat(sep=' '),
                'actual': record.value,
                'decoded': decoded,
                'signed_error': error,
                'absolute_error': abs(error),
                'absolute_percentage_error': abs(error) / abs(record.value) if record.value else '',
                'code_sha256': stable_sha256([(event.column, event.time) for event in code.events]),
            }
        )
    predictions = [float(row['decoded']) for row in rows]
    targets = [float(row['actual']) for row in rows]
    errors = [float(row['signed_error']) for row in rows]
    absolute = [abs(error) for error in errors]
    summary = {
        'records': len(rows),
        'unique_source_values': len(set(targets)),
        'unique_ordered_codes': len(decode_cache),
        'ideal_roundtrip_mape_repository_formula': mape(predictions, targets),
        'ideal_roundtrip_pointwise_mape': sum(
            abs(prediction - target) / abs(target)
            for prediction, target in zip(predictions, targets)
            if target
        ) / sum(bool(target) for target in targets),
        'mae': sum(absolute) / len(absolute),
        'max_absolute_error': max(absolute),
        'mean_signed_bias': sum(errors) / len(errors),
    }
    return rows, summary


def decoder_partial_pattern_rows() -> list[dict[str, object]]:
    passenger = build_fig9_encoder(Fig9StrictConfig()).encoders[2]
    rows: list[dict[str, object]] = []
    for value in (0.0, 10000.0, 15000.0, 20000.0, 30000.0, 40000.0):
        original = passenger.encode(value)  # type: ignore[attr-defined]
        for retained in range(10, 0, -1):
            partial = SymbolCode(original.events[:retained])
            decoded = passenger.decode_likelihood(partial)  # type: ignore[attr-defined]
            rows.append(
                {
                    'input_value': value,
                    'variant': f'first_{retained}_of_10_events',
                    'retained_events': retained,
                    'wrong_columns': 0,
                    'order_reversed': False,
                    'decoded_value': decoded,
                    'absolute_error': abs(decoded - value),
                }
            )
        reversed_code = SymbolCode(
            tuple(
                SpikeEvent(event.column, target_time)
                for event, target_time in zip(reversed(original.events), passenger.event_times)  # type: ignore[attr-defined]
            )
        )
        decoded = passenger.decode_likelihood(reversed_code)  # type: ignore[attr-defined]
        rows.append(
            {
                'input_value': value,
                'variant': 'same_columns_reversed_order',
                'retained_events': 10,
                'wrong_columns': 0,
                'order_reversed': True,
                'decoded_value': decoded,
                'absolute_error': abs(decoded - value),
            }
        )
        for wrong in (1, 2):
            events = list(original.events)
            used = {event.column for event in events}
            replacements = [
                column for column in range(88, 570) if column not in used
            ][:wrong]
            for offset, replacement in enumerate(replacements, start=1):
                target = len(events) - offset
                events[target] = SpikeEvent(replacement, events[target].time)
            changed = SymbolCode(tuple(events))
            decoded = passenger.decode_likelihood(changed)  # type: ignore[attr-defined]
            rows.append(
                {
                    'input_value': value,
                    'variant': f'{wrong}_wrong_columns',
                    'retained_events': 10,
                    'wrong_columns': wrong,
                    'order_reversed': False,
                    'decoded_value': decoded,
                    'absolute_error': abs(decoded - value),
                }
            )
    return rows


def prediction_alignment(records: Sequence[TaxiRecord], anchor: int = 200) -> dict[str, object]:
    horizon = Fig9StrictConfig().horizon
    if anchor <= 0 or anchor + horizon >= len(records):
        raise ValueError('anchor must have a previous record and a complete horizon')
    current = records[anchor]
    last_observed = records[anchor - 1]
    compared_target = records[anchor + horizon]
    semantic_steps = [
        {
            'rollout_step': step,
            'causal_continuation_index': anchor + step - 1,
            'causal_continuation_timestamp': records[anchor + step - 1].timestamp.isoformat(sep=' '),
        }
        for step in range(1, horizon + 1)
    ]
    reported_delta = compared_target.timestamp - current.timestamp
    causal_delta = compared_target.timestamp - last_observed.timestamp
    expected_step5_index = semantic_steps[-1]['causal_continuation_index']
    return {
        'anchor_zero_based_index': anchor,
        'output_input_index_one_based': anchor + 1,
        'reported_input_timestamp': current.timestamp.isoformat(sep=' '),
        'last_observed_zero_based_index_before_rollout': anchor - 1,
        'last_observed_timestamp_before_rollout': last_observed.timestamp.isoformat(sep=' '),
        'current_record_observed_before_rollout': False,
        'rollout_steps': semantic_steps,
        'compared_target_zero_based_index': anchor + horizon,
        'compared_target_timestamp': compared_target.timestamp.isoformat(sep=' '),
        'reported_anchor_to_target_hours': reported_delta.total_seconds() / 3600,
        'last_observed_to_target_hours': causal_delta.total_seconds() / 3600,
        'causal_step5_expected_zero_based_index': expected_step5_index,
        'causal_step5_expected_timestamp': records[int(expected_step5_index)].timestamp.isoformat(sep=' '),
        'target_index_offset_from_causal_step5': anchor + horizon - int(expected_step5_index),
        'off_by_one_detected': int(expected_step5_index) != anchor + horizon,
        'code_evidence': [
            'experiments/fig9_strict_reproduction.py:3394 loop index',
            'experiments/fig9_strict_reproduction.py:3410 target index+horizon',
            'experiments/fig9_strict_reproduction.py:3418 rollout before observe',
            'experiments/fig9_strict_reproduction.py:3659 observe current record',
        ],
    }


def evaluation_window_rows(root: Path) -> list[dict[str, object]]:
    path = root / 'results/fig9_strict/diagnostic_500_20260727_205411/original_predictions.csv'
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    output = []
    for start in (200, 250, 300, 400):
        scoped = [
            row for row in rows
            if row.get('prediction') not in {'', None} and int(row['input_index']) >= start
        ]
        predictions = [float(row['prediction']) for row in scoped]
        targets = [float(row['target']) for row in scoped]
        pointwise = [
            abs(prediction - target) / abs(target)
            for prediction, target in zip(predictions, targets)
            if target
        ]
        output.append(
            {
                'evaluation_start_input_index': start,
                'prediction_count': len(scoped),
                'repository_mape': mape(predictions, targets),
                'conventional_pointwise_mape': sum(pointwise) / len(pointwise),
                'source': str(path.relative_to(root)).replace('\\', '/'),
                'diagnostic_only': True,
            }
        )
    return output


def classify_run_family(path: str) -> str:
    lowered = path.lower().replace('\\', '/')
    if any(token in lowered for token in ('etth1', 'weather', 'multivariate')):
        return 'ALTERNATIVE_MODEL'
    if any(token in lowered for token in ('context-table', 'sparse-knn', 'sparse_context')):
        return 'ENGINEERING_BASELINE'
    if 'historical_compensated' in lowered or 'generic' in lowered:
        return 'OLD_PROTOTYPE'
    if any(token in lowered for token in (
        'competition', 'lmatch', 'ambiguity', 'oracle', 'teacher', 'diagnostic'
    )):
        return 'DIAGNOSTIC_ABLATION'
    if 'fig9_strict' in lowered:
        return 'PAPER_STRICT_ATTEMPT'
    return 'OLD_PROTOTYPE'


def run_family_registry() -> list[dict[str, object]]:
    entries = (
        ('results/fig9_strict/stage_a_250', 'raw L4, 250 records, warmup 200', 0.5054500721931258),
        ('results/fig9_strict/diagnostic_500_20260727_205411', 'raw L4, 500 records, warmup 200', 0.5392795787736315),
        ('results/fig9_diagnostics/lmatch_stack_500_completion_20260810_resume2/500/STRICT_RAW_L4', 'raw L4 confirmation', 0.5392795787736315),
        ('results/fig9_diagnostics/l2_stack_component_500_20260810/500/P2_COMPETITION_ONLY_L2', 'nonpaper competition and L2', 0.48966059282732727),
        ('results/fig9_historical_compensated', 'winner/eventwise compensated historical implementation', ''),
        ('experiments/diagnostics/fig9_taxi_prediction.py', 'context-table/sparse diagnostic baseline', ''),
        ('experiments/diagnostics/etth1_paper_snn.py', 'ETTh1 transfer', ''),
        ('experiments/diagnostics/weather_paper_snn.py', 'Weather transfer', ''),
    )
    return [
        {
            'path': path,
            'family': classify_run_family(path),
            'description': description,
            'reported_mape_if_available': value,
            'directly_comparable_to_paper_fig9': False,
        }
        for path, description, value in entries
    ]


def parity_matrix_rows(root: Path) -> list[dict[str, object]]:
    """Build a deterministic, evidence-bearing matrix with over 100 checks."""

    paper = paper_protocol_fingerprint()['protocol']
    current = current_protocol_fingerprint(root)['protocol']
    rows: list[dict[str, object]] = []

    def add(
        category: str,
        item: str,
        paper_value: object,
        paper_evidence: str,
        page: object,
        current_value: object,
        location: str,
        status: str,
        severity: str,
        confidence: str,
        impact: str,
        action: str,
    ) -> None:
        rows.append(
            {
                'category': category,
                'item': item,
                'paper_value': json.dumps(_jsonable(paper_value), ensure_ascii=False) if isinstance(paper_value, (dict, list)) else paper_value,
                'paper_evidence': paper_evidence,
                'paper_page': page,
                'current_value': json.dumps(_jsonable(current_value), ensure_ascii=False) if isinstance(current_value, (dict, list)) else current_value,
                'code_location': location,
                'status': status,
                'severity': severity,
                'confidence': confidence,
                'likely_mape_impact': impact,
                'action': action,
            }
        )

    data_specs = [
        ('source agency', paper['data_source'], current['data_provenance'], 'PARTIAL', 'HIGH', 'Resolve author-processed dataset provenance'),
        ('record count', 17520, current['records'], 'MATCH', 'CRITICAL', 'Keep full-year candidate separate from 10,320-row subset'),
        ('duration days', 365, 365, 'MATCH', 'HIGH', 'None'),
        ('sampling interval minutes', 30, 30, 'MATCH', 'HIGH', 'None'),
        ('exact start timestamp', None, current['first_timestamp'], 'UNKNOWN', 'HIGH', 'Do not backfill paper year range'),
        ('exact end timestamp', None, current['last_timestamp'], 'UNKNOWN', 'HIGH', 'Do not backfill paper year range'),
        ('timezone', None, 'naive datetime', 'UNKNOWN', 'MEDIUM', 'Locate author preprocessing'),
        ('raw source granularity', 'TLC trip records', 'already aggregated bins', 'PARTIAL', 'HIGH', 'Document transformation provenance'),
        ('passenger aggregation formula', None, 'not present in repository', 'UNKNOWN', 'CRITICAL', 'Locate author aggregation or contact authors'),
        ('passenger semantic unit', 'number of passengers', 'aggregate named passenger_count', 'PARTIAL', 'HIGH', 'Distinguish SUM passenger_count from COUNT trips'),
        ('pickup/dropoff anchor', None, 'not recoverable from aggregate', 'UNKNOWN', 'HIGH', 'Locate generation code'),
        ('yellow/green/FHV scope', None, 'not recorded', 'UNKNOWN', 'HIGH', 'Locate generation code'),
        ('geographic scope', None, 'not recorded', 'UNKNOWN', 'HIGH', 'Locate generation code'),
        ('missing-bin treatment', None, 'no missing bins in finished CSV', 'UNKNOWN', 'MEDIUM', 'Locate generation code'),
        ('duplicate treatment', None, 'no duplicates in finished CSV', 'UNKNOWN', 'LOW', 'None'),
        ('DST treatment', None, 'fixed 30-minute naive grid', 'UNKNOWN', 'MEDIUM', 'Locate timezone/preprocessing'),
        ('normalization before encoding', None, 'none; clipped to 0..40000', 'UNKNOWN', 'HIGH', 'Treat as implementation assumption'),
        ('outlier treatment', None, 'clipping only in encoder', 'UNKNOWN', 'MEDIUM', 'Treat as implementation assumption'),
        ('data min', None, 8, 'UNKNOWN', 'MEDIUM', 'Do not treat observed min as paper bound'),
        ('data max', None, 39197, 'UNKNOWN', 'MEDIUM', 'Do not treat observed max as paper bound'),
        ('April 2015 included', 'logically required if same stream', True, 'MATCH', 'HIGH', 'Preserve logical/inferred label'),
        ('modified morning rule', 'weekday 07:00-11:00 -20%', 'same', 'MATCH', 'MEDIUM', 'None'),
        ('modified evening wording', 'weekday 21:00-23:00 +20%', 'generator applies 21:00 through 23:30 bins', 'PARTIAL', 'MEDIUM', 'Audit endpoint interpretation'),
        ('modified start index', None, 13152, 'UNKNOWN', 'LOW', 'Current value comes from [58], not Zhang text'),
        ('NAB 10,320 relationship', None, 'exact prefix of current full-year file', 'UNKNOWN', 'MEDIUM', 'Do not call seven-month NAB file paper data'),
        ('author code located', None, False, 'UNKNOWN', 'HIGH', 'Contact authors or locate supplement'),
    ]
    for item, pv, cv, status, severity, action in data_specs:
        add('DATA', item, pv, 'Section III-D or not specified', 10, cv, 'data/ and experiments/fig9/data.py:19', status, severity, 'HIGH', severity, action)

    encoding_specs = [
        ('weekday encoding type', 'periodic Gaussian ring', 'periodic Gaussian ring', 'MATCH', 'HIGH'),
        ('time encoding type', 'periodic Gaussian ring', 'periodic Gaussian ring', 'MATCH', 'HIGH'),
        ('passenger encoding type', 'real Gaussian population', 'real Gaussian population', 'MATCH', 'HIGH'),
        ('weekday columns', 30, 30, 'MATCH', 'CRITICAL'),
        ('time columns', 58, 58, 'MATCH', 'CRITICAL'),
        ('passenger columns', 482, 482, 'MATCH', 'CRITICAL'),
        ('total columns', 570, 570, 'MATCH', 'CRITICAL'),
        ('K per field', 10, 10, 'MATCH', 'CRITICAL'),
        ('record event count', 30, 30, 'MATCH', 'HIGH'),
        ('distinct event times per field', 10, 10, 'MATCH', 'HIGH'),
        ('spike order retained in code', True, True, 'MATCH', 'CRITICAL'),
        ('spike times first half-cycle', True, '0.0 through 0.5 inclusive', 'MATCH', 'HIGH'),
        ('spike-time spacing', 'even', '0.5/(K-1)', 'MATCH', 'HIGH'),
        ('real center interval l', None, current['passenger_spacing'], 'UNKNOWN', 'CRITICAL'),
        ('real Gaussian sigma', None, current['passenger_sigma'], 'UNKNOWN', 'CRITICAL'),
        ('passenger minimum bound', None, current['passenger_min'], 'UNKNOWN', 'HIGH'),
        ('passenger maximum bound', None, current['passenger_max'], 'UNKNOWN', 'HIGH'),
        ('482 derivation', None, 'hard-coded paper topology; spacing derived from 0..40000', 'UNKNOWN', 'HIGH'),
        ('Gaussian response normalization', None, 'standard exp distance/sigma', 'UNKNOWN', 'MEDIUM'),
        ('top-K ranking tie break', None, 'lower column index', 'UNKNOWN', 'LOW'),
        ('passenger clipping', None, 'clip to 0..40000', 'UNKNOWN', 'MEDIUM'),
        ('weekday convention', 'day of week', 'Python Monday=0', 'PARTIAL', 'MEDIUM'),
        ('time unit', 'time of day', 'half-hour slot 0..47', 'PARTIAL', 'MEDIUM'),
        ('weekday circular boundary', 'minimum/maximum adjacent', 'Sunday/Monday codes overlap', 'MATCH', 'HIGH'),
        ('time circular boundary', 'minimum/maximum adjacent', '23:30/00:00 codes overlap', 'MATCH', 'HIGH'),
        ('field column disjointness', 'three field populations', 'offsets 0/30/88', 'MATCH', 'HIGH'),
        ('encoding resolution', 'l/2', current['passenger_spacing'] / 2, 'PARTIAL', 'HIGH'),
        ('same columns different order capacity', True, True, 'MATCH', 'HIGH'),
    ]
    for item, pv, cv, status, severity in encoding_specs:
        add('ENCODING', item, pv, 'Section II-C and Fig. 3', 5, cv, 'src/seqmem/encoding.py; experiments/fig9_strict_reproduction.py:1454', status, severity, 'HIGH', severity, 'Keep assumption explicit' if status != 'MATCH' else 'None')

    network_specs = [
        ('neurons per column', 32, 32, 'MATCH', 'CRITICAL'),
        ('one-layer Fig9 topology', 'single memory layer implied', 'one SequentialMemory layer', 'MATCH', 'HIGH'),
        ('initial weight w0', 0.5, 0.5, 'MATCH', 'HIGH'),
        ('correct delta weight', 0.1, 0.1, 'MATCH', 'HIGH'),
        ('incorrect delta weight', 0.01, 0.01, 'MATCH', 'HIGH'),
        ('dendrite threshold', 1.0, 1.0, 'MATCH', 'HIGH'),
        ('soma threshold', 1.0, 1.0, 'MATCH', 'HIGH'),
        ('Vdep', 0.5, 0.5, 'MATCH', 'HIGH'),
        ('oscillation amplitude', 0.5, 0.5, 'MATCH', 'HIGH'),
        ('L_weight', 10, 10, 'MATCH', 'HIGH'),
        ('L_age', 1, 1, 'MATCH', 'HIGH'),
        ('L_match taxi', 4, 4, 'MATCH', 'CRITICAL'),
        ('forgetting threshold taxi', 65, 65, 'MATCH', 'CRITICAL'),
        ('tau_m', None, 0.1, 'UNKNOWN', 'CRITICAL'),
        ('tau_s', None, 0.02, 'UNKNOWN', 'CRITICAL'),
        ('response scale V0', None, 'normalized kernel (None)', 'UNKNOWN', 'CRITICAL'),
        ('integration step', None, 0.005, 'UNKNOWN', 'HIGH'),
        ('timing tolerance', None, 0.03, 'UNKNOWN', 'HIGH'),
        ('cycle duration normalization', None, 1.0, 'UNKNOWN', 'HIGH'),
        ('depolarization duration', 'half cycle', 0.5, 'MATCH', 'HIGH'),
        ('refractory duration', 'rest of current cycle', 'one normalized cycle', 'PARTIAL', 'MEDIUM'),
        ('dendrite reset after spike', True, 'first crossing modeled per prediction', 'PARTIAL', 'HIGH'),
        ('intracolumn inhibition', 'winner suppresses same column', 'first-spike selection', 'PARTIAL', 'CRITICAL'),
        ('intercolumn inhibition', 'selected input columns inhibit others', 'strict rollout competition off', 'PARTIAL', 'CRITICAL'),
        ('proximal membrane integration', 'Eq. 3 component', 'external code handled algorithmically', 'PARTIAL', 'HIGH'),
        ('apical input', 'general model component', 'not used in one-layer Fig9', 'PARTIAL', 'LOW'),
        ('continuous implementation', 'continuous DS dynamics', 'reference Python integrator', 'PARTIAL', 'HIGH'),
        ('candidate threshold crossing', 'Vd >= threshold', 'grid first crossing', 'PARTIAL', 'HIGH'),
        ('tie-break seed', None, 0, 'UNKNOWN', 'LOW'),
    ]
    for item, pv, cv, status, severity in network_specs:
        add('NETWORK', item, pv, 'Eq. 1-3, Table I, Section III-A', '4,8', cv, 'src/seqmem/dynamics.py; src/seqmem/model.py:375', status, severity, 'HIGH', severity, 'Mechanism diagnostic only after protocol alignment' if status != 'MATCH' else 'None')

    learning_specs = [
        ('empty initial network', True, True, 'MATCH', 'HIGH'),
        ('synapse weight state', True, True, 'MATCH', 'HIGH'),
        ('synapse delay state', True, True, 'MATCH', 'HIGH'),
        ('synapse age state', True, True, 'MATCH', 'HIGH'),
        ('delay immutable until deletion', True, True, 'MATCH', 'HIGH'),
        ('Scenario 1 predictive-cell gate', True, True, 'MATCH', 'CRITICAL'),
        ('Scenario 1 contributing strengthen', '+delta', '+0.1', 'MATCH', 'HIGH'),
        ('Scenario 1 contributing rejuvenate', 'age=0', 'age=0', 'MATCH', 'HIGH'),
        ('Scenario 1 noncontributing weaken', '-delta', '-0.1', 'MATCH', 'HIGH'),
        ('Scenario 1 noncontributing age', '+1', '+1', 'MATCH', 'HIGH'),
        ('Scenario 1 contribution definition', 'synapses causing predictive state', 'arrival-window strict default', 'PARTIAL', 'CRITICAL'),
        ('Scenario 2 no predictive neuron', True, True, 'MATCH', 'HIGH'),
        ('Scenario 2 L_match eligibility', 'active arrivals >= L_match', 'timed overlap >= 4', 'PARTIAL', 'HIGH'),
        ('Scenario 2 grow previous winners', True, True, 'MATCH', 'HIGH'),
        ('Scenario 3 new least-used neuron segment', True, True, 'MATCH', 'HIGH'),
        ('new segment source set', 'previous-cycle winners', 'previous_winners', 'MATCH', 'HIGH'),
        ('new delay equation', 'tj-ti-pi', 'normalized half-cycle alignment', 'PARTIAL', 'HIGH'),
        ('incorrect prediction decrement', '-delta prime', '-0.01', 'MATCH', 'HIGH'),
        ('incorrect prediction age', '+1', '+1', 'MATCH', 'HIGH'),
        ('forgetting formula', 'Lw(1-w)+La*age', 'same', 'MATCH', 'CRITICAL'),
        ('forget threshold comparison', 'reaches threshold', 'delete when score >= threshold', 'MATCH', 'HIGH'),
        ('burst active cells', 'all neurons share feedforward item before contextual selection', 'all-cell active; one winner', 'PARTIAL', 'CRITICAL'),
        ('winner tie break', None, 'seeded least-used tie break', 'UNKNOWN', 'MEDIUM'),
        ('online one-pass learning', 'online no train/test switch', 'one observe per record', 'MATCH', 'HIGH'),
        ('learning during rollout', 'prediction without proximal input', False, 'MATCH', 'CRITICAL'),
    ]
    for item, pv, cv, status, severity in learning_specs:
        add('LEARNING', item, pv, 'Section II-D and III-B/D', '6,8,10', cv, 'src/seqmem/model.py:2031,3291', status, severity, 'HIGH', severity, 'Keep strict defaults unchanged during audit')

    prediction_specs = [
        ('target quantity', 'passenger count', 'passenger decode', 'MATCH', 'CRITICAL'),
        ('horizon steps', 5, 5, 'MATCH', 'CRITICAL'),
        ('horizon wall time', '2.5 hours', 'reported input to target is 2.5 hours', 'MATCH', 'CRITICAL'),
        ('prediction state anchor', 'historical data through current anchor implied', 'state ends at anchor-1', 'MISMATCH', 'CRITICAL'),
        ('step1 causal meaning', 'next record after anchor', 'current unobserved record', 'MISMATCH', 'CRITICAL'),
        ('step5 causal index', 'anchor+5', 'raw step5 corresponds to anchor+4', 'MISMATCH', 'CRITICAL'),
        ('compared target index', 'anchor+5', 'anchor+5', 'MATCH', 'CRITICAL'),
        ('last observed to target duration', '2.5 hours expected', '3.0 hours', 'MISMATCH', 'CRITICAL'),
        ('autonomous retrieval', 'method implies predictive spikes propagate', 'raw predict/advance', 'MATCH', 'CRITICAL'),
        ('exact five-cycle mapping', None, 'five autonomous cycles', 'UNKNOWN', 'HIGH'),
        ('decoded proximal replay', 'not part of contextual retrieval', False, 'MATCH', 'CRITICAL'),
        ('future covariates', None, False, 'UNKNOWN', 'MEDIUM'),
        ('transient state restoration', None, True, 'UNKNOWN', 'MEDIUM'),
        ('final readout step', 'five steps in advance', 'only final rollout code', 'MATCH', 'HIGH'),
        ('step1-5 mean used', 'not specified', False, 'PARTIAL', 'MEDIUM'),
        ('no prediction handling', None, 'excluded from target/prediction MAPE; coverage separate', 'UNKNOWN', 'HIGH'),
    ]
    for item, pv, cv, status, severity in prediction_specs:
        add('PREDICTION', item, pv, 'Section II-E/F and Section III-D', '7,10', cv, 'experiments/fig9_strict_reproduction.py:3394-3659', status, severity, 'HIGH', severity, 'Fix only after isolated parity test and user approval')

    decoder_specs = [
        ('paper numerical decoder equation', None, 'custom likelihood codebook', 'UNKNOWN', 'CRITICAL'),
        ('most-likely criterion', 'most likely prediction', 'max timed then column overlap', 'PARTIAL', 'HIGH'),
        ('spike order used', True, 'timed overlap uses event times', 'MATCH', 'CRITICAL'),
        ('column set used', True, 'secondary column overlap', 'MATCH', 'HIGH'),
        ('timing tolerance', None, 0.03, 'UNKNOWN', 'HIGH'),
        ('candidate grid', None, 'half-spacing 963 values', 'UNKNOWN', 'HIGH'),
        ('unique legal code count', None, 940, 'UNKNOWN', 'MEDIUM'),
        ('tie handling', None, 'mean all tied candidate values', 'UNKNOWN', 'HIGH'),
        ('perfect-code roundtrip WAPE', None, 0.0014126706909136887, 'UNKNOWN', 'LOW'),
        ('perfect-code roundtrip MAE', None, 21.296961476756007, 'UNKNOWN', 'LOW'),
        ('perfect-code max error', None, 83.04573804573738, 'UNKNOWN', 'LOW'),
        ('decoder lower-bound implication', None, 'far below observed 0.4-0.6 error', 'PARTIAL', 'LOW'),
        ('ground truth used by decoder', False, False, 'MATCH', 'CRITICAL'),
    ]
    for item, pv, cv, status, severity in decoder_specs:
        add('DECODING', item, pv, 'Section III-D; decoder details omitted', 10, cv, 'src/seqmem/encoding.py:142', status, severity, 'HIGH', severity, 'Contact authors or run controlled codebook parity test')

    evaluation_specs = [
        ('metric label', 'MAPE', 'MAPE', 'MATCH', 'HIGH'),
        ('metric equation in Zhang', None, 'sum abs error / sum abs target', 'UNKNOWN', 'CRITICAL'),
        ('metric equation in cited [58]', 'ratio of mean absolute error to mean absolute target', 'same', 'MATCH', 'HIGH'),
        ('conventional pointwise MAPE', None, 'not used', 'PARTIAL', 'MEDIUM'),
        ('warmup in Zhang', None, 'default 5904; diagnostics 200', 'UNKNOWN', 'CRITICAL'),
        ('warmup 5904 provenance', None, 'Numenta [58] plotting utility', 'UNKNOWN', 'HIGH'),
        ('warmup 200 provenance', None, 'local short diagnostic design', 'UNKNOWN', 'CRITICAL'),
        ('evaluation record range', None, 'index >= warmup through limit-horizon', 'UNKNOWN', 'CRITICAL'),
        ('full-year evaluation completed', 'paper uses one year', False, 'MISMATCH', 'CRITICAL'),
        ('250 run prediction count', None, 45, 'UNKNOWN', 'HIGH'),
        ('500 run prediction count', None, 295, 'UNKNOWN', 'HIGH'),
        ('rolling window in Zhang', None, 400, 'UNKNOWN', 'MEDIUM'),
        ('rolling window provenance', None, 'local/reference [58] interpretation', 'UNKNOWN', 'MEDIUM'),
        ('overall MAPE horizon', 'five steps in advance', 'final step only', 'MATCH', 'HIGH'),
        ('coverage reporting', None, 'separate coverage field', 'UNKNOWN', 'MEDIUM'),
        ('paper original Ours MAPE', 'about 0.10', '0.50545 (250) / 0.53928 (500)', 'MISMATCH', 'CRITICAL'),
        ('paper MAPE precision', 'figure digitized approximate', 'local exact CSV calculation', 'PARTIAL', 'LOW'),
        ('modified panel isolation', 'Fig9c modified; Fig9d post-change curve', 'separate perturbed stream supported', 'MATCH', 'MEDIUM'),
        ('evaluation window sensitivity', None, '0.499-0.539 WAPE across starts in existing 500 run', 'UNKNOWN', 'HIGH'),
        ('current500 direct comparability', 'one-year paper result', '500 rows, warmup 200', 'MISMATCH', 'CRITICAL'),
    ]
    for item, pv, cv, status, severity in evaluation_specs:
        add('EVALUATION', item, pv, 'Section III-D and Fig. 9; details omitted unless [58]', 10, cv, 'experiments/fig9/metrics.py; experiments/fig9_strict_reproduction.py:3394', status, severity, 'HIGH', severity, 'Do not compare short diagnostic MAPE as full paper reproduction')

    return rows


def root_cause_rows() -> list[dict[str, object]]:
    candidates = [
        (1, 'FIVE_STEP_PROTOCOL_MISMATCH', 'Code state is through t-1, but output labels t and compares raw step5 to t+5.', 'Reported timestamp difference is still exactly 2.5 h.', 'EXPLICIT horizon; anchor timing inferred', 'Direct code-level off-by-one', 'CRITICAL', 'HIGH', 'Shift anchor/target in isolated non-model parity A/B.'),
        (2, 'EVALUATION_WINDOW_MISMATCH', 'Current 250/500 uses warmup 200 and only 45/295 predictions; paper reports a year.', 'Current default 5904 follows cited [58], but no full strict run exists.', 'Warmup/range not specified by Zhang', 'Not directly comparable', 'CRITICAL', 'HIGH', 'After horizon fix, run a small alignment validation before any full year.'),
        (3, 'PASSENGER_AGGREGATION_MISMATCH', 'Paper cites raw TLC; exact aggregation, fleet and timestamp rule are absent.', 'Current 17,520 file matches all published structural facts and [58] assets.', 'NOT_SPECIFIED_IN_PAPER', 'Provenance unresolved, not proven wrong', 'HIGH', 'MEDIUM', 'Locate author processed dataset or ask authors for preprocessing.'),
        (4, 'READOUT_MECHANISM_MISMATCH', 'Paper gives no real-value decoder equation; current tie averaging can be broad for dense predictions.', 'Perfect encoded patterns decode at only 0.00141 repository MAPE.', 'NOT_SPECIFIED_IN_PAPER', 'Implementation assumption', 'HIGH', 'HIGH', 'Use ideal and partial-pattern audit before changing readout.'),
        (5, 'CONTINUOUS_DYNAMICS_UNDERSPECIFIED', 'tau, V0 normalization, timing tolerance and integration grid are unpublished.', 'Topology and thresholds are aligned; prior diagnostics already isolated density failure.', 'NOT_SPECIFIED_IN_PAPER', 'Implementation assumption', 'HIGH', 'MEDIUM', 'Resume mechanism diagnostics only after protocol alignment.'),
        (6, 'INTERCOLUMN_INHIBITION_MISMATCH', 'Paper describes intercolumn inhibition; strict rollout has no continuous intercolumn competition.', 'The paper does not publish scheduling or current equations.', 'IMPLIED_BY_METHOD', 'Partial implementation', 'HIGH', 'MEDIUM', 'Only test after anchor/evaluation parity is fixed.'),
        (7, 'ENCODER_SCALE_MISMATCH', 'Paper omits l, sigma, min and max; current values come from [58].', 'Exact topology/K/order match and ideal roundtrip error is small.', 'NOT_SPECIFIED_IN_PAPER', 'Implementation assumption', 'MEDIUM', 'HIGH', 'Obtain author scale or report sensitivity as diagnostic.'),
        (8, 'LEARNING_RULE_MISMATCH', 'Exact contributing-synapse timing and solver behavior are not fully specified.', 'Scenario 1/2/3, updates, L_match and forgetting values broadly align.', 'PARTIAL', 'Partial implementation', 'MEDIUM', 'MEDIUM', 'Defer until protocol mismatch is resolved.'),
        (9, 'FORGETTING_MISMATCH', 'Pruning timing and tie details are implementation choices.', 'Formula, weights and threshold 65 are explicit and matched.', 'EXPLICIT core rule', 'Minor partial mismatch', 'LOW', 'MEDIUM', 'No immediate action.'),
        (10, 'DATASET_ROW_COUNT_MISMATCH', 'A 10,320-row file exists historically.', 'Current strict default actually reads the complete 17,520-row file.', 'EXPLICIT_IN_PAPER', 'No mismatch in current default', 'LOW', 'HIGH', 'Keep old NAB subset classified separately.'),
    ]
    return [
        dict(zip(
            ('rank', 'candidate', 'evidence_for', 'evidence_against', 'paper_explicitness', 'current_mismatch', 'estimated_importance', 'testability', 'next_test'),
            row,
        ))
        for row in candidates
    ]


def severity_sort_key(row: dict[str, object]) -> tuple[int, str, str]:
    order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
    return (
        order.get(str(row.get('severity')), 4),
        str(row.get('category', '')),
        str(row.get('item', '')),
    )
