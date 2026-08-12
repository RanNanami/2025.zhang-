"""Generate the read-only Fig.9 paper-to-code parity audit bundle."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

from experiments.fig9.data import read_records
from experiments.fig9.parity import (
    CURRENT_DATA_RELATIVE,
    CURRENT_PERTURBED_RELATIVE,
    PAPER_CHANGE_DATE,
    PAPER_FIG9_APPROX_MAPE,
    PAPER_FIG9_DIGITIZATION_UNCERTAINTY,
    current_protocol_fingerprint,
    dataset_inventory,
    decoder_partial_pattern_rows,
    encoding_fixture_rows,
    evaluation_window_rows,
    file_sha256,
    ideal_roundtrip_rows,
    paper_protocol_fingerprint,
    parity_matrix_rows,
    prediction_alignment,
    root_cause_rows,
    run_family_registry,
    severity_sort_key,
)


ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    fields = list(materialized[0]) if materialized else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def _paper_source_audit() -> str:
    return """# Fig.9 Paper Protocol Source Audit

This audit uses the supplied English IEEE paper as the primary source. It
separates direct statements from method implications and omissions. Current
code or Numenta reference values never fill a paper-side unknown.

## Direct Fig.9 Protocol

| Fact | Evidence | Level |
| --- | --- | --- |
| Public NYC taxi passenger source, cited as TLC [66] | PDF p.10, Section III-D; reference [66] on p.12 | EXPLICIT_IN_PAPER |
| Half-hour aggregation, 17,520 records, one year | PDF p.10, Section III-D | EXPLICIT_IN_PAPER |
| Record fields: day of week, time of day, passenger number | PDF p.10, Section III-D | EXPLICIT_IN_PAPER |
| Periodic weekday/time and real-value passenger encoding | PDF p.10, Section III-D; algorithm on p.5, Section II-C | EXPLICIT_IN_PAPER |
| K=10 per field; 30/58/482 columns; 570 total; 32 neurons/column | PDF p.10, Section III-D | EXPLICIT_IN_PAPER |
| Predict 2.5 hours, five steps ahead | PDF p.10, Section III-D | EXPLICIT_IN_PAPER |
| MAPE between most-likely prediction and actual data | PDF p.10, Section III-D | EXPLICIT_IN_PAPER |
| L_match=4 and forgetting threshold=65 for taxi | PDF p.8, Section III-A after Table I | EXPLICIT_IN_PAPER |
| w0=.5, delta=.1, incorrect delta=.01, Vdep=.5, Aosc=.5, thresholds=1, Lw=10, La=1 | PDF p.8, Table I | EXPLICIT_IN_PAPER |
| Modified stream after 2015-04-01: weekday 07:00-11:00 -20%, 21:00-23:00 +20% | PDF p.10, Section III-D | EXPLICIT_IN_PAPER |

## Method Evidence

- PDF p.5, Section II-C: real-value encoding uses Gaussian receptive fields
  separated by interval `l`, selects top K outputs, has resolution `l/2`, and
  assigns ordered spike times evenly in the first half-cycle.
- PDF p.5: periodic encoders arrange the Gaussian population on a ring.
- PDF p.6, Section II-D: Scenario 1/2/3, structural growth, delay assignment,
  weight/age updates, and the forgetting equation are explicit.
- PDF p.7, Section II-E/F: a predictive neuron may fire without feedforward
  input and drive further prediction. Using raw autonomous activity is
  strongly implied; an exact statement that one normalized cycle equals one
  30-minute record is absent.

## Explicit Unknowns

`NOT_SPECIFIED_IN_PAPER`: exact year range, timezone, TLC fleet/geographic
scope, pickup/dropoff anchor, passenger aggregation formula, missing-bin and
DST treatment, passenger min/max, numerical `l`, Gaussian sigma, real-value
decoder equation, tau_m, tau_s, response normalization, integration grid,
timing tolerance, random seed, evaluation warmup/range, rolling window, and
the exact MAPE normalization equation.

The Fig.9(b) `Ours` bar is approximately 0.10. This is
`FIGURE_DIGITIZED_APPROXIMATE`, with a conservative visual uncertainty of
about +/-0.01; it is not an exact table value.

Public searches found the official TLC source but did not identify an author
release containing the processed 17,520-row stream or preprocessing code:
`AUTHOR_PROCESSED_DATASET_NOT_PUBLICLY_IDENTIFIED`.
"""


def _data_pipeline(inventory: list[dict[str, object]]) -> str:
    current = inventory[0]
    return f"""# Current Data Preprocessing Pipeline

1. Strict CLI default: `data/paper_nyc_taxi.csv`.
2. `read_records()` skips the two NuPIC metadata rows because they are not
   parseable timestamps/numbers, then reads `timestamp` and `passenger_count`.
3. No resampling, normalization, interpolation, zero fill, outlier removal,
   timezone conversion, or train/test scaling occurs in the loader.
4. `record_values()` computes Python weekday (`Monday=0`) and half-hour slot
   (`hour*2 + minute//30`); it forwards passenger count unchanged.
5. The passenger encoder clips only at its configured range 0..40,000.

Current file SHA256: `{current['sha256']}`. It has {current['row_count']}
records from {current['first_timestamp']} through {current['last_timestamp']},
with no missing half-hours or duplicate timestamps.

The file is byte-identical to `.deps/reference_nyc_taxi.csv`, the bundled
Numenta [58] reference asset. It was committed as a finished aggregate; no
raw TLC-to-half-hour generation script exists in this repository. Therefore
its exact aggregation semantics remain unresolved.
"""


def _data_constraints() -> str:
    return """# Dataset Temporal Constraints

## Direct Facts

- The paper states one year, 17,520 records, and half-hour aggregation.
- `17,520 = 365 * 48`; this is exactly a non-leap 365-day half-hour grid.
- The modified experiment changes records after April 1, 2015.

## Logical Constraints

If original and modified experiments operate on the same one-year stream, the
stream must include records after 2015-04-01. This does not establish a
specific first date.

## Unknown

`PAPER_YEAR_RANGE_NOT_EXPLICITLY_SPECIFIED`: the paper does not state the
first timestamp, last timestamp, timezone, or whether the year is July-June,
January-December, or another 365-day interval.
"""


def _encoder_report(roundtrip: dict[str, object]) -> str:
    return f"""# Passenger Encoder Parity

The current implementation matches the published topology and algorithmic
shape: 482 passenger mini-columns, K=10, Gaussian population ranking, ordered
events, and ten evenly spaced times from 0 through 0.5 cycle.

It is not numerically strict to the paper because the paper omits passenger
range, interval `l`, and sigma. Current code imports the Numenta-style range
0..40,000, spaces centers by `40000/(482-1) = 83.160083...`, and defaults sigma
to one spacing. Thus `ENCODER_SCALE_UNDERSPECIFIED` remains.

The ideal full-dataset code-to-decoder audit gives repository-formula MAPE
`{roundtrip['ideal_roundtrip_mape_repository_formula']:.12f}`, MAE
`{roundtrip['mae']:.6f}`, and maximum absolute error
`{roundtrip['max_absolute_error']:.6f}`. Quantization alone is far too small to
explain the observed 0.4-0.6 strict diagnostic MAPE.
"""


def _decoder_report(roundtrip: dict[str, object]) -> str:
    return f"""# Current Real-Value Decoder

`SSTDRealValueEncoder.decode_likelihood()` filters raw prediction events to
passenger columns 88..569. It compares them with every half-spacing legal
codebook value. Ranking is lexicographic: timed overlap first, column overlap
second. All values tied at the best score are averaged.

Temporal order is genuinely used: candidate event times must match predicted
times within tolerance 0.03. Column membership remains a secondary fallback.
This is not a set-only decoder, although dense raw predictions can create many
ties and weaken order selectivity.

The paper gives only “most likely prediction results” and no numerical decode
equation, codebook, tolerance, or tie rule. Therefore
`REAL_VALUE_DECODER_NOT_FULLY_SPECIFIED_IN_PAPER`.

Ideal ordered-code audit: {roundtrip['records']} records,
{roundtrip['unique_ordered_codes']} unique dataset codes, repository MAPE
{roundtrip['ideal_roundtrip_mape_repository_formula']:.12f}, pointwise MAPE
{roundtrip['ideal_roundtrip_pointwise_mape']:.12f}, mean signed bias
{roundtrip['mean_signed_bias']:.6f}.
"""


def _alignment_report(alignment: dict[str, object]) -> str:
    steps = "\n".join(
        f"- step {row['rollout_step']}: causal continuation index "
        f"{row['causal_continuation_index']} at {row['causal_continuation_timestamp']}"
        for row in alignment['rollout_steps']  # type: ignore[index]
    )
    return f"""# Prediction Alignment Example

For zero-based loop index 200, the CSV labels
`{alignment['reported_input_timestamp']}` as the input and compares against
`{alignment['compared_target_timestamp']}`, exactly +2.5 hours.

However, prediction executes before record 200 is observed. The model state
contains actual observations only through index 199 at
`{alignment['last_observed_timestamp_before_rollout']}`. Raw autonomous steps
therefore have this causal interpretation:

{steps}

The fifth raw continuation corresponds to index
{alignment['causal_step5_expected_zero_based_index']}, while evaluation uses
index {alignment['compared_target_zero_based_index']}. Relative to the last
actually observed record, the compared target is
{alignment['last_observed_to_target_hours']} hours ahead, not 2.5.

Verdict: `PREDICTION_HORIZON_SEMANTICS_MISMATCH` is detected in the current
loop. This audit does not alter it. A separate isolated correction and A/B is
required before another expensive run.
"""


def _network_rows() -> list[dict[str, object]]:
    matrix = parity_matrix_rows(ROOT)
    return [row for row in matrix if row['category'] == 'NETWORK']


def _learning_report() -> str:
    return """# Learning and Forgetting Parity

The current one-layer implementation broadly follows paper Section II-D:

- Scenario 1 reuses the matching prediction candidate; contributing synapses
  strengthen/rejuvenate and noncontributing synapses weaken/age.
- Scenario 2 requires timed overlap at L_match=4, reinforces the best matching
  segment, and grows missing previous-winner synapses.
- Scenario 3 grows a segment on a least-used neuron when no eligible segment
  exists; wrong predictions are punished by delta'=0.01.
- Forgetting uses `10*(1-weight) + age` and removes a synapse at threshold 65.

L_match=4 and threshold=65 are paper-explicit on PDF page 8. Earlier project
documentation that treated them as inferred was incorrect.

Remaining partial assumptions concern the exact continuous definition of a
“contributing” arrival, delay/time normalization, winner tie-breaking,
burst/context representation, pruning timing, and the numerical continuous
solver. Those details are not sufficient grounds to rewrite the paper rule.
"""


def _assumption_registry() -> list[dict[str, object]]:
    entries = (
        ('DATA_YEAR_RANGE', '2014-07-01 to 2015-06-30', 'Numenta asset, not Zhang text', 'HIGH'),
        ('PASSENGER_AGGREGATION', 'Finished aggregate accepted as passenger count', 'No TLC aggregation script', 'CRITICAL'),
        ('PASSENGER_RANGE', '0..40000', 'Numenta [58] configuration', 'HIGH'),
        ('GAUSSIAN_SIGMA', 'one center spacing', 'local implementation default', 'HIGH'),
        ('REAL_VALUE_DECODER', 'timed/column codebook overlap and tie average', 'local implementation', 'CRITICAL'),
        ('TIMING_TOLERANCE', '0.03 cycle', 'local implementation', 'HIGH'),
        ('TAU_M', '0.10 cycle', 'local implementation', 'HIGH'),
        ('TAU_S', '0.02 cycle', 'local implementation', 'HIGH'),
        ('RESPONSE_NORMALIZATION', 'kernel normalized to peak 1', 'local implementation', 'CRITICAL'),
        ('INTEGRATION_STEP', '0.005 cycle', 'local implementation', 'HIGH'),
        ('CYCLE_RECORD_MAPPING', 'one autonomous call per record continuation', 'method interpretation', 'CRITICAL'),
        ('WARMUP_DEFAULT', '5904', 'Numenta [58] plotting utility', 'HIGH'),
        ('WARMUP_SHORT_RUN', '200', 'local diagnostic design', 'CRITICAL'),
        ('MAPE_FORMULA', 'global normalized absolute error', 'Numenta [58] utility', 'HIGH'),
        ('ROLLING_WINDOW', '400', 'local/reference interpretation', 'MEDIUM'),
        ('RNG_SEED', '0', 'reproducibility choice', 'LOW'),
        ('INTRACOLUMN_SOLVER', 'first firing candidate retained', 'local continuous implementation', 'HIGH'),
        ('INTERCOLUMN_COMPETITION', 'off in strict default', 'paper scheduling absent', 'CRITICAL'),
    )
    return [
        {
            'assumption': name,
            'current_value': value,
            'provenance': provenance,
            'severity': severity,
            'paper_explicit': False,
            'strict_default_changed_by_audit': False,
        }
        for name, value, provenance, severity in entries
    ]


def _data_parity_rows(inventory: list[dict[str, object]]) -> list[dict[str, object]]:
    current = inventory[0]
    rows = (
        ('source', 'NYC TLC [66]', current['source'], 'PARTIAL', 'Same upstream agency is plausible; author processed asset unverified'),
        ('record count', 17520, current['row_count'], 'MATCH', 'Direct file audit'),
        ('time span', 'one year; dates unspecified', f"{current['first_timestamp']} to {current['last_timestamp']}", 'PARTIAL', 'Duration matches; dates cannot be checked'),
        ('sampling frequency', '30 minutes', current['interval_minutes'], 'MATCH', 'Direct file audit'),
        ('fields', 'weekday,time,passenger', current['columns'], 'MATCH', 'Loader derives weekday/time from timestamp'),
        ('passenger definition', 'number of passengers', 'half-hour passenger_count aggregate', 'PARTIAL', 'Raw aggregate formula absent'),
        ('aggregation rule', None, 'unknown', 'UNKNOWN', 'SUM passenger_count vs trip count unresolved'),
        ('missing treatment', None, 'finished file has no gaps', 'UNKNOWN', 'Generation code absent'),
        ('timezone', None, 'naive timestamps', 'UNKNOWN', 'No timezone metadata'),
        ('normalization', None, 'none before encoder; 0..40000 clipping', 'UNKNOWN', 'Paper does not specify bounds'),
        ('modified experiment support', 'after 2015-04-01', True, 'MATCH', 'Current stream includes date and paired perturbation file'),
    )
    return [
        {
            'item': item,
            'PAPER': paper,
            'CURRENT_STRICT': current_value,
            'MATCH_STATUS': status,
            'EVIDENCE': evidence,
        }
        for item, paper, current_value, status, evidence in rows
    ]


def _protocol_diff(matrix: list[dict[str, object]]) -> str:
    selected = [
        row for row in sorted(matrix, key=severity_sort_key)
        if row['status'] in {'MISMATCH', 'UNKNOWN', 'PARTIAL'}
    ]
    lines = [
        '# Paper vs Current Protocol Diff',
        '',
        'Only mismatches, unknowns, and implementation assumptions are shown.',
        '',
        '| Severity | Category | Item | Status | Current | Action |',
        '| --- | --- | --- | --- | --- | --- |',
    ]
    for row in selected:
        lines.append(
            f"| {row['severity']} | {row['category']} | {row['item']} | "
            f"{row['status']} | {str(row['current_value']).replace('|', '/')} | "
            f"{str(row['action']).replace('|', '/')} |"
        )
    return '\n'.join(lines) + '\n'


def _comparability() -> str:
    return """# Current vs Paper Comparability

| Axis | Status | Reason |
| --- | --- | --- |
| DATASET | PARTIAL | Full 17,520-row structure matches, but author aggregation/provenance is unresolved. |
| ENCODING | PARTIAL | Topology/K/order match; l, sigma and range are unpublished assumptions. |
| NETWORK | PARTIAL | Explicit topology/thresholds match; continuous constants and inhibition solver are unpublished. |
| LEARNING | PARTIAL | Scenario structure and taxi thresholds match; exact timing/contribution implementation remains interpretive. |
| PREDICTION | MISMATCH | Rollout runs before current anchor observation, creating a one-record causal offset. |
| DECODING | UNKNOWN | Paper does not publish the numeric real-value decoder. |
| EVALUATION | MISMATCH | Short 250/500 warmup-200 windows are not the paper's one-year evaluation. |

**COMPARABILITY LEVEL: NOT DIRECTLY COMPARABLE.**

The current label “strict reproduction” is too strong for numerical claims.
`PAPER_CONSTRAINED_IMPLEMENTATION` or `STRICT_TO_CURRENT_INTERPRETATION` is
more accurate until the anchor/evaluation protocol and unpublished data/readout
details are resolved. No code rename is performed by this audit.
"""


def _final_report(
    inventory: list[dict[str, object]],
    roundtrip: dict[str, object],
    alignment: dict[str, object],
    matrix: list[dict[str, object]],
) -> str:
    current = inventory[0]
    counts: dict[str, int] = {}
    for row in matrix:
        counts[str(row['status'])] = counts.get(str(row['status']), 0) + 1
    return f"""# Fig.9 Paper-to-Code Parity Audit Report

## Primary Conclusion

`MULTIPLE_CRITICAL_PROTOCOL_MISMATCHES_FOUND`

The current implementation is a serious paper-constrained implementation of
the same task family, but the existing 250/500 MAPE values are not a direct
reproduction of the paper's Fig.9 number. The strongest directly demonstrated
problem is a one-record prediction-anchor mismatch. Evaluation also uses a
local 200-record warmup and a short slice, whereas the paper reports a
one-year experiment and does not state that protocol.

## What Is Aligned

- One-year candidate file with 17,520 contiguous half-hour records.
- Three fields and 30/58/482 columns, K=10, 570 total, 32 neurons/column.
- Periodic weekday/time and real-valued ordered Gaussian SSTD coding.
- Taxi L_match=4 and forgetting threshold=65 are paper-explicit and matched.
- Core Scenario 1/2/3 and weight/age/forgetting equations broadly match.
- Retrieval uses raw predictive neural activity, not decoded proximal replay.

## What Is Not Established

- Current data SHA `{current['sha256']}` is byte-identical to the bundled
  Numenta [58] asset, but no author preprocessing release was found. TLC raw
  records contain per-trip passenger counts; paper aggregation scope/formula
  is unspecified. Current data is a candidate, not proven author data.
- Passenger `l`, sigma, range and real-value decoder are absent from the paper.
- Zhang does not state warmup, exact MAPE equation, evaluation range, rolling
  window, random seed, tau values, timing tolerance, or integration grid.

## High-Impact Findings

1. **Five-step anchor mismatch.** At loop index 200, the state ends at 03:30,
   five autonomous continuations causally reach 06:00, but evaluation compares
   to 06:30. The output label still displays 04:00 -> 06:30 (+2.5 h), hiding
   that the 04:00 record was not observed before rollout.
2. **Short-window evaluation mismatch.** The completed strict 250 result
   (0.505450) evaluates 45 points; the completed raw L4 500 result (0.539280)
   evaluates 295 points. Both use warmup 200, a local diagnostic setting.
3. **Data/readout underspecification.** Exact passenger aggregation and decoder
   are unknown. They remain material reproducibility risks, although ideal
   encode/decode MAPE `{roundtrip['ideal_roundtrip_mape_repository_formula']:.6f}`
   shows current quantization alone does not produce the large gap.

## Paper Value

Fig.9(b) `Ours` is approximately {PAPER_FIG9_APPROX_MAPE:.2f} +/-
{PAPER_FIG9_DIGITIZATION_UNCERTAINTY:.2f} by visual digitization. It is not an
exact reported table value. Fig.9(c) is modified data; Fig.9(d) is the
post-modification error trajectory and must not be mixed with panel (b).

## Matrix Summary

The matrix has {len(matrix)} checks: {json.dumps(counts, sort_keys=True)}.

## Does Protocol Mismatch Plausibly Explain the Gap?

Yes, it can explain a substantial and currently unquantified part of the gap,
especially the causal horizon offset and incomparable evaluation window. The
audit cannot claim all error comes from protocol: dense raw predictions and
continuous/inhibition assumptions remain genuine model-mechanism risks.

## Recommendation

`FIX_FIVE_STEP_PREDICTION_PARITY`

First add an isolated, explicit anchor convention where the current record is
observed before forecasting t+5 (or equivalently align target to the fifth
causal continuation), then run a small A/B. Do not run another 500/full-year
experiment until that parity decision is tested. The Windows native runtime
crash investigation is operationally important but causally separate from
this paper-protocol audit.
"""


def _direct_answers() -> str:
    return """# Direct Answers to the 67 Audit Questions

1. **Paper records:** 17,520.
2. **One year:** yes, explicit.
3. **Sampling:** every 30 minutes.
4. **Exact year range:** not stated.
5. **April 1 constraint:** if original/modified use one stream, it must extend past 2015-04-01; it does not prove the start date.
6. **Paper source:** NYC TLC trip-record source [66].
7. **Current source:** a finished Numenta [58] half-hour aggregate bundled in `.deps`; exact author use is unverified.
8. **Current rows:** 17,520 in the strict default; a historical 10,320-row prefix also exists.
9. **Current range:** 2014-07-01 00:00 through 2015-06-30 23:30.
10. **Same dataset:** structurally plausible but not proven identical to the author's processed stream (`PARTIAL`).
11. **Is NAB the paper source:** no. The paper cites TLC; NAB/Numenta is a processed derivative candidate.
12. **NAB/TLC relation:** common TLC origin is plausible, but the repository lacks the raw aggregation chain.
13. **Passenger aggregation:** unspecified by the paper and absent locally.
14. **Exact reconstruction possible:** no, not without author data or aggregation rules.
15. **Weekday encoding:** periodic Gaussian SSTD ring.
16. **Time encoding:** periodic Gaussian SSTD ring.
17. **Passenger encoding:** real-value Gaussian population SSTD.
18. **K:** 10 per field.
19. **Columns:** 30/58/482, 570 total.
20. **Neurons:** 32 per mini-column.
21. **Current topology:** those values match.
22. **Gaussian interval `l`:** no numeric value in paper.
23. **Sigma/receptive width:** no numeric value in paper.
24. **Passenger range:** not stated.
25. **Current scale:** 0..40,000 from [58], 482 evenly spaced centers, sigma=one spacing.
26. **Strict encoder parity:** algorithmically aligned, numerically only partial because scale is unpublished.
27. **Spike order:** preserved in `SymbolCode` and timed-overlap decoding.
28. **Paper decoder:** not fully specified.
29. **Current decoder:** maximize `(timed overlap, column overlap)` over a half-spacing codebook; average ties.
30. **Perfect-code roundtrip:** repository MAPE 0.001412670691; MAE 21.29696.
31. **Does decoder alone create huge error:** ideal quantization does not; dense/partial prediction tie behavior can still matter.
32. **Paper five-step statement:** passenger count after 2.5 h, five steps in advance.
33. **Five autonomous cycles explicit:** no; it is strongly implied by retrieval, not stated as a Fig.9 implementation detail.
34. **Reported target timestamp:** exactly +2.5 h from the CSV input label.
35. **Off-by-one:** yes causally; the labelled current record is observed only after rollout.
36. **Overall current MAPE step:** final rollout step only.
37. **Paper horizon metric:** most reasonably t+5, but exact step aggregation is not described.
38. **Warmup=200:** local short-run diagnostic design.
39. **Paper warmup:** not specified.
40. **Current 500 vs paper:** not directly comparable.
41. **Paper Fig.9(b) Ours:** about 0.10.
42. **Value precision:** visual digitization, approximately +/-0.01, not exact.
43. **Paper L_match:** yes.
44. **L_match=4 source:** PDF page 8 explicitly says taxi task uses 4.
45. **Paper forgetting threshold:** yes, taxi threshold 65 on page 8.
46. **Current forgetting:** formula and threshold match; pruning timing is an implementation detail.
47. **Scenario 1/2/3:** broad rule structure is aligned.
48. **Learning timing:** one online observe per actual record; exact continuous contribution timing is partial.
49. **Network config:** explicit topology/threshold values match; continuous constants/inhibition are partial or unknown.
50. **Dataset match:** `PARTIAL`.
51. **Encoder match:** `PARTIAL`.
52. **Decoder match:** `UNKNOWN`.
53. **Prediction match:** `MISMATCH` because of anchor semantics.
54. **Evaluation match:** `MISMATCH` for completed short runs.
55. **“strict” naming:** too strong for numerical reproduction; “paper-constrained implementation” is more accurate.
56. **Largest MAPE-gap suspect:** five-step anchor mismatch.
57. **Second suspect:** evaluation-window/warmup mismatch.
58. **Third suspect:** unresolved passenger aggregation/provenance.
59. **Direct mismatches:** causal anchor/target alignment and short-run-vs-one-year comparison.
60. **Paper-unspecified items:** aggregation, range, l/sigma, decoder, warmup/formula/range, tau/V0/grid/tolerance, seed.
61. **Runtime crash relation:** unrelated to the scientific parity finding.
62. **Run another 500 now:** no.
63. **Next experiment:** isolated five-step anchor correction A/B on a small range.
64. **Primary conclusion:** `MULTIPLE_CRITICAL_PROTOCOL_MISMATCHES_FOUND`.
65. **Recommended next step:** `FIX_FIVE_STEP_PREDICTION_PARITY`.
66. **Tests:** 37 new parity tests plus the full existing suite.
67. **Commit SHA:** recorded after the final commit; see repository HEAD and final handoff.
"""


def generate(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    inventory = dataset_inventory(ROOT)
    fixtures = encoding_fixture_rows(ROOT)
    roundtrip_rows, roundtrip_summary = ideal_roundtrip_rows(ROOT)
    partial_rows = decoder_partial_pattern_rows()
    records = read_records(ROOT / CURRENT_DATA_RELATIVE, 0)
    alignment = prediction_alignment(records)
    matrix = parity_matrix_rows(ROOT)
    causes = root_cause_rows()
    current_fingerprint = current_protocol_fingerprint(ROOT)
    paper_fingerprint = paper_protocol_fingerprint()
    eval_windows = evaluation_window_rows(ROOT)

    original = inventory[0]
    perturbed = inventory[1]
    current_provenance = {
        'path': str(CURRENT_DATA_RELATIVE).replace('\\', '/'),
        'sha256': original['sha256'],
        'source': original['source'],
        'rows': original['row_count'],
        'first_timestamp': original['first_timestamp'],
        'last_timestamp': original['last_timestamp'],
        'interval_minutes': original['interval_minutes'],
        'strict_cli_evidence': 'experiments/fig9_strict_reproduction.py:5349',
        'loader_evidence': 'experiments/fig9/data.py:19',
        'byte_identical_reference_copy': original['sha256'] == inventory[3]['sha256'],
        'perturbed_path': str(CURRENT_PERTURBED_RELATIVE).replace('\\', '/'),
        'perturbed_sha256': perturbed['sha256'],
        'author_processed_dataset_publicly_identified': False,
        'passenger_aggregation_rule': None,
    }

    (output_dir / 'FIG9_PAPER_PROTOCOL_SOURCE_AUDIT.md').write_text(_paper_source_audit(), encoding='utf-8')
    (output_dir / 'CURRENT_DATA_PREPROCESSING_PIPELINE.md').write_text(_data_pipeline(inventory), encoding='utf-8')
    (output_dir / 'dataset_temporal_constraints.md').write_text(_data_constraints(), encoding='utf-8')
    (output_dir / 'passenger_encoder_parity.md').write_text(_encoder_report(roundtrip_summary), encoding='utf-8')
    (output_dir / 'current_real_value_decoder.md').write_text(_decoder_report(roundtrip_summary), encoding='utf-8')
    (output_dir / 'prediction_alignment_example.md').write_text(_alignment_report(alignment), encoding='utf-8')
    (output_dir / 'learning_forgetting_parity.md').write_text(_learning_report(), encoding='utf-8')
    (output_dir / 'paper_current_protocol_diff.md').write_text(_protocol_diff(matrix), encoding='utf-8')
    (output_dir / 'current_vs_paper_comparability.md').write_text(_comparability(), encoding='utf-8')
    direct_answers = _direct_answers()
    (output_dir / 'FIG9_PARITY_67_ANSWERS.md').write_text(direct_answers, encoding='utf-8')
    (output_dir / 'FIG9_PAPER_TO_CODE_PARITY_AUDIT_REPORT.md').write_text(
        _final_report(inventory, roundtrip_summary, alignment, matrix)
        + '\n'
        + direct_answers,
        encoding='utf-8',
    )

    write_json(output_dir / 'paper_fig9_protocol_fingerprint.json', paper_fingerprint)
    write_json(output_dir / 'current_fig9_protocol_fingerprint.json', current_fingerprint)
    write_json(output_dir / 'CURRENT_STRICT_DATA_PROVENANCE.json', current_provenance)
    write_json(output_dir / 'encoding_roundtrip_summary.json', roundtrip_summary)
    write_json(output_dir / 'prediction_alignment.json', alignment)
    write_csv(output_dir / 'dataset_inventory.csv', inventory)
    write_csv(output_dir / 'fig9_data_parity_matrix.csv', _data_parity_rows(inventory))
    write_csv(output_dir / 'encoding_fixture_audit.csv', fixtures)
    write_csv(output_dir / 'encoding_roundtrip_audit.csv', roundtrip_rows)
    write_csv(output_dir / 'decoder_roundtrip_audit.csv', partial_rows)
    write_csv(output_dir / 'network_parity_matrix.csv', _network_rows())
    write_csv(output_dir / 'implementation_assumption_registry.csv', _assumption_registry())
    write_csv(output_dir / 'fig9_run_family_registry.csv', run_family_registry())
    write_csv(output_dir / 'mape_gap_root_cause_ranking.csv', causes)
    write_csv(output_dir / 'evaluation_window_sensitivity.csv', eval_windows)
    write_csv(output_dir / 'FIG9_PAPER_TO_CODE_PARITY_MATRIX.csv', matrix)

    summary = {
        'primary_conclusion': 'MULTIPLE_CRITICAL_PROTOCOL_MISMATCHES_FOUND',
        'recommended_next_step': 'FIX_FIVE_STEP_PREDICTION_PARITY',
        'comparability_level': 'NOT_DIRECTLY_COMPARABLE',
        'matrix_items': len(matrix),
        'critical_mismatches': [
            row['item'] for row in matrix
            if row['severity'] == 'CRITICAL' and row['status'] == 'MISMATCH'
        ],
        'paper_records': 17520,
        'current_records': original['row_count'],
        'current_dataset_sha256': original['sha256'],
        'paper_exact_year_range_known': False,
        'author_processed_dataset_publicly_identified': False,
        'paper_ours_mape_approx': PAPER_FIG9_APPROX_MAPE,
        'paper_bar_uncertainty': PAPER_FIG9_DIGITIZATION_UNCERTAINTY,
        'paper_bar_status': 'FIGURE_DIGITIZED_APPROXIMATE',
        'ideal_roundtrip': roundtrip_summary,
        'off_by_one_detected': alignment['off_by_one_detected'],
        'current_250_mape': 0.5054500721931258,
        'current_500_raw_l4_mape': 0.5392795787736315,
        'runtime_crash_related_to_parity_finding': False,
        'new_model_mechanism_added': False,
        'strict_defaults_changed': False,
        'formal_500_or_full_year_run_executed': False,
    }
    write_json(output_dir / 'FINAL_FIG9_PARITY_SUMMARY.json', summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT / 'results' / 'fig9_diagnostics' / f'paper_code_parity_{timestamp}'
    )
    if not output.is_absolute():
        output = ROOT / output
    summary = generate(output)
    print(json.dumps({'output_dir': str(output), **summary}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
