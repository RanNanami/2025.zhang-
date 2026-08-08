# Fig.9 Segment Context Representation Audit

## Scope

This audit compares the real `L_match=2` and `L_match=4` runs under the same
250-record Fig.9 configuration:

- `warmup=200`
- `K=10`, `neurons_per_column=32`
- `competitive_raw`, `strength=0.1`, `tau=0.02`
- `simultaneous_policy=batched`, `bin_width=0.005`
- `intracolumn_selection_policy=max_candidate_score`
- continuous implementation `reference`
- raw autonomous propagation, seed `0`

The composition diagnostic is read-only. It does not change prediction,
candidate selection, reinforcement, RNG state, or checkpoint/model state. The
source trace is explicitly `trajectory_kind=actual_observation`; autonomous
rollout context is not mixed into these rows.

## Formal Results

| metric | L2 | L4 | L2 - L4 |
|---|---:|---:|---:|
| Mean MAPE | 0.3927571082 | 0.4086558112 | -0.0158987530 |
| coverage | 1.000000 | 1.000000 | 0 |
| final segments | 1,945 | 2,377 | -432 |
| final synapses | 115,473 | 130,270 | -14,797 |
| runtime seconds | 1,011.21 | 1,165.87 | -154.65 |

Stepwise MAPE/raw-column means:

| horizon step | L2 MAPE | L4 MAPE | L2 raw columns | L4 raw columns |
|---:|---:|---:|---:|---:|
| 1 | 0.399240 | 0.329294 | 230.47 | 262.76 |
| 2 | 0.340884 | 0.437043 | 237.31 | 274.58 |
| 3 | 0.390564 | 0.463993 | 227.69 | 297.04 |
| 4 | 0.401540 | 0.423901 | 228.13 | 312.96 |
| 5 | 0.392757 | 0.408656 | 215.89 | 334.27 |

This directly answers the apparent paradox: L2 is worse at Step 1, but it
emits a much sparser recurrent rollout. Its later steps therefore accumulate
less dense-context error. That is an observed association under this strict
ablation, not proof that sparsity alone is causal.

## Context Composition

The composition trace contains one row per inspected observation x candidate
segment. The formal row counts are:

| trace quantity | L2 | L4 |
|---|---:|---:|
| candidate segment rows | 22,015 | 28,260 |
| L2-only rows (`overlap=2`, not 3) | 1,398 | 2,005 |
| selected L2-only rows | 402 | 124 |
| source rows | 2,435,940 | 2,752,579 |

For the L2-only candidate rows:

| quantity | L2 | L4 |
|---|---:|---:|
| mean overlap-source IDF | 2.0657 | 2.4476 |
| mean source incidence | 282.87 | 270.83 |
| rare-source fraction | 0.0701 | 0.0835 |
| common-source fraction | 0.0705 | 0.0706 |
| very-common-source fraction | 0.7804 | 0.7733 |
| ambiguity rate | 0.7854 | 0.1756 |

L2 does not look like a clean sparse-memory regime. Its L2-only candidates are
mostly built from very-common sources and are much more ambiguous. The
important difference is that more of those relaxed candidates are actually
selected and reinforced, while the overall rollout density is lower.

Field split for L2-only candidates:

| field | L2 rows | selected | ambiguity | mean IDF | very-common fraction |
|---|---:|---:|---:|---:|---:|
| Passenger | 336 | 222 | 0.384 | 2.178 | 0.768 |
| Time | 837 | 129 | 0.902 | 2.107 | 0.787 |
| Weekday | 225 | 51 | 0.951 | 1.745 | 0.776 |

Passenger is the field where L2 changes most usefully: relaxed L2-only
segments are selected much more often than in L4. Time and Weekday remain
highly ambiguous, so the effect is not a uniformly better representation.

The new trace reports 92 selected L2-only rows in records 200--244 under its
candidate-segment join. The earlier `actual_branch` report's 92 total / 86
Passenger numbers used a different branch-summary denominator and cannot be
treated as the same row unit. This audit keeps both definitions separate
rather than silently forcing them to agree.

## Scenario And Temporal Evidence

Scenario counts below use the composition trace's unique
`(actual_record_index, encoded_column, scenario)` event unit, not the older
field-level summary unit:

| scenario | L2 | L4 |
|---|---:|---:|
| Scenario 1 | 985 | 1,077 |
| Scenario 2 | 4,390 | 3,866 |
| Scenario 3 | 1,543 | 1,975 |

The direction matches the observed training behavior: L2 increases Scenario 2
and reduces Scenario 3. The lower final segment count is therefore consistent
with more existing contexts being accepted before new segments are grown.

The source-frequency and context-quality tables are split by Passenger, Time,
Weekday and record ranges. They are written under the audit output's `analysis`
directories. No ground-truth suffix, future covariate, or target label enters
the model path.

## Error Association And Hypotheses

The Step1 error grouping is posthoc only. For L2-only rows, the bootstrap
95%-CI for LOW_ERROR minus HIGH_ERROR common-source fraction is
`[-0.1619, -0.0261]`; the corresponding IDF CI is
`[-0.2153, 0.1979]`. In this trace, high-error L2-only rows are associated with
more common sources, while IDF itself is not separated reliably. For L4 the
common-fraction CI is `[0.0184, 0.1393]` and the IDF CI is
`[-0.2675, 0.1033]`; this is not a stable causal ranking signal across both
ablations.

The evidence supports **H3_MIXED_REGIME**:

1. L2 permits useful weak matches, especially in Passenger, and turns more
   existing contexts into Scenario 2 reinforcement.
2. The same L2 threshold admits many shared/common contexts. Ambiguity rises
   sharply, particularly for Time and Weekday.
3. Batched competition and the resulting lower emitted/raw-column density
   partially contain that ambiguity during recurrent rollout.

This explains why Step 1 worsens while Steps 2--5 improve without claiming
that the current strict model has a solved segment representation. No
mechanism experiment was enabled because this evidence is mixed and a choice
such as IDF weighting or common-source downweighting would be a separate
nonpaper ablation.

The earlier composition-only runs did not claim target-vs-false context
quality because they had no lossless candidate identity join. That gap is now
closed by the separate `context_oracle_join_20260808` diagnostic: its formal
L2 and L4 traces use `row_unit=CANDIDATE_SEGMENT`, stable candidate IDs, and
zero duplicate/orphan rows. The new offline result is mixed rather than a
uniform context-quality signal. In particular, Passenger score favors false
candidates in both L2 and L4, Weekday favors targets, and Time is near zero or
changes sign. The oracle label remains posthoc and never enters prediction,
selection, competition, reinforcement, or RNG. See the generated
`FIG9_CONTEXT_ORACLE_DISCRIMINATION_REPORT.md` for the paired bootstrap
intervals and the exact formal output paths.

## Files Added Or Changed

- `experiments/diagnostics/fig9_segment_context_composition.py`
  - read-only live-segment/source snapshot;
  - source incidence, IDF-like diagnostic, quantile bins and role labels;
  - match, source, quality, event and L2-only traces;
  - bounded batch CSV/GZ streaming.
- `experiments/diagnostics/analyze_fig9_segment_context_composition.py`
  - streaming source aggregation;
  - stage funnel, L2-only, field, temporal, reinforcement, error association,
    bootstrap CI and H1/H2/H3 report.
- `experiments/fig9_strict_reproduction.py`
  - default-off CLI and read-only call-chain hook.
- `tests/test_fig9_segment_context_composition.py`
  - read-only snapshot, role labels, real observation join and tracker-only
    history tests.
- `docs/FIG9_SEGMENT_CONTEXT_REPRESENTATION_AUDIT.md`
  - source math and exact strict call-chain audit.

## Validation

- `compileall`: passed.
- Full unittest suite: **356 tests passed**.
- 20-record diagnostic on/off: predictions SHA256, MAPE, model/RNG
  fingerprints, segment count and synapse count identical.
- 50-record diagnostic on/off: predictions SHA256, MAPE, model/RNG
  fingerprints, segment count and synapse count identical.
- GZIP source traces: read to EOF successfully in smoke validation and formal
  analyzer runs.
- Formal L2/L4: both completed with exit code 0 and coverage 1.0.

The strict defaults remain unchanged. No mechanism was merged into the strict
path, no `L_match` default was changed, and no main branch merge was made.
