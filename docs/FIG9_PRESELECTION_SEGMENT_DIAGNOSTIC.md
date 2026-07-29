# Fig.9 Preselection Segment Diagnostic

## Status

This is a nonpaper, read-only diagnostic. It does not change strict defaults,
segment response, candidate selection, competition, propagation, decoding,
learning, RNG order, or checkpoint v1.

Ground truth is encoded only after the prediction and competition result are
fixed. It labels target columns for offline analysis and never enters a score,
threshold, winner, emitted code, rollout state, or learning call.

Both existing internal selection groups contain one column. The SSTD encoder
defines a target column but not a target neuron. Consequently, target-vs-false
replacement inside one internal group is not identifiable from the available
oracle: every segment in that group shares the same target-column label. The
diagnostic can still measure loss before valid firing, compression ratios,
segment-level separability across columns, and candidate-to-emitted loss.

## CLI

```text
--preselection-segment-diagnostic
--preselection-segment-level summary|crossing|full
--preselection-segment-compress
--preselection-max-rows N
```

The diagnostic is off by default.

- `summary` writes only per-rollout funnel rows.
- `crossing` writes all threshold-crossing segments and is recommended for the
  manual 250-record run.
- `full` writes every touched segment and is intended only for tiny smoke tests.
- Compression writes segment, group, and replacement traces as `.csv.gz`.
- `--preselection-max-rows` is explicit debug truncation. Funnel totals remain
  complete, and protocol metadata records requested and actual row counts.

## Outputs

- `preselection_funnel_trace.csv`
- `preselection_segment_trace.csv[.gz]`
- `preselection_group_trace.csv[.gz]`
- `preselection_replacement_trace.csv[.gz]`
- `preselection_summary.json`
- `preselection_protocol.json`

The traces remain separate from `competition_trace.csv`.

The offline analyzer writes:

- `preselection_stage_summary.csv`
- `preselection_field_summary.csv`
- `preselection_horizon_summary.csv`
- `target_segment_survival.csv`
- `target_loss_stage_summary.csv`
- `internal_group_competition.csv`
- `segment_score_separability.csv`
- `segment_context_separability.csv`
- `winner_vs_discarded_summary.csv`
- `preselection_bootstrap_ci.csv`
- `preselection_analysis_summary.json`
- `FIG9_PRESELECTION_SEGMENT_REPORT.md`

## Stable Identity

Segments reuse the experiment-side provenance ID:

```text
SHA256(
  creation_transition_index,
  target_column,
  target_neuron,
  creation_source_fingerprint,
  stable_creation_ordinal
)
```

Selection groups use:

```text
SHA256(
  input_index,
  horizon_step,
  actual_group_type,
  group_column,
  group_neuron,
  stable_time_key
)
```

Object identities are used only for transient joins inside the same prediction
call. They are never written as persistent IDs.

## Interpretation Boundaries

- Low target crossing recall indicates activation, learning, segment matching,
  or threshold-scale failure before candidate selection.
- High crossing recall but low candidate survival implicates the internal
  same-time score or earliest-column selection.
- High candidate survival but low emitted survival implicates the later
  intercolumn competition.
- If segment-level target/false metrics are already inseparable, a different
  selector cannot recover information that the segment representation does not
  contain.
- High unknown elimination rate means the hook is incomplete and no root-cause
  conclusion should be made.

No counterfactual offline ranking is connected back to the model, and this
diagnostic does not produce a new MAPE claim.

## 50-Record Validation

The allowed batched run used `limit=50`, `warmup=20`, horizon 5, reference
continuous dynamics, and the existing competition settings.

```text
inspected segments       58,619
response computed        55,671
positive response        55,671
threshold crossed        54,039
event score winners      45,976
saved candidates         21,737
competition emitted      13,388
```

Thus 40.22% of threshold-crossing segments became saved candidates. Much of
this compression is expected same-column event reduction and does not itself
prove loss of a correct target column.

Target-column survival was:

| Field | Crossing recall | Candidate recall | Emitted recall |
|---|---:|---:|---:|
| weekday | 0.9904 | 0.7048 | 0.5536 |
| time | 0.5616 | 0.5376 | 0.3496 |
| passenger | 0.5672 | 0.5352 | 0.3648 |

For passenger target columns, 532/1250 had no inspected segment, 9/1250 had
only below-threshold segments, 40/1250 crossed but had no valid soma firing
time, 213/1250 became candidates but were suppressed by intercolumn
competition, and 456/1250 were emitted.

Passenger segment-score PR-AUC by rollout step was approximately
`0.0680, 0.0622, 0.0447, 0.0354, 0.0549`, close to the corresponding target
prevalence. Candidate-score PR-AUC was approximately
`0.0803, 0.0495, 0.0468, 0.0332, 0.0652`. The signal is weak before and after
internal selection.

The most important diagnosed loss is therefore not a target column losing to
a false column inside the preselection group. That comparison is structurally
unavailable. Passenger and time frequently have no touched target-column
segment at all, and later competition removes additional target candidates.

The diagnostic produced zero unknown eliminations. Predictions, oracle rows,
branch provenance rows, model fingerprint, and RNG fingerprint matched the
diagnostic-off run. Competition rows matched after excluding wall-clock timing
fields; raw competition CSV SHA differs because prediction runtime includes
trace-copy overhead.
