# Fig.9 Candidate Score Separability Diagnostic

## Scope

This is a nonpaper, offline-only diagnostic on
`experiment/fig9-competitive-inhibition`. Ground truth is used only to label
completed candidates. It never affects prediction, competition, decoding,
learning, RNG, or checkpoint state.

The existing 250-record sequential and batched outputs were audited first.
Their `competition_trace.csv` and `oracle_candidate_trace.csv` files contain
only rollout-step aggregates. They do not contain candidate column, neuron,
predicted time, original score, inhibition, effective score, and emitted state
on the same row. Candidate-level 250 analysis therefore cannot be reconstructed
from those historical files.

A new opt-in `--candidate-separability-trace` was added. It is disabled by
default and requires `--oracle-candidate-diagnostic`. The 250-record experiment
was not rerun by Codex.

## Aggregation

- Candidate level: one saved `CompetitionDecision` is one sample.
- Column level: choose the candidate with maximum `original_score`; ties use
  `candidate_original_index`, then neuron index. Scores are not averaged.
- Field offsets come from protocol encoder sizes: weekday `[0, 30)`, time
  `[30, 88)`, passenger `[88, 570)`.
- Bootstrap unit: one rollout start (`input_index`), not one candidate.

In the 50-record trace each competition candidate mapped to one distinct raw
event column, so candidate and column row counts were both 43,210. The
aggregation still handles duplicate columns and is covered by tests.

## Read-Only Verification

| Policy | Trace rows | Predictions SHA on/off equal | Model equal | RNG equal |
|---|---:|---|---|---|
| Sequential | 21,473 | Yes | Yes | Yes |
| Batched | 21,737 | Yes | Yes | Yes |

Sequential predictions SHA:
`EB34EB343E57C267BA49AD4742CA9887DA78794B95D8EDD70EF5FE01F432A241`.

Batched predictions SHA:
`50036D50769C8C22F16AB0465F58BBCE3F21DE8B6BE9E0AE5B0BC5695D373257`.

Checkpoint format remains `fig9-strict-v1`, and stable competition-trace fields
are unchanged by the trace switch.

## Total Column Separability

PR-AUC is shown as `original / effective`. Positive prevalence is the random
ranking baseline.

| Policy | Step | Prevalence | PR-AUC |
|---|---:|---:|---:|
| Sequential | 1 | 0.098 | 0.118 / 0.130 |
| Sequential | 2 | 0.157 | 0.601 / 0.614 |
| Sequential | 3 | 0.104 | 0.413 / 0.479 |
| Sequential | 4 | 0.105 | 0.586 / 0.617 |
| Sequential | 5 | 0.088 | 0.532 / 0.560 |
| Batched | 1 | 0.098 | 0.118 / 0.117 |
| Batched | 2 | 0.107 | 0.396 / 0.416 |
| Batched | 3 | 0.116 | 0.281 / 0.289 |
| Batched | 4 | 0.103 | 0.472 / 0.481 |
| Batched | 5 | 0.090 | 0.456 / 0.464 |

These total values are inflated by field identity. Nearly every predicted
weekday candidate is a target candidate, so total PR-AUC does not demonstrate
that passenger branches are separable.

## Passenger Separability

| Policy | Step | Prevalence | Original PR-AUC | Effective PR-AUC |
|---|---:|---:|---:|---:|
| Sequential | 1 | 0.047 | 0.080 | 0.079 |
| Sequential | 2 | 0.071 | 0.078 | 0.077 |
| Sequential | 3 | 0.039 | 0.059 | 0.061 |
| Sequential | 4 | 0.039 | 0.051 | 0.060 |
| Sequential | 5 | 0.034 | 0.040 | 0.042 |
| Batched | 1 | 0.047 | 0.080 | 0.072 |
| Batched | 2 | 0.047 | 0.050 | 0.050 |
| Batched | 3 | 0.048 | 0.047 | 0.047 |
| Batched | 4 | 0.040 | 0.033 | 0.036 |
| Batched | 5 | 0.036 | 0.065 | 0.067 |

Step 1 contains modest passenger information, but it is not strong. Batched
Steps 2-4 are approximately prevalence-level. Effective score is not
consistently worse than original score, so inhibition is not the sole cause of
poor passenger ranking.

Time-field original-score PR-AUC is also only moderate:

- Sequential Steps 1-5: `0.251, 0.237, 0.271, 0.171, 0.131`.
- Batched Steps 1-5: `0.251, 0.163, 0.172, 0.107, 0.107`.

Offline field z-score and percentile normalization do not consistently improve
separability. They often reduce total PR-AUC after Step 1. This does not support
installing a field-normalized score in the actual model.

## Threshold Feasibility

The following are pooled false-column counts across 25 rollout starts for
`original_score`.

| Policy | Step | False columns at recall 0.5 | False columns at recall 0.7 |
|---|---:|---:|---:|
| Sequential | 1 | 2,253 | 3,372 |
| Sequential | 2 | 235 | 1,081 |
| Sequential | 3 | 760 | 2,160 |
| Sequential | 4 | 187 | 1,189 |
| Sequential | 5 | 356 | 1,457 |
| Batched | 1 | 2,253 | 3,372 |
| Batched | 2 | 1,145 | 2,212 |
| Batched | 3 | 1,442 | 2,210 |
| Batched | 4 | 777 | 1,932 |
| Batched | 5 | 823 | 2,155 |

At Step 1, retaining 50% of target columns also retains about 90 false columns
per rollout; retaining 70% retains about 135. High recall therefore requires a
large false context even before recurrent trajectories diverge.

## Batch Competition

The fraction of candidates in multi-candidate batches is:

`0.951, 0.928, 0.950, 0.961, 0.976` for Steps 1-5.

Among mixed target/false batches:

- target is top-1: `0.432, 0.338, 0.363, 0.622, 0.632`;
- target is top-3: `0.705, 0.669, 0.688, 0.756, 0.732`;
- a false candidate outranks all targets:
  `0.543, 0.662, 0.632, 0.374, 0.364`.

Batching removes order bias, but many mixed batches still cannot be resolved
reliably by original score.

## Trajectory Interpretation

Sequential and batched Step-1 candidate pools match for all 25 rollout starts.
For Steps 2-5 the matching rate is zero because emitted context has already
changed. Later differences are policy-level trajectory differences, not a
fixed-pool re-ranking result.

The 50-record evidence supports these conclusions:

1. Score has limited information at Step 1, especially for passenger columns.
2. Effective score does not consistently destroy a strong original ordering;
   often neither score is sufficiently discriminative.
3. Field normalization alone does not repair the ranking.
4. A better within-batch WTA may recover some targets, but a scalar score cannot
   simultaneously preserve high target recall and sparse false activity.
5. The next high-value diagnostic is read-only candidate-to-segment provenance
   and recurrent branch coherence, rather than further global threshold tuning.

These are smoke-test findings, not formal 250-record conclusions.
