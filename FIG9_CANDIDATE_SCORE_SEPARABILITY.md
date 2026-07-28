# Fig.9 Candidate Score Separability Diagnostic

## Scope

This is a nonpaper, offline-only diagnostic on
`experiment/fig9-competitive-inhibition`. Ground truth labels candidates only
after prediction. They do not affect prediction, competition, decoding,
learning, RNG, or checkpoint state.

The formal analysis uses the user-run 250-record outputs in:

`results/fig9_diagnostics/candidate_score_separability_250_20260728_222514/`

The sequential and batched prediction SHA256 values exactly match their
pre-existing 250-record runs:

- Sequential: `3DD597607BEF41E657615C19938B38F15622710A1D80C08AEB206921549D7AC6`
- Batched: `8E015313FF5CEB968419C7CA63CD2936BD22A63EA60205E66A63D87B0F290A8C`

Both policies also have identical final model and RNG fingerprints. The trace
is therefore read-only. A separate 50-record on/off test reached the same
conclusion.

## Method

- Candidate level: one saved `CompetitionDecision` is one sample.
- Column level: select maximum `original_score`; ties use
  `candidate_original_index`, then neuron. Scores are not averaged.
- Field offsets come from protocol encoder sizes: weekday `[0, 30)`, time
  `[30, 88)`, passenger `[88, 570)`.
- Bootstrap unit: one rollout start (`input_index`), with 1,000 samples and
  seed 0.
- Step 1 candidate pools match for all 45 rollouts. Steps 2-5 are policy-level
  comparisons after recurrent trajectory divergence.
- Candidate-to-segment provenance is unavailable in this trace.

The analysis contains 123,788 candidate rows and 123,788 column rows. The
one-to-one counts mean no duplicate candidate columns occurred in these runs,
although duplicate-column aggregation is implemented and tested.

## Prediction Results

| Policy | MAPE | Final rolling MAPE | Coverage |
|---|---:|---:|---:|
| Sequential | 0.608449 | 0.541128 | 1.0 |
| Batched | 0.485179 | 0.431497 | 1.0 |

Batched competition is better than sequential competition on this run, but
this diagnostic does not compare it against a new strict/raw run.

## Total Separability

Column-level PR-AUC is shown as `original / effective`. Prevalence is the
random-ranking baseline.

| Policy | Step | Prevalence | PR-AUC |
|---|---:|---:|---:|
| Sequential | 1 | 0.092 | 0.149 / 0.152 |
| Sequential | 2 | 0.123 | 0.345 / 0.357 |
| Sequential | 3 | 0.086 | 0.117 / 0.139 |
| Sequential | 4 | 0.080 | 0.137 / 0.147 |
| Sequential | 5 | 0.081 | 0.128 / 0.140 |
| Batched | 1 | 0.092 | 0.149 / 0.148 |
| Batched | 2 | 0.099 | 0.214 / 0.218 |
| Batched | 3 | 0.088 | 0.159 / 0.165 |
| Batched | 4 | 0.089 | 0.137 / 0.141 |
| Batched | 5 | 0.079 | 0.143 / 0.146 |

Total PR-AUC mixes fields with very different target prevalence and is not
evidence that passenger branches are separable.

## Field Separability

Original-score PR-AUC by field:

| Policy | Field | Step 1 | Step 2 | Step 3 | Step 4 | Step 5 |
|---|---|---:|---:|---:|---:|---:|
| Sequential | Weekday | 0.585 | 0.738 | 0.372 | 0.400 | 0.381 |
| Sequential | Time | 0.253 | 0.288 | 0.183 | 0.174 | 0.158 |
| Sequential | Passenger | 0.045 | 0.040 | 0.036 | 0.031 | 0.033 |
| Batched | Weekday | 0.585 | 0.590 | 0.486 | 0.417 | 0.432 |
| Batched | Time | 0.253 | 0.267 | 0.201 | 0.170 | 0.152 |
| Batched | Passenger | 0.045 | 0.040 | 0.043 | 0.039 | 0.056 |

Passenger prevalence ranges from 0.032 to 0.047. Its original-score PR-AUC is
at or close to prevalence from Step 1 onward. Effective score and the simple
`score + predicted_time` two-dimensional diagnostic do not consistently
improve it. Predicted time alone is somewhat informative only at Step 1
(passenger PR-AUC 0.065), and remains far too weak for sparse retrieval.

Offline field z-score and percentile normalization mostly reduce total
PR-AUC, so the result does not support installing field normalization in the
model.

## Threshold Feasibility

Pooled false-column counts required by `original_score`:

| Policy | Step | False at recall 0.5 | False at recall 0.7 |
|---|---:|---:|---:|
| Sequential | 1 | 5,759 | 7,769 |
| Sequential | 2 | 2,586 | 5,016 |
| Sequential | 3 | 6,752 | 9,391 |
| Sequential | 4 | 6,682 | 9,587 |
| Sequential | 5 | 6,091 | 9,000 |
| Batched | 1 | 5,759 | 7,769 |
| Batched | 2 | 4,416 | 6,646 |
| Batched | 3 | 5,012 | 7,805 |
| Batched | 4 | 5,314 | 7,841 |
| Batched | 5 | 5,012 | 7,926 |

At Step 1 this is about 124 false columns per rollout at target recall 0.5
(95% bootstrap CI 113-135), or 172 at recall 0.7 (CI 159-185). No scalar
threshold provides both high target recall and sparse false activity.

## Batch Competition

| Step | Candidates per batch | In multi-candidate batches | Mixed target/false | Target top-1 | Target top-3 | False outranks all targets |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.79 | 0.935 | 0.259 | 0.306 | 0.639 | 0.689 |
| 2 | 4.65 | 0.937 | 0.292 | 0.315 | 0.622 | 0.671 |
| 3 | 5.25 | 0.943 | 0.278 | 0.259 | 0.605 | 0.728 |
| 4 | 5.36 | 0.947 | 0.292 | 0.248 | 0.596 | 0.743 |
| 5 | 6.49 | 0.960 | 0.291 | 0.267 | 0.551 | 0.724 |

Batching removes sequential order bias, but in mixed batches a false candidate
outranks every target in 67-74% of cases. Only 22-30% of mixed batches are
separable by an original-score threshold.

## Conclusion

The formal 250-record evidence supports case B/C/F from the diagnostic plan:

1. Step 1 passenger score is already effectively inseparable, so recurrent
   context mixing is not the original cause.
2. `effective_score` does not consistently destroy a strong
   `original_score` ordering; both are weak for the passenger field.
3. Batched competition improves MAPE relative to sequential competition, but
   further scalar WTA or global-threshold tuning cannot provide sparse,
   high-recall context.
4. Field normalization and the tested score/time combination do not repair
   the ranking.
5. The next useful diagnostic is candidate-to-segment provenance and recurrent
   branch coherence, including whether candidates share the causal source
   segment and neuron identity expected for the active branch.

This is a diagnostic result, not a paper reproduction claim. No oracle
threshold, normalization, or ground-truth label is applied to prediction.

Detailed generated outputs are under:

`results/fig9_diagnostics/candidate_score_separability_250_20260728_222514/analysis/`
