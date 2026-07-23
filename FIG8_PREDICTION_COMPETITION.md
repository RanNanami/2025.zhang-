# Fig.8 Prediction Competition Diagnostic

## Scope

This fourth-stage diagnostic tests two hypotheses without changing the strict
Fig.8 defaults:

1. continuous predictions might return nearly identical threshold-level scores;
2. learned delays might omit the double-exponential PSP rise time.

All runs are labeled **nonpaper diagnostic** and use CBT, seed 11, one-pass
online learning, V0/`response_scale=1.0`, 100 columns x 10 neurons, K=10,
L_match=3, forgetting threshold 500, raw predictive neurons, and no decoded-word
replay. The strict defaults remain `current-score`, `current-delay`, raw neural
propagation, and Scenario-1 `arrival-window`.

The ranking rollouts apply diagnostic eventwise top-1 selection to the raw
PredictionCandidates. All three rankings reuse the same trained model and the
same candidate objects. They differ only in which real prediction cells are
allowed to advance the autonomous context.

## Implementation and Verification

`PredictionCandidate` now exposes the soma firing time saved by the same
`predict_code()` call as its score, actual dendritic peak, first crossing, and
PSP contributors. `select_prediction_events()` accepts three explicit read-only
rankings:

- `current-score`: descending returned score;
- `actual-peak`: descending saved peak dendritic potential;
- `earliest-crossing`: ascending saved crossing time.

The delay diagnostic has two explicit modes:

- `current-delay`: existing strict formula;
- `peak-aligned-delay`: subtracts `kernel_peak_time=0.0402359478` from each new
  synaptic delay exactly as proposed. It is not clamped and is not a strict mode.

`compileall` passed and the complete suite passed **87/87 tests**: all 80 prior
tests plus 7 new tests for trace invariance, same-candidate metadata, absence of
ground-truth ranking inputs, read-only ranking, delay-only parameter changes,
autonomous A-B-C retrieval, and unchanged defaults.

The auditable output is under
`results/fig8_diagnostics/prediction_competition/`. Every accepted prediction
event has nonempty peak, crossing, and soma metadata. Expected-event row counts
are 1,800 for 20 sentences, 9,000 for 100, and 18,000 for 200 per mode.

## Ranking Results

The strict raw rows below are the preserved Stage-3 arrival-window baseline.
The other rows are this stage's same-model, eventwise top-1 ranking rollouts.

| Sentences | Propagation/ranking | Mean Lev. | Step 1/2/3/4 error | Expected word present | Next-step raw columns | Cue burst cells | No prediction / early stop |
|---:|---|---:|---|---:|---:|---:|---:|
| 100 | strict raw baseline | 1.590 | .170/.330/.410/.680 | .720 | 24.935 | 16.696 | .040/.070 |
| 100 | current-score | **1.550** | .170/.260/.440/.690 | .680 | 14.887 | 16.696 | .115/.250 |
| 100 | actual-peak | 1.590 | .160/.320/.440/.670 | .678 | 15.723 | 16.696 | .143/.300 |
| 100 | earliest-crossing | 1.870 | .200/.400/.520/.750 | .613 | 13.979 | 16.696 | .160/.290 |
| 200 | strict raw baseline | 2.600 | .370/.555/.745/.930 | .514 | 35.255 | 31.045 | .026/.060 |
| 200 | current-score | **2.570** | .365/.545/.745/.925 | .454 | 20.793 | 31.045 | .295/.520 |
| 200 | actual-peak | 2.580 | .340/.555/.760/.925 | .451 | 21.344 | 31.045 | .274/.500 |
| 200 | earliest-crossing | 2.940 | .480/.655/.840/.965 | .364 | 22.436 | 31.045 | .404/.640 |

The tiny Levenshtein change from strict raw to current-score is accompanied by
much worse presence and early stopping, so it is not a robust improvement.
Neither alternative ranking improves both accuracy and context contraction.
Therefore no 500-sentence run was started.

## Score Competition

| Sentences | Accepted candidates | Near threshold | Score values | Mean values/group | Top-tie groups | Mean top tie | Corr(score, peak/crossing/contributors) |
|---:|---:|---:|---:|---:|---:|---:|---|
| 100 | 18,042 | 0.0000% | 2,840 | 2.334 | 1.931% | 1.020 | .827/-.018/.687 |
| 200 | 54,552 | 0.0238% | 8,423 | 4.046 | 1.306% | 1.014 | .809/-.020/.680 |

The score is not collapsing to the threshold or to one shared value. It is
strongly correlated with actual peak and contributor count, but almost
uncorrelated with crossing time. Eventwise selection is deterministic and only
about 1-2% of competition groups have a top-score tie, so the previous top-1
selector was not literally random. Its deeper problem is that a strong segment
does not necessarily identify the correct contextual branch.

For correct timed candidates, post-hoc mean ranks were:

| Sentences | Current score | Actual peak | Earliest crossing |
|---:|---:|---:|---:|
| 100 | 1.359 | **1.325** | 1.483 |
| 200 | 1.757 | **1.689** | 2.117 |

Actual peak slightly improves the local rank of a correct event, but selecting
it for rollout does not improve sentence recall and increases next-step raw
columns. Local event quality is therefore insufficient to resolve branch
identity over multiple autonomous steps.

## Timing Results

| Sentences | Column missing | Column present, timing mismatch | Timed presence | Residual mean/p50/p90/p99 |
|---:|---:|---:|---:|---|
| 100 | 17.278% | 11.889% | 70.833% | -.0006/.0064/.0263/.1751 |
| 200 | 28.917% | 20.211% | 50.872% | -.0195/.0064/.0300/.2872 |

Column absence is the larger failure category, although timing mismatch also
grows with capacity. Mismatch residuals are not a common positive offset:

- at 100 sentences, 47.7% are positive and 52.3% negative; only 11.8% lie within
  `kernel_peak_time +/- 0.005`;
- at 200 sentences, 35.1% are positive and 64.9% negative; only 7.2% lie within
  that interval.

Thus the long residual tail reflects wrong or ambiguous temporal branches, not
a uniform omitted PSP rise time.

Cue burst cells / raw columns for current-delay were:

| Sentences | 1->2 | 2->3 | 3->4 | 4->5 | 5->6 |
|---:|---|---|---|---|---|
| 100 | 26.92/47.47 | 13.15/20.12 | 14.80/21.16 | 12.60/19.13 | 16.01/19.84 |
| 200 | 42.29/65.40 | 27.54/39.10 | 27.74/37.34 | 27.75/37.34 | 29.91/34.96 |

The cue is already dense and burst-contaminated before autonomous suffix
ranking begins.

## Delay Ablation

Both rows below use current-score eventwise retrieval; only the learned delay
formula differs.

| Sentences | Delay | Mean Lev. | Step 1/2/3/4 error | Expected word present | Raw columns | Cue burst | No prediction / early stop |
|---:|---|---:|---|---:|---:|---:|---:|
| 100 | current | **1.550** | .170/.260/.440/.690 | .680 | 14.887 | 16.696 | .115/.250 |
| 100 | peak-aligned | 3.990 | 1.00/1.00/1.00/1.00 | .023 | 17.735 | 88.014 | .190/.340 |
| 200 | current | **2.570** | .365/.545/.745/.925 | .454 | 20.793 | 31.045 | .295/.520 |
| 200 | peak-aligned | 3.980 | .985/.995/1.00/1.00 | .039 | 24.881 | 84.305 | .351/.530 |

Peak alignment moves the timing-residual median from about `+0.0064` to
`-0.0343`, just outside the fixed `0.03` tolerance. Correct timed presence falls
to 2.2% at 100 and 5.0% at 200, causing near-global cue bursts. The literal
formula also creates negative delays for about 0.98% of learned synapses when a
late source event connects to an early target event. It does not reduce burst or
prediction error and is not suitable as a strict default.

## Root-Cause Judgment

1. Accepted scores do **not** degenerate to one threshold-level value.
2. Previous eventwise top-1 is not tie-driven random selection; its score lacks
   enough neuron/branch identity information to choose a stable context.
3. Correct-event failure is primarily column absence, with a substantial but
   smaller timing-mismatch component.
4. Timing residuals do not concentrate near positive `kernel_peak_time`.
5. Actual-peak and earliest-crossing do not improve autonomous context
   contraction through 200 sentences.
6. Literal peak-aligned delay sharply increases timing mismatch and burst.
7. No mode satisfies the 200-sentence gate, so 500 and 1000 were not run.
8. The remaining capacity failure is more consistent with burst/shared-context
   contamination, neuron-identity branch ambiguity, and missing continuous
   intercolumn inhibition than with score collapse or one constant delay bias.

These diagnostics do not constitute a complete numerical reproduction of
Fig.8 and do not alter the strict paper-facing configuration.
