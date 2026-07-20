# Fig.9 Protocol Audit and Correction

## Paper Target

Fig.9 uses 17,520 half-hour NYC taxi records. Weekday, time of day, and
passenger count use 30, 58, and 482 mini-columns, respectively. Each field
activates ten columns, every column has 32 neurons, `L_match=4`, and the
forgetting threshold is 65. The task predicts passenger count five steps, or
2.5 hours, ahead after one online pass through the stream.

The plotted paper values are approximate because the authors do not publish
their result arrays. Fig.9(b) places the original-data Ours bar near `0.10`.
Fig.9(c) places the changed-data bar near `0.10`. In Fig.9(d), Ours rises from
about `0.07` to `0.12` after April 1 and falls to roughly `0.05` by early May.

## Errors Found

1. Actual records were observed without first calculating the prediction from
   the previous record. That bypassed the paper's correct-prediction and
   incorrect-prediction learning cases.
2. Recursive retrieval fed each predicted code back as a new proximal input.
   The paper instead lets predictive neurons fire without external input and
   directly drive the next retrieval cycle.
3. An intermediate prediction was forced to equal a legal external Gaussian
   code. Neural prediction and event-wise inhibition can produce a partial or
   noisy sparse pattern; only the final value needs likelihood decoding.
4. Rollout consumed the learning tie-break RNG and changed later online
   learning even though retrieval was supposed to be read-only.
5. Literal burst context exposed up to 960 simultaneous cells to event-level
   distal matching. Without continuous membrane and inhibition integration,
   these spikes generated excessive false matches in the high-overlap taxi
   encoding.
6. A previous audit incorrectly changed rolling MAPE to use a target sum from
   the same 400-point window. Reference [58] divides the recent mean absolute
   error by the global mean absolute target.

## Retained Corrections

- Every actual record now predicts before applying the three paper learning
  cases.
- Continuous retrieval advances through the identities of predictive neurons,
  without proximal replay or learning.
- Strict retrieval advances raw predictive-neuron spikes. Coherent and
  eventwise projection remain diagnostics only.
- The final passenger value is decoded over the real-value likelihood grid.
- Rollout restores all transient state and cannot perturb later learning.
- Rolling MAPE now uses the exact global normalization from reference [58].
- All neurons in an unpredicted active mini-column drive next-cycle context,
  first-spike intracolumn inhibition is enabled, and scenario 1 reinforces a
  correct predictive segment without an extra `L_match` gate.

## Historical Compensated Full-Year Result

| Stream | Predictions | Coverage | Overall MAPE | Post-change MAPE | Final rolling MAPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| original | 11,611 | 1.000 | 0.131 | 0.089 | 0.066 |
| perturbed | 11,611 | 1.000 | 0.131 | 0.088 | 0.074 |

These files used winner-only context and eventwise projection. There are zero
pre-change prediction mismatches between the two runs. With the corrected
reference formula, perturbed rolling error is `0.089` on April 1, peaks near
`0.133` around April 8, falls to `0.090` by April 22, and reaches `0.065` by
April 29. The shape is close to Fig.9(d), but this is not a strict-default run.

Artifacts:

- `results/fig9_corrected_original.csv` and `.png`
- `results/fig9_corrected_perturbed.csv` and `.png`
- `results/fig9_corrected_comparison/summary.csv`
- `results/fig9_corrected_comparison/checkpoints.csv`
- `results/fig9_corrected_comparison/comparison.png`
- `results/fig9_strict_dynamics_2000.csv`
- `results/fig9_reference_comparison/bars.csv` and `comparison.png`

## Historical Diagnostics

Short and 6500-record checks were used before the full runs:

| Diagnostic | Result | Decision |
| --- | ---: | --- |
| literal full-burst context, 6500 records | 0.283 | exposed missing continuous competition |
| full burst plus first-spike inhibition, 2000 records | 0.437 | former strict diagnostic |
| winner context, 6500 records | 0.082 | historical compensation only |
| known future weekday/time, 6500 records | 0.278 | rejected; unnecessary future covariate clamp |
| direct lag-5 learning, 6500 records | 0.239 | rejected; not specified by paper |
| coherent intermediate code, 6500 records | 0.329 | rejected; contradicts neural retrieval |
| old growth-only/proximal replay, 1200 records | 0.115 | historical diagnostic only |

The old stack can achieve a visually attractive short-run number by skipping
two paper learning cases and replaying predictions as external input. It is not
used in the corrected result.

Paper-explicit bursting, within-column inhibition, raw neural retrieval, and
scenario-1 reinforcement are now defaults. Prediction now integrates distal
responses, dendritic threshold crossing, depolarization, phase-precessed soma
firing, and first-spike competition at `0.005` normalized-cycle resolution.
The article does not publish `V0`, `tau_m`, or `tau_s`, so the exact trajectory
is not recoverable from the supplied PDF.

A 250-record smoke run with 200 warmup records improved from event-score MAPE
`0.915` to continuous-integration MAPE `0.480`, with full coverage. This is a
mechanism check, not a paper-scale result. A pre-integration 2000-record strict
run obtained MAPE `0.745`; it is retained only as historical evidence.

### Continuous-integration runtime

The response-cache lookup is inlined inside the fine-grid potential loop. This
does not change the double-exponential kernel, rounding precision, integration
step, thresholds, or accumulation order. On the same 140-record/100-warmup
strict run, elapsed time fell from `264.740 s` to `212.053 s` (`19.9%`), while
all 35 prediction rows and MAPE `0.464` matched exactly. More aggressive input
grouping and analytic pre-filter experiments were rejected because they either
changed predictions or made this benchmark slower.

## Run

```powershell
# Run from the cloned repository root.
$env:PYTHONPATH = "$PWD\src;$PWD"
.\.venv\Scripts\python.exe experiments\fig9_paper_snn.py `
  --output-csv results\fig9_strict_default_original.csv `
  --output-plot results\fig9_strict_default_original.png
.\.venv\Scripts\python.exe experiments\fig9_paper_snn.py `
  --data data\paper_nyc_taxi_perturb.csv `
  --output-csv results\fig9_strict_default_perturbed.csv `
  --output-plot results\fig9_strict_default_perturbed.png
```

The historical compensated full-year runs took approximately 50 and 66 minutes.
The corrected fine-grid continuous implementation is substantially slower; a
strict full-year timing is not yet available. It therefore does not reproduce
the paper's real-time efficiency claim and remains an engineering gap.
