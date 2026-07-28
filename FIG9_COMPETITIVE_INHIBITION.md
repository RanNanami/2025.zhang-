# Fig.9 Competitive Inhibition Diagnostic

## Scope

This branch adds a nonpaper `competitive_raw` rollout mode. The strict default
remains `raw`; prediction, observation, learning, RNG order, floating-point
order, and checkpoint v1 are unchanged.

Every competitive output is marked:

```text
diagnostic_only = true
competition_is_local_choice = true
```

## Candidate Mapping

`predict_code()` returns a raw `SymbolCode` and saves the corresponding neural
candidates in `last_prediction_candidates`. Each raw event is matched to the
same highest-score candidate that `prediction_active_cells()` would activate.
A candidate therefore has:

- mini-column and neuron index;
- predicted firing time;
- current dendritic score;
- the original segment and continuous-PSP provenance.

No expected value, future covariate, decoded replay, or ground truth enters
candidate selection.

## Competition Rule

Candidates are processed in stable order by predicted time, column, neuron,
and original raw-event order. Previously emitted candidates from other columns
contribute:

```text
inhibition_i =
    inhibition_strength
    * sum(exp(-(t_i - t_j) / inhibition_tau))

effective_score_i = original_score_i - inhibition_i
```

Candidate `i` emits only when `effective_score_i >= dendrite_threshold`.
Candidates in the same column do not inhibit one another. Time differences
inside `simultaneous_tolerance` are treated as zero but still processed in the
documented stable order. This is event-driven competition, not top-k pruning.

## Parameter Scale

The 50-record zero-strength trace observed step-level mean candidate scores
from `1.1080` to `1.9273`, with the existing threshold fixed at `1.0`.
Predicted times ranged from `0.0` to `0.5279` cycle units. The diagnostic grid
therefore used strengths `0.1, 0.25, 0.5, 1.0` and time constants
`0.01, 0.02, 0.05`. No value is promoted as a new default.

## Verification

- `compileall`: passed.
- Full unit suite: 143 tests passed.
- Added competition tests cover off/raw identity, zero-strength byte identity,
  deterministic ordering, exponential decay, same-column exclusion, model and
  RNG read-only behavior, trace independence, checkpoint v1, strict protocol
  isolation, and explicit nonpaper labels.
- The raw and zero-strength predictions CSV files have identical SHA256:
  `7d47fa82c803e82c939748892c80cb507d64af6d57a76efe3ffc400b2a3645cd`.
- All 14 small runs share model fingerprint
  `8e86c6fac808a07c845cf8919ac1c0f1b6da96e811a65eb26842c91212718130`
  and RNG fingerprint
  `146a05aaca1b7e5f27e4787e9a67d50ec003af268c832bdd254132d8a40e5e00`.

## 50-Record Grid

All runs used the same first 50 taxi records, warmup 20, horizon 5, seed 0,
reference continuous implementation, and original stream. This yields 25
rollout attempts. Runtime is reported only as a smoke diagnostic.

| Strength | Tau | MAPE | Coverage | Raw columns | Emitted columns |
|---:|---:|---:|---:|---:|---:|
| off | - | 0.6450 | 1.00 | 132.664 | 132.664 |
| 0.00 | 0.02 | 0.6450 | 1.00 | 132.664 | 132.664 |
| 0.10 | 0.01 | 0.3712 | 1.00 | 169.920 | 65.776 |
| 0.10 | 0.02 | **0.3199** | 1.00 | 171.784 | 60.880 |
| 0.10 | 0.05 | 0.4138 | 1.00 | 170.032 | 46.192 |
| 0.25 | 0.01 | 0.4151 | 1.00 | 164.160 | 36.688 |
| 0.25 | 0.02 | 0.4660 | 0.92 | 162.824 | 31.952 |
| 0.25 | 0.05 | 0.4454 | 1.00 | 162.712 | 24.144 |
| 0.50 | 0.01 | 0.4391 | 1.00 | 162.312 | 22.696 |
| 0.50 | 0.02 | 0.4443 | 1.00 | 161.952 | 19.568 |
| 0.50 | 0.05 | 0.5547 | 1.00 | 161.688 | 15.008 |
| 1.00 | 0.01 | 0.5334 | 1.00 | 160.768 | 15.320 |
| 1.00 | 0.02 | 0.5924 | 0.56 | 160.160 | 11.824 |
| 1.00 | 0.05 | 0.4625 | 0.16 | 155.056 | 9.744 |

For the lowest-MAPE grid point (`strength=0.1`, `tau=0.02`), emitted columns
by horizon step were `60.32, 45.92, 64.44, 63.72, 70.00`. Step coverage stayed
at `1.0`; step MAPE rose from `0.2068` at step 1 to `0.3199` at step 5.

## Interpretation

The result matches outcome A at this small scale: weak competition sharply
reduces propagated density while MAPE improves and coverage remains complete.
This supports missing continuous intercolumn competition as one plausible
cause of density explosion.

It is not sufficient to call competition the sole root cause. Stronger
inhibition continues to reduce density but stops improving MAPE and eventually
collapses coverage. Competition also changes later raw candidate density,
showing that branch identity and recurrent context remain important. Formal
250-record behavior has not been tested, and `competitive_raw` must remain a
diagnostic mode rather than a strict default.

Compact machine-readable results are stored in
`results/fig9_diagnostics/competitive_inhibition_20260728/grid_summary.csv`
and `grid_summary.json`. Full traces remain local and reproducible.
