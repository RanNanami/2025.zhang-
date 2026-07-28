# Fig.9 Oracle Candidate Survival Diagnostic

## Safety Boundary

`--oracle-candidate-diagnostic` is an analysis-only switch for
`competitive_raw`. It is off by default and requires competition to be
enabled. Ground truth is encoded with a separate Fig.9 encoder only after the
raw prediction and competition result have already been fixed.

The oracle trace and summary both state:

```text
diagnostic_only = true
uses_ground_truth_for_analysis_only = true
ground_truth_does_not_affect_prediction = true
```

Ground truth is never passed to candidate competition, emitted-code
construction, rollout propagation, decoding, learning, ranking, or parameter
selection. Strict protocol JSON and checkpoint v1 are unchanged.

## Recorded Evidence

Each horizon row records:

- input, target, decoded prediction, error, and prediction availability;
- raw, emitted, and inhibited candidate sizes;
- target weekday, time, passenger, and total column sets;
- raw and emitted target hits and recall by field;
- suppressed target columns, survival ratio, and suppression ratio;
- raw and emitted false-column counts;
- target and false candidate original/effective scores and inhibition;
- target candidate score ranks using descending saved score with the
  competition stable order as the tie-break;
- available segment IDs, creation metadata, source fingerprints, and source
  cell IDs;
- provenance availability, branch counts, and coherence;
- one documented classification.

Strict Fig.9 does not enable segment creation provenance capture. The
diagnostic therefore records structural source cell IDs but reports
`provenance_available=false` instead of inventing branch coherence.

## Classification Thresholds

Classifications are applied in this order:

1. `TARGET_ABSENT_FROM_RAW`: raw total target recall is exactly zero.
2. `TARGET_PRESENT_BUT_SUPPRESSED`: target survival ratio is below `0.5`.
3. `TARGET_BRANCH_INCOHERENT`: provenance exists and target branch coherence
   is below `0.5`.
4. `TARGET_SURVIVES_BUT_DECODE_WRONG`: emitted passenger recall is at least
   `0.7`, but relative absolute error exceeds `0.25`.
5. `TARGET_SURVIVES_BUT_FALSE_DOMINATES`: false emitted ratio exceeds `0.5`,
   the best false score exceeds the best target score, or emitted total target
   recall remains below `0.7`.
6. `TARGET_PRESERVED`: none of the preceding conditions applies.

Continuous recall, score, suppression, and coherence statistics remain the
primary evidence; labels are only compact summaries.

## 50-Record Validation

Both runs used 50 records, warmup 20, horizon 5, reference continuous
prediction, strength `0.1`, tau `0.02`, and simultaneous tolerance `0.0`.

| Check | Oracle off | Oracle on |
|---|---|---|
| Predictions SHA256 | `eb34eb343e57...a241` | `eb34eb343e57...a241` |
| MAPE | 0.3198588 | 0.3198588 |
| Coverage | 1.0 | 1.0 |
| Model fingerprint | `8e86c6fac808...8130` | same |
| RNG fingerprint | `146a05aaca1b...5e00` | same |
| Stable competition trace SHA256 | `a7e278f4a08c...da94` | same |

Runtime fields were excluded from the competition trace hash because wall
clock measurements naturally differ between separate processes. Every
behavioral and candidate field was identical. The enabled run added only
`oracle_candidate_trace.csv` and `oracle_candidate_summary.json`.

## Preliminary 50-Record Evidence

| Step | Raw target recall | Emitted target recall | Target suppression | False emitted ratio |
|---:|---:|---:|---:|---:|
| 1 | 0.532 | 0.209 | 0.614 | 0.894 |
| 2 | 0.644 | 0.441 | 0.284 | 0.708 |
| 3 | 0.660 | 0.456 | 0.313 | 0.790 |
| 4 | 0.643 | 0.492 | 0.209 | 0.752 |
| 5 | 0.584 | 0.425 | 0.249 | 0.817 |

At Step 1, 20 of 25 rows were classified
`TARGET_PRESENT_BUT_SUPPRESSED`; competition removed a substantial fraction of
already incomplete true-column support. At Steps 2-5, most rows were
`TARGET_SURVIVES_BUT_FALSE_DOMINATES`: true columns remained, but false columns
still occupied most emitted activity.

This small run suggests both mechanisms matter:

- raw prediction is already incomplete before competition;
- the current local competition can suppress correct columns, especially at
  Step 1;
- later rollout still emits many false columns even when target candidate
  scores are stronger.

Passenger recall after competition remained only `0.184-0.268`, so this sample
does not support the specific hypothesis that passenger columns are mostly
correct and decoding alone causes the error. Creation provenance is
unavailable, so branch/chimera consistency cannot yet be judged honestly.

These are 50-record diagnostics, not a formal conclusion. The existing
250-record result was not modified or rerun. The user can produce the required
250 oracle evidence with:

```powershell
.\scripts\fig9_competitive_oracle_250.ps1
```
