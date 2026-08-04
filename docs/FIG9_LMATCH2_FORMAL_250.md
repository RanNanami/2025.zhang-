# Fig.9 Real L_match=2 Formal 250 Diagnostic

## Execution

- Branch: `experiment/fig9-competitive-inhibition`
- Formal attempts: 2
- Native crashes: 1 (`0xC0000005`)
- Failed last phase: index 225, rollout step 4, `rollout_step_enter`
- Recovery checkpoint: `next_index=220`
- Final exit code: 0

Core-only and full-diagnostic gzip replays from 220 to 226 both completed.
Their predictions, MAPE, model fingerprint, and RNG fingerprint were identical.
No diagnostic family or compression mode was a stable minimal crash trigger.
The formal run was therefore resumed from the original atomic checkpoint without
changing model semantics.

## L2 Result

- MAPE: 0.3927571082122346
- Final rolling MAPE: 0.34930143306795164
- Step MAPE: 0.399240, 0.340884, 0.390564, 0.401540, 0.392757
- Predictions: 45/45
- Coverage: 1.0
- Logical cumulative runtime: 1215.750 seconds
- Final segments: 1945
- Final synapses: 115473
- Mean/peak raw columns: 227.898 / 335
- Mean/peak emitted columns: 91.582 / 155
- Suppression ratio: 0.598143

Formal Scenario 1/2/3 counts are 181/925/244. Passenger-only counts are
35/179/236. Scenario 3 decreased, but Scenario 2 and repeated existing-segment
reuse increased.

## Comparison

| L_match | MAPE | Final segments | Ambiguity rate | Mean emitted columns |
|---:|---:|---:|---:|---:|
| 4 | 0.408656 | 2377 | 0.155160 | 124.018 |
| 3 | 0.437359 | 2225 | 0.202673 | 114.902 |
| 2 | 0.392757 | 1945 | 0.285078 | 91.582 |

L2 improves MAPE by 3.89% relative to L4 and 10.20% relative to L3. However,
multiple-segment/neuron ambiguity rises by 83.7% relative to L4. Matching
segment count rises from 1235 to 1558 while mean best overlap falls from 11.799
to 10.747. This is evidence of broader, less selective matching rather than an
unqualified memory improvement.

Step 2 onward has `recurrent_trajectory_divergence=true`; those differences are
paired recurrent outcomes, not fixed-state local threshold effects.

## Validation

- Independent MAPE recomputation: exact match
- Prediction CSV SHA256: `3cc2dfa2f4840ac1d7c70e1f028b57aa85ce5accb483ddb8b90d7a001581873a`
- Model fingerprint: `e069baa10a77abc3957ed6808432a51f798c1369cccfdebc1df5b003034a1e82`
- RNG fingerprint: `1c56cf7fd42bf31a9dbf59993d5b3ee248b72423984b99c56888fa2ac37753fe`
- Checkpoint reload: verified twice
- Gzip EOF: all files verified
- NaN/Inf: 0

## Decision

`LMATCH2_IMPROVES_ERROR_WITH_INSTABILITY`

Recommended next step: `DIAGNOSE_WRONG_SEGMENT_REINFORCEMENT`.

This is a nonpaper diagnostic. Strict defaults remain L_match=4.
