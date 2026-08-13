# Instrumentation Repair Report

## Scope

This is a Phase-A/Phase-B read-only audit over 20 CBT sentences, seed 11, response scale 1.0, raw training observations, and the unchanged local timing tolerance 0.03. No tolerance sweep or strict/default change was made.

## Repair

The previous delta histogram was aggregated from post-hoc punishment rows whose nearest-candidate timing fields were unavailable for `NO_PROXIMAL_EVENT` cases. It therefore wrote zero counts even though the source trace contained timing-rejected same-column candidates. This run computes deltas directly from the gate-before observation snapshot and uses one row per active column timing rejection. The total is `123` and the V2 bins sum to the same number.

The previous SSTD rank was a global candidate ordering and could exceed K=10. This run defines `predicted_sstd_rank` from the same raw prediction code's ordered selected column events and `actual_sstd_rank` from the encoded proximal event's ordered K events. These are post-hoc labels only and never affect training.

## Phase-B counts

| quantity | count |
|---|---:|
| active column events | 2000 |
| any prediction candidate | 172 |
| unique candidate | 172 |
| multiple candidates | 0 |
| current Scenario 1 | 49 |
| timing rejected | 123 |
| unique timing rejected | 123 |
| unique rejected and exact candidate punished | 123 |
| unique rejected to current S2A | 13 |
| unique rejected to current S2B | 110 |
| branch-replacement/double-credit cases | 123 |

## Gate decision

The numerical gate is not paper-explicit in the reviewed source. The paper does explicitly describe a predictive/depolarized neuron firing before its same-column peers, but multi-candidate soma competition remains only partially specified. This sample has nonzero unique candidates rejected only by the local time gate and those cases create a measurable missed-S1/punishment or fallback-branch population. Phase-C narrow candidate gate: **PASS**.

Primary audit conclusion: `LOCAL_ABSOLUTE_TIMING_GATE_CAUSES_DOUBLE_NEGATIVE_CREDIT`.

The complete source trace is `PRE_TIME_GATE_ACTIVE_COLUMN_TRACE.csv`; all rates use event-specific denominators in `TEMPORAL_CREDIT_RATE_SUMMARY_V2.csv`.
