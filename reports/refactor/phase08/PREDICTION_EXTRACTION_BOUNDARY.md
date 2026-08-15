# Phase 08 Prediction Extraction Boundary

## Allowed mechanical helpers

The only proposed production helpers are small and explicit:

1. upper-bound threshold eligibility from already accumulated scalar values;
2. valid soma firing-window predicate with the original short-circuit order;
3. exact event key packaging using the existing 12-digit `round`;
4. `PredictionCandidate` construction from an explicit factory, exact Segment,
   selected scalars, and metadata from the same prediction;
5. prediction stats dictionary packaging;
6. emitted `(column, time)` value packaging before `SpikeEvent` construction.

These helpers may receive factories or scalar/container values, but never a
model, RNG, encoder, experiment, diagnostic loader, or ground truth. They do not
search, sort, select, sum PSP values, or copy segments/candidates.

## Retained in `SequentialMemory`

- `_active_sources()` and `_live_incoming()` traversal;
- candidate-segment aggregation and contribution ordering;
- all continuous/event response calculations;
- all response caches and memoization;
- reference/optimized implementation dispatch;
- threshold crossing and soma firing solvers;
- event score replacement and tie/order behavior;
- existing/alternative intracolumn selection;
- trace/preselection/competition diagnostics;
- publication ownership of `last_prediction_candidates` and stats;
- `advance_prediction()` and all autonomous rollout behavior.

## Identity rule

Candidate construction must receive the exact selected Segment object and pass
it directly to the existing `seqmem.model.PredictionCandidate` constructor. No
deep copy, structural clone, diagnostic ID lookup, or causal-segment reselection
is allowed. Crossing contribution objects are filtered in original metadata
order and retained by identity.

## Numerical rule

No helper may enter `_continuous_segment_prediction()`,
`reference_continuous_prediction()`, either optimized path, `_segment_score()`,
or `_cached_spike_response()`. Scalar expressions moved to helpers retain their
parenthesization and built-in call (`round`) exactly. Regression comparison uses
raw serialized float values and byte-exact trace files, not tolerances.
