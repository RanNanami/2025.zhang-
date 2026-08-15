# Phase 08 Prediction Flow Map

## Scope

This map records the real prediction path at Phase 08 start commit `57319b5`.
It is an architecture audit, not a proposed numerical rewrite. Line ranges are
the pre-extraction locations in `src/seqmem/model.py`.

## Actual execution order

| Step | Current location | State reads | State writes | RNG | Float-order sensitive | Identity sensitive | Classification / extraction decision |
|---:|---|---|---|---|---|---|---|
| 1 | `predict_code()` 1061-1075; `_active_sources()` 769 | previous active cells/winners, mode flags | clears prediction candidates/stats and optional traces | no | no | containers only | orchestration; retain publication owner |
| 2 | `predict_code()` 1087-1115; `_live_incoming()` 4237 | active-source dict, incoming index, segment synapses | lazy incoming-index compaction only when dirty | no | yes: source order accumulates upper bound | exact Segment object | traversal/search; retain |
| 3 | `predict_code()` 1201-1223 | candidate tuple, `v_rest`, upper bound, threshold | trace-only fields | no | yes: existing addition/comparison | Segment key via runtime identity | scalar predicate may be extracted; traversal stays |
| 4 | `_continuous_segment_prediction()` 3058-3173 | exact segment, active sources, delays, weights, dynamics/cache | response cache and optional diagnostic metadata | no | yes | exact Segment and synapses | retain completely |
| 5 | `reference_continuous_prediction()` 3195 onward and optimized variants | ordered arrivals, PSP parameters, thresholds, time grid | response cache and diagnostic mapping | no | extremely sensitive | no new model objects | retain all reference/optimized numerical paths |
| 6 | `predict_code()` 1345-1411 | timed score, soma capability, firing window parameters | trace-only fields | no | yes | exact Segment | pure window predicate may be extracted; event/continuous math stays |
| 7 | `predict_code()` 1412-1485 | column, rounded target time, score, prior event winner | `best_by_event`, trace-only replacements | no | round and score comparison sensitive | exact winner Segment | preserve iteration, `round(12)`, and strict `>` replacement in model |
| 8 | `predict_code()` 1534-1627; `_select_intracolumn_event_winners()` 882-1042 | ordered event winners, policy, metadata | selected-event list and trace-only ranking | no | sorting/key sensitive | exact selected Segment | retain completely; optional competitive boundary unchanged |
| 9 | `predict_code()` 1679-1717 | selected values and segment metadata | new `PredictionCandidate` objects in `last_prediction_candidates` | no | threshold margin subtraction | candidate must retain exact Segment | deterministic construction/field packaging is safe if factory and object are passed explicitly |
| 10 | `predict_code()` 1718-1752 | selected events, active/candidate counts | trace flags, `last_prediction_stats`, returned `SymbolCode`/`SpikeEvent` | no | event-time subtraction | candidate structure already fixed | deterministic stats/event packaging is safe |

## Continuous path boundary

`_continuous_segment_prediction()` constructs matched arrivals in active-source
order and dispatches through `_continuous_prediction_from_arrivals()` to
`reference`, `optimized_v1`, or `optimized_v2`. The strict/default value remains
`reference`. Potential loops, rounded cache keys, dictionary lookups, threshold
crossing bisection, peak search, soma integration, memoization, and diagnostic
contribution reconstruction are all excluded from Phase 08 extraction.

## Selection boundary

The established path first retains the highest score for an exact
`(column, round(target_time, 12))` key, replacing only on strict `>`. With
intracolumn inhibition and policy `existing`, it then keeps the earliest firing
time per column in insertion order. Alternative policies continue through the
existing `_select_intracolumn_event_winners()` method. No selection helper or
competition behavior will be redesigned.

## Publication boundary

Every saved candidate is newly constructed for the current prediction but holds
the exact selected `Segment` object. Its crossing contribution objects are
reused from the same prediction metadata. `advance_prediction()` later consumes
that published candidate map to create autonomous transient context; it remains
outside this phase.
