# Phase 06 Learning Decision / Mutation Boundary

## Function classification

| Function or block | Classification | Phase 06 action | Reason |
|---|---|---|---|
| proposed branch classifier over precomputed booleans | PURE_DECISION | extract | no traversal, RNG, floats, identity lookup, or mutation |
| timing confirmation list/max in `observe_code` | MIXED | keep | candidate traversal, timing gate, score selection, identity semantics |
| `_best_matching_neuron` | RNG_DECISION / READ_ONLY_SEARCH | keep | traversal and exact-tie learning RNG draw |
| matching `timed_overlap >= l_match` expression | READ_ONLY_SEARCH / float-sensitive | keep calculation in place | summation/traversal and tolerance must not move |
| `_least_used_neuron_index` | RNG_DECISION | keep | consumes learning RNG |
| `_reinforce_segment` | PERSISTENT_MUTATION | keep | weights, ages, growth, counters, prune order |
| `_grow_segment` | PERSISTENT_MUTATION | keep | segment/synapse creation, delays, index mutation |
| `_punish_wrong_predictions` | PERSISTENT_MUTATION / READ_ONLY_SEARCH | keep | eligibility, contribution timing, weight/age and prune order |
| transient publication at end of `observe_code` | TRANSIENT_PUBLISH | keep | established Phase 05 state owner/order |
| observation and parity trace construction | DIAGNOSTIC_ONLY | keep | artifact compatibility and identity evidence |
| `_prune_neuron` | PERSISTENT_MUTATION | keep | forgetting and container mutation order |

## Allowed helper contract

The helper may consume only values already resolved by the existing path. It
must not accept a model, candidate, Segment, column, RNG, tolerance, score,
overlap iterable, or active-source container. It may return only the historical
branch string used by the existing code.

Canonical paper names remain a documentation/test mapping. They are not written
to existing scientific artifacts.

## Identity and order invariants

- S1 receives the same `PredictionCandidate` and its exact `Segment` object.
- S2A receives the exact tuple returned by `_best_matching_neuron()`.
- S2B still calls least-used selection at the same point and with the same RNG.
- Paper S3 punishment remains after all proximal events and before transient
  publication.
- Reinforce, grow, punish, publish, and prune calls remain in their current
  relative order.
- No helper recalculates timed overlap, score, response, max, sort, or sum.
