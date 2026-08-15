# Phase 05 Transient State Mutation Map

This map records the current execution path before extraction. `SNAPSHOT` and
`RESTORE` refer only to the ten values in `TransientStateSnapshot`.

| Stage | previous active/winners | prediction candidates/stats | observe stats | symbol ranking | source labels | RNG | persistent memory |
|---|---|---|---|---|---|---|---|
| 1. encode | no model mutation | no mutation | no mutation | no mutation | no mutation | no draw | no mutation |
| 2. predict | READ context | CLEAR then WRITE candidates/stats | no mutation | no mutation | READ for diagnostics | no learning draw | READ segments/synapses |
| 3. observe | READ previous context | READ same-call candidates | prepares WRITE | no mutation | prepares WRITE | possible learning draw through selection | READ and possibly WRITE when learning is enabled |
| 4. learn | READ winners as growth/contribution sources | READ candidates for Scenario credit | WRITE Scenario counters | no mutation | READ provenance labels | possible `_learning_rng` draw | WRITE segment/synapse weight/age/structure |
| 5. publish state | WRITE active cells and winners | retained until next predict/reset | WRITE `last_observe_stats` | retained | WRITE predicted/burst-only sets | no reorder | no additional mutation |
| 6. propagate | WRITE from prediction-active cells | READ same-call candidates | no mutation | decode may WRITE ranking | WRITE predicted set and CLEAR burst-only set | decode may draw `_decode_rng` | no mutation |
| 7. forget | READ current context only for trace metadata | READ candidates for punishment path | no mutation | no mutation | no mutation | no draw | WRITE ages/weights/index invalidation and prune structures |

## Snapshot operations

| Operation | Exact action |
|---|---|
| SNAPSHOT maps | `dict.copy()` |
| SNAPSHOT candidate map | dict comprehension with `candidates.copy()` |
| SNAPSHOT ranking | `list.copy()` |
| SNAPSHOT source labels | `set.copy()` |
| SNAPSHOT RNG | decode `getstate()` then learning `getstate()` |
| RESTORE maps | assign fresh `dict.copy()` objects |
| RESTORE candidate map | assign new dict and shallow-copied lists |
| RESTORE ranking | assign fresh list copy |
| RESTORE source labels | assign fresh set copies |
| RESTORE RNG | decode `setstate()` then learning `setstate()` |
| RESET | assign fresh empty dict/list/set containers; RNG untouched |

## Read/write ownership

- Prediction owns clearing and repopulating its candidate/stat outputs.
- Decode owns only `last_symbol_ranking` and decode RNG consumption.
- Observation owns publishing the next proximal context and Scenario stats.
- Autonomous propagation owns publishing the next raw-neural context.
- Learning owns persistent segment/synapse mutation and learning RNG consumption.
- Evaluation owns SNAPSHOT/RESTORE orchestration but never persistent rollback.
- `SequentialMemory` remains the owner of all fields after extraction.
