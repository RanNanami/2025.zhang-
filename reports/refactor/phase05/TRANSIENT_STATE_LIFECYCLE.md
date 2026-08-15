# Phase 05 Transient State Lifecycle

Status: pre-extraction audit at commit `3d0076768a2e9e697133b5feb64bd458ff5ee096`.

## Boundary

`SequentialMemory` remains the direct owner of every field. Phase 05 may extract
only pure copy/reset helpers for the existing `TransientStateSnapshot` boundary.
It does not introduce a `StateManager`, a full model-state object, or a persistent
memory rollback mechanism.

The existing snapshot contains exactly ten values:

1. `previous_active_cells`
2. `previous_winners`
3. `last_prediction_candidates`
4. `last_prediction_stats`
5. `last_observe_stats`
6. `last_symbol_ranking`
7. `previous_predicted_sources`
8. `previous_burst_only_sources`
9. `_decode_rng.getstate()`
10. `_learning_rng.getstate()`

It excludes `columns`, neurons, segments, synapses, weights, ages, delay values,
the incoming index, numeric caches, configuration, callbacks, and diagnostic
provenance counters. A snapshot therefore protects evaluation/rollout context;
it cannot undo learning.

## Container and identity contract

| Value | Current capture depth | Current restore depth | Required identity behavior |
|---|---|---|---|
| active/winner maps | `dict.copy()` | `dict.copy()` | New dict; scalar keys/values equal |
| candidate map | new dict plus `list.copy()` per column | same | Candidate objects are identical |
| candidate segment | not copied | not copied | Segment object is identical |
| prediction/observe stats | `dict.copy()` | `dict.copy()` | New dict; scalar values equal |
| symbol ranking | `list.copy()` | `list.copy()` | New list; tuple items equal |
| predicted/burst source sets | `set.copy()` | `set.copy()` | New set; integer members equal |
| RNG state | `getstate()` | `setstate()` | Future draws reproduce exactly |

No helper may call `deepcopy`, rebuild a `PredictionCandidate`, clone a `Segment`,
or alter dict/list iteration order.

## Lifecycle

### Construction

`SequentialMemory.__init__` creates both RNG instances first, then the eight
empty transient containers in their current order. Callback and provenance
fields are separate and are not part of `TransientStateSnapshot`.

### Prediction

`predict_code()` replaces `last_prediction_candidates` and
`last_prediction_stats` with new empty dicts before traversing persistent
segments. It reads the active context selected by `_active_sources()` and then
publishes candidates and scalar prediction statistics. It does not update
`previous_active_cells` or `previous_winners`.

### Decode

`decode_symbols_from_prediction()` reads the candidates produced by the same
prediction call. Tie handling can consume `_decode_rng`; it finally stores a
shallow copy in `last_symbol_ranking`. It does not observe or learn.

### Observation and learning

`observe_code()` reads the prior candidate/context state while applying the
existing Scenario and learning rules. At the end it publishes new
`previous_active_cells`, `previous_winners`, predicted/burst source labels, and
`last_observe_stats`. Learning may consume `_learning_rng` and mutate persistent
segments/synapses; transient restore cannot reverse those mutations.

### Autonomous propagation

`advance_prediction()` derives actual active prediction cells from the current
raw code and the same `last_prediction_candidates`, then publishes active cells,
winners, and source labels. It does not provide proximal replay and does not
learn.

### Evaluation isolation

Fig.8 and Fig.9 capture one snapshot before no-learning evaluation/rollout and
restore it in `finally`. Restore assigns the eight copied containers in the
current order, then restores decode RNG followed by learning RNG. This ordering
is part of the Phase 05 parity contract.

### Reset

`reset_state()` assigns fresh empty containers to the same eight fields. It does
not reset either RNG and does not touch persistent memory, caches, callbacks,
configuration, or diagnostic provenance counters.

## Checkpoint boundary

`SequentialMemory.__getstate__()` copies `__dict__`, empties only the reproducible
spike-response cache, and removes runtime-only callbacks/context. Phase 05 does
not alter this payload, class module paths, or pickle schema. In particular,
`TransientStateSnapshot` remains defined as `seqmem.model.TransientStateSnapshot`.

## Fields deliberately left out

The provenance indices, diagnostic callbacks, caches, persistent memory, and the
`UNKNOWN` `_next_segment_diagnostic_id` remain direct fields on
`SequentialMemory`. Their lifecycle is not widened or normalized in this phase.
