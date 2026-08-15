# Phase 07 Forgetting Boundary

## Current behavior

`Synapse.forgetting_score()` computes:

```text
l_weight * (1.0 - weight) + l_age * age
```

`SequentialMemory._prune_neuron()` retains a synapse only when that score is
strictly less than `forgetting_threshold`. It reconstructs each segment's
synapse dictionary in existing item order, marks removed incoming sources dirty,
sets `segment.active`, retains active segments in existing order, then replaces
the neuron's segment list.

Pruning is invoked at the end of each reinforcement and after each failed
prediction punishment. It occurs before `observe_code()` publishes the next
transient context.

## Classification

| Responsibility | Classification | Phase 07 action |
|---|---|---|
| score arithmetic from weight/age/scalars | PURE | eligible for extraction |
| strict `< forgetting_threshold` decision | PURE | eligible for extraction |
| iterate synapses in dictionary order | READ_ONLY_SEARCH / FLOAT_SENSITIVE | retain orchestration order |
| replace `segment.synapses` | PERSISTENT_MUTATION | retain in model |
| update `_dirty_incoming_sources` | PERSISTENT_INDEX_MUTATION | retain in model |
| set `segment.active` | PERSISTENT_MUTATION | retain in model |
| rebuild `neuron.segments` | PERSISTENT_MUTATION | retain in model |
| emit prune diagnostics | DIAGNOSTIC_ONLY / ORDER_SENSITIVE | retain in model |

## Proposed helper boundary

`seqmem._forgetting_helpers` may own plain functions for the exact score and
retention predicate. Inputs are only scalar values. The functions must preserve
the existing parenthesization and strict comparison. They must not accept a
model, neuron, segment, callback, RNG, or index.

The mutating prune loop remains in `SequentialMemory._prune_neuron()` because
moving it would combine container replacement, object activity, incoming-index
maintenance, and diagnostic callback order. This is a deliberate risk boundary,
not an incomplete attempt to redesign forgetting.
