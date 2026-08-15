# Phase 07 Learning Mutation Map

## Scope

This audit describes the real Phase 06 implementation at commit `a340461`.
Phase 07 may mechanically extract mutation bodies, but `SequentialMemory`
remains the state, RNG, and orchestration owner. The audit does not reinterpret
historical Scenario labels.

## Actual observation order

| Order | Location before Phase 07 | Reads | Writes | RNG | Identity/order sensitivity |
|---:|---|---|---|---|---|
| 1 | `observe_code()` 2142-2314 | previous context, saved candidates, actual event, segment graph | diagnostic-only locals | best-match tie RNG | candidate traversal, exact candidate/segment, timing and float sensitive |
| 2 | `classify_observation_learning_branch()` | already resolved booleans | none | no | pure Phase 06 decision |
| 3A | `observe_code()` 2315-2350 | least-used neuron result, previous winners | new S2B segment/synapses and incoming index | least-used RNG before mutation | source iteration and delay assignment sensitive |
| 3B | `observe_code()` 2356-2378 -> `_reinforce_segment()` | same saved S1 candidate and exact causal segment | weights, ages, counters, optional growth, incoming index | no new RNG | same object and mutation order required |
| 3C | `observe_code()` 2379-2396 -> `_reinforce_segment()` | selected S2A segment and previous winners | weights, ages, new synapses, incoming index | no new RNG | dict/source traversal and delay order required |
| 3D | `observe_code()` 2397-2413 | selected least-used neuron, previous winners | new S2B segment/synapses and incoming index | least-used RNG before mutation | same as 3A |
| 4 | `_reinforce_segment()` final call | mutated winner neuron | pruned synapses/segments, dirty incoming-source set, segment active flags | no | pruning happens immediately, before punishment and publish |
| 5 | `_punish_wrong_predictions()` | saved candidates, actual events, prior active context | failed-prediction weights/ages | no | candidate and contributing-source iteration order sensitive |
| 6 | `_punish_wrong_predictions()` per candidate | punished neuron | pruned synapses/segments, dirty incoming-source set | no | pruning happens immediately after each punishment |
| 7 | `observe_code()` 2736-2750 | completed active/winner/source sets and Scenario counts | transient context and `last_observe_stats` | no | publication order retained |

There is no independent post-publication forgetting pass. Moving pruning after
transient publication would change the real mutation order and is prohibited.

## Mutation formulas and ownership

### Reinforcement

`_reinforce_segment()` first records diagnostic state and resolves contributing
sources. For S1 it requires the same saved `PredictionCandidate` and exact
candidate segment. For S2A it uses the existing arrival-window calculation.
Missing S2A winner sources are appended before weight updates and registered in
the incoming index.

The selected segment is then traversed in existing dictionary order:

- contributor: `weight = min(1.0, weight + delta_w)`, then `age = 0`;
- noncontributor when requested: `weight = max(0.0, weight - delta_w)`, then
  `age += 1`.

Other segments on the same neuron are traversed afterward and depressed/aged
only when `depress_noncontributing` is true. Diagnostics are emitted after the
mutation; `_prune_neuron()` runs last.

### New segment creation

`_grow_segment()` returns early without active sources. Otherwise it constructs
one `Segment` whose synapse dictionary follows `active_sources.items()` order,
assigns each delay through `_new_synapse_delay()`, appends the segment to the
chosen neuron, then registers each source in `_incoming_index`. Diagnostic IDs
are allocated only under the existing branch-diagnostic flag.

### Failed-prediction punishment

`_punish_wrong_predictions()` traverses
`last_prediction_candidates.items()` and each candidate list in their existing
order. Confirmed identities and timing-confirmed events are skipped. For each
remaining candidate it computes the existing arrival-window contributors,
depresses those sources by `delta_w_bad`, increments age, and immediately
prunes that candidate's neuron.

## Phase 07 extraction boundary

Safe mechanical candidates are the already-resolved selected-segment
weight/age loop, the same-neuron other-segment depression loop, and the
already-resolved punishment weight/age loop. They accept explicit mutable
objects and scalar parameters, draw no RNG, search no model, and do not create
or replace segments.

The following stay in `SequentialMemory`: contributing-source resolution,
S1 candidate identity check, S2A growth and delay construction, incoming-index
maintenance, new-segment construction, least-used selection, diagnostics,
pruning orchestration, and all Scenario counters/reasons.
