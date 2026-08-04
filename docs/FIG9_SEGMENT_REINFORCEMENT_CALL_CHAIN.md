# Fig.9 Segment Reinforcement Call Chain

This audit describes the real execution path. The new trace copies values from
that path and never supplies values back to matching, selection, or learning.

## Observation and matching

1. `SequentialMemory.observe_code` starts at `src/seqmem/model.py:1997`. The
   proximal event is compared with `last_prediction_candidates` at line 2057.
2. A time-matched predictive candidate enters Scenario 1. If none exists,
   `_best_matching_neuron` (`src/seqmem/model.py:2431`) enumerates live incoming
   segments reached by the currently active source cells.
3. Each Scenario-2 candidate is scored once. Selection compares
   `(timed_overlap, candidate_score)` lexicographically at line 2477. Higher
   overlap wins before score.
4. Exact overlap-and-score ties use the learning RNG coin flip at lines
   2490-2491. The diagnostic records the already-computed candidate tuple; it
   does not repeat this traversal or consume another random value.
5. A selected old segment with overlap at least `L_match` enters Scenario 2.
   Scenario 2 calls `_reinforce_segment` with `grow_missing=True` and
   `depress_noncontributing=False` at lines 2170-2171. Otherwise Scenario 3
   creates a new segment on the least-used neuron.

## Reinforcement and forgetting

1. `_reinforce_segment` starts at `src/seqmem/model.py:3216`.
2. Existing per-segment Scenario counters are read and then incremented at
   lines 3236-3243. These counters are diagnostic metadata only when a segment
   has a diagnostic ID.
3. Scenario 1 strengthens contributing synapses and may weaken
   noncontributing synapses. Scenario 2 grows missing winner-context synapses,
   strengthens contributors, and does not weaken noncontributors.
4. A contributing synapse is increased by `delta_w`, capped at 1.0, at line
   3297. The callback receives immutable copies of weights and counts after the
   normal update, before pruning.
5. `_punish_wrong_predictions` starts at line 3386 and is separate from the
   selected-segment reinforcement event. `_prune_neuron` starts at line 3419;
   it is the path that removes expired synapses/segments.

## Trace timing and identity

`experiments/fig9_strict_reproduction.py` installs the callback immediately
before the real observation at line 2629 and removes it at line 2679, before a
checkpoint can pickle the model. The callback cannot select or reinforce a
segment. `build_segment_reinforcement_rows` is called after observation and
provenance registration at line 2803.

Stable output identity comes from `BranchProvenanceRegistry`, which binds a
segment's object identity to a deterministic ID based on its creation
transition, target column/neuron, source fingerprint, and creation ordinal.
Object IDs are used only for the same-process join; CSV output uses the stable
provenance ID. Checkpoints store and rebind this registry.

## Teacher/reference validity

`reference_applicability_from_trace` is defined at
`experiments/diagnostics/fig9_teacher_forced_identity.py:201`; teacher rows are
built at line 348, after the normal observation. The existing teacher segment
for Scenario 2 is therefore the segment selected by the very decision under
audit. Treating it as an independent correct label would be circular.

The reinforcement diagnostic applies these rules in
`experiments/diagnostics/fig9_segment_reinforcement.py:47`:

- Scenario 1: a unique time-matched predictive identity existed before the
  proximal observation, so reference compatibility is applicable.
- Scenario 1 with multiple predictive identities: the exact segment reference
  is ambiguous.
- Scenario 2: the post-observation operational reference is invalid for a
  wrong-selection judgment.
- Scenario 3: no existing segment is reinforced, so it has no reinforcement
  event row.

Consequently, `REFERENCE_UNAVAILABLE` and
`POST_OBSERVATION_REFERENCE_INVALID` are never labelled `wrong`. Ground truth
is used only to name the observed field/column and correlate later errors; it
does not affect matching, reinforcement, prediction, or RNG.
