# Fig.9 Autonomous Context Provenance Call-Chain Audit

This document records the source-level semantics used by the read-only
autonomous-context provenance diagnostic.  It is intentionally separate from
the model protocol and does not change matching, selection, learning, RNG, or
checkpoint model state.

## Call chain

1. `experiments/fig9_strict_reproduction.py:2811` enters the per-record loop.
   For an index at or after warmup it calls `rollout_raw_autonomous(...)`
   before the current real record is observed.
2. `rollout_raw_autonomous` at line `1273` snapshots transient
   model state, then obtains the current context from
   `SequentialMemory._active_sources()` (`src/seqmem/model.py:630`). With
   `burst_context=True`, this is `previous_active_cells`, falling back to
   `previous_winners` only when the active set is empty.
3. Each rollout step at `experiments/fig9_strict_reproduction.py:1403` calls
   `SequentialMemory.predict_code(...)` once. The
   predictor reads the active source map and fills
   `model.last_prediction_candidates`. Candidates contain the segment,
   crossing time, PSP contribution metadata, score, and predicted time.
4. The raw `SymbolCode` is optionally passed through the diagnostic-only
   `compete_prediction_candidates(...)`. The returned emitted candidate list
   is converted to a new raw `SymbolCode` by
   `emitted_prediction_code(...)`; decoded passenger values are never encoded
   back into the model.
5. `model.prediction_active_cells(propagated)` maps the emitted column/time
   events to actual predictive neuron identities. Those active cell IDs are
   copied into `previous_active_cells` and `previous_winners` immediately
   before the next rollout step. `BranchProvenanceRegistry.set_autonomous_sources`
   records the same emitted IDs as a diagnostic label only.
6. In the `finally` block of `rollout_raw_autonomous`, the saved transient
   state is restored. Segment/synapse structure, weights, ages, and persistent
   learning state are not changed by autonomous rollout.
7. After rollout, `run_strict_stream` calls `model.observe_code(...)` at
   `experiments/fig9_strict_reproduction.py:3189` for the
   real record. This is the only proximal input in the strict record loop and
   is the point where learning, winners, burst cells, segments, synapses,
   weights, and ages may change.

## Actual versus autonomous state

The provenance tracker maintains two separate concepts:

- **Source identity provenance**: whether a source cell has appeared in the
  actual observation history before the current rollout anchor.
- **Current activation provenance**: whether the current activation was present
  before rollout, or was generated at rollout step 1--5.

The same cell can therefore have `source_identity_actual_history_count=1`
while its current activation is `ROLLOUT_STEP2_GENERATED`. That is not a
contradiction; it is the distinction needed to audit recurrent self-refresh.

Actual history is updated only after `observe_code` completes, at
`experiments/fig9_strict_reproduction.py:3419`, using the current active
sources and learning winners. Autonomous activation timestamps
are held only inside the tracker for the current rollout. They do not update
`actual_last_activation` and are not written into model state.

## Context semantics

`burst_context=True` means prediction context is an all-cell active context,
not winner-only context. The emitted raw prediction cells become the next
rollout context. A source-level trace is optional and capped at 32 contributors
per candidate; formal runs should use candidate-level output to avoid a
source-by-candidate expansion.

## Ground-truth boundary

The target code is encoded only after a candidate prediction exists and is used
to add post-hoc target/false labels. It does not enter candidate generation,
ranking, competition, propagation, or learning. This is why provenance output
cannot itself improve MAPE.

## Stable identifiers and resume

`rollout_id` is derived from stream, anchor timestamp, seed, and a trajectory
ordinal. Candidate IDs use the existing stable candidate-event hash and the
registry's stable segment provenance ID when available. Python object IDs are
not written to output. The tracker checkpoint stores only actual-history
activation indices and counters, so a checkpoint resume does not serialize
file handles or model views.
