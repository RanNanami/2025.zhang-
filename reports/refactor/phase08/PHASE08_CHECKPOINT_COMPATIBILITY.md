# Phase 08 Checkpoint Compatibility

## Result

PASS. Phase 08 did not move or rename any pickle-visible `seqmem.model.*`
class, did not change `MemoryParams`, transient-state fields, checkpoint schema,
or candidate ownership.

## Evidence

- `test_pre_refactor_checkpoint_loads_with_stable_model_path`: PASS.
- `test_current_checkpoint_round_trip_is_exact_in_fresh_process`: PASS.
- The current round trip preserves the model, full-state, RNG, previous-active,
  previous-winner, candidate, and segment/synapse fingerprints exactly.
- `PredictionCandidate` remains defined at `seqmem.model.PredictionCandidate`.
- The extracted constructor receives and stores the exact selected `Segment`
  object; no clone, lookup, or reconstruction occurs.
- The helper module contains plain functions only and contributes no persistent
  object to the checkpoint.

The six-fixture Golden comparison also reports exact candidate, full-state,
RNG, and structure fingerprints after the extraction.
