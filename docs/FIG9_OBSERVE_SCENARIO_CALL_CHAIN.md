# Fig.9 Observe Scenario Call Chain

This document audits the real code path used by the read-only observation
scenario diagnostic. Line numbers refer to the implementation on
`experiment/fig9-competitive-inhibition`.

## Entry And Inputs

`SequentialMemory.observe_code()` starts at `src/seqmem/model.py:1632`. Its
inputs are the externally observed `SymbolCode`, the existing
`last_prediction_candidates`, previous active cells, previous learning
winners, and the optional `ObservationTrace`.

The diagnostic never calls `predict_code()` or `observe_code()` itself.
`experiments/fig9_strict_reproduction.py` creates one trace before the normal
`learn_actual_code()` call and serializes its copied results afterwards.

## Scenario 1

The observed column and SSTD time must match a saved `PredictionCandidate`
within `timing_tolerance`, and the candidate's segment must still be active.
The branch records `CORRECTLY_PREDICTED_CELL_AVAILABLE` at
`src/seqmem/model.py:1772`.

The winner neuron and reinforced segment both existed before observation.
The same saved candidate is passed into Scenario-1 reinforcement. No
additional prediction or matching call occurs.

## Scenario 2

When no valid predictive candidate matches, `_best_matching_neuron()` at
`src/seqmem/model.py:2049` inspects active incoming segments in the observed
column. It compares `(timed_overlap, score)` and retains the lexicographically
largest pair. Equal pairs invoke the existing learning RNG at
`src/seqmem/model.py:2100`.

If the selected segment's timed overlap reaches `L_match`, observation records
`MATCHING_SEGMENT_FOUND` at `src/seqmem/model.py:1796`. The selected neuron and
segment existed before observation.

## Scenario 3

The actual branches record one of:

- `NO_EXISTING_SEGMENT_IN_COLUMN`
- `EXISTING_SEGMENTS_BELOW_L_MATCH`
- `MATCHING_SEGMENT_NOT_ELIGIBLE`
- `PREDICTED_TIME_INVALID`

These branches are at `src/seqmem/model.py:1734-1751` and
`src/seqmem/model.py:1815`. The reason is copied from the branch that actually
ran; it is not inferred from the final model.

Scenario 3 calls `_least_used_neuron_index()` at
`src/seqmem/model.py:3041`. Equal least-used neurons are selected by the
existing learning RNG. `_grow_segment()` at `src/seqmem/model.py:2115` may
then create a segment from the pre-observation learning winners.

The resulting winner is a post-observation operational identity, not a valid
pre-existing autonomous reference.

## Context And Forgetting

Before observation, `previous_active_cells` represents the complete active
context, including burst cells. `previous_winners` contains only learning
winners used for segment growth. Observation replaces both only after all
events have been processed.

Pruning occurs in `_prune_neuron()` at `src/seqmem/model.py:3018`. The model
does not retain a deletion history that can prove a removed segment was
relevant to a later observation. Therefore the trace reports:

- `forgetting_evidence_available = false`
- `matching_segment_missing_due_to_forgetting_known = false`
- `recently_deleted_relevant_segment_count` unavailable

No forgetting cause is guessed.

## Read-Only Guarantee

`ObservationTrace.capture_scenario_details` defaults to false. When enabled,
the code copies objects and scalar values already participating in the normal
observation decision. It does not change comparison order, floating-point
operations, tie-breaking, segment growth, reinforcement, pruning, or RNG
calls.

