# Phase 05 Final Report

## Status

`PHASE05_PASS`

Phase 05 extracted only the existing transient-state copy/reset boundary. It did
not change prediction, learning, Scenario handling, rollout, forgetting,
checkpoint IO, strict defaults, or persistent memory ownership. This is an
architecture regression result, not a paper-result improvement.

## Forty Required Answers

| # | Question | Answer |
|---:|---|---|
| 1 | SequentialMemory state fields | 28 direct instance fields were inventoried. |
| 2 | Persistent fields | 1 `PERSISTENT_MEMORY` owner: `columns`, which owns neurons/segments/synapses. |
| 3 | Transient fields | 15 total: 5 `TRANSIENT_CONTEXT` and 10 `TRANSIENT_DIAGNOSTIC`. Only 8 container fields belong to the established snapshot. |
| 4 | RNG fields | 2: `_decode_rng` and `_learning_rng`. |
| 5 | Derived/cache fields | 7. |
| 6 | UNKNOWN fields | 1: `_next_segment_diagnostic_id`; it was deliberately left in place. |
| 7 | Extracted helpers | `shallow_copy_dict`, `shallow_copy_list`, `shallow_copy_set`, `shallow_copy_candidate_map`, `build_transient_snapshot`, `new_empty_dict`, `new_empty_list`, and `new_empty_set`. |
| 8 | Fields deliberately untouched | Persistent columns/memory, configuration, dynamics/cache/index fields, provenance indices, callbacks, runtime context, and the UNKNOWN diagnostic ID. |
| 9 | Why untouched | They are outside the established evaluation snapshot, have checkpoint or numerical implications, or do not yet have a proven grouping boundary. |
| 10 | New StateManager | **NO**. |
| 11 | SequentialMemory still owns all state | **YES**. Helpers receive values and return copies; they retain no model reference. |
| 12 | Pickle-visible class paths changed | **NO**. All compatibility classes remain `seqmem.model.*`. |
| 13 | Snapshot fields changed | **NO**. The same eight containers and two RNG states remain in the same dataclass order. |
| 14 | Snapshot copy depth changed | **NO**. Dict/list/set shallow copies are preserved. |
| 15 | Candidate identity semantics changed | **NO**. Candidate objects are not rebuilt or deep-copied. |
| 16 | Segment identity changed | **NO**. Candidate `segment` references and persistent segment objects are identical. |
| 17 | `reset_state()` semantics changed | **NO**. It assigns fresh built-in empty containers to the same eight fields and leaves RNG/persistent memory untouched. |
| 18 | RNG draw order changed | **NO**. Helpers do not draw RNG values. |
| 19 | RNG restore order | Unchanged: decode RNG `setstate()` first, learning RNG second. |
| 20 | Persistent memory copied by snapshot | **NO**. |
| 21 | Prediction algorithm changed | **NO**. |
| 22 | Learning algorithm changed | **NO**. |
| 23 | Scenario semantics changed | **NO**. |
| 24 | Rollout changed | **NO**. |
| 25 | Forgetting changed | **NO**. |
| 26 | Six Golden exact | **YES**. Five stages each passed all six fixtures; 30/30 fixture comparisons were exact. |
| 27 | Full-state SHA exact | **YES**, for every retained fixture and stage. |
| 28 | Transient SHA exact | **YES** for all frozen transient component SHA fields: active, winner, candidate, combined RNG, learning RNG, and decode RNG. The baseline has no separate aggregate field named `TRANSIENT_SHA256`. |
| 29 | Diagnostic ON/OFF parity | **YES** for all three Phase 03 fixtures. Two gzip files differed only in header metadata; decompressed scientific content was exact. |
| 30 | Old checkpoint load | **YES**. The pre-refactor 26-key `fig9-strict-v1` checkpoint loaded without rewriting. |
| 31 | Current save/load roundtrip | **EXACT** in a fresh Python process for payload keys, model module, model fingerprint, and RNG fingerprint. |
| 32 | Compileall | **PASS**, exit 0. |
| 33 | Relevant tests | **PASS**. New tests cover field equality, copy depth, object identity, future RNG sequences, evaluation isolation, advance/restore, reset, old checkpoint, and fresh-process roundtrip. |
| 34 | Single-process x3 | **PASS/PASS/PASS**, 732/732 each; 198.332 s, 197.245 s, and 197.282 s. |
| 35 | Isolated suite | **PASS**, 55/55 modules and 732/732 tests; zero isolated native crashes. |
| 36 | Native crash claimed fixed | **NO**. This phase did not reproduce it, but the historical Windows native failure remains unresolved. |
| 37 | `model.py` LOC | 4214 before, 4225 after, net +11 physical lines. Explicit imports/calls grew the file slightly; the new 101-line helper module creates the testable boundary. |
| 38 | Commits | `03befd6`, `35d7382`, `368115e`, `79c6157`, `aaf932a`, plus the final report commit. |
| 39 | Phase status | **PHASE05_PASS**. No main merge and no push were performed. |
| 40 | Recommended next phase | **EXTRACT_LEARNING_DECISION_HELPERS**. It should begin only after human approval and must retain Scenario and RNG parity. |

## State Boundary

Long-term memory is the neuron/segment/synapse graph under `columns`, including
weights, ages, delays, activity, and target times. Transient context is the
active/winner identity carried between transitions plus the latest prediction,
ranking, and Scenario statistics. Random state is separately owned by the
decode and learning RNG objects.

`TransientStateSnapshot` is an evaluation/rollout guard, not a transaction over
long-term memory. Restoring it after `learn=True` would not undo synaptic
mutation, and the API documentation continues to say so.

## Regression Evidence

- Golden summary: `PHASE05_GOLDEN_REGRESSION.json`.
- Checkpoint evidence: `PHASE05_CHECKPOINT_COMPATIBILITY.md`.
- Diagnostic ON/OFF evidence:
  `diagnostic_regression/diagnostic_regression.json`.
- Full test ledger: `PHASE05_TEST_RUNS.csv`.
- Field and lifecycle audit: `MODEL_STATE_INVENTORY.csv`,
  `TRANSIENT_STATE_LIFECYCLE.md`, and `TRANSIENT_STATE_MUTATION_MAP.md`.

## Residual Risk and Stop

All three long-lived test processes passed in this run, unlike Phase 04. That is
useful evidence but not proof that the intermittent Windows `0xC0000005` problem
is fixed. No native-crash code was changed, and no native debugging phase was
started.

Phase 05 stops here. It does not begin learning extraction, prediction
extraction, forgetting extraction, Fig.9 orchestration work, checkpoint redesign,
or native-crash debugging.
