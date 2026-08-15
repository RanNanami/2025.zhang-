# Phase 08 Prediction Pipeline Final Report

## Decision

Phase 08 is PASS. The prediction pipeline now has an explicit audit map and a
small deterministic packaging boundary. Numerically sensitive traversal,
continuous PSP integration, threshold crossing, soma timing, candidate
competition, and autonomous propagation remain in their established owners.

Final recommendation: `CORE_REFACTOR_COMPLETE`.

## Scope and commits

Phase 08 started from `57319b5` on `refactor/architecture-cleanup-v2`.

- `4770972` maps the real prediction flow and safe extraction boundary.
- `a4508df` adds stable prediction-sequence capture without persisted `id()`.
- `761a74d` extracts deterministic prediction packaging helpers and six tests.
- final evidence/report commit records this report and compatibility artifacts.

No merge to `main` and no push were performed.

## Mechanical extraction

`src/seqmem/_prediction_helpers.py` contains six plain functions:

1. the existing upper-bound eligibility expression;
2. the existing short-circuit firing-window predicate;
3. the existing `(column, round(time, 12))` event key;
4. `PredictionCandidate` packaging from the selected values and same-call
   metadata;
5. prediction-stat dictionary packaging with the established key order;
6. emitted `SpikeEvent` packaging in selected-event order.

`SequentialMemory.predict_code()` remains the orchestration and publication
owner. Its physical size fell from 4,248 to 4,234 lines at the file level; the
new helper is 126 lines including documentation and type annotations. Line
count was not used as a success criterion.

## Deliberately retained behavior

- Active-source and incoming-segment traversal remain inline so dictionary and
  accumulation order cannot change.
- `_continuous_segment_prediction()`, the reference implementation, and both
  optimized implementations remain separate and unmodified.
- The strict default remains `continuous_prediction_impl="reference"`; existing
  reference/optimized equality tests pass.
- Potential loops, caches, rounded lookup keys, threshold solvers, soma
  integration, and floating-point summation remain untouched.
- Exact event score replacement (`>`), insertion order, earliest-column
  selection, and optional diagnostic competition remain untouched.
- Every candidate retains the exact selected `Segment` and same-call crossing
  contribution objects. Scenario 1 therefore sees the same causal object.
- Decoder, observe/learning, Scenario selection, forgetting, CLI defaults,
  checkpoint schema, and pickle-visible class paths are unchanged.
- `advance_prediction()` was not edited. Raw neural prediction and autonomous
  transient propagation remain separate; decoded-symbol replay was not added.
- Core-to-experiments imports remain zero, enforced by architecture tests.

## Exact prediction evidence

The complete Fig.8 20/50 and Fig.9 20/50 prediction trace contains 770 calls and
120,321,278 bytes. Before and after SHA-256 is identical:
`bbd6681a41f216d0aeb5af1b85f7a0251d49d355f67b019ed404ec932bff5dcb`.

The byte-exact trace covers source-context sequence, candidate count, stable
segment path, exact neuron, raw serialized score/crossing/soma/peak/margin and
contribution values, emitted column/time order, selected-candidate identity,
and `last_prediction_candidates`/stats structure. No tolerance comparison and
no persisted runtime `id()` were used.

## Golden and learning parity

- Fig.8 10/20/50 and Fig.9 10/20/50: **6/6 EXACT**, all first attempt.
- Prediction SHA, metrics, Scenario ledger, science model, full state,
  learning/decode RNG, previous active/winner, candidate fingerprint, and
  segment/synapse structure: **EXACT**.
- Phase 06 branch trace: 12,690,305 bytes and SHA-256
  `b49bcaaa225932696ddb68cab2578c89b64dc07ee1ddcc1b6c638fc2a685e0fb`,
  byte exact.
- Phase 07 mutation trace: 177,246,335 bytes and SHA-256
  `d783af223879090556fc573f216fc131434aad373c47c571dd29d6be3cab0c5a`,
  exact.
- Winner/branch selection, selected segments, creation, punishment, mutation,
  and forgetting/prune sequence therefore did not drift.

## Checkpoint and diagnostics

The old pre-refactor checkpoint loads through the stable model paths. A current
save loaded in a fresh process reproduces all required fingerprints exactly.
Phase 08 introduces no persistent helper object or schema field.

Three diagnostic ON/OFF fixtures preserve prediction, science state, RNG,
Scenario ledger, and segment/synapse structure exactly. CSV/JSON artifacts are
byte exact. Two gzip files differ only in header timestamp metadata; their
decompressed scientific content hashes are exact.

## Tests and native status

- Focused prediction/temporal/Scenario suite: 141/141 PASS.
- Checkpoint tests: 2/2 PASS.
- Final compileall and `git diff --check`: PASS.
- Single-process discover: 752/752 PASS three times (297.572 s, 293.471 s,
  292.239 s).
- Isolated runner: 59/59 modules and 752/752 tests PASS; native crash count 0.

No native crash occurred in this Phase 08 matrix. This does **not** prove the
historical intermittent Windows `0xC0000005` issue is fixed: no native runtime
or numerical implementation was changed, and no fixed claim is made.

## Stop

Phase 08 is the final core behavior-preserving refactor phase. This work stops
without starting Phase 09, redesigning checkpoints, cleaning historical code,
tuning scientific parameters, changing runners, or debugging native runtime
behavior.

`CORE_REFACTOR_COMPLETE`
