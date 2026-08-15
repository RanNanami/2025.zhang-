# Phase 07 Final Report

## Status

`PHASE07_PASS`

Phase 07 mechanically extracted three already-resolved weight/age mutation
loops and two pure forgetting calculations. `SequentialMemory` remains the
state, RNG, search, identity, growth, index, prune, diagnostic, and orchestration
owner. This is a behavior-preserving architecture result, not an accuracy or
paper-result improvement.

## Extracted boundaries

`seqmem._learning_helpers` now performs, on explicitly passed mutable objects:

- selected-segment contributor reinforcement and optional noncontributor
  depression;
- subsequent same-neuron other-segment depression;
- failed-prediction contributor depression.

The helpers use the existing collection iteration order and original formulas.
They do not resolve contributors, search candidates, draw RNG, construct
segments/synapses, assign delays, update incoming indices, emit diagnostics, or
prune.

`seqmem._forgetting_helpers` owns only the exact scalar score expression and the
strict `< forgetting_threshold` predicate. The mutating prune loop remains in
`SequentialMemory._prune_neuron()`.

## Deliberately retained behavior

- S1 still uses the same saved `PredictionCandidate` and exact causal segment.
- S2A best-match traversal and growth remain in the model.
- S2B least-used RNG selection, segment construction, source order, delay
  assignment, diagnostic IDs, and incoming-index registration remain in the
  model.
- Paper S3 candidate/timing traversal and contributor resolution remain in the
  model.
- Historical labels (`scenario1`, `scenario2`, observation `scenario3`) remain
  unchanged.
- Prediction, rollout, decoder, PSP/SSTD, strict defaults, checkpoint schema,
  and pickle module paths are unchanged.

The real pruning order is important: `_reinforce_segment()` prunes immediately
after its mutation, and `_punish_wrong_predictions()` prunes after each punished
candidate. Both occur before `observe_code()` publishes transient state. Phase
07 documents and preserves this order instead of moving pruning after publish.

## Exact regression evidence

- Three Golden stages, each with Fig.8/Fig.9 10/20/50: **18/18 EXACT**.
- Scenario ledger, science model, full state, learning/decode RNG, previous
  active/winner, candidates, and segment/synapse structure: **EXACT**.
- Branch trace before/after: complete 12,690,305-byte JSON is byte exact,
  SHA-256 `b49bcaaa225932696ddb68cab2578c89b64dc07ee1ddcc1b6c638fc2a685e0fb`.
- Mutation trace before/after: complete 177,246,335-byte JSON is byte exact,
  SHA-256 `d783af223879090556fc573f216fc131434aad373c47c571dd29d6be3cab0c5a`.
- Mutation trace covers reinforced segment, weight/age updates, new segment,
  new synapse, punishment updates, and forgetting/prune calls.
- Checkpoint: old load PASS; current fresh-process round trip EXACT.
- Three diagnostic ON/OFF fixtures: science/state/RNG/Scenario/structure EXACT;
  gzip differences are header metadata only and decompressed content is exact.
- Core-to-experiments imports remain zero.

## Tests

- Compileall: PASS.
- Focused mutation suite: 110/110 PASS.
- Focused forgetting suite: 110/110 PASS.
- Checkpoint/temporal/architecture suite: 91/91 PASS.
- Single-process discover: 746/746 PASS three times (245.268 s, 249.239 s,
  263.688 s).
- Isolated suite: 58/58 modules and 746/746 tests PASS; zero isolated native
  crashes.

No native crash occurred in the Phase 07 final matrix. This does **not** prove
the historical intermittent Windows `0xC0000005` issue is fixed; no native
runtime code was changed and no such claim is made.

## Size and commits

`model.py` changed from 4,238 to 4,248 physical lines. Explicit imports and
calls increased it by 10 lines; LOC reduction was not the goal.
`_learning_helpers.py` is 112 lines and `_forgetting_helpers.py` is 26 lines.

Phase commits:

- `eba64ac` docs(refactor): map learning mutation and forgetting
- `421c039` test(refactor): capture learning mutation sequences
- `6649749` refactor(learning): extract mutation update loops
- `67a4c43` refactor(forgetting): extract scalar retention rules
- final evidence/report commit

No merge to `main` and no push were performed.

## Stop and next phase

Phase 07 stops here. It does not begin prediction refactoring. The only
recommended next phase is `PHASE08_PREDICTION_PIPELINE`, subject to explicit
human approval.
