# Phase 06 Final Report

## Status

`PHASE06_PASS`

Phase 06 extracted only the pure observation-side branch classification. All
prediction evidence resolution, matching traversal, RNG decisions, learning
mutation, punishment, transient publication, and forgetting remain owned by
`SequentialMemory`. This is an architecture regression result, not a paper
result or an accuracy improvement.

Windows native-runtime status is separately
`NATIVE_CRASH_REPRODUCED_IN_PHASE06_NOT_FIXED`: one of three long-lived test
processes ended with `0xC0000005`; the other two completed 737/737 tests. No
learning code was changed in response to that crash.

## What Changed

`seqmem._learning_helpers` now names the existing branch decision with one plain
pure function. It consumes four booleans that were already resolved at the
original point in `observe_code()` and returns the existing historical label:
`scenario1`, `scenario2`, `scenario3`, or the empty no-learning label.

The same module provides a read-only mapping from historical labels to paper
names. Historical observation `scenario3` remains Paper S2B; Paper S3 remains
the separate failed-prediction punishment path. Canonical names do not enter
checkpoints, counters, CSV schemas, or historical artifacts.

The following deliberately stayed in `model.py`: temporal confirmation,
candidate lookup, `_best_matching_neuron()`, timed-overlap calculation,
least-used neuron RNG selection, exact candidate/segment object use,
reinforcement, segment/synapse growth, punishment, pruning, and transient-state
publication.

## Thirty-Eight Required Answers

| # | Question | Answer |
|---:|---|---|
| 1 | Major observe/learning decision stages | Six: prediction-evidence resolution, fallback matching search, branch classification, branch mutation execution, failed-prediction punishment, and transient publication/forgetting. |
| 2 | Pure decisions | Historical branch classification and canonical paper-name lookup. |
| 3 | Read-only searches | Candidate confirmation, timing match, best-matching neuron/segment traversal, and timed-overlap eligibility calculation. |
| 4 | RNG decisions | Best-match exact ties and least-used neuron ties consume `_learning_rng`; decoding separately owns `_decode_rng`. |
| 5 | Persistent mutations | Segment reinforcement, weight/age updates, synapse growth, segment creation, failed-prediction punishment, and forgetting/pruning. |
| 6 | Extracted helpers | `classify_observation_learning_branch` and immutable `CANONICAL_PAPER_BRANCH_BY_INTERNAL`. |
| 7 | Risky candidates retained | Temporal confirmation, best-match search, timed overlap, least-used selection, all mutation, punishment, pruning, and publication. They are traversal-, identity-, RNG-, float-, or order-sensitive. |
| 8 | Scenario 1 decision semantics changed | **NO**. |
| 9 | Scenario 2A changed | **NO**. |
| 10 | Scenario 2B changed | **NO**. |
| 11 | Paper Scenario 3 punishment changed | **NO**. |
| 12 | Historical scenario labels changed | **NO**. |
| 13 | `timing_tolerance` semantics changed | **NO**. |
| 14 | Temporal confirmation changed | **NO**. |
| 15 | Candidate identity changed | **NO**. The same saved object remains in use. |
| 16 | Segment identity changed | **NO**. No copy or ID relookup was introduced. |
| 17 | Best-match traversal changed | **NO**. |
| 18 | Least-used RNG changed | **NO**. |
| 19 | Learning RNG draw order changed | **NO**. The helper has no RNG access. |
| 20 | Float accumulation order changed | **NO**. The helper performs no sum, sort, max, PSP, response, or overlap calculation. |
| 21 | Mutation order changed | **NO**. Reinforce/grow remains before punishment; punishment remains before transient publication and forgetting. |
| 22 | Branch sequence before/after exact | **YES**. All four real traces were event-for-event and punishment-for-punishment exact. |
| 23 | Six Golden exact | **YES** before and after the extraction commit; 12/12 fixture comparisons exact. |
| 24 | Scenario ledger exact | **YES** for all Golden and diagnostic parity fixtures. |
| 25 | Full state exact | **YES**. |
| 26 | RNG exact | **YES** for combined, learning, and decode RNG fingerprints. |
| 27 | Checkpoint compatibility | **PASS**: old load and fresh-process roundtrip exact. |
| 28 | Diagnostic ON/OFF parity | **PASS** for all three Phase 03 fixtures. Gzip byte differences are header metadata only; decompressed scientific content is exact. |
| 29 | Temporal confirmation tests | **PASS**, including current mode, identity, multiple-candidate behavior, and exact candidate reuse. |
| 30 | Compileall | **PASS**. |
| 31 | Single-process x3 | Run 1 native `0xC0000005`; runs 2 and 3 PASS with 737/737 tests each. |
| 32 | Isolated suite | **PASS**, 56/56 modules and 737/737 tests; zero isolated native crashes. |
| 33 | Native crash claimed fixed | **NO**. |
| 34 | `model.py` LOC before/after | 4,225 before; 4,238 after; net +13 physical lines. Clarity, not LOC reduction, was the goal. |
| 35 | `_learning_helpers.py` LOC | 48 physical lines. |
| 36 | Commits | `ddcbd5f`, `537dc22`, `487cd48`, plus the final evidence/report commit. |
| 37 | Phase status | **PHASE06_PASS**, with native runtime instability explicitly unresolved. No main merge and no push. |
| 38 | Recommended next phase | **EXTRACT_FORGETTING_PURE_HELPERS**. Human approval is required before starting it. |

## Regression Evidence

- Learning flow and boundaries: `LEARNING_FLOW_MAP.md`,
  `LEARNING_DECISION_MUTATION_BOUNDARY.md`, and
  `SCENARIO_DECISION_TABLE.csv`.
- Extracted/retained responsibilities: `PHASE06_EXTRACTION_MAP.csv`.
- Four real trace comparisons: `PHASE06_BRANCH_SEQUENCE_PARITY.csv`.
- Golden summary: `PHASE06_GOLDEN_REGRESSION.json`.
- Checkpoint evidence: `PHASE06_CHECKPOINT_COMPATIBILITY.md`.
- Diagnostic evidence: `diagnostic_regression/diagnostic_regression.json`.
- Complete test ledger: `PHASE06_TEST_RUNS.csv`.

## Stop

Phase 06 stops here. It does not move reinforcement, creation, punishment,
prediction, forgetting, Fig.9 orchestration, checkpoint code, or native-crash
debugging. The sole next-phase recommendation is
`EXTRACT_FORGETTING_PURE_HELPERS`; it has not been started.
