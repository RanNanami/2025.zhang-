# Refactor Phase Plan

## Phase 0: freeze and backup - complete

- Goal: immutable scientific snapshot and independent recovery paths.
- Touched: backup refs, external bundle/archive, Phase 0 reports.
- Forbidden: scientific source files.
- Commit: `chore(refactor): freeze pre-refactor scientific snapshot`.
- Gate: branch/tag/bundle/ZIP validation.
- Rollback: restore `1864f46` or clone the bundle.
- Risk: scientific low; checkpoint low; RNG none; complexity low.

## Phase 1: audit and golden baselines - complete

- Goal: dependency, credibility, dataset, artifact, CLI, test, and exact-state
  baselines.
- Touched: reports, audit tools, baseline tools, baseline manifest test.
- Forbidden: `src/seqmem`, experiment behavior, historical outputs.
- Commits: architecture/credibility audit and golden regression baselines.
- Gate: repeated 10/20/50 prediction/model/RNG equality.
- Rollback: revert Phase 1 commits; frozen science remains unchanged.
- Risk: scientific none; checkpoint none; RNG observational only; complexity medium.

## Phase 2: pure infrastructure extraction - next

- Goal: extract canonical hashing, atomic JSON, CSV/artifact IO, and reporting
  helpers from experiment runners.
- Expected files: new `experiments/common/*`, narrow imports in Fig.9 runner,
  focused utility tests.
- Forbidden: `src/seqmem/model.py`, rollout, prediction, observation, learning,
  checkpoints, defaults, native crash supervision.
- Expected commit: `refactor(infra): extract behavior-neutral artifact utilities`.
- Regression: compileall, unit tests, six exact golden fixtures.
- Larger gate: no.
- Rollback: revert the single extraction commit on any byte/hash mismatch.
- Risk: scientific low; checkpoint low; RNG low; complexity low-to-medium.

## Phase 3: runtime and diagnostic IO isolation

- Goal: separate logging/compression/trace sinks from strict orchestration and
  remove eager strict-to-diagnostic imports.
- Forbidden: model and selection logic; diagnostic content semantics.
- Regression: diagnostic OFF/ON exact state, failure-log tests, golden suite.
- Larger gate: optional 50 only; no formal 250 yet.
- Rollback: restore eager adapters if process supervision or output parity fails.
- Risk: scientific low; checkpoint medium; RNG low; complexity medium.

## Phase 4: transient state helpers

- Goal: extract pure copy/restore payload helpers while keeping
  `TransientStateSnapshot` and public methods compatible.
- Forbidden: class module paths, RNG ownership, rollout behavior.
- Regression: snapshot roundtrip, evaluation read-only, exact golden suite.
- Larger gate: no.
- Rollback: revert if full-state/RNG hashes differ.
- Risk: scientific medium; checkpoint high; RNG high; complexity medium.

## Phase 5: forgetting and pruning helpers

- Goal: isolate pure forgetting-score and eligibility calculations; mutation
  remains in `SequentialMemory`.
- Forbidden: iteration order and segment container replacement semantics.
- Regression: pruning fixtures, checkpoints, golden suite.
- Larger gate: no unless graph structure changes unexpectedly.
- Rollback: any segment/synapse fingerprint mismatch.
- Risk: scientific high; checkpoint high; RNG low; complexity medium.

## Phase 6: learning helper extraction

- Goal: expose explicit calculations for matching, growth inputs,
  reinforcement, and punishment while preserving mutation order in facade.
- Forbidden: Scenario taxonomy/logic changes and RNG tie-break changes.
- Regression: Scenario tests, exact golden suite, checkpoint roundtrip.
- Larger gate: Fig.8 100 after exact small gates.
- Rollback: prediction/state/RNG/Scenario mismatch.
- Risk: scientific very high; checkpoint high; RNG high; complexity high.

## Phase 7: prediction helper extraction

- Goal: isolate continuous PSP calculations and candidate records without
  changing search grid, caching, accumulation, sorting, or tie-breaks.
- Forbidden: optimization or candidate pruning mixed with refactor.
- Regression: reference-path parity, exact golden suite.
- Larger gate: Fig.8 100 and manually authorized Fig.9 250.
- Rollback: any timing/candidate/prediction/full-state mismatch.
- Risk: scientific very high; checkpoint medium; RNG high; complexity very high.

## Phase 8: experiment runner decomposition

- Goal: split Fig.7/8/9 protocol, execution, checkpoint, and reporting layers;
  keep old CLI files as shims.
- Forbidden: CLI/default and checkpoint payload changes.
- Regression: CLI inventory parity, resume parity, golden suite.
- Larger gate: manually authorized Fig.9 250.
- Rollback: restore old runner orchestration.
- Risk: scientific high; checkpoint very high; RNG high; complexity very high.

## Phase 9: historical/result cleanup

- Goal: archive confirmed historical code/results and delete only reviewed,
  backed-up, regenerable cache candidates.
- Forbidden: canonical checkpoints/reports and compatibility shims still used.
- Regression: import graph and documented restore drill.
- Larger gate: none.
- Rollback: restore archive or critical-artifact ZIP.
- Risk: scientific low; evidence-retention high; complexity medium.

Phase 2 requires human approval after reviewing this plan and the golden
baseline manifest. This Phase 0/1 task stops here.

