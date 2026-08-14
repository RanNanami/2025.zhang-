# Phase 03 Final Report

## Status

`PHASE03_PASS`

The scientific and architecture acceptance gates passed. The Windows native
runtime issue did not pass: two of three long-lived single-process test runs
terminated natively. Phase 03 does **not** claim that `0xC0000005` or any other
native crash is fixed.

Frozen scientific reference:
`1864f46aee5cc0aee814fd4e6154b06f8bfeb08f`.

Working branch: `refactor/architecture-cleanup-v2`.

## What Changed

- Added explicit cached loaders in `experiments/fig9/diagnostic_loader.py`.
- Moved CLI-only level values to `experiments/fig9/diagnostic_options.py` so
  parser construction does not load trace implementations.
- Moved generic diagnostic CSV/GZIP ownership to
  `experiments/common/diagnostic_io.py` without changing row formatting,
  pending-file compression, flush, or close order.
- Preserved disabled independent-reference and actual-branch checkpoint
  payloads exactly while avoiding construction of unused tracker objects.
- Kept all historical diagnostic implementations and all CLI names.
- Kept `fig9_competitive_inhibition` direct because it owns an optional
  prediction transform and cannot honestly be classified as pure IO.

The strict runner changed from 6,040 to 5,917 lines. Its diff is 226 inserted
and 349 deleted lines, for a net reduction of 123 lines. The removed body is
generic writer ownership, not model or Scenario logic.

## Import Boundary

| Measure | Before | After |
| --- | ---: | ---: |
| Direct strict-to-diagnostic dependencies | 18 | 1 |
| Diagnostic implementation modules loaded by strict import | 18 | 1 |
| Loaded entries including diagnostics package | 19 | 2 |

The one retained module is `fig9_competitive_inhibition`. Oracle,
teacher-forced, provenance, trace, formatter, and gzip/report families are no
longer loaded merely by importing the strict runner.

## Dependency Classification

The complete symbol-level map is in
`FIG9_DIAGNOSTIC_DEPENDENCY_MAP.csv`.

- `DIAGNOSTIC_MODEL_TRANSFORM`: competitive inhibition (1).
- `ORACLE`: oracle candidate, candidate-context oracle, teacher-forced
  identity, and independent reference (4). These remain default-off and are
  used only for post-prediction/post-observation labels.
- `READ_ONLY_TRACE`: the other 13 direct families.
- Direct `POSTHOC_ANALYZER` imports: 0. Analyzer scripts were already separate.
- Module-level `PURE_IO`: 0 in the original 18 because writers were mixed with
  trace logic. The extracted `experiments/common/diagnostic_io.py` is now the
  pure IO boundary.

No dependency remains `UNKNOWN`. Competition is the only scientific-risk
family and was deliberately not migrated. No trace family was allowed to
select candidates, alter learning, consume RNG, or use expected/future values
for model decisions.

## Scientific Regression

All six frozen fixtures were regenerated after each migration stage and match
exactly:

| Stage | Fig8 10 | Fig8 20 | Fig8 50 | Fig9 10 | Fig9 20 | Fig9 50 |
| --- | --- | --- | --- | --- | --- | --- |
| Readout lazy load | EXACT | EXACT | EXACT | EXACT | EXACT | EXACT |
| Remaining safe families | EXACT | EXACT | EXACT | EXACT | EXACT | EXACT |
| Diagnostic writer extraction | EXACT | EXACT | EXACT | EXACT | EXACT | EXACT |
| Disabled tracker import deferral | EXACT | EXACT | EXACT | EXACT | EXACT | EXACT |

Compared fields include prediction SHA, metrics, Scenario summary,
science/full-state SHA, combined/learning/decode RNG SHA, previous active and
winner SHA, candidate SHA, and segment/synapse structure.

## Diagnostic Regression

Three 20-record fixtures cover a pure trace, provenance, and a complex trace:

| Fixture | Prediction | Science/full state | RNG | Scenario ledger | Segments/synapses |
| --- | --- | --- | --- | --- | --- |
| Segment reinforcement | EXACT | EXACT | EXACT | EXACT | EXACT |
| Actual branch provenance | EXACT | EXACT | EXACT | EXACT | EXACT |
| Match overlap | EXACT | EXACT | EXACT | EXACT | EXACT |

Output compatibility covered 11 artifacts. Seven are byte-for-byte identical.
All 11 have identical scientific content. The four raw-byte exceptions are two
gzip header timestamps and two `strict_protocol_sha256` values derived from the
different before/after commit SHA. Decompressed CSV rows and normalized
protocol content are exact.

## Tests

- Compileall: PASS.
- New Phase 03 tests: 9/9 PASS.
- Single-process run 1: native `0xC0000409` after 59.634 seconds near the
  context-trajectory noninterference test.
- Single-process run 2: 711/711 PASS in 211.205 seconds.
- Single-process run 3: native `0xC0000005` after 53.587 seconds in
  `model.py:3187 potential()`, again during the context-trajectory test.
- Isolated modules: 51/51 PASS, 711/711 tests, 0 assertion failures, 0 native
  crashes.

The isolated result shows that every test module passes independently. It does
not replace the two failed single-process results. Native runtime status is:
`NATIVE_CRASH_REPRODUCED_IN_PHASE03_NOT_FIXED`.

## Required Answers

1. Original direct dependencies: 18.
2. Classification: 1 model transform, 4 oracle, 13 read-only trace; see CSV.
3. Pure IO: no original whole module; generic writer functions are now pure IO.
4. Read-only traces: 13 families.
5. Direct posthoc analyzers: 0.
6. Scientific risk: competitive inhibition only; it remains direct.
7. Not moved: competition science, native crash instrumentation,
   tracker-specific streaming sinks, analyzers, and all diagnostic algorithms.
8. Default-loaded diagnostic implementations before: 18.
9. Default-loaded diagnostic implementations after: 1.
10. Lazy-loaded families: the other 17 original families.
11. CLI changed: NO.
12. Defaults changed: NO.
13. Diagnostic semantics changed: NO.
14. Scientific behavior changed: NO.
15. Six golden baselines: all EXACT.
16. Diagnostic ON/OFF science parity: all EXACT for three fixtures.
17. Output compatibility: 7/11 raw-byte exact; 11/11 scientific-content exact,
    with only gzip metadata or commit provenance differences.
18. Core to experiments imports: still 0.
19. `src/seqmem/model.py` modified: NO.
20. Scenario logic modified: NO.
21. Prediction modified: NO.
22. Rollout modified: NO.
23. Checkpoint format modified: NO.
24. Single-process x3: native crash, 711 PASS, native crash.
25. Isolated tests: 51/51 modules and 711/711 tests PASS.
26. `0xC0000005` reproduced: YES, in single-process run 3.
27. Native crash claimed fixed: NO.
28. Strict runner LOC: 6,040 to 5,917, net -123.
29. Direct strict-to-diagnostic dependencies: 18 to 1.
30. Commits: `55c73a7`, `4a8d2ca`, `1b55d03`, `b2a740e`, `f745b84`,
    `62fc514`, plus this report's containing commit.
31. Phase status: `PHASE03_PASS`, with native runtime explicitly still open.
32. Recommended next phase: `DECOMPOSE_FIG9_NONSCIENTIFIC_RUNNER_IO`.

## Next Phase

`DECOMPOSE_FIG9_NONSCIENTIFIC_RUNNER_IO`

This is the single recommended next step. Diagnostic loading and generic
writer ownership now have explicit boundaries, while the 5,917-line strict
runner still owns substantial protocol-neutral orchestration. Model state,
forgetting, learning, prediction, CLI redesign, and main-branch merge should
remain out of scope until that phase is separately approved.
