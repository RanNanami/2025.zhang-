# Phase 04 Final Report

## Status

`PHASE04_PASS`

Scientific and architecture gates passed. Windows native-runtime status is
`NATIVE_CRASH_REPRODUCED_IN_PHASE04_NOT_FIXED`: all three final long-lived
single-process test runs ended with `0xC0000005`. Phase 04 does not claim that
the native crash was fixed.

Baseline commit: `88dfbd897f30869ada3e8d4a8452005feaf71ee3`.  
Final implementation commit: `ac2e630f903870989d621e61581f12af6053513d`.  
Branch: `refactor/architecture-cleanup-v2`.

## Runner Size

The audited Phase 04 starting runner had 5,943 physical lines. The attachment's
5,917 value was an approximate/stale count from the Phase 03 report. The final
runner has 5,905 physical lines, a net reduction of 38 lines. LOC was not the
goal: low-level ordinary artifact ownership moved out while protocol and
scientific construction remained explicit in the runner.

## Extracted Responsibilities

`experiments/fig9/outputs.py` now owns:

- strict stream prediction/summary/protocol paths and output directory creation;
- strict summary discovery excluding historical compensated results;
- initial prediction CSV plus summary/protocol publication;
- final enriched summary publication;
- interval and debug result writing;
- runtime JSON plus byte-preserving canonical protocol copy;
- ordinary original/perturbed adaptation plot naming and finalization;
- legacy density and long-sequence writer compatibility implementations.

Helpers accept completed rows, payloads, paths and labels. They do not accept a
model, encoder or RNG and do not import `seqmem`.

## Deliberately Retained Responsibilities

The runner still constructs every prediction row, protocol field, metric,
summary value and diagnostic payload. It also retains scientific observation,
autonomous rollout, decode, competition, Scenario logic, tracker dispatch,
progress logging, profiling and all diagnostic compression/close ordering.

Checkpoint save/load, resume, payload schema and sidecars remain untouched
because they serialize scientific state and RNG. Native crash handles,
faulthandler, background traceback thread, process-memory probes and breadcrumbs
remain untouched because their lifetime is part of the unresolved runtime
investigation. The function-source hashes for both families match the Phase 04
baseline exactly.

## Compatibility

- CLI names/defaults changed: **NO**.
- Protocol schema/content changed: **NO**.
- Metric definitions changed: **NO**.
- Rollout changed: **NO**.
- Prediction changed: **NO**.
- Learning changed: **NO**.
- `src/seqmem/model.py` changed: **NO**.
- Checkpoint schema/semantics changed: **NO**.
- Diagnostic defaults or lazy-load boundary changed: **NO**.
- Core-to-experiments imports: **0**.
- Direct strict-to-diagnostic imports: **1**, still only competitive inhibition.

All six Golden Baselines were exact after every retained extraction. The final
compatibility run completed all six on the first attempt and matched prediction,
science/full state, learning/decode RNG, Scenario, previous active/winner,
candidate and segment/synapse fingerprints.

The independent 20-record artifact A/B compared eight artifacts. Three were
byte exact and all eight were scientifically content exact. Non-byte differences
were limited to commit provenance, timestamps, elapsed/runtime rates, absolute
output paths and runtime/RSS trace columns. Prediction CSV field order, values,
float text, CRLF and encoding were byte exact. The reduced 20-record protocol
does not accumulate a 400-record rolling MAPE, so it does not emit plot PNGs;
the plot helper delegates unchanged to the existing plotting implementation.

## Tests

- Compileall: PASS.
- Phase 04 architecture tests: 9/9 PASS.
- Relevant strict tests after each extraction: 29/29 PASS.
- Final compatibility-focused modules: 24/24 PASS.
- Single-process run 1: native `0xC0000005` after 80.761 seconds in
  `model.py:3187 potential()` during an intracolumn-selector test.
- Single-process run 2: native `0xC0000005` after 102.146 seconds during the
  paper-parity roundtrip, without a Python traceback.
- Single-process run 3: native `0xC0000005` after 28.646 seconds during a
  competitive-inhibition test; faulthandler was in model construction.
- Isolated modules: 52/52 PASS, 720/720 tests, zero isolated native crashes.

An earlier preliminary single-process run exposed a removed historical writer
export and a long-process dynamics type corruption. The writer import was
restored as an output-module compatibility export. The dynamics/native issue
was not modified; its isolated module passes.

## Commits

- `32c35c6` docs(refactor): map Fig9 runner output responsibilities
- `f397058` refactor(fig9): extract strict artifact paths
- `abb7287` refactor(fig9): extract strict metadata writers
- `29bed63` refactor(fig9): extract strict result writers
- `4469ba9` refactor(fig9): extract strict plot finalization
- `1e4f8cd` test(refactor): audit native retries in phase 4 golden runs
- `9a8556d` test(refactor): lock Fig9 output architecture parity
- `ac2e630` refactor(fig9): preserve strict writer compatibility

No merge to `main` and no push were performed.

## Required Answers

1. Runner start: 5,943 physical lines.
2. Runner end: 5,905 physical lines.
3. Extracted: ordinary paths, prediction/metadata/result writers, runtime and
   protocol publication, and plot finalization.
4. Retained: all scientific/protocol construction, diagnostics, checkpoint,
   native instrumentation, CLI and profiling.
5. Checkpoint stayed because it is state/RNG/schema sensitive.
6. Native crash IO stayed because the unresolved handle/thread lifetime is
   crash-sensitive.
7. New production modules: none; the existing cohesive `fig9/outputs.py` was
   expanded. One Golden runner and one architecture test module were added.
8. CLI changed: NO.
9. Protocol changed: NO.
10. Metric changed: NO.
11. Rollout changed: NO.
12. Prediction changed: NO.
13. Learning changed: NO.
14. `model.py` changed: NO.
15. Checkpoint schema changed: NO.
16. Six Golden fixtures: all EXACT at every retained stage.
17. Artifact scientific content: 8/8 EXACT; raw bytes 3/8 EXACT.
18. Diagnostic lazy-load boundary: preserved.
19. Strict-to-diagnostic direct dependencies: still 1.
20. Core-to-experiments dependencies: still 0.
21. Single-process x3: native crash, native crash, native crash.
22. Isolated tests: 52/52 modules and 720/720 tests PASS.
23. Native crash reproduced: YES.
24. Native crash claimed fixed: NO.
25. Commits: listed above plus the report commit.
26. Phase status: `PHASE04_PASS`, native runtime separately unresolved.
27. Recommended next phase: `DECOMPOSE_FIG9_PROTOCOL_ORCHESTRATION`.

## Next Phase

`DECOMPOSE_FIG9_PROTOCOL_ORCHESTRATION`

This is the single recommendation. Output ownership is now explicit, while the
runner still combines protocol validation, diagnostic-option assembly and
stream orchestration. Any next phase must remain separate from transient-state,
forgetting, learning, prediction, checkpoint and native-crash work.
