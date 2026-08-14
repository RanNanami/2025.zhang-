# Architecture Test Plan

## Existing constraints to retain

- `seqmem` does not import experiments or diagnostics.
- Pickle-visible model classes remain under `seqmem.model`.
- Stable Fig.9 helper shims continue to resolve.
- Diagnostic imports do not alter `MemoryParams` defaults.

## New static boundary tests

1. Parse imports and fail if `src/seqmem` targets `experiments`.
2. Fail if the future strict entrypoint imports `experiments.diagnostics`.
3. Fail if strict imports `experiments.historical` or the historical shim.
4. Fail if oracle/teacher/GT modules are reachable from core learning.
5. Assert diagnostic flags all default off in strict protocol config.
6. Assert CLI defaults exactly match the protocol config projection.
7. Assert the six checkpoint-visible class `__module__` values remain
   `seqmem.model`.

The current strict runner has 18 direct diagnostic imports, so rule 2 is a
target-state test introduced only when that dependency is intentionally
removed. It must not be added as a knowingly failing test beforehand.

## Behavioral tests

- Run the six 10/20/50 golden fixtures and compare every stored SHA exactly.
- Verify trace OFF/ON has identical prediction, science model, full state, and
  RNG hashes for each extracted diagnostic adapter.
- Verify checkpoint save/load/save preserves protocol, state, RNG, and class
  paths.
- Verify evaluation and autonomous rollout restore transient state exactly.
- Verify expected/teacher/oracle data is absent from prediction/helper
  signatures and does not appear in runtime candidate selection.

## Major stage gates

Do not run a larger experiment after every commit. Run one curated larger gate
only after a high-risk boundary changes:

- Fig.8 canonical 100-sentence neural retrieval after learning/prediction
  extraction.
- Fig.9 canonical 250-record strict raw L_match=4 after strict-runner or
  rollout extraction.

The major gate requires existing dataset/protocol hashes and exact predictions
against a trusted anchor. It is manually authorized and must not overwrite the
pre-refactor artifact.

## Windows runtime status

The current acceptance matrix has two independent dimensions:

- scientific assertions: isolated module suite;
- process stability: single-process suite.

Phase 0/1 records `696/696` isolated assertions passing and the single-process
suite crashing with `0xC0000005`. Architecture work must not describe that
native defect as fixed unless a separate reproducible crash investigation does
so.

