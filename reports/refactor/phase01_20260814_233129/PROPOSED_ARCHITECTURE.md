# Proposed Architecture

## Design objective

Reduce review and maintenance cost while preserving prediction bytes, graph
state, RNG state, floating-point order, checkpoint class paths, CLI defaults,
and historical evidence. Extraction precedes abstraction.

## Target layout

```text
src/seqmem/
  encoding.py                 # existing encoders
  dynamics.py                 # existing PSP and neuron dynamics
  model.py                    # compatibility facade and pickle-visible classes
  _state_helpers.py           # later: pure transient-state transformations
  _forgetting_helpers.py      # later: pure eligibility/score helpers
  _learning_helpers.py        # later: explicit-argument learning calculations
  _prediction_helpers.py      # last core extraction; no hidden RNG/state

experiments/
  common/
    artifacts.py              # atomic JSON/CSV and artifact metadata
    fingerprints.py           # canonical SHA-256 utilities
    runtime_logging.py        # process/native breadcrumbs, no model semantics
  fig7/
    protocol.py
    runner.py
    reporting.py
  fig8/
    protocol.py
    runner.py
    retrieval.py
    reporting.py
  fig9/
    data.py                    # existing stable helper
    metrics.py                 # existing stable helper
    outputs.py                 # existing stable helper
    learning.py                # existing stable helper
    protocol.py                # future typed paper-facing config
    checkpoints.py             # wrappers preserving exact payload/version
    strict_runner.py           # future orchestration after regression gates
  diagnostics/                # explicitly nonpaper; strict does not import eagerly
  historical/                 # retained compatibility/history, never strict dependency

scripts/
  run_logged_process.py
  capture_phase01_baselines.py
  run_isolated_unittest_modules.py

reports/                      # committed audits and curated conclusions
results/                      # generated experiment outputs and checkpoints
```

## Compatibility facade

The following definitions remain in `seqmem.model` until a versioned checkpoint
migration is designed and tested:

- `Synapse`
- `Segment`
- `Neuron`
- `MiniColumn`
- `MemoryParams`
- `SequentialMemory`

`SequentialMemory.predict_code()`, `observe_code()`, and public decoding/state
methods keep their current signatures. Later helpers receive explicit values
and return calculations; `model.py` continues to own mutation and call order.
This avoids changing pickle globals or hiding RNG draws behind new objects.

## Configuration boundary

Future typed configs should separate:

- `Fig7ProtocolConfig`
- `Fig8ProtocolConfig`
- `Fig9ProtocolConfig`
- `DiagnosticConfig`

Paper-facing config contains only protocol choices. Diagnostic config contains
trace, oracle, teacher-forced, profiler, compression, and debug switches. The
existing CLI remains the compatibility surface until an exact default-parity
test maps every one of the 505 inventoried options.

## Dependency rules

1. `src/seqmem` cannot import `experiments`.
2. Strict runners cannot import oracle, teacher-forced, or historical modules.
3. Diagnostics may import strict read-only interfaces; strict loads diagnostic
   adapters only in a separate diagnostic entrypoint or explicit lazy path.
4. Historical shims may import historical implementations, but strict code may
   not import the shim.
5. Ground-truth-bearing objects cannot enter model prediction, candidate
   selection, learning, or autonomous rollout.

## First extraction target

Phase 2 should extract the pure artifact utilities currently mixed into the
Fig.9 runner: canonical hashing, atomic JSON writing, CSV writing/appending,
and output-path metadata. These functions do not own model objects or RNG and
can be verified byte-for-byte. Native crash logging is a separate operational
subphase because file-handle lifetime has previously been unstable.

## Deferred work

Prediction, learning, transient state, forgetting, and Fig.9 stream
orchestration remain in place until lower-risk utilities have exact golden
parity. No Strategy/Factory hierarchy is proposed. Small functions with
explicit arguments are sufficient until stable boundaries are demonstrated.

