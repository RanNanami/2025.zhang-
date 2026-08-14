# Core Flow Proposal

## Decision

The Phase 01 proposal is directionally sound but its full target tree is too
large to treat as a near-term blueprint. Creating protocol classes, reporting
layers, checkpoint wrappers, runtime logging, and per-figure packages at once
would add boundaries before their stability is proven. The safer design is a
small explicit Zhang state machine, extracted one action at a time behind the
existing `SequentialMemory` facade.

No manager, strategy hierarchy, registry, service container, or plugin layer
is justified. Diagnostics must consume read-only trace objects; the core must
not acquire diagnostic-specific interfaces merely to support possible future
experiments.

## Recommended researcher-facing flow

A researcher should be able to follow one computation through seven actions:

```text
1. encode external item
       SSTD encoder -> SymbolCode(column, time)

2. predict from prior lateral context
       previous active/winner cells
       -> delayed segment responses
       -> raw PredictionCandidate objects
       -> raw predicted SymbolCode

3. observe real proximal input
       match the saved prediction candidate by column/time
       -> predicted activation or burst
       -> choose one learning winner per input event

4. learn from the resolved event
       Scenario 1: reinforce the same predictive segment
       Scenario 2A: reinforce/grow the best timed matching segment
       Scenario 2B: create a segment on a least-used neuron
       Scenario 3: punish failed predictions

5. publish next transient state
       previous_active_cells <- predicted cells or burst cells
       previous_winners <- learning winners

6. propagate during autonomous retrieval
       raw prediction -> actual predicted neuron identities
       -> next transient context, without decoded-symbol replay

7. forget/prune after the existing learning point
       preserve the current weight/age equation and mutation order
```

The current public call order remains unchanged. In online training it is
typically `predict_code()` followed by `observe_code()`. In autonomous
retrieval it is `predict_code()` followed by `advance_prediction()`.

## Minimal future modules

This is a dependency direction, not authorization to perform the extraction in
Phase 2.

```text
src/seqmem/
  encoding.py          # existing SSTD code creation and decoding support
  dynamics.py          # existing PSP, dendrite, soma, oscillation behavior
  model.py             # stable facade, state ownership, pickle-visible classes
  _state_helpers.py    # later: pure transient snapshots/transformations only
  _learning_helpers.py # later: explicit calculations; model owns mutation/RNG
  _prediction_helpers.py # last: explicit calculations; preserve float order
  _forgetting_helpers.py # later: pure score/eligibility calculations

experiments/common/
  hashing.py           # Phase 2 candidate
  jsonio.py            # Phase 2 candidate
  csvio.py             # Phase 2 candidate only where byte-compatible
  artifacts.py         # Phase 2 candidate only for generic path creation
```

The four underscore-prefixed core helpers are future possibilities, not a
framework contract. A file should exist only after an exact golden regression
proves that one cohesive pure calculation can move safely.

## State ownership

`SequentialMemory` continues to own:

- persistent columns, neurons, segments, and synapses;
- `previous_active_cells` and `previous_winners`;
- `last_prediction_candidates` and symbol ranking;
- learning and decode RNG instances;
- the exact order of mutation, floating-point accumulation, and RNG calls.

Helpers may receive explicit values and return calculations. They must not
hide model mutation or random draws. This keeps the control flow visible and
avoids creating nominally clean modules whose behavior is actually coupled by
implicit state.

## Reduced Phase 01 proposal

| Earlier proposal | Phase 1.5 decision | Reason |
| --- | --- | --- |
| `experiments/common/artifacts.py` | Keep as a small Phase 2 candidate | Pure path/IO behavior can be tested byte-for-byte |
| `experiments/common/fingerprints.py` | Rename/scope to generic hashing only | Model fingerprint field selection is scientific audit policy and must stay put |
| `experiments/common/runtime_logging.py` | Defer | Native file-handle/runtime behavior has an unresolved crash history |
| Per-figure `protocol.py` classes | Defer | 505 CLI options and checkpoint compatibility make this a separate protocol migration |
| Per-figure `runner.py` / `reporting.py` | Defer | Would create a broad diff before the extraction method is proven |
| `fig9/checkpoints.py` | Defer | Checkpoint payload and restore semantics are explicitly out of Phase 2 |
| Fig.9 strict runner decomposition | Defer | The known 18 strict-to-diagnostic dependencies need a dedicated phase |
| `_state_helpers.py` | Candidate for a later approved phase | Transient state is lower risk than prediction, but still affects evaluation/RNG isolation |
| `_forgetting_helpers.py` | Later | Pruning controls persistent graph identity and checkpoint contents |
| `_learning_helpers.py` | Much later | Scenario and mutation order are high-risk scientific behavior |
| `_prediction_helpers.py` | Last | Candidate ordering and floating accumulation are the highest parity risk |

## Diagnostic boundary

The strict Fig.9 runner currently has 18 direct dependencies on diagnostics.
Phase 1.5 records that debt and does not move the imports. A future diagnostic
isolation phase should make trace production optional and read-only while
leaving strict execution independent of analyzers and artifact formatting.
That work must have its own on/off fingerprints and native-runtime checks.

## Phase 2 boundary

Phase 2 may extract only generic hashing, canonical serialization, atomic JSON,
truly generic CSV IO, and simple artifact-directory helpers. It must not split
`src/seqmem/model.py` or `experiments/fig9_strict_reproduction.py`. Each small
migration must pass all six Phase 01 golden baselines exactly before the next
migration starts.

This proposal intentionally makes the next code change modest. The first goal
is to prove that refactoring discipline works; reducing the two large modules
comes only after that proof.
