# Phase 02 Final Report

## Status

`PHASE02_PASS`

Phase 1.5 aligned the proposed architecture with Numenta's Temporal Memory as
an architecture reference. Phase 2 then moved three classes of behavior-neutral
IO helpers in separate commits. All six frozen regression fixtures matched
exactly after every migration.

Frozen scientific reference:
`1864f46aee5cc0aee814fd4e6154b06f8bfeb08f`.

Working branch: `refactor/architecture-cleanup-v2`.

## 1. Reference alignment

NuPIC and the current Zhang implementation share a useful structural pattern:

- prior active cells and winners are explicit inputs to the next transition;
- a predicted column/cell follows a different activation and learning path
  from an unpredicted bursting column;
- a matching segment can select a winner;
- absence of a matching segment can select a least-used cell and create a
  segment from previous winners;
- distal segment activation prepares prediction for the next transition;
- failed predictions have a separate punishment path.

The complete mapping is in
`reports/refactor/REFERENCE_ARCHITECTURE_ALIGNMENT.md`.

## 2. Behavior that cannot be copied from NuPIC

NuPIC is not Zhang implementation truth. Its active-column sets, permanence
overlap thresholds, and predicted-segment decrement cannot replace Zhang's:

- ordered SSTD column-time events;
- delayed double-exponential PSP integration;
- dendritic threshold crossing and soma firing time;
- depolarized state, oscillation, refractory behavior, and phase precession;
- autonomous raw-neuron propagation;
- Zhang Scenario 1/2A/2B/3 weight and age semantics;
- the existing forgetting equation and pruning order.

These are recorded as `REFERENCE_BEHAVIOR_DIFFERENCE`; no behavior was changed
to imitate NuPIC.

## 3. Architecture reduction

The Phase 01 proposal was directionally correct but too broad for near-term
execution. Protocol classes, checkpoint wrappers, runtime logging, per-figure
runner/reporting packages, and core model helpers should not all be introduced
at once. The reduced proposal uses small explicit functions and retains
`SequentialMemory` as state and mutation owner. It introduces no manager,
strategy hierarchy, registry, or speculative interface.

The recommended seven-action reading path is:

```text
encode
-> predict from prior lateral context
-> observe real proximal input
-> resolve and apply Scenario learning
-> publish next transient state
-> propagate raw neural prediction during retrieval
-> forget/prune at the existing point
```

Details are in `reports/refactor/CORE_FLOW_PROPOSAL.md`.

## 4. Functions migrated

| Function | Previous location(s) | New location | Current callers | Output compatibility |
| --- | --- | --- | --- | --- |
| `sha256_file()` | local `file_sha256()` copies in `experiments/fig9_strict_reproduction.py`, `experiments/fig9/parity.py`, and `scripts/capture_phase01_baselines.py` | `experiments/common/hashing.py` | the same three modules | Exact digest; same 1 MiB read chunks |
| `write_json()` | `experiments/fig9_strict_reproduction.py` | `experiments/common/jsonio.py` | strict Fig.9 runner | Exact UTF-8 text-mode bytes, sorted keys, indent 2, no added trailing newline |
| `write_json_atomic()` | `experiments/fig9_strict_reproduction.py` | `experiments/common/jsonio.py` | strict Fig.9 runner | Exact bytes and `.tmp` -> `fsync` -> `os.replace` sequence |
| `write_csv_rows()` | five equivalent local `write_csv()` bodies | `experiments/common/csvio.py` | Fig.8 branch-coherence, false-positive, prediction-competition, response-scale, and Scenario-1 diagnostics | Exact field discovery, quoting, newline, and both historical empty-input behaviors |

The old local public `write_csv()` names remain as one-line compatibility
wrappers in the five diagnostics. The strict runner remains a single module;
only two pure JSON function bodies and one pure file-hash body were replaced by
imports.

## 5. Candidates deliberately not migrated

- Canonical model serialization and fingerprint field selection stayed in the
  baseline/runner owners because moving or normalizing them could alter audit
  schema or bytes.
- Strict diagnostic CSV/GZIP append sinks stayed in place because their file
  lifecycle is tied to prior Windows native-runtime investigations.
- Prediction CSV writers stayed local because their columns carry experiment
  protocol semantics.
- Artifact-path and report helpers were not created: the audit found only
  trivial `mkdir` repetition or incompatible formatting policies. A generic
  abstraction would add parameters without establishing a stable boundary.
- The 18 direct strict-to-diagnostics dependencies were recorded and left
  untouched.

## 6. Protocol and scientific invariants

The migration changed no protocol, model parameter, strict default, diagnostic
default, checkpoint payload, checkpoint class path, RNG call, floating-point
accumulation, candidate order, winner selection, Scenario branch, rollout, or
learning order.

`src/seqmem/model.py` was not modified. In particular, no scientific logic in
`predict_code()`, `observe_code()`, Scenario 1/2A/2B/3, PSP dynamics, SSTD,
inhibition, forgetting, or autonomous retrieval changed.

No historical result or checkpoint was deleted. `main` was not merged.

## 7. Golden regression

Each migration independently regenerated all six fixtures and compared these
fields exactly:

- prediction SHA-256;
- metric payload;
- Scenario summary;
- science model SHA-256;
- full-state SHA-256;
- combined, learning, and decode RNG SHA-256;
- previous-active, previous-winner, and candidate SHA-256;
- all recorded structure counts, including segment and synapse counts.

| Migration | Fig8 10 | Fig8 20 | Fig8 50 | Fig9 10 | Fig9 20 | Fig9 50 |
| --- | --- | --- | --- | --- | --- | --- |
| Hashing | EXACT | EXACT | EXACT | EXACT | EXACT | EXACT |
| JSON IO | EXACT | EXACT | EXACT | EXACT | EXACT | EXACT |
| CSV IO | EXACT | EXACT | EXACT | EXACT | EXACT | EXACT |

Evidence:

- `reports/refactor/phase02_regression/hashing.json`
- `reports/refactor/phase02_regression/json_io.json`
- `reports/refactor/phase02_regression/csv_io.json`

Therefore prediction SHA, science/full-state SHA, all RNG SHA values, Scenario
counts, and segment/synapse counts are all unchanged in all six fixtures.

## 8. Tests

- `COMPILEALL_STATUS`: `PASS`
- `SINGLE_PROCESS_STATUS`: `PASS_THIS_RUN`, 702/702 tests in 232.642 seconds
- `ISOLATED_MODULE_STATUS`: `PASS`, 50/50 modules and 702/702 tests
- isolated native crash count: 0
- infrastructure helper tests added: 6

The Phase 01 one-process run exited with Windows `0xC0000005`. The Phase 2
one-process run did not reproduce it. **The native crash is not claimed fixed.**
No native-runtime fix was attempted, and one successful run is insufficient to
establish resolution.

## 9. Commits

- `d374e80 docs(refactor): align architecture with Temporal Memory reference`
- `08b5d11 refactor(io): extract deterministic hashing helpers`
- `099dad5 refactor(io): extract atomic json helpers`
- `9520c5f refactor(io): extract generic csv helpers`
- final report and concise test evidence: this document's containing commit

## 10. Required answers

1. **Which NuPIC architecture is similar?** Explicit previous state,
   predicted/burst/punishment branches, matching-segment winner selection,
   least-used-cell creation, and distal activation for next-step prediction.
2. **What cannot be copied?** SSTD, PSP, delay, soma/dendrite dynamics,
   oscillation, autonomous propagation, Zhang learning/ageing, and forgetting.
3. **Was the old proposal overdesigned?** Yes as a near-term plan; it was
   reduced to small explicit modules introduced only after parity evidence.
4. **Recommended core flow?** The seven actions in Section 3.
5. **Functions migrated?** `sha256_file`, `write_json`, `write_json_atomic`,
   and `write_csv_rows`.
6. **Previous paths?** Listed in Section 4.
7. **New paths?** `experiments/common/{hashing,jsonio,csvio}.py`.
8. **Callers?** Listed in Section 4.
9. **Any function output bytes changed?** No for the migrated compatibility
   cases; JSON and CSV bytes have direct tests.
10. **Any protocol changed?** No.
11. **Any scientific behavior changed?** No.
12. **All six golden baselines exact?** Yes after each migration.
13. **Prediction SHA all identical?** Yes.
14. **Science model SHA all identical?** Yes.
15. **RNG SHA all identical?** Yes, including learning and decode RNG.
16. **Scenario counts all identical?** Yes.
17. **Segment/synapse counts identical?** Yes.
18. **Compileall?** PASS.
19. **Single-process tests?** PASS this run, 702/702.
20. **Isolated tests?** PASS, 50/50 modules and 702/702 tests.
21. **Native crash claimed fixed?** NO.
22. **Scientific logic in `model.py` modified?** NO.
23. **Fig.9 runner decomposed?** NO; only pure helper imports changed.
24. **Historical results deleted?** NO.
25. **`main` merged?** NO.
26. **Commits?** Listed in Section 9.
27. **Next phase?** `ISOLATE_DIAGNOSTIC_IO`.

## 11. Single next recommendation

`ISOLATE_DIAGNOSTIC_IO`

This is the narrowest next step because the strict Fig.9 runner still has 18
direct diagnostic dependencies and prior native-runtime evidence is entangled
with diagnostic writers and file lifetimes. It should be a separately approved
phase with diagnostic on/off fingerprints and repeated native-runtime tests.
It must not begin model state, learning, prediction, or runner-orchestration
extraction at the same time.
