# Fig.9 Strict Runner Section Map

Audit baseline: `88dfbd897f30869ada3e8d4a8452005feaf71ee3`  
Branch: `refactor/architecture-cleanup-v2`  
Runner: `experiments/fig9_strict_reproduction.py`  
Starting size: 5,943 physical lines

This map classifies the runner before Phase 04 edits. Line numbers refer to the
audit baseline and are expected to drift after extraction.

| Lines | Responsibility | Classification | Phase 04 decision |
| --- | --- | --- | --- |
| 1-181 | Imports, protocol constants, diagnostic empty payloads, native globals | `PROTOCOL_SEMANTIC`, `NATIVE_CRASH_SENSITIVE` | Keep; only output imports may change |
| 183-252 | Strict config, rollout result, resumable run state | `PROTOCOL_SEMANTIC`, `CHECKPOINT_SENSITIVE` | Keep unchanged |
| 255-336 | Percentiles, correlations, field counts, passenger decode details | `PURE_FORMATTING`, `PROTOCOL_SEMANTIC` | Keep; decode details are not pure output |
| 339-366 | Segment count and transient fingerprint | `CHECKPOINT_SENSITIVE` | Keep unchanged |
| 369-909 | Process memory, environment capture, integrity checks, faulthandler, breadcrumbs | `NATIVE_CRASH_SENSITIVE`, `RUNTIME_INFRASTRUCTURE` | Keep unchanged |
| 915-928 | Density and long-sequence CSV forwarding wrappers | `PURE_OUTPUT` | Candidate for removal in favor of the existing Fig.9 writer |
| 931-1123 | Checkpoint save/load, sidecars, metadata, resume validation | `CHECKPOINT_SENSITIVE` | Explicitly excluded |
| 1126-1236 | Density summary aggregation | `PURE_FORMATTING` | Keep for now; it computes diagnostic content rather than only writing it |
| 1239-1288 | Encoder and model construction | `SCIENTIFIC_CORE` | Keep unchanged |
| 1291-1495 | State hashes, implementation version, protocol fingerprint and validation | `PROTOCOL_SEMANTIC`, `CHECKPOINT_SENSITIVE` | Keep unchanged |
| 1498-2611 | Autonomous raw rollout, prediction competition and transient restore | `SCIENTIFIC_CORE` | Keep unchanged |
| 2614-2620 | Strict summary path discovery | `PURE_OUTPUT` | Move to Fig.9 output module; preserve runner re-export |
| 2623-2818 | Stream API and diagnostic/protocol validation | `PROTOCOL_SEMANTIC` | Keep unchanged |
| 2820-2941 | Checkpoint restore or model initialization | `CHECKPOINT_SENSITIVE`, `SCIENTIFIC_CORE` | Keep unchanged |
| 2942-3148 | Diagnostic tracker setup and protocol fingerprint validation | `PROTOCOL_SEMANTIC`, `RUNTIME_INFRASTRUCTURE` | Keep unchanged |
| 3150-3205 | Native debug callback, integrity boundaries and prune hook | `NATIVE_CRASH_SENSITIVE` | Keep unchanged |
| 3206-4142 | Online prediction, observation, learning, diagnostics and periodic checkpoints | `SCIENTIFIC_CORE`, `CHECKPOINT_SENSITIVE` | Keep unchanged |
| 4143-4232 | Diagnostic flush/close and final checkpoint | `CHECKPOINT_SENSITIVE`, `NATIVE_CRASH_SENSITIVE` | Keep unchanged |
| 4233-4419 | Core summary and optional diagnostic summary construction | `PURE_FORMATTING`, `PROTOCOL_SEMANTIC` | Keep row/payload construction in runner |
| 4420-4447 | Output directory, prediction CSV, initial summary and protocol JSON | `PURE_OUTPUT` | Extract as one ordinary artifact block |
| 4448-5148 | Optional diagnostic artifact summaries and trace finalization | `RUNTIME_INFRASTRUCTURE`, `PURE_FORMATTING` | Keep diagnostic algorithms; migrate only generic writer calls if cohesive |
| 5149-5166 | Republish final summary, adaptation plots and close writers | `PURE_OUTPUT`, `NATIVE_CRASH_SENSITIVE` | Extract ordinary summary/plot writes; keep diagnostic close order in runner |
| 5169-5198 | Pre-change prediction CSV comparison | `PROTOCOL_SEMANTIC` | Keep unchanged |
| 5201-5558 | Argparse CLI and defaults | `PROTOCOL_SEMANTIC` | Explicitly excluded |
| 5561-5618 | April branch orchestration and summary artifact | `PROTOCOL_SEMANTIC`, `PURE_OUTPUT` | Keep orchestration; output helper may write its completed payload |
| 5621-5900 | Main protocol orchestration and runtime metadata construction | `PROTOCOL_SEMANTIC`, `RUNTIME_INFRASTRUCTURE` | Keep construction and ordering unchanged |
| 5901-5912 | Runtime JSON and canonical protocol copy | `PURE_OUTPUT` | Extract as one run artifact block |
| 5915-5943 | Profiling lifecycle and report write | `RUNTIME_INFRASTRUCTURE` | Keep unchanged |

## State Boundaries

- `previous_winners`, `previous_active_cells`, prediction candidates, Scenario
  state and RNG are mutated only inside the scientific/checkpoint regions above.
- Candidate output helpers may consume completed `rows`, `summary`, `fingerprint`
  and `runtime` values, but must not receive the model or encoder.
- Checkpoint payload construction, save/load order and sidecar writes remain in
  the runner even though they perform IO.
- Native crash file handles, background traceback thread, process-memory probes,
  breadcrumbs and diagnostic writer close order remain in the runner.

## Proposed Low-Risk Cuts

1. Move ordinary path helpers and core stream artifact writing to
   `experiments/fig9/outputs.py`.
2. Move completed runtime payload writing and canonical protocol copying to the
   same module.
3. Move adaptation plot selection into one explicit stream-output helper while
   retaining the existing plotting implementation and bytes.
4. Keep all payload construction, prediction row construction, diagnostics,
   checkpoint code, CLI and native instrumentation in place.
