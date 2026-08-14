# Fig.9 Runner Responsibility Map

The strict runner is a 6,000-line orchestration module. It imports stable Fig.9
helpers and many nonpaper diagnostics at module import time.

| Responsibility | Current symbol/range | Scientific or operational risk | Proposed treatment |
| --- | --- | --- | --- |
| Strict configuration | `Fig9StrictConfig` 240-266 | Protocol/default risk | Keep public shape stable; later map CLI to config explicitly |
| Resume state | `StrictRunState` 292-309 | Checkpoint compatibility | Do not move before migration tests |
| Native crash logging | lines 584-1129 | Runtime-only, file-handle sensitive | Future infrastructure extraction |
| Checkpoint IO | `save_strict_checkpoint` 1146-1223; `load_strict_checkpoint` 1286-1291 | Highest operational compatibility risk | Extract only as wrappers preserving payload |
| Encoder/model construction | `build_fig9_encoder` 1454-1482; `build_strict_model` 1485-1503 | Protocol and RNG initialization | Add exact fingerprint gate before moving |
| Fingerprinting/protocol | lines 1506-1720 | Pure audit utilities | Best early extraction candidate |
| Autonomous rollout | `rollout_raw_autonomous` 1721-2811 | Core scientific trajectory | Do not extract early |
| Stream training/evaluation | `run_strict_stream` 2823-5312 | Combines prediction, observation, learning, diagnostics, output, resume | Decompose late and incrementally |
| CLI | `parse_args` 5347-5704 | Hundreds of paper/diagnostic options share one namespace | Inventory now; separate config later without default changes |
| Top-level orchestration | `run_main` 5767-6058 | Dataset, streams, branches, profiling, summaries | Extract after runner regression gate |
| Diagnostic hooks | imports at lines 50-162 and branches in `run_strict_stream` | Strict depends on nonpaper modules | Future adapter boundary; defaults remain off |

## State transition timing

`rollout_raw_autonomous` snapshots transient state, advances raw predicted cell
identities for each horizon step, and restores the observation trajectory.
Actual observation then enters `model.observe_code(..., learn=True)` inside
`run_strict_stream`, updating long-term graph state and `previous_winners`.
Diagnostic traces must distinguish autonomous rollout state from the actual
observation state and must never feed oracle/teacher references into either.
