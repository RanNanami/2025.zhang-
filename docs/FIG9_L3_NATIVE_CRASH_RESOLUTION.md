# Fig.9 L=3 Native Crash Resolution

## Failure evidence

- Failed run: `lmatch_ablation_250_20260803_032526/L3_20260803_032526_110`
- Windows status: `0xC0000005` (`STATUS_ACCESS_VIOLATION`)
- Persisted checkpoint: `next_index=200`
- Last native breadcrumb: index 207, rollout step 2, `rollout_step_enter`
- Python traceback: none

The failed process had advanced past its last persisted checkpoint. Progress output
therefore was not treated as recoverable model state.

## Isolation result

Ten copied-checkpoint runs covered core-only, each diagnostic family, all diagnostics
without gzip, and all diagnostics with gzip. Every run advanced from 200 to 202 with
exit code 0. Predictions, MAPE, final model fingerprint, and RNG fingerprint were
identical. No diagnostic family or gzip combination reproduced the crash.

Periodic native logs from the failed run showed short-lived `subprocess.py`
`_readerthread` frames created by the Windows `tasklist` memory query. The crash
occurred near the next periodic faulthandler scan. The operational fix therefore:

1. reads the small `tasklist` pipe on the calling thread;
2. disables periodic all-thread traceback scanning by default;
3. retains synchronous phase breadcrumbs and fatal faulthandler logging.

This is strong evidence for a run-supervision thread-lifecycle race. The exact
faulting native module remains unproven because ProcDump was unavailable.

## Formal recovery

The original checkpoint and failed directory were not modified. A copied checkpoint
resumed at index 200 and completed at index 245 (45 prediction anchors):

- exit code: 0
- predictions: 45/45
- coverage: 1.0
- MAPE: 0.43735914619285043
- final rolling MAPE: 0.3889685847467485
- runtime: 914.8619519000058 seconds (model runtime)
- final segments: 2225
- final synapses: 124663

Independent MAPE recomputation matched exactly. The saved model and RNG fingerprints
matched the final summary, and every gzip trace was readable to EOF.

The fix changes logging, process-memory measurement, and supervision only. It does
not change prediction, learning, L_match semantics, RNG, rollout, or strict protocol
parameters.
