# Fig.9 Windows Native Crash Audit

## Crash Classification

`3221225477` is `0xC0000005` (`STATUS_ACCESS_VIOLATION`). Windows terminated
the process after native code accessed invalid memory. Python cannot catch this
reliably as a normal exception, which explains the missing traceback.

## Recoverable State

The failed run reports `last_completed_index=210`, but its checkpoint contains
`next_index=200`. `progress.jsonl` is telemetry, not serialized model state.
There is therefore no valid index-210 model to use for a 210-to-220 isolation
run. Replaying 200-to-210 is prohibited by the requested protocol, so the
isolation script refuses this checkpoint.

## Warmup-boundary Audit

- Autonomous rollout starts at index 200 and snapshots only Python containers,
  dataclasses, RNG tuples, and model references.
- `restore_transient_state` copies dictionaries and restores Python RNG state.
  It does not retain external buffers.
- Prediction candidates and traces are Python lists/tuples/dataclasses. The
  Fig.9 runner contains no NumPy slice/view, `memoryview`, `mmap`, or shared
  mutable native array.
- Checkpoints contain the encoder, model, metrics, Python rows, and provenance.
  They do not pickle gzip handles, crash-log handles, ctypes objects, threads,
  or native buffers.
- gzip writers are created inside `with` blocks and closed before their row
  batches are released. No writer is reused across checkpoint restore.
- The dedicated faulthandler file is held by a module global. Shutdown first
  cancels delayed dumps and disables faulthandler, then closes the descriptor.

## Native Surfaces

The relevant native surfaces in the failed process were:

1. CPython 3.12 runtime;
2. zlib through gzip compression;
3. `_ctypes.pyd` and Windows PSAPI through the working-set measurement;
4. native modules loaded transitively by plotting/scientific dependencies.

Windows Error Reporting classified the crash as `BEX64`, with exception data
`8` (execute access violation), `StackHash_ac46`, and an unknown faulting
module. Its loaded-module list explicitly includes `_ctypes.pyd` and
`psapi.dll`. The working-set helper was the only direct ctypes call introduced
immediately before this crash series, so it has been removed from the Python
process. Memory is now queried by a separate `tasklist.exe` process once at
observation entry and that value is reused in the rollout breadcrumbs.

`native_environment.json` records Python, NumPy, Pandas, SciPy,
Matplotlib, NumExpr, threadpoolctl, zlib, platform, thread limits, and loaded
`.pyd`/`.dll` modules visible at startup.

## Isolation Matrix

`scripts/fig9_lmatch_native_crash_isolation.ps1` defines core-only,
full-without-compression, plain/gzip comparison, individual diagnostic family,
and full single-thread profiles. Every profile copies the source checkpoint and
is limited by `--debug-end-index` to at most ten records. It refuses any source
whose real `next_index` is not 210.

ProcDump is optional. When enabled, the supervisor attaches to the exact child
PID with `-e 1 -ma` and stores dumps under that profile's `dumps` directory.

## Current Conclusion

The faulting module and minimum reproducing diagnostic family remain unknown:
the old process produced neither a faulthandler file nor a dump, and no
index-210 checkpoint exists. No model or protocol conclusion should be inferred
until a valid state or an approved replay is available.
