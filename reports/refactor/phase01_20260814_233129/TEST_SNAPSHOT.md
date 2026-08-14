# Pre-refactor Test Snapshot

## Compile

- Command: `.\.venv\Scripts\python.exe -m compileall -q src experiments tests`
- Exit code: `0`
- Status: `PASS`

## Single-process discovery

- Command: `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`
- Exit code: `3221225477` (`0xC0000005`, Windows access violation)
- Elapsed: `151.740` seconds
- Python traceback: none
- Last started test in the durable log:
  `test_fig9_preselection_segments.test_summary_full_and_gzip_levels`
- Status: `NATIVE_CRASH`

This is not a Python assertion failure and is not fixed by Phase 0/1. The
single-process suite must not be reported as passing.

## Isolated module runner

- Modules: `49`
- Passed modules: `49`
- Failed modules: `0`
- Native crashes: `0`
- Reported tests: `696`
- Status: `PASS_ISOLATED_ONLY`

Each `tests/test_*.py` module ran in a fresh Python process. This demonstrates
that all current assertions pass when native-library/process lifetime is
isolated. It is not equivalent to one-process discovery and does not resolve
the Windows runtime defect.

## Evidence

- `test_snapshot/compileall_result.json`
- `test_snapshot/single_result.json`
- `test_snapshot/single_failure.json`
- `test_snapshot/single_combined.log`
- `test_snapshot/isolated/summary.json`
- Per-module stdout/stderr logs below `test_snapshot/isolated/`

`SINGLE_PROCESS_STATUS=NATIVE_CRASH_0xC0000005`

`ISOLATED_MODULE_STATUS=PASS_49_MODULES_696_TESTS`

