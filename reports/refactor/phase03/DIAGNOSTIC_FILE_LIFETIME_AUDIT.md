# Fig.9 Diagnostic File Lifetime Audit

## Ownership Inventory

| Owner | Handle lifetime | Close path | Initial finding |
| --- | --- | --- | --- |
| Strict `write_diagnostic_csv` | lexical context manager | automatic | bounded and explicit |
| Strict `_PendingCsvAppendSink` | process-level registry | explicit close plus `atexit` | long-lived by design; pending CSV is compressed only after close |
| Strict native crash log | process lifetime | `close_native_crash_logging` plus `atexit` | intentionally outside diagnostic writer refactor |
| Strict native runtime log | process lifetime when enabled | same native close path | intentionally outside diagnostic writer refactor |
| Temporal-context `_CsvAppendSink` | tracker lifetime | runner calls tracker `close()` | bounded by enabled diagnostic |
| Autonomous-provenance `_CsvSink` | tracker lifetime | runner calls tracker `close()` | bounded by enabled diagnostic |
| Candidate-context writer | lexical context managers | automatic | bounded and explicit |
| Readout writer | lexical context manager | automatic | bounded and explicit |
| Independent-reference writer | lexical context manager | automatic | bounded and explicit |
| Actual-branch writer | lexical context manager | automatic | bounded and explicit |
| Segment-composition writer | lexical context manager | automatic | bounded and explicit |

## Callback Inventory

- `model.runtime_debug_callback` is installed only by native-runtime debug and
  is excluded from the scientific fingerprint/checkpoint compatibility path.
- `model.prune_diagnostic_callback` records scalar breadcrumb data for runtime
  diagnostics. It must not be moved in Phase 03.
- `model.reinforcement_trace_callback` is assigned only around the observation
  that produces a segment-reinforcement trace and is cleared immediately
  afterward.
- Provenance and trajectory trackers are passed as explicit arguments; the
  strict runner does not install them globally at module import time.

## Risks

1. `_PendingCsvAppendSink` has split ownership: the registry owns open sinks,
   while each sink owns an uncompressed pending file until final compression.
   Moving it requires byte-compatibility tests and explicit close verification.
2. Streaming tracker sinks own gzip handles across observations. Their runner
   close calls are required before summaries read the files.
3. Native crash handles and the periodic traceback thread have separate
   lifetime rules. They are historical crash instrumentation and are outside
   this extraction.
4. Empty diagnostic trackers currently contribute compatibility payloads even
   while disabled. Eliminating them is not an IO-only change.

## Initial Action

Only the generic strict diagnostic CSV functions are candidates for extraction
after lazy-loading work is proven exact. Native crash logging, checkpoint
writing, tracker payload structure, and tracker-specific streaming sinks stay
in place during the first migration.

## Final Phase 03 Result

The strict one-shot writer, pending append sink, close registry, and append
writer moved unchanged to `experiments/common/diagnostic_io.py`. Direct byte
tests cover plain CSV, one-shot gzip content, multi-batch gzip content, pending
file removal, and explicit close. Three real 20-record diagnostic fixtures
also preserve all scientific CSV content.

Tracker-specific temporal/autonomous streaming sinks did not move. Native crash
logging, checkpoint atomic writes, process memory monitoring, `atexit` crash
cleanup, and native traceback instrumentation did not move. The Phase 03
single-process suite still reproduced native terminations, including an access
violation inside continuous prediction. The writer extraction is therefore
not presented as a native-crash fix or root-cause result.
