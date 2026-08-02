# Fig.9 Real L_match Mid-run Failure Diagnosis

## Evidence

- The two failed L4 runs reached different checkpoints (`next_index=100` and
  `next_index=150`). This rules against a deterministic malformed input row.
- The latest process left no Python traceback and the old launcher did not
  preserve a child-process result record.
- At index 150 the checkpoint is internally consistent: 1,607 live segments
  and 1,607 provenance bindings, with no unknown binding.
- A copied checkpoint resumed from 150 to 160 in 22.61 seconds. At index 160 it
  contained 1,691 segments, 88,188 synapses, and used 235.65 MB of working-set
  memory. The source checkpoint remained at index 150.

## Root Cause

The most likely cause is resource growth in the optional diagnostic layer,
followed by external process termination. It is not evidence of an L_match=4
model exception. The previous run retained every wide context-trajectory,
preselection-segment, branch-segment, and match-overlap-segment row in Python
lists until completion. Context dependency analysis also replayed the same
column history independently for duplicate source dependencies.

The exact terminating resource (memory limit, host termination, or another
external condition) cannot be proven retroactively because the old launcher
did not preserve stderr, a native exit code, or resource progress. The new
launcher records all three, so a recurrence will be attributable.

## 150-200 Complexity Audit

- **Branch provenance capture:** the old path built a set of every segment
  object ID and rescanned every live segment after each observation. When an
  `ObservationTrace` is already present, the implementation now consumes its
  exact newly-created segment objects. Stable provenance ordering and IDs are
  unchanged.
- **Creation source fingerprint:** sources are sorted once per observation and
  reused for all exact new segments. The hash contents are unchanged.
- **Checkpoint rebind signature:** all segment source/delay signatures are still
  generated at checkpoint boundaries. This remains linear in total segment and
  synapse count, but no longer occurs per record.
- **Context trajectory:** duplicate `(trajectory kind, source column, creation
  index, current index)` queries now share one history summary. Previously each
  dependency scanned all retained transition snapshots, producing
  dependency-count times history-length growth.
- **Candidate enumeration and threshold crossing:** `_live_incoming` lookup,
  continuous PSP integration, threshold crossing, and candidate selection were
  not modified. They are model behavior, not diagnostic overhead.
- **Trace serialization:** the largest read-only trace tables are appended and
  flushed every progress interval, then released. This bounds resident trace
  memory without changing row order or values.

## Recovery Guarantees

Checkpoint pickle and provenance JSON use temporary files followed by atomic
replacement. Provenance is embedded in new checkpoints and retained as a
sidecar for compatibility. Resume validates L_match and requires an exact
segment-to-provenance rebind before processing another record.

The original failed checkpoint predates incremental trace state. Its model,
predictions, density state, and provenance resume exactly, but diagnostic CSVs
created after resumption cover only the continuation. This limitation affects
offline trace completeness, not model state or predictions.

## Behavioral Scope

No prediction, learning, L_match, K, network size, forgetting threshold,
propagation, continuous PSP, RNG, rollout, or ground-truth isolation behavior
was changed. The patch changes logging, checkpoint durability, resume auditing,
trace memory management, and equivalent diagnostic bookkeeping only.
