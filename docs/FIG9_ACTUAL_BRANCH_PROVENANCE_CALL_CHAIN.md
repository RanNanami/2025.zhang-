# Fig.9 Actual Branch Provenance Call Chain

This document describes the diagnostic path in the current checkout. The
actual branch tracker is external metadata; it does not enter the model.

## Call order

1. `experiments/fig9_strict_reproduction.py:run_strict_stream`, around
   lines 2600-2740, encodes the current record and runs the autonomous rollout.
   Rollout uses raw emitted cells and restores the transient state before the
   actual observation.
2. The actual transition calls `model.predict_code()` and then
   `model.observe_code()` around lines 2700-2760. The actual-branch prematch
   capture runs after this prediction call and before `observe_code`; therefore
   it sees only pre-existing segments and the previous actual-history tracker.
3. `src/seqmem/model.py:observe_code` (around lines 2000-2460) assigns
   Scenario 1/2/3, selects matching segments, reinforces, creates segments,
   and updates winners. Those results are not read to create prematch anchors.
4. `experiments/diagnostics/fig9_branch_provenance.py:BranchProvenanceRegistry`
   records newly visible segments after observation. Its stable segment ID is
   derived from creation transition, target column/neuron, creation source
   fingerprint, and stable creation ordinal.
5. Around lines 2870-2930, the actual-branch trace is joined to the completed
   `ObservationTrace`, then the tracker records the selected pre-existing
   segment as an actual-history event. A segment created by the current
   observation is excluded from this current anchor update.
6. Around lines 3020-3140, the model checkpoint stores the diagnostic tracker,
   event rows, and segment rows alongside the existing diagnostic state. The
   model checkpoint format and model object are unchanged.

## What an anchor means

An `ACTUAL_HISTORY_ANCHOR` is created only after an earlier actual observation
selected or reinforced a pre-existing segment with a stable segment provenance
ID. Its ID is a deterministic hash of the stable segment ID, transition index,
field, target column/neuron, and source provenance signature. It is not a
Python object ID, row number, random UUID, floating-time join, current rank, or
candidate score.

The first actual source signature creates the first anchor. If the same stable
segment is later observed with a distinct actual source signature, a second
anchor is retained and the segment is marked `mixed_actual_history`. The
tracker never chooses one history to make a segment appear unique.

## Lifecycle boundaries

- Segment creation is performed by the existing learning rules. Creation
  sources are copied from the pre-observation `previous_winners` context for
  diagnostic labeling only.
- Segment reinforcement can increase the diagnostic reinforcement count, but
  reinforcement cannot retroactively create the current prematch anchor.
- Autonomous rollout does not add actual anchors. Its transient state is
  restored before actual observation and its trajectory is not joined into the
  actual-history branch.
- Segment deletion or checkpoint rebinding can make a chain unavailable. The
  stable ID does not silently attach to a newly recreated segment; the report
  must classify that case as unresolved or chain-broken.
- The current selected segment, matching score, candidate rank, and current
  reinforcement are post-observation joins. They are never reference inputs.

## Continuity classes

`UNIQUE_ACTUAL_BRANCH_CONTINUATION` means the prematch history had one anchor
and the selected pre-existing segment carries that same one. `ACTUAL_BRANCH_SWITCH`
means a unique prematch history existed but the selected pre-existing segment
carried a different existing history. `MULTIPLE_ACTUAL_BRANCH_COMPATIBLE`,
`ACTUAL_BRANCH_MIXED_HISTORY`, `ACTUAL_BRANCH_NOT_FOUND`, and
`ACTUAL_BRANCH_CHAIN_BROKEN` preserve ambiguity. None of these classes means
correct or wrong.

Tier A prematch predicted references remain separate from actual-history
anchors. Tier A compatibility is prediction compatibility only, not
correctness. Actual branch continuity is descriptive evidence and cannot be
used to claim causality.
