# Fig.9 Actual Branch Identity Granularity Audit

This audit is diagnostic-only and offline. It does not load a model checkpoint
for prediction, does not change matching or learning, and does not use future
observations or ground truth for model selection.

## Scope and source lines

The implementation was audited at commit `245a68b`, with the explicit
pre-match lineage trace added in the working tree as a diagnostic-only
extension.

- `experiments/diagnostics/fig9_actual_branch_provenance.py:102` defines
  `_new_anchor`.
- `:114` defines `actual_anchor_id` from segment, field, target column, target
  neuron, and the current source signature.
- `:115` immediately defines `actual_branch_provenance_id` from that anchor,
  segment, and source signature.
- `:124` in the legacy branch record stores `parent_actual_anchor_ids` as an
  empty string; the v2 anchor record stores parent anchors captured before the
  actual transition.
- `:136-137` stores `parent_branch_ids` as an empty string and `branch_depth`
  as `0`.
- `:156-234` adds the anchor and branch to segment state and increments
  reinforcement counters, but never resolves a historical parent.
- `:490-517` records actual history after the observation trace identifies the
  selected or reinforced segment.
- `experiments/fig9_strict_reproduction.py:2885-2895` captures pre-matching
  history and then calls `model.observe_code`.
- `experiments/fig9_strict_reproduction.py:3070-3089` joins the observation
  trace and records actual history. The v2 tracker receives the pre-match
  parent-anchor set at this point; autonomous rollout remains separate.

## Identity definitions

### Actual anchor

An anchor is one stable hash of a segment provenance ID, field, target column,
target neuron, and the source-cell signature available at the actual event.
Repeated reinforcement with the exact same tuple reuses the anchor. A changed
source signature creates a different anchor, even when the target segment is
the same.

### Actual branch

The current branch ID is a second hash built immediately from the new anchor.
It is not inherited from a previous branch. Its root is the new anchor, its
parent list is empty, and its depth is zero.

### Event, anchor, lineage

- **EVENT**: one actual observation/reinforcement event.
- **ANCHOR**: one concrete historical provenance node.
- **LINEAGE**: a sequence of anchors connected by explicit historical parent
  relations.

The legacy branch trace records the first two. The v2 tracker also records a
lineage relation when it can be resolved from the pre-match state. This
relation is not read by matching, prediction, learning, or selection.

## Formal trace sanity statistics

The results below were generated offline by
`experiments/diagnostics/analyze_fig9_branch_identity.py` from the existing
formal traces:

| run | anchor count | branch count | reinforcement count | P(branch = anchor) | P(new reinforcement creates branch) | P(inherits branch) | P(depth = 0) |
|---|---:|---:|---:|---:|---:|---:|---:|
| L4 | 4943 | 4943 | 4943 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| L2 | 5375 | 5375 | 5375 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

Every branch row has one root anchor, no parent branch, depth zero, and one
reinforcement. Therefore the current `actual_branch_provenance_id` is
effectively an anchor alias, not a continuing lineage identity.

## Meaning of mixed history

`ACTUAL_BRANCH_MIXED_HISTORY` should be interpreted as
`ANCHOR_LEVEL_MIXED_HISTORY`. It means that multiple anchor IDs are associated
with a segment under the current diagnostic definition. It does **not** prove
that multiple independent historical lineages merged.

The legacy trace cannot distinguish repeated anchors from one continuing
lineage, a true merge, a split, or unresolved history because parent relations
were never recorded. The v2 trace resolves a subset of continuations and keeps
`LINEAGE_UNRESOLVED` when the parent lineage is unavailable.

The v2 classification uses `LINEAGE_ROOT` when no pre-existing anchor is
visible, `LINEAGE_CONTINUATION_UNIQUE` when one existing lineage is resolved,
and `LINEAGE_UNRESOLVED` when parent anchors are present but cannot be mapped
to a known lineage. It never invents `ACTUAL_LINEAGE_ID` from prediction error,
a future observation, or the decoded target.

## Output files

The audit outputs are under
`results/fig9_diagnostics/branch_identity_joint_20260808/`. They include
branch depth, parent counts, anchors per branch, branches per anchor, segment
growth, event-level lineage classification, explicit lineage registry data, and
the joint summary. The earlier `branch_identity_sanity_20260807` directory is
kept as the pre-v2 baseline.

## Conclusion and next diagnostic

The valid legacy-branch conclusion is:

`ACTUAL_BRANCH_ID_IS_EFFECTIVELY_ANCHOR_ID`

For the v2 diagnostic trace, the valid lineage conclusion is:

`ACTUAL_LINEAGE_PARTIALLY_RECOVERED_FROM_EXPLICIT_PREMATCH_TRACE`

Formal L4 has 740 lineage roots, 471 unique continuations, and 3732
unresolved events. Formal L2 has 690 roots, 418 unique continuations, and
4267 unresolved events. No `LINEAGE_MERGE` was observed in either run. This
rejects the old branch ID as a lineage identity, but does not justify a model
change because the unresolved majority remains.

The next step is a pure diagnostic trace that records parent-anchor candidates
from state available before each actual transition. No strict model mechanism
should be changed until that trace can separate same-lineage continuation from
true lineage merge.
