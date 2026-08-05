# Fig.9 Independent Reference Provenance

The strict path is `experiments/fig9_strict_reproduction.py:run_strict_stream`.
For each actual record it encodes the record, runs the five-step autonomous
rollout, restores the transient rollout state, and then performs the actual
transition as `model.predict_code()` followed by `model.observe_code()`.

The independent reference hook is placed after that actual `predict_code()`
returns and immediately before `observe_code()`. It reads only the
same-call `last_prediction_candidates`, the pre-existing segment registry,
and the previous actual-reference history. It does not call matching helpers,
read the selected candidate, read reinforcement traces, or score segments.

`BranchProvenanceRegistry` is populated only after the observation completes,
when newly created segments are visible. Its stable segment IDs are derived
from creation transition, target column/neuron, source fingerprint, and the
stable creation ordinal. The diagnostic-only branch ID is a deterministic hash
of that stable segment ID and creation metadata; it is not model state.

Actual observation state and autonomous rollout state are separate: rollout
uses raw emitted prediction cells to advance context and then restores the
transient snapshot. The later actual observation updates
`previous_active_cells` and `previous_winners` through `observe_code`. Only
selected pre-existing actual segments are added to the diagnostic history,
and only after the current observation has completed. A segment created or
reinforced by the current observation therefore cannot become its own
reference.

Implementation anchors in the current checkout:

- `experiments/fig9_strict_reproduction.py:2610-2730`: per-record pre-state,
  actual prediction, independent capture, and `observe_code`.
- `experiments/fig9_strict_reproduction.py:2760-2890`: created-segment
  registration, post-observation join, and actual-history update.
- `experiments/fig9_strict_reproduction.py:3020-3095`: checkpoint persistence
  for diagnostic rows and tracker state.
- `experiments/fig9_strict_reproduction.py:3530-3565`: trace and protocol
  output.
- `src/seqmem/model.py:900-1600`: prediction candidate production and
  `last_prediction_candidates` update.
- `src/seqmem/model.py:2000-2460`: actual observation scenario assignment,
  matching, winners, and reinforcement.

The event trace records the pre-matching reference set and post-observation
join. The segment trace records every live pre-existing segment in the
observed column. Current winner, score, reinforcement, and future
compatibility are analysis joins only; they never feed prediction, selection,
learning, RNG, or checkpoint model state.

## Stable keys

Trace joins use `segment_provenance_id`, derived `branch_provenance_id`,
creation transition, target column/neuron, and source fingerprint. Python
object IDs are used only during the in-memory join and are excluded from all
CSV, JSON, and checkpoint provenance rows.

## Limitations

Tier A is available only when the same prediction call produced a candidate
for the observed column. Tier B comes from previous actual observations.
Tier C chain continuity is represented by the stable provenance fields but is
not promoted to a reference unless its chain is present. Tier D future
compatibility is post-hoc analysis and is never a current reference. The
diagnostic is therefore evidence about provenance, not a change to the strict
Fig.9 algorithm.
