# Fig.9 Matching Semantics Implementation Audit

## Repository implementation

- Matching context is `previous_active_cells` under strict all-cell burst
  context, with an empty-state fallback to `previous_winners`.
- Exact source identity and delayed arrival within `timing_tolerance` determine
  `Segment.timed_overlap`.
- Weight and continuous PSP response determine the secondary segment score.
- A target prediction also has an independent firing-time validity check.
- `L_match=4` gates Scenario-2 reuse after a best segment is found.
- Segment creation and growth use `previous_winners`, which is narrower than
  the strict matching context.

## Naming versus behavior

The old `SOURCE_TIMING_OR_ELIGIBILITY_EXCLUDED` label combined several
possibilities in its name. In the current strict path, an exact source in the
matching context fails contribution specifically when its delayed arrival is
outside the timed-overlap window. Segment-level eligibility and target
prediction-time validity are downstream or separate conditions and must not be
silently attributed to an individual synapse.

## Paper-supported statements

Existing repository comments mark the three learning scenarios, bursting,
distal prediction, and winner-based segment growth as paper-explicit. This
audit does not re-read or quote the paper, so it does not claim that the exact
Python timing window, active-cell fallback, or tie-break implementation is
paper-mandated.

## Implementation choices requiring paper-text verification

- Whether contextual matching should use all active cells or winners only.
- Whether source contribution should use the discrete arrival window exactly
  as implemented.
- Whether the target prediction-time and source-arrival windows should share
  the same tolerance.
- Whether PSP response score should influence matching after overlap.
- Whether winner-only growth combined with all-cell matching is intended.

These are implementation-semantic suspicions, not established
paper/reproduction mismatches.

