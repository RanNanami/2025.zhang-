# Fig.9 Timing / Eligibility Decomposition

This is a read-only diagnostic. It does not modify matching, prediction,
learning, L_match, burst context, RNG, or strict defaults.

- Source rows: 210225
- Unique observations: 1041
- Unique segments: 1975
- Unknown reason rate: 0.000000
- Dominant primary reason: `SOURCE_COLUMN_NOT_ACTIVE`
- Recommended next step: `CONTEXT_TRAJECTORY_DIAGNOSTIC_REQUIRED`

The repository's actual overlap uses exact source-cell membership **and**
`source_time + synaptic_delay` within the dendritic target time plus/minus
`timing_tolerance`. Weight and PSP score rank candidate segments but do not
make an individual source contribute to `timed_overlap`. With strict
`burst_context=True`, matching uses `previous_active_cells` (falling back to
`previous_winners` only when active cells are empty), not winner-only context.

Rates in the CSV files use pre-observation source-synapse rows in the named
scope. This small diagnostic cannot establish the late 200-244 distribution
unless the manually filtered 250-record trace passes the 329-observation and
2437-segment coverage checks.

The context-membership counterfactual is fixed-state and read-only. It never
uses ground truth to select a segment and never produces a new MAPE. Identity
modes that ignore arrival timing are diagnostic upper bounds, not alternate
model trajectories.
