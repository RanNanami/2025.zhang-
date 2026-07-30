# Source-Level Match-Overlap Report

This is a small-scale, read-only diagnostic. It is not a formal 250-record
result and does not change strict defaults, matching, observation, or learning.

## Coverage

- Source-synapse rows: 210225
- Unknown rows: 0
- Unknown rate: 0.000000
- Passenger Scenario-3 source rows: 8926
- Passenger records 200-244 Scenario-3 rows: 0 (unavailable in runs shorter than 245 records)

## Passenger Scenario 3

The denominator below is all pre-observation source-synapse rows belonging to
existing passenger segments assigned to Scenario 3.

- Exact-cell match: 63/8926
  (0.007058)
- Active column, wrong neuron: 127/8926
  (0.014228)
- Source column inactive: 4312/8926
  (0.483083)
- Timing/eligibility excluded: 4424/8926
  (0.495631)

## Diagnostic judgment

- Dominant passenger Scenario-3 reason: `SOURCE_TIMING_OR_ELIGIBILITY_EXCLUDED`
- Neuron identity drift supported by this small run: `False`

Neuron identity drift requires active-column/wrong-neuron loss to dominate,
passenger to exceed weekday/time with its bootstrap interval, and a low unknown
rate. When timing exclusion or inactive columns dominate, the evidence instead
points toward matching-timing semantics or context-trajectory drift. This
50-record result remains a small diagnostic and cannot replace a formal
250-record source trace.
