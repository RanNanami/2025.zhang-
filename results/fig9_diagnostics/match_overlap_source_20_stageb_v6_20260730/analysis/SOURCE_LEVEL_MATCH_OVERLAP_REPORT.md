# Source-Level Match-Overlap Report

This is a small-scale, read-only diagnostic. It is not a formal 250-record
result and does not change strict defaults, matching, observation, or learning.

## Coverage

- Source-synapse rows: 30542
- Unknown rows: 0
- Unknown rate: 0.000000
- Passenger Scenario-3 source rows: 2398
- Passenger records 200-244 Scenario-3 rows: 0 (unavailable in runs shorter than 245 records)

## Passenger Scenario 3

The denominator below is all pre-observation source-synapse rows belonging to
existing passenger segments assigned to Scenario 3.

- Exact-cell match: 22/2398
  (0.009174)
- Active column, wrong neuron: 31/2398
  (0.012927)
- Source column inactive: 844/2398
  (0.351960)
- Timing/eligibility excluded: 1501/2398
  (0.625938)

## Diagnostic judgment

- Dominant passenger Scenario-3 reason: `SOURCE_TIMING_OR_ELIGIBILITY_EXCLUDED`
- Neuron identity drift supported by this small run: `False`

Neuron identity drift requires active-column/wrong-neuron loss to dominate,
passenger to exceed weekday/time with its bootstrap interval, and a low unknown
rate. When timing exclusion or inactive columns dominate, the evidence instead
points toward matching-timing semantics or context-trajectory drift. This
50-record result remains a small diagnostic and cannot replace a formal
250-record source trace.
