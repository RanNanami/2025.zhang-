# Fig.9 Match-Overlap Metric Denominators

This document applies to the record-range-fixed, offline match-overlap
analysis. It does not change model execution or the strict protocol.

## Record ranges

Ranges are derived only from `actual_record_index`:

- `0-49`
- `50-99`
- `100-149`
- `150-199`
- `200-244`

Missing, non-integral, negative, and values at or above 245 are classified as
`UNKNOWN_RECORD_RANGE`. A formal 250-record analysis fails its consistency
check if any input row has that label.

## Observation metrics

An observation is one encoded field-column event identified by record,
timestamp, field, and encoded column. Counts and unconditional overlap
statistics use all observations in the named field, range, and Scenario scope.

`existing_segment_observation_count` is the subset of observations that had at
least one preexisting segment before the real observation.

## Conditional overlap metrics

L1-L4 match rates, ambiguity rates, multiple-neuron ambiguity rates, and
context-variant best overlaps use existing-segment observations as their
denominator. They are fixed-state counterfactual measurements, not rerun
training trajectories and not new MAPE values.

## Segment metrics

Segment source composition, actual overlap, source retention, and
creation/current Jaccard metrics use individual preexisting segment rows.
Observations with more segments therefore contribute more rows to these
segment-level summaries.

## Teacher-reference metrics

Teacher-reference summaries use rows from
`teacher_forced_observation_trace.csv`. They are grouped independently by
field, actual record range, and observed Scenario.

## Audit files

The exact row units and denominators are also recorded in
`denominator_manifest.json`. `analysis_consistency_checks.json` verifies range
coverage, count conservation, L-threshold monotonicity, ambiguity bounds, zero
unknown ranges, and preservation of the old global totals.
