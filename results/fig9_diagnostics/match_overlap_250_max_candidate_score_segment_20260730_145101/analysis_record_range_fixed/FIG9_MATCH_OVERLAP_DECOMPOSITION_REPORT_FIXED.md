# Fig.9 Match Overlap Decomposition (Record Range Fixed)

This is an offline re-analysis of existing traces. The model was not rerun.
No prediction, matching, learning, RNG, checkpoint, or strict default changed.

## Audit identity

- Fix version: `fig9-match-overlap-record-range-v2`
- Analysis-code commit: `17f308f9bfbe266f4a782409240855af11867160`
- Trace-producing commit: `f159d979e8d1f2eabba07e38853712139465c13b`
- Old analysis directory: `results\fig9_diagnostics\match_overlap_250_max_candidate_score_segment_20260730_145101\analysis`
- Fixed analysis directory: `results\fig9_diagnostics\match_overlap_250_max_candidate_score_segment_20260730_145101\analysis_record_range_fixed`
- Column trace SHA256: `a32fe855ecace487994c8bc0f711390eab8e18bc00e274010f478ab4a53933c0`
- Segment trace SHA256: `e092be8fcbd37a87ca188f387ea219541ec12273fe555c6b30403b49e792d8e1`
- Teacher trace SHA256: `42ffda14abf1336da50431b815032cc141dda207b579ac2e373358fcece7ec91`

## Record-range correction

The old generic grouping path read a nonexistent `record_range` column and
coerced the missing value to zero. The fixed path classifies only
`actual_record_index`: 0-49, 50-99, 100-149, 150-199, and 200-244. Invalid,
missing, negative, and out-of-protocol values remain
`UNKNOWN_RECORD_RANGE`; they are never silently assigned to 0-49.

## Denominators

All-observation metrics use encoded-column observations in the named
field/range/scenario scope. L-match, ambiguity, and context-variant rates are
conditional on observations that had at least one preexisting segment.
Segment composition and retention use individual preexisting segment rows.
Teacher-reference metrics use teacher-forced reference rows. Exact definitions
are machine-readable in `denominator_manifest.json`.

## Late passenger range

{
  "L1_ambiguity_observation_count": 172,
  "L1_ambiguity_rate_existing_segment_observations": 0.5227963525835866,
  "L1_match_observation_count": 267,
  "L1_match_rate_existing_segment_observations": 0.8115501519756839,
  "L1_multiple_neuron_ambiguity_count": 172,
  "L1_multiple_neuron_ambiguity_rate_existing_segment_observations": 0.5227963525835866,
  "L2_ambiguity_observation_count": 16,
  "L2_ambiguity_rate_existing_segment_observations": 0.0486322188449848,
  "L2_match_observation_count": 83,
  "L2_match_rate_existing_segment_observations": 0.25227963525835867,
  "L2_multiple_neuron_ambiguity_count": 16,
  "L2_multiple_neuron_ambiguity_rate_existing_segment_observations": 0.0486322188449848,
  "L3_ambiguity_observation_count": 1,
  "L3_ambiguity_rate_existing_segment_observations": 0.00303951367781155,
  "L3_match_observation_count": 18,
  "L3_match_rate_existing_segment_observations": 0.0547112462006079,
  "L3_multiple_neuron_ambiguity_count": 1,
  "L3_multiple_neuron_ambiguity_rate_existing_segment_observations": 0.00303951367781155,
  "L4_ambiguity_observation_count": 0,
  "L4_ambiguity_rate_existing_segment_observations": 0.0,
  "L4_match_observation_count": 0,
  "L4_match_rate_existing_segment_observations": 0.0,
  "L4_multiple_neuron_ambiguity_count": 0,
  "L4_multiple_neuron_ambiguity_rate_existing_segment_observations": 0.0,
  "all_cell_current_best_overlap_mean_existing_segment_observations": 1.1185410334346504,
  "best_overlap_mean_all_observations": 1.1185410334346504,
  "best_overlap_median_all_observations": 1.0,
  "best_overlap_p10_all_observations": 0.0,
  "best_overlap_p25_all_observations": 1.0,
  "best_overlap_p75_all_observations": 2.0,
  "best_overlap_p90_all_observations": 2.0,
  "burst_only_best_overlap_mean_existing_segment_observations": 1.1063829787234043,
  "creation_current_jaccard_mean_segment_rows": 0.00799348065403461,
  "cross_field_only_best_overlap_mean_existing_segment_observations": 0.9148936170212766,
  "existing_segment_denominator_scope": "observations_in_scope_with_at_least_one_preexisting_segment",
  "existing_segment_observation_count": 329,
  "existing_segment_rate_all_observations": 1.0,
  "field": "passenger",
  "gap_to_L_match_mean_all_observations": 2.8814589665653494,
  "observation_count": 329,
  "observation_denominator_scope": "field_x_record_range_x_scenario_all_observed_columns",
  "observe_scenario": "scenario3",
  "predicted_only_best_overlap_mean_existing_segment_observations": 0.04559270516717325,
  "predicted_plus_winner_best_overlap_mean_existing_segment_observations": 0.6048632218844985,
  "record_range": "200-244",
  "same_field_only_best_overlap_mean_existing_segment_observations": 0.40425531914893614,
  "segment_row_count": 2437,
  "segment_row_denominator_scope": "all_preexisting_segment_rows_for_observations_in_scope",
  "source_retention_mean_segment_rows": 0.2255368622623444,
  "winner_only_best_overlap_mean_existing_segment_observations": 0.6048632218844985,
  "without_burst_only_best_overlap_mean_existing_segment_observations": 0.6048632218844985
}

## Consistency

- All checks passed: `True`
- Unknown range counts: `{"column_rows": 0, "segment_rows": 0, "teacher_rows": 0}`
- Existing global totals preserved: `True`

These outputs correct grouping and denominator labels only. They are not a new
trajectory and do not establish a new MAPE or a completed reproduction.
