# Fig.8 Scenario and Temporal Credit Call Chain

The audit path is intentionally the existing training path:

```text
predict_code()
  -> last_prediction_candidates[column]
  -> observe_code(encoded proximal SymbolCode, learn=True)
  -> observe_code timing gate
       -> legacy scenario1 -> _reinforce_segment(saved PredictionCandidate.segment)
       -> no timing match -> _best_matching_neuron() -> legacy scenario2 or legacy scenario3/new segment
  -> _punish_wrong_predictions(active events)
       -> failed saved candidates are depressed/aged
  -> previous_active_cells / previous_winners updated
```

The audit callback is attached only when `MemoryParams.capture_intralayer_parity_diagnostics`
is enabled. It runs after the existing decision and catches callback errors, so it
does not affect the model's mathematical path or RNG path.

## Functions and locations

Line numbers are generated from the current checkout by
`experiments/diagnostics/fig8_scenario_temporal_credit_parity.py` and repeated in
`CODE_SCENARIO_TAXONOMY_MAP.csv`.

- `SequentialMemory.predict_code`: creates the current `PredictionCandidate` objects and stores them in `last_prediction_candidates`.
- `SequentialMemory.observe_code`: compares each proximal event to candidates in the same column using `abs(candidate.time - event.time) <= timing_tolerance`.
- `SequentialMemory._best_matching_neuron`: searches active incoming segments after no timing-matched prediction was found.
- `SequentialMemory._reinforce_segment`: receives the already selected segment and, for Scenario 1, the same `PredictionCandidate` object.
- `SequentialMemory._grow_segment`: creates the no-match segment from `previous_winners`.
- `SequentialMemory._punish_wrong_predictions`: checks every saved candidate against the actual column event and depresses/ages failed candidates.
- `SequentialMemory.advance_prediction`: in Fig.8 neural retrieval, sets the next context directly from prediction-active cells; it does not re-encode the decoded word.

## State separation

The audit's 20-sentence run uses actual observations for training. It does not mix
autonomous rollout states with the observation trace. Fig.8's neural retrieval
caller uses `advance_prediction(raw_code)` and reserves decoded symbols for output
and scoring. `previous_winners` is updated after `observe_code`; autonomous
prediction uses the raw predicted cells as the next context.

## Label warning

The historical `scenario3` label in the observation branch is overloaded: it is
used for the no-match/new-segment fallback, which maps to paper S2B. The distinct
`wrong_prediction_punishment` phase is the observable failed-prediction path that
maps to paper S3. Raw labels are preserved; the audit adds a post-hoc paper class.
