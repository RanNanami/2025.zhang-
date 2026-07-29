# Fig.9 Reference Neuron Selection Call Chain

This diagnostic explains how a future-observed, pre-existing reference neuron
progressed through an earlier autonomous prediction. The reference is joined
offline and never participates in prediction or selection.

## Prediction Stages

`SequentialMemory.predict_code()` begins at `src/seqmem/model.py:574`.
Continuous segment responses are evaluated and recorded by the existing
preselection trace. The real selection stages are:

1. segment inspected and response computed;
2. dendritic threshold crossed;
3. firing time accepted;
4. event-level winner selected;
5. earliest event retained for the column;
6. `PredictionCandidate` saved;
7. candidate emitted or suppressed by later competition.

The model marks event winners near `src/seqmem/model.py:1020`, column winners
and saved candidates near `src/seqmem/model.py:1130`, and records the actual
replacement comparisons near `src/seqmem/model.py:956` and
`src/seqmem/model.py:1077`.

## Actual Ordering And Tie-Breaks

Event selection uses the existing score comparison and its existing stable
tie fields. Column compression retains the earliest predicted time. The
preselection trace records:

- comparison field and values;
- previous and replacement segment identities;
- tie-break values;
- replacement stage;
- inspection order and group identifiers.

`experiments/diagnostics/fig9_preselection_segments.py:584` serializes these
real replacement events. The reference analyzer does not reconstruct a
replacement from the final winner.

## Future Reference Join

The future observation reference is captured by
`build_teacher_forced_observation_rows()` at
`experiments/diagnostics/fig9_teacher_forced_identity.py:348`.
Only `PREEXISTING_PREDICTED_REFERENCE` and
`PREEXISTING_MATCHING_SEGMENT_REFERENCE` rows are eligible.

`build_reference_selection_trace()` at
`experiments/diagnostics/analyze_fig9_reference_neuron_selection.py:96` joins
that identity to the already-saved segment trace by input index, target
timestamp, horizon step, and target column. It records every participating
segment and the stages it actually reached.

`build_reference_replacement_trace()` at
`experiments/diagnostics/analyze_fig9_reference_neuron_selection.py:676`
retains only captured comparisons in which the reference neuron was the
previous or replacement winner.

## Loss Classification

The funnel distinguishes:

- no inspected reference segment;
- nonpositive response;
- below threshold;
- crossing with no valid firing time;
- valid event lost during event selection;
- event winner lost during column selection;
- saved candidate suppressed by intercolumn competition;
- emitted reference.

If the source trace was captured at `crossing` level, pre-crossing losses
cannot be subdivided. They remain explicitly unavailable instead of being
inferred.

## Offline Ranking Boundary

Hit@K, MRR, pairwise win rate, and metric comparisons are conditioned on the
already-known target column. They measure separability inside a real saved
competition set, but they do not generate a new `SymbolCode`, MAPE result, or
selector. No ranking rule is installed into the model.

