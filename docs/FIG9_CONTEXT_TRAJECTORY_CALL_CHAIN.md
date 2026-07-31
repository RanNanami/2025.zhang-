# Fig.9 Context Trajectory Call Chain

This document audits the repository implementation. It does not claim that
every implementation choice is required by the paper.

## Prediction candidate generation

1. `SequentialMemory.predict_code`
   (`src/seqmem/model.py:879`) reads `_active_sources()` and enumerates live
   incoming segments for every active source (`:925`).
2. Each eligible segment is evaluated by
   `_continuous_segment_prediction` (`:1074`, implementation at `:2576`).
   The optional `PreselectionTrace` is populated by the same call; it does not
   invoke prediction again.
3. The continuous response records dendritic crossing and soma firing time.
   A segment below threshold stops at `RESPONSE_COMPUTED`. A crossing without
   a valid soma time stops at `SOMA_FIRING`.
4. The valid firing window is checked before the segment enters
   `raw_eligible_segments` (`:1244-1247`).
5. `best_by_event` keeps the highest score for one rounded
   `(column, firing_time)` event (`:1255-1306`).
6. Intracolumn selection keeps one event for each column. The selected
   preselection row is marked `became_prediction_candidate` (`:1476`).
7. The same selected segment is stored in `last_prediction_candidates`
   (`:1520`) and returned as a raw `SymbolCode`.

The diagnostic stage meanings are therefore:

| Stage | Repository evidence |
|---|---|
| segment inspected | segment appears in the same-call `PreselectionTrace` |
| overlap positive | at least one active source is a segment synapse |
| response computed | `PreselectionSegmentTrace.response_computed` |
| threshold crossed | `crossed_threshold` |
| firing time valid | `valid_firing_time` |
| saved candidate | `became_prediction_candidate` |
| intracolumn selected | saved candidate appears in the raw code |

## Diagnostic intercolumn competition

`rollout_raw_autonomous` starts at
`experiments/fig9_strict_reproduction.py:801`.

1. It executes one `predict_code()` call at `:957`.
2. `candidates_for_prediction` binds raw events to the same
   `last_prediction_candidates`.
3. `compete_prediction_candidates` runs at `:1197`.
4. `emitted_prediction_code` creates the propagated raw-neural code at
   `:1206`.
5. `prediction_active_cells` resolves only emitted events to neuron
   identities at `:1233`.

No decoded passenger value is re-encoded or used to choose a candidate.

## Autonomous state propagation

After competition, emitted prediction cells are copied into both transient
contexts:

```text
previous_active_cells = emitted prediction cells
previous_winners      = emitted prediction cells
```

This occurs at
`experiments/fig9_strict_reproduction.py:1602-1603`.
The next autonomous step therefore reads exactly those emitted cells.

The entire rollout starts from `snapshot_transient_state()` and restores it
in `finally` at `:1629`. Autonomous rollout changes no long-term segment,
synapse, weight, delay, or age state.

## Actual observation state propagation

Actual online learning uses a separate path:

1. The strict runner calls `predict_code()` before observation
   (`experiments/fig9_strict_reproduction.py:2154`).
2. It then calls `observe_code(..., learn=True)` at `:2174`.
3. `SequentialMemory.observe_code` begins at
   `src/seqmem/model.py:1977`.
4. Proximal events that were predicted activate the predicted neuron.
   Unpredicted proximal events burst the complete mini-column.
5. At the end, actual observation replaces the transient contexts:
   `previous_active_cells = active_cells` and
   `previous_winners = learning_winners`
   (`src/seqmem/model.py:2301-2302`).

Consequently, a raw prediction for an unobserved column is not propagated
through the actual-observation path. This is not the same event as autonomous
intercolumn suppression and is recorded separately as
`EMITTED_NOT_PROPAGATED`.

## Read-only trajectory hook

`ContextTrajectoryTracker.record_transition` consumes:

- the same-call `PreselectionTrace`;
- the raw returned code;
- the already-computed `CompetitionResult`;
- the resulting active cells and winners.

It performs no predict, observe, matching, response, RNG, or model-state
operation. For each historical segment source classified as
`SOURCE_COLUMN_NOT_ACTIVE`, `dependency_rows` joins prior column histories
without feeding the observed target back into the model.

The two values of `trajectory_kind` are:

- `actual_observation`: proximal observation determines the next context;
- `autonomous_rollout`: emitted predictive neurons determine the next context.

Their histories and loss distributions must never be pooled without retaining
this label.
