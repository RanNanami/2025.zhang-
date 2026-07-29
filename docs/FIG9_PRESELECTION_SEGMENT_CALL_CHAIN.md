# Fig.9 Preselection Segment Call Chain

This document audits the implemented call chain. It does not infer stages from
names and does not propose a new selector.

## Actual Funnel

```text
previous active source cells
  -> live incoming segments touched by at least one active source
  -> active-weight upper-bound eligibility
  -> continuous dendritic response integration
  -> dendritic threshold crossing and valid soma firing window
  -> highest score for (column, round(soma_time, 12))
  -> earliest soma firing time in each column
  -> one saved PredictionCandidate and raw event per selected column
  -> raw-event-to-candidate mapping
  -> sequential or batched intercolumn competition
  -> emitted SymbolCode
```

There is no independent per-neuron winner group in the current code.

## Implemented Stages

| Stage | Input | Existing selection key | Tie behavior | Output / loss |
|---|---|---|---|---|
| `INSPECTED` | `_active_sources()` and `_live_incoming(source)` | Segment object reached by any active source | Existing dict insertion order | One touched segment entry |
| `UPPER_BOUND_ELIGIBILITY` | Sum of active matched synapse weights | `v_rest + upper_bound >= dendrite_threshold` | None | Ineligible segments skip numerical response |
| `RESPONSE_COMPUTED` | Segment and active source times | Existing continuous PSP integration | Existing floating-point order | Peak/crossing/soma metadata |
| `THRESHOLD_CROSSED` | Timed score | `score >= dendrite_threshold` | None | Crossing segment |
| `VALID_FIRING_WINDOW` | Predicted soma time | Existing cycle window bounds | None | Event-eligible segment |
| `EVENT_SCORE_WINNER` | Same column and rounded soma time | Replace only when `score > previous_score` | Equal score keeps first traversal entry | `best_by_event` |
| `COLUMN_EARLIEST_SELECTION` | Event winners in same column | Replace only when `time < previous_time` | Equal time keeps first traversal entry | `winner_by_column` |
| `PREDICTION_CANDIDATE` | Column winner | No additional numerical selection | Existing iteration order | `PredictionCandidate` and raw event |
| `COMPETITION_EMITTED` | Candidate bound to raw event | Existing sequential/batched effective score | Policy-specific stable ordering | Emitted event |

## Source Locations

- Segment discovery and active-weight accumulation:
  `SequentialMemory.predict_code`, `src/seqmem/model.py:517`, especially
  lines 554 onward.
- Upper-bound pruning: `src/seqmem/model.py:675`.
- Continuous segment response: `src/seqmem/model.py:701` and
  `_continuous_segment_prediction` at line 1948.
- Threshold and firing-window checks: `src/seqmem/model.py:829`.
- Same-column/same-time score selection: `src/seqmem/model.py:875-883`.
- Earliest event per column: `src/seqmem/model.py:999-1002`.
- `PredictionCandidate` construction: `src/seqmem/model.py:1121`.
- Raw `SymbolCode`: `src/seqmem/model.py:1176`.
- Raw-event candidate binding:
  `experiments/diagnostics/fig9_competitive_inhibition.py:119-149`.
- Sequential/batched competition:
  `experiments/diagnostics/fig9_competitive_inhibition.py:152`.

Line numbers describe the implementation at the diagnostic commit and may move
with later documentation or test changes.

## Auditable Elimination Reasons

The hook records only reasons corresponding to real branches:

- `active_weight_upper_bound_below_threshold`
- `below_dendritic_threshold`
- `no_valid_soma_firing_time`
- `predictive_soma_cannot_fire`
- `predicted_time_outside_valid_window`
- `lower_score_same_column_time`
- `lower_or_equal_score_same_column_time`
- `later_firing_time_same_column`
- `later_or_equal_firing_time_same_column`

An unclassified discarded segment is counted as unknown. The diagnostic does
not silently invent a reason.
