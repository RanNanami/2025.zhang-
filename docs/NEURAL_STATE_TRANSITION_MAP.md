# Neural State Transition Map

| State | Produced by | Leaves by | Fires? | Suppresses peers? | Propagates? |
|---|---|---|---|---|---|
| `PREDICTIVE_NEURON` | distal PSP crosses dendritic threshold in `reference_continuous_prediction()` | depolarization window expires or soma test fails | not by definition | not by itself | not by definition |
| `CANDIDATE_NEURON` | valid crossing and valid predicted soma time enter `best_by_event` | event/column selection | not by definition | not by itself | not by definition |
| `THRESHOLD_CROSSING_NEURON` | first dendritic crossing saved in prediction metadata | candidate filtering | not by definition | no | no |
| `WINNER_NEURON` | strict current path: one event/column representative survives selection | next prediction/observation state | assumed by current API | represented implicitly by one selected event | yes, if `advance_prediction()` is called |
| `FIRED_NEURON` | paper: soma potential is maximal in its column and above theta; code: not separately materialized | refractory/cycle reset | yes | yes, paper semantics | yes |
| `BURST_NEURON` | proximal event has no matching prediction | end of observation cycle | yes, all neurons in the active column | no single-neuron WTA | yes through `previous_active_cells`; learning uses `previous_winners` |
| `EMITTED_NEURON` | current returned `SymbolCode` event and selected candidate | `prediction_active_cells()` | assumed for raw propagation | no separate membrane inhibition | yes |
| `PROPAGATED_NEURON` | `advance_prediction()` copies resolved selected cells into `previous_active_cells` and `previous_winners` | next `predict_code()` | assumed | no additional firing step | yes |

## Key distinction

The current implementation does not propagate every valid segment candidate:
`best_by_event` first keeps one segment for a rounded column/time event and the
strict `existing` branch keeps the earliest event per column. However, the
selected object is still a prediction candidate resolved through the candidate
score, not a separately recorded winner produced by a full soma membrane WTA.
That is why the audit records `PAPER_SOMA_WINNER_NOT_RECONSTRUCTABLE` rather
than asserting that all predictive cells are propagated.
