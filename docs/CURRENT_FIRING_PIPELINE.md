# Current Firing Pipeline

## Proximal observation

`observe_code()` receives a known SSTD code. For each active column it first
looks for a timing-matched `last_prediction_candidates` entry. A matched entry
selects its stored neuron; an unmatched active column enters the paper burst
branch and activates every neuron in that column. Learning then records one
winner per observed event while `previous_active_cells` retains the burst
population.

## Prediction

1. `_active_sources()` reads the current transient context.
2. `_live_incoming()` enumerates segments reached by active source cells.
3. The continuous solver finds the first dendritic threshold crossing and a
   simplified soma firing time.
4. Valid results are grouped by `(column, round(time, 12))` and replaced by the
   greater returned `score`.
5. With strict `intracolumn_inhibition=True` and policy `existing`, one event
   per column survives, chosen by earliest predicted firing time.
6. `PredictionCandidate` stores the selected segment and diagnostic crossing
   metadata.
7. `prediction_active_cells()` resolves the emitted event against those stored
   candidates and `advance_prediction()` copies only those selected cells into
   the next transient context.

## Parity finding

This pipeline has a real same-column reduction and does not equal “all
predictive segments propagate”. The unresolved parity point is earlier: the
code does not expose the full soma terms required to prove that its event score
and earliest-time decisions are the paper's max-soma-potential firing WTA.
