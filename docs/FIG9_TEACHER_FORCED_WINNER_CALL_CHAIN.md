# Fig.9 Teacher-Forced Winner Call Chain

This audit describes the implemented winner semantics. The future-observed
winner is an operational reference under the normal real-data stream. It is
not a biological ground-truth neuron and never participates in prediction.

## Prediction Before Observation

1. `run_strict_stream` calls `rollout_raw_autonomous` before learning the
   current record in `experiments/fig9_strict_reproduction.py`.
2. The rollout snapshots transient state, calls `predict_code` once per step,
   advances only emitted predictive cells, and restores the snapshot.
3. Only after restoration does `learn_actual_code` call `predict_code`, then
   `observe_code(..., learn=True)` in `experiments/fig9/learning.py`.
4. Observation rows are built after that real observation and after new
   segments have been registered.

No future record is observed early. Target timestamps and columns are used only
after prediction for trace labelling and offline joins.

## Winner Semantics

`SequentialMemory.observe_code` is the authoritative selection point in
`src/seqmem/model.py`.

- Before observation, `previous_winners` represents the previous cycle's one
  learning winner per encoded event/column. `_active_sources()` may also
  include all burst-active cells.
- Scenario 1 uses the matching predicted neuron and predicted segment.
- Scenario 2 uses the best matching existing neuron/segment when timed overlap
  reaches `L_match`.
- Scenario 3 chooses the least-used neuron and may create a segment from the
  previous winners.
- `winner_id` is fixed before it is written to `learning_winners`.
- Predicted events put only that winner in `active_cells`. Unpredicted events
  burst every neuron in the column, while `learning_winners` still receives
  only the selected winner.
- At exit, `previous_winners = learning_winners`; it is distinct from the
  potentially all-cell `previous_active_cells`.

The optional `ObservationTrace` copies these existing local decisions. It does
not match, predict, learn, sort, or call RNG.

## Segment Semantics

- Scenario 1: selected and reinforced segment are the same predicted segment.
- Scenario 2: selected and reinforced segment are the same matched segment.
- Scenario 3: when context exists, selected and created segment are the newly
  created segment.
- A segment created only after seeing the real target is marked as such and is
  excluded from exact autonomous segment-match applicability.

Stable provenance comes from `BranchProvenanceRegistry`. Its external sidecar
rebinds IDs after checkpoint load without changing strict checkpoint v1.

## Audited Source Locations

- Trace records: `src/seqmem/model.py:344` and `:359`.
- Real observation entry: `src/seqmem/model.py:1599`.
- Winner identity fixation: `src/seqmem/model.py:1726`.
- Persist current winners: `src/seqmem/model.py:1760`.
- Scenario 3 segment creation: `src/seqmem/model.py:1897`.
- Strict predict-then-observe wrapper: `experiments/fig9/learning.py:9`.
- Experiment-side trace allocation:
  `experiments/fig9_strict_reproduction.py:1895`.
- New-segment provenance registration:
  `experiments/fig9_strict_reproduction.py:1912`.
- Reference row serialization:
  `experiments/fig9_strict_reproduction.py:1923` and
  `experiments/diagnostics/fig9_teacher_forced_identity.py:159`.
- Source-label update for the next transition:
  `experiments/fig9_strict_reproduction.py:1939`.

## Capture Order

```text
autonomous rollout and transient restore
  -> snapshot pre-observe segments and context
  -> learn_actual_code
  -> observe_code populates ObservationTrace
  -> register newly created segments
  -> serialize teacher-forced rows
  -> update source labels for the next transition
```

Use these searches for current line numbers:

```powershell
rg -n "def observe_code|learning_winners|winner_id" src/seqmem/model.py
rg -n "def learn_actual_code" experiments/fig9/learning.py
rg -n "teacher_forced_trace|learn_actual_code" experiments/fig9_strict_reproduction.py
```
