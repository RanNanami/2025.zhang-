# Fig.9 Match-Overlap Call Chain

This document describes the implementation on
`experiment/fig9-competitive-inhibition`. It is an audit of executable code,
not an inference from variable names.

## Strict learning entry

`experiments/fig9/learning.py:9` defines `learn_actual_code`.

1. `model.predict_code()` runs at line 26.
2. `model.observe_code(code, learn=True)` runs at lines 28-34.

The match-overlap diagnostic preserves this order. It copies read-only state
between these calls and does not invoke either operation a second time.

## Context definitions

`src/seqmem/model.py:604` defines `_active_sources`.

- With `burst_context=True`, matching receives
  `previous_active_cells`, falling back to `previous_winners` only when the
  active-cell dictionary is empty.
- With `burst_context=False`, matching receives `previous_winners`.
- Fig.9 strict uses `burst_context=True`.

`observe_code` stores this result in `previous_active` at line 1991 and passes
it to `_best_matching_neuron`.

After observation:

- predicted winner cells are added at line 2286;
- every neuron in an unpredicted mini-column is added as a burst cell at line
  2294;
- this all-cell set becomes `previous_active_cells` at line 2301;
- exactly one learning winner per observed event becomes `previous_winners`
  at line 2302.

Therefore `ALL_CELL_CURRENT` is not equal to `WINNER_ONLY` under normal
bursting. The former is the real matching context.

## Exact overlap

`Segment.timed_overlap` is defined at `src/seqmem/model.py:71`.

For segment \(S\), active source-time mapping \(A\), dendritic target time
\(t_d\), and tolerance \(\epsilon\):

```
overlap(S, A) =
  count(source in A
        where source is a key in S.synapses
        and abs(A[source] + S.synapses[source].delay - t_d) <= epsilon)
```

Consequences:

- it counts unique stable integer cell IDs;
- a cell ID encodes both mini-column and neuron identity;
- it counts matching synapses/cells, not weight sum;
- duplicate source IDs cannot count twice because both structures are maps;
- synaptic delay and source firing time are part of eligibility;
- only currently active source cells are considered.

## Best-match search

`_best_matching_neuron` starts at `src/seqmem/model.py:2394`.

1. The incoming index is queried only for current active source IDs.
2. Candidate segments must target the observed column and remain active.
3. Each candidate receives `timed_overlap` at line 2426 and a continuous
   response score.
4. Candidates are ordered lexicographically by `(timed_overlap, score)`.
5. Exact ties use `learning_rng.random() < 0.5`.

The returned value is `(neuron_index, segment)` or `None`. The incoming index
only exposes segments sharing at least one exact active source ID; the
diagnostic scans every existing segment in the observed column with read-only
membership checks so it can explain both indexed candidates and zero-source
overlap segments without rerunning the selector or RNG.

## Scenario assignment

`observe_code` starts at `src/seqmem/model.py:1977`.

- Scenario 1: a timing-valid `PredictionCandidate` exists. Its same segment is
  directly reinforced.
- Scenario 2: no valid prediction exists, but the selected existing segment
  passes `segment.timed_overlap(...) >= self.params.l_match` at lines
  2132-2136.
- Scenario 3: no reusable segment passes the threshold, so a new segment is
  created on a least-used neuron.

The Scenario-3 threshold in strict Fig.9 is `L_match=4`.

## Creation and reinforcement sources

`_grow_segment` starts at `src/seqmem/model.py:2460`.

- Scenario 3 calls it with `self.previous_winners` at lines 2101 and 2164.
- Every winner source becomes one synapse.
- Delay is derived from the source time and target event time.

`_reinforce_segment` starts at line 3170.

- Scenario 1 evaluates contribution against real `previous_active`, but does
  not grow missing sources.
- Scenario 2 evaluates the selected segment against real `previous_active`
  and passes `previous_winners` as `growth_sources` at line 2149.

Thus creation/growth uses winner-only identity while later best matching uses
all active cells. The diagnostic measures both sets independently.

## Deletion

`_prune_neuron` starts at `src/seqmem/model.py:3363`.

Synapses whose weight/age forgetting score reaches the configured threshold
are removed. A segment with no remaining synapses is marked inactive and
removed from the neuron's segment list. A deleted historical segment cannot be
reconstructed from current model state; the trace reports unavailable
provenance instead of inventing it.

## Diagnostic boundary

`experiments/diagnostics/fig9_match_overlap.py`:

- copies pre-observation contexts and current segment/synapse values;
- mirrors the exact timed-membership formula;
- never calls `_best_matching_neuron`, `predict_code`, `observe_code`, or RNG;
- attaches Scenario/winner/segment identities from the one real
  `ObservationTrace` after observation;
- computes L1-L4 and context variants offline only.

Ground truth supplies the observed field/value labels and does not select a
segment, neuron, prediction candidate, or matching context.
