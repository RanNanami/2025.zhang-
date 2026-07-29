# Fig.9 Intracolumn Selector Call Chain

This document describes the code path at commit `745e6d6` plus the gated
diagnostic selector implementation. The ablation is not a strict reproduction
claim.

## Real prediction path

1. `SequentialMemory.predict_code()` starts at
   `src/seqmem/model.py:879`.
2. Candidate segments are reached from the current lateral active sources.
   `_continuous_segment_prediction()` (`model.py:2576`) integrates each real
   segment and returns its first threshold crossing, selected potential and
   soma firing time.
3. A firing time outside the existing valid window is discarded before any
   selector policy runs.
4. Valid segments are grouped in `best_by_event` (`model.py:1023`) by:

   ```text
   (column, round(predicted_soma_firing_time, 12))
   ```

5. Within one column-time event, the existing code replaces the provisional
   winner only when the new `score` is strictly greater. Equal scores keep the
   first segment in candidate traversal order. No RNG is called.
6. The event winners form the same-column candidate pool. With the strict
   `existing` policy, `winner_by_column` (`model.py:1375`) replaces a
   provisional representative only when the new predicted soma time is
   strictly earlier. Equal times keep the first event winner.
7. The selected representative becomes a `PredictionCandidate` at
   `model.py:1520` and is stored in `last_prediction_candidates`.
8. `candidates_for_prediction()` in
   `experiments/diagnostics/fig9_competitive_inhibition.py:119` converts those
   representatives to competition inputs.
9. `compete_prediction_candidates()` at the same file's line 152 performs
   sequential or batched intercolumn inhibition.
10. `emitted_prediction_code()` at line 483 constructs the code propagated to
    the next autonomous horizon step.

The selector therefore runs after threshold crossing, valid-time filtering and
event-level selection, but before intercolumn competition and recurrent state
propagation.

## Existing selector semantics

- Primary key: predicted soma firing time ascending.
- Secondary key: none.
- Tie-break: stable event-winner insertion order; the first equal-time member
  remains selected.
- RNG: none.
- Number of members: one real event winner for every distinct valid rounded
  soma time in that column. A one-member group is unchanged by every policy.
- Why one representative: `intracolumn_inhibition=True` enforces one predicted
  neuron identity per mini-column before intercolumn competition.

The `existing` path still executes the original `winner_by_column` loop. The
new helper is used only to audit that result and assert equivalence when trace
capture is enabled.

## Candidate metrics

- `candidate_score`: the existing `PredictionCandidate.score`. In continuous
  mode this is the potential returned at the selected crossing/firing
  calculation. It is not recomputed by the selector.
- `response_peak`: the maximum dendritic potential recorded by the same
  continuous calculation. It can differ from `candidate_score`.
- `contributor_count`: number of synapses represented in the crossing
  contribution tuple. This is the same definition used by the existing
  offline report. `positive_contributor_count` is traced separately but is not
  the max-contributor policy key.
- `current_context_jaccard`: Jaccard similarity between the segment's current
  synapse source IDs and the current prediction call's active source IDs. It
  uses no target column, future observation or teacher reference.
- `candidate_original_index`: insertion index in the real `best_by_event`
  event-winner stream. It is the final deterministic tie-break.

Nonfinite or missing numeric metrics sort after finite values. Candidates are
never removed because a diagnostic metric is missing.

## Diagnostic policies

| Policy | Ordered keys |
|---|---|
| `existing` | predicted time ascending, stable original order |
| `max_candidate_score` | score descending, time ascending, stable order |
| `max_response_peak` | peak descending, score descending, time ascending, stable order |
| `max_contributor_count` | contributor count descending, peak descending, score descending, time ascending, stable order |
| `context_then_score` | current-context Jaccard descending, score descending, time ascending, stable order |

All policies receive the same event-winner pool. Runtime assertions verify that
the selected column set equals the event-winner column set. Step 1 can therefore
be compared as a fixed-pool local choice. From Step 2 onward, a changed neuron
identity changes lateral context, so later pools represent recurrent trajectory
divergence rather than simple reranking.

## Trace and state safety

`IntracolumnSelectionTrace` is optional and cleared per prediction call. It
copies already computed metadata and does not:

- call `predict_code()` again;
- integrate a segment again;
- read ground truth or a teacher reference;
- call an RNG;
- observe or learn;
- mutate a segment, synapse, checkpoint, or strict protocol fingerprint.

The strict default remains `existing`, and both the policy and trace CLI
features are explicitly marked diagnostic-only in their separate protocol.
