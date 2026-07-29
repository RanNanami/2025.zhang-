# Fig.9 Provenance Call Chain

## Actual Prediction Chain

```text
previous active source cells and spike times
    -> incoming active dendritic segments
    -> one continuous PSP integration per segment
    -> threshold-crossing segment candidates
    -> best segment per (column, rounded soma time)
    -> earliest event per column
    -> one PredictionCandidate with one Segment
    -> CompetitionCandidate column representative
    -> sequential or batched inhibition
    -> emitted SymbolCode event
    -> prediction_active_cells()
    -> autonomous next-step context
```

`SequentialMemory.predict_code()` first gathers every live segment touched by
the current active sources. `_continuous_segment_prediction()` integrates each
segment independently. Its return value is `(soma_firing_time, score)`, where
the reference implementation returns the maximum potential encountered before
the first crossing, clamped to at least the dendritic threshold.

The model then keeps the highest-scoring segment for each
`(column, rounded firing time)` and, with strict intracolumn inhibition, the
earliest event per column. Only at that point is a `PredictionCandidate`
created. It stores exactly one `Segment` plus crossing time, peak potential,
and per-synapse PSP contributions from that same prediction call.

Therefore:

- `PredictionCandidate.score` is not a multi-segment sum.
- `last_prediction_candidates` is already post segment/event/column selection.
- `candidates_for_prediction()` normally maps each raw event to that same
  column representative.
- Candidate rows equalled column rows in the earlier separability trace because
  the model had already reduced the raw neural segment pool before creating
  `PredictionCandidate` objects.

## Reused Fig.8 Semantics

The Fig.9 diagnostic reuses these Fig.8 concepts:

- creation source-cell fingerprint;
- predicted-supported, burst-assisted, burst-only, and unavailable source
  support classes;
- PSP contribution copied from the same `PredictionCandidate`;
- current source-context overlap and Jaccard;
- diagnostic-only provenance that never participates in selection.

Fig.8 stores optional provenance directly on `Segment`. Fig.9 instead uses an
experiment-side registry so on/off runs retain identical model and RNG
fingerprints. The registry observes newly created segments after learning and
records their exact pre-learning source context.

## Stable IDs

Persistent IDs are:

```text
SHA256(
    creation_transition_index,
    target_column,
    target_neuron,
    creation_source_fingerprint,
    stable_creation_ordinal
)
```

Python `id()`, `repr(object)`, and `hash()` never enter persisted output.
Object identity is used only as an in-process lookup key. A checkpoint sidecar
stores stable IDs and a separate structural rebinding signature; the latter is
not the public branch ID.

## Limits

The current model cannot test a candidate assembled from several simultaneous
segments because such a candidate does not exist. Exact creation-branch
entropy for one candidate consequently degenerates to zero when provenance is
available. The useful remaining question is whether a single segment's source
set has become mixed relative to its creation context and the current
autonomous context.
