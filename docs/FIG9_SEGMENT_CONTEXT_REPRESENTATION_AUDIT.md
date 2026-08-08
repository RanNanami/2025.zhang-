# Fig.9 Segment Context Representation Audit

This document records the strict code path inspected by the composition
diagnostic.  The diagnostic is read-only and is disabled unless
`--segment-context-composition-diagnostic` is supplied.

## Real Source Representation

The stored source is an exact presynaptic **cell ID**, not a column ID, a
neuron class, or a symbolic spike event.  `Synapse.source` is an `int` in
[`src/seqmem/model.py`](../src/seqmem/model.py:22-32).  A `Segment` stores a
dictionary keyed by those IDs (`Segment.synapses`) and the long-term fields
`weight`, `delay`, and `age` live on each `Synapse`
([model.py](../src/seqmem/model.py:37-53)).  A cell ID is decoded for analysis
as `column = cell_id // neurons_per_column` and
`neuron = cell_id % neurons_per_column`; this is an audit projection, not a
matching key replacement.

Segment creation occurs in `SequentialMemory._grow_segment()`
([model.py](../src/seqmem/model.py:2455-2511)).  The source set is copied from
the `active_sources` mapping into one synapse per source.  The strict Fig.9
caller passes `model.previous_winners` as the creation context.  The same
source set is also recorded by the experiment-side
`BranchProvenanceRegistry` after the real observation.  No source is added by
the composition diagnostic.

## Matching Expression

The basic overlap is a binary unique-cell intersection:

```text
overlap(segment, active_sources)
    = sum(1 for source in active_sources if source in segment.synapses)
```

This is `Segment.overlap()` in
[model.py](../src/seqmem/model.py:57-61).  The actual timed matching path is
`Segment.timed_overlap()` ([model.py](../src/seqmem/model.py:63-74):

```text
count(source for source, source_time in active_sources
      if source has a synapse and
         abs(source_time + synapse.delay - target_time) <= timing_tolerance)
```

There is no multiplicity term and no synapse-weight term in this eligibility
count.  `L_match` is applied by `_best_matching_neuron()` and again in the
Scenario-2 branch of `observe_code()` ([model.py](../src/seqmem/model.py:
2035-2121)).  A prediction candidate is preferred first when its time matches;
otherwise the real matching path selects an existing segment only when its
timed overlap reaches `params.l_match`.

## Learning And Context Mutation

Scenario 1 reinforces the exact predicted segment and uses the matching
`PredictionCandidate` ([model.py](../src/seqmem/model.py:2094-2115)).
Scenario 2 calls `_reinforce_segment(..., grow_missing=True)` and can add
missing `previous_winners` synapses ([model.py](../src/seqmem/model.py:
2116-2135 and 3147-3224).  Scenario 3 grows a new segment from
`previous_winners` ([model.py](../src/seqmem/model.py:2137-2147).  Thus the
segment's synapse set is not immutable over its lifetime: reinforcement can
add sources in Scenario 2, while Scenario 1 changes weights and ages.  The
creation context remains available separately through the provenance fields;
the diagnostic compares it with current synapse membership.

At the end of `observe_code()`, `previous_active_cells` receives the exact
active cell map and `previous_winners` receives one winner per encoded event
([model.py](../src/seqmem/model.py:2292-2311).  For autonomous rollout,
`advance_prediction()` copies only the cells represented by the raw prediction
code into both transient contexts ([model.py](../src/seqmem/model.py:2323-2335).
The composition trace is intentionally tagged `trajectory_kind=
actual_observation`; it does not mix those autonomous states into actual
observation rows.

## Diagnostic Call Chain

The strict runner preserves the real order:

1. `run_strict_stream()` calls `rollout_raw_autonomous()` for the prediction
   horizon (`experiments/fig9_strict_reproduction.py:2548-2600`).  This is the
   autonomous state path and ends before the actual record is observed.
2. For the actual record, the runner calls `model.predict_code()` once and
   then captures the current matching state using
   `capture_match_overlap()` (`fig9_strict_reproduction.py:2817-2899`).
3. `capture_match_overlap()` mirrors `_active_sources`: with `burst_context`
   it uses `previous_active_cells` with a `previous_winners` fallback;
   otherwise it uses `previous_winners`.  It computes the same timed overlap
   and only copies rows (`experiments/diagnostics/fig9_match_overlap.py:332-520`).
4. The strict runner then makes the one real call to
   `model.observe_code(code, learn=True)` (`fig9_strict_reproduction.py:2900-2905`
   in the current branch).
5. `finalize_match_overlap()` joins the captured rows with the completed
   `ObservationTrace`, attaching Scenario 1/2/3, selected segment, and
   reinforced segment without re-running matching
   (`fig9_match_overlap.py:1045-1145`).
6. `SegmentContextCompositionTracker.consume_transition()` copies source
   composition and joins those already-produced outcomes.  It updates only
   Python dictionaries/lists owned by the diagnostic.
7. The strict runner updates provenance labels and the tracker records the
   returned actual context.  No diagnostic result is sent back to
   `predict_code()` or `observe_code()`.

The prediction rollout and actual observation therefore have separate state
boundaries.  The output protocol explicitly reports that autonomous rollout is
not captured by this composition trace rather than presenting it as actual
context evidence.

## Diagnostic Quantities

For each live segment inspected at a transition, the diagnostic records its
target field, source cell IDs, creation transition, current synapse count,
reinforcement count, and source-origin labels.  A source can have multiple
labels at once:

- `ACTUAL_WINNER`
- `ACTUAL_BURST`
- `PREDICTED`
- `PROPAGATED`
- `UNKNOWN`

The labels are descriptive projections of the already-existing transient
state.  They are not mutually exclusive and do not affect matching.

For the current live segment population:

```text
source_segment_incidence(s) = number of live segments containing s
source_idf(s) = log((N_segments + 1) / (source_segment_incidence(s) + 1))
normalized_incidence(s) = source_segment_incidence(s) / N_segments
```

The RARE/MEDIUM/COMMON/VERY_COMMON bins use Q25/Q50/Q75/Q90 of the current
incidence distribution.  These are diagnostic statistics only; no IDF score,
source gate, or candidate filter is applied to the strict model.

The per-candidate trace includes L2/L3/L4 flags, overlap source IDs, source
incidence/IDF summaries, same-field/cross-field ratios, active-role ratios,
ambiguity, selection, and reinforcement.  `l2_only_context_composition.csv`
contains candidates with overlap exactly 2 and not 3; the passenger-only file
is a field projection, not a different experiment.

## Interpretation Boundary

The trace can distinguish sparse source reuse from common/shared source reuse,
and it can compare those associations across Passenger, Time, and Weekday and
across record ranges.  It cannot by itself establish causality.  In
particular, Step1 error grouping and target-vs-false grouping are posthoc
labels.  Any mechanism based on IDF weighting, common-source downweighting, or
minimum discriminative source gates must remain a separately labelled
nonpaper diagnostic until a controlled ablation proves that it changes the
failure mechanism without using ground truth or future context.
