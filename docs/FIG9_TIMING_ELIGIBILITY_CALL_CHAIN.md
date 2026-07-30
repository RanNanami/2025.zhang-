# Fig.9 Timing and Eligibility Call Chain

This audit describes the repository implementation. It does not claim a
paper-text mismatch.

## Online entry

`experiments/fig9/learning.py:9-31`, `learn_actual_code`

- Input: model, actual encoded code, optional observation trace.
- Calls `predict_code()` once, then `observe_code(..., learn=True)`.
- The order is prediction before the real proximal observation.

## Matching context

`src/seqmem/model.py:604-614`, `_active_sources`

- With strict `burst_context=True`, returns `previous_active_cells`, falling
  back to `previous_winners` only when active cells are empty.
- With burst context disabled, returns `previous_winners`.

`src/seqmem/model.py:1941-1961`, `observe_code`

- Copies `_active_sources()` into `previous_active` before observation state,
  learning, segment creation, or reinforcement is changed.
- This `previous_active` object is passed to matching and reinforcement.

## Prediction-time gate

`src/seqmem/model.py:1995-2007`, `observe_code`

- A target-column prediction is valid when
  `abs(candidate.time - event.time) <= timing_tolerance`.
- An invalid target prediction can lead to Scenario 2/3 matching.
- This target prediction-time gate is separate from source-synapse arrival
  timing.

## Candidate segment search

`src/seqmem/model.py:2346-2401`, `_best_matching_neuron`

- Candidate segments are reached through the incoming index of any cell in
  `active_sources`.
- Deleted/inactive segments are excluded at lines 2359-2362.
- Each candidate receives:
  - a PSP/weight response score from `_segment_score`;
  - a discrete timed overlap from `Segment.timed_overlap`.
- Candidates are ordered by `(timed_overlap, score)`.
- An RNG tie-break is used only when both values are exactly equal.

## Source overlap

`src/seqmem/model.py:71-79`, `Segment.timed_overlap`

A source contributes exactly one count only when:

1. its exact cell ID is present in `active_sources`;
2. the segment has a synapse from that cell;
3. `abs(source_time + synapse.delay - target_time) <= timing_tolerance`.

Thus actual overlap uses source identity, source firing time, synaptic delay,
arrival time, and the timing window. Synaptic weight is not part of this
per-source count.

## Response score

`src/seqmem/model.py:2478-2496`, `_segment_score`

- Uses synaptic weight, delayed PSP response, and kernel peak time.
- It ranks candidate segments after they are reached.
- It does not turn a source into a `timed_overlap` contributor.

## Segment eligibility and Scenario assignment

`src/seqmem/model.py:2009-2056`, `observe_code`

- No timing-valid prediction causes `_best_matching_neuron` to run.
- No reachable segment produces Scenario 3.
- Target prediction validity and matching-segment availability have separate
  trace reasons.

`src/seqmem/model.py:2133-2167`, `observe_code`

- Scenario 2 requires the selected segment's timed overlap to be at least
  `L_match`.
- Otherwise the event falls back to Scenario 3 and may create a segment.
- Scenario 1 uses the timing-valid `PredictionCandidate`.

## State phase

The source hook runs after the required learning-side `predict_code()` and
before the single real `observe_code()`. It captures pre-observation active,
winner, predicted, burst, segment, synapse, delay, and arrival evidence.
Scenario labels are attached afterward from the already-completed observation;
no post-observation state is used to reconstruct matching inputs.

