# Reference Architecture Alignment

## Scope and authority

This document compares three things without treating them as interchangeable:

1. Zhang et al. (2025), which defines the scientific target;
2. the frozen paper-constrained implementation at
   `1864f46aee5cc0aee814fd4e6154b06f8bfeb08f`;
3. Numenta's legacy Temporal Memory implementation, used only as an
   architecture and state-machine reference.

The NuPIC source is the official
[`numenta/nupic-legacy`](https://github.com/numenta/nupic-legacy) repository,
especially
[`src/nupic/algorithms/temporal_memory.py`](https://github.com/numenta/nupic-legacy/blob/master/src/nupic/algorithms/temporal_memory.py).
Its `compute()` method calls `activateCells()` and then
`activateDendrites()`. `activateCells()` explicitly dispatches predicted,
burst, and punishment branches; `activateDendrites()` computes the active and
matching segments for the next time step. This separation is useful as a code
organization reference. It is not evidence for Zhang-specific dynamics.

## Reference state machine

NuPIC's high-level state transition is compact:

```text
compute(activeColumns)
  -> preserve activeCells/winnerCells as previous state
  -> activateCells(activeColumns)
       -> predicted column: activate predicted cells and learn
       -> unpredicted column: burst, choose a winner, learn/create
       -> inactive predicted column: punish matching segments
  -> activateDendrites()
       -> recompute activeSegments and matchingSegments
       -> those segments represent predictions for the next step
```

In `_burstColumn()`, an existing matching segment selects its cell as winner;
otherwise `_leastUsedCell()` chooses among cells with the fewest segments and
a new segment grows from previous winner cells. These are useful explicit
boundaries for future Zhang code organization.

## Three-way mapping

`Can reuse behavior?` is deliberately conservative. `No` means the NuPIC
algorithm must not replace the frozen Zhang behavior, even when the names look
similar.

| Zhang concept | Current implementation | NuPIC analogue | Semantic similarity | Critical differences | Future module | Can reuse architecture? | Can reuse behavior? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Active mini-column | Each `SpikeEvent.column` handled by `SequentialMemory.observe_code()` (`src/seqmem/model.py:2116`); event time remains part of the input | `activeColumns` passed to `compute()` / `activateCells()` | Medium | Zhang uses an ordered column-time SSTD event, not a timeless active-column set | `encoding.py` + future activation helper | Yes | No |
| Predictive neuron | `PredictionCandidate.neuron_index` saved by `predict_code()` (`src/seqmem/model.py:1018`) after dendritic/soma timing checks | Cell owning an `activeSegment`; exposed as predictive cell | Medium | Zhang prediction has crossing time, soma firing time, score, delay, PSP evidence, and event time | Future prediction helper, with `model.py` facade | Yes | No |
| Predictive segment | Candidate's exact `Segment` object and crossing metadata; the same object is reused by Scenario 1 | `activeSegments` | High structurally | NuPIC activation is overlap/permanence thresholding; Zhang uses delayed continuous PSP and dendrite/soma dynamics | Future prediction helper | Yes | No |
| Winner neuron | `learning_winners` selected per observed SSTD event and assigned to `previous_winners` at end of `observe_code()` | `winnerCells` | High structurally | Zhang winner carries SSTD time; predicted and burst activation rules differ | Future activation helper | Yes | No |
| Burst | For an unpredicted event, all neurons in the mini-column enter `previous_active_cells`; one learning winner is selected (`model.py:2685-2694`) | `burstColumn()` activates all cells and chooses one winner | High structurally | Zhang burst context stores firing times and is coupled to Scenario 2A/2B rules and spiking dynamics | Future activation helper | Yes | No |
| Previous active cells | `previous_active_cells`, selected by `_active_sources()` according to `burst_context` (`model.py:743`) | `prevActiveCells` local captured before replacing current state | High structurally | Zhang maps cell IDs to spike times and may use all burst cells | Future transient-state helper | Yes | No |
| Previous winners | `previous_winners`, used as growth sources and autonomous context (`model.py:2116`, `2739`) | `prevWinnerCells`, used for synapse growth | High structurally | Zhang values retain event times; autonomous propagation can set winners from predicted spikes | Future transient-state helper | Yes | No |
| Matching segment | `_best_matching_neuron()` uses timed overlap and `L_match` (`model.py:2799`) | `matchingSegments` from potential-synapse activity and `minThreshold` | Medium | Zhang matching includes source timing, delay target, and SSTD tolerance; NuPIC uses permanence/overlap counts | Future learning helper | Yes | No |
| Least-used neuron | `_least_used_neuron_index()` chooses the neuron with the fewest segments (`model.py:4200`) | `_leastUsedCell()` with random tie break | High conceptually | Current tie behavior and RNG path are frozen and must not be replaced by NuPIC's RNG behavior | Future learning helper | Yes | No |
| Segment creation | `_grow_segment()` creates a target-time segment from previous winners (`model.py:2880`) | `_createSegment()` plus `_growSynapses()` from previous winners | High structurally | Zhang assigns delays, target time, weights, ages, and Scenario provenance | Future learning helper | Yes | No |
| Scenario 1 | A time-matched saved `PredictionCandidate` is observed; its exact segment is reinforced (`model.py:2330-2354`) | `activatePredictedColumn()` reinforces active predictive segments | Medium-high | Zhang distinguishes causal PSP contributors and noncontributors and preserves the exact prediction candidate | Future learning helper | Yes | No |
| Scenario 2A | Historical label `scenario2`: no confirmed prediction, but an existing segment passes timed `L_match`; reinforce and grow missing sources (`model.py:2355-2373`) | Matching-segment branch inside `burstColumn()` | High structurally | Zhang uses timed overlap, delay assignment, and its own weight/age rules | Future learning helper | Yes | No |
| Scenario 2B | Historical label `scenario3`: no eligible existing segment; choose least-used neuron and create a segment (`model.py:2287-2318`, `2374-2390`) | No-matching-segment branch inside `burstColumn()` | High structurally | Current raw label is historically overloaded; rename would change reports/tests and is deferred | Future learning helper | Yes | No |
| Scenario 3 | Failed predictive candidates are handled by `_punish_wrong_predictions()` (`model.py:3956`) | `punishPredictedColumn()` | Medium | Zhang's failure and age/weight semantics are not NuPIC's `predictedSegmentDecrement` rule | Future learning helper | Yes | No |
| Punishment | `_punish_wrong_predictions()` updates failed candidates after real input; Scenario 1 also depresses noncontributing synapses | `_punishPredictedColumn()` weakens active synapses on matching segments in inactive columns | Medium | Candidate identity, SSTD timing gate, ageing, and contribution semantics differ | Future learning helper | Yes | No |
| Distal activation | `_active_sources()` -> `_live_incoming()` -> per-segment response in `predict_code()` | `connections.computeActivity(activeCells)` in `activateDendrites()` | Medium | Zhang integrates delayed PSPs and checks dendritic/soma times; NuPIC counts connected/potential synapses | Future prediction helper | Yes | No |
| Prediction | `predict_code()` returns raw SSTD events and stores `last_prediction_candidates` | Predictive cells are cells owning active segments after `activateDendrites()` | Medium | Zhang emits timed spikes, applies intra/intercolumn choices, and preserves candidate metadata | Future prediction helper | Yes | No |
| Autonomous propagation | `advance_prediction()` resolves raw events back to actual predicted cells and advances transient context without proximal replay (`model.py:2739`) | No direct equivalent in this Python TM step API | Low | Continuous raw neural rollout is a central Zhang retrieval protocol; decoding cannot be fed back | Future propagation helper | Limited | No |
| SSTD | `SpikeEvent`, `SymbolCode`, and SSTD encoders in `src/seqmem/encoding.py` | Sparse active columns | Low | Ordered spike times encode identity/value; NuPIC input is a set of active column indices | Keep `encoding.py` | Limited | No |
| PSP | Double-exponential `spike_response()` and `dendritic_potential()` (`src/seqmem/dynamics.py:101`, `122`) | No equivalent continuous PSP in Temporal Memory | None | NuPIC overlap counts cannot substitute for membrane potential integration | Keep `dynamics.py` | No | No |
| Synaptic delay | `Synapse.delay`; arrival time participates in continuous prediction and `_grow_segment()` assignment | No equivalent per-synapse temporal delay in this TM implementation | None | Delay is part of Zhang's temporal computation and checkpoint state | Future learning/prediction helpers | No | No |
| Phase precession | `DSNeuronState` depolarization, oscillation, membrane potential and firing (`dynamics.py:134-197`) | No equivalent | None | Zhang's soma timing and oscillation are scientific behavior | Keep `dynamics.py` | No | No |
| Forgetting | Weight/age score and `_prune_neuron()` (`model.py:39`, `4106`) | Permanence adaptation, segment use bookkeeping, and capacity eviction | Low-medium | Equations, threshold, age update timing, and inactive representation differ | Future forgetting helper | Limited | No |

## Explicit behavior differences

Every item below is a `REFERENCE_BEHAVIOR_DIFFERENCE`. It is documentation,
not a request to alter the implementation.

- `REFERENCE_BEHAVIOR_DIFFERENCE`: NuPIC treats input as active columns;
  Zhang uses ordered column-time SSTD events.
- `REFERENCE_BEHAVIOR_DIFFERENCE`: NuPIC segment activation is based on
  connected/potential overlap counts; Zhang prediction uses delayed
  double-exponential PSP integration and dendritic/soma timing.
- `REFERENCE_BEHAVIOR_DIFFERENCE`: NuPIC predicted-segment punishment is a
  permanence decrement; Zhang has Scenario-specific weight and age semantics.
- `REFERENCE_BEHAVIOR_DIFFERENCE`: NuPIC has no direct autonomous raw-spike
  rollout equivalent to `advance_prediction()`.
- `REFERENCE_BEHAVIOR_DIFFERENCE`: NuPIC's random tie break in
  `_leastUsedCell()` cannot replace the frozen current tie/RNG behavior.
- `REFERENCE_BEHAVIOR_DIFFERENCE`: the current historical `scenario3` label
  combines a paper Scenario 2B creation branch with legacy reporting language;
  changing that label now would alter artifacts, not just architecture.
- `REFERENCE_BEHAVIOR_DIFFERENCE`: NuPIC does not model Zhang's SSTD, synaptic
  delays, continuous dendritic state, soma oscillation, phase precession, or
  forgetting equation.

The stop-gate question is therefore answered **YES**: matching NuPIC behavior
would require scientific algorithm changes. No such changes are permitted or
made. Only its small, explicit state-machine organization is reusable.

## Architecture lesson

The reusable idea is a readable transition skeleton with named state inputs
and outputs. The nonreusable part is almost all numerical and learning
behavior. A future refactor may make Zhang's existing actions visible as
separate functions, but `SequentialMemory` remains the mutation owner and its
pickle-visible classes remain in `seqmem.model` until a separately approved
checkpoint migration exists.
