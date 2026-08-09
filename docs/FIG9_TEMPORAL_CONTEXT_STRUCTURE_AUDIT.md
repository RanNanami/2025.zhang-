# Fig.9 Temporal Context Structure Audit

This document records the source-level time semantics used by the diagnostic.
The audit is post-hoc and read-only. It does not alter matching, candidate
selection, competition, learning, the RNG path, or strict defaults.

## Call-chain facts

| Stage | File and function | State or formula |
|---|---|---|
| Strict prediction entry | `experiments/fig9_strict_reproduction.py:run_strict_stream` | For an eligible record, autonomous rollout runs before the current real observation is encoded into long-term memory. |
| Autonomous rollout | `experiments/fig9_strict_reproduction.py:rollout_raw_autonomous` | Each step calls `model.predict_code()`, resolves the raw code to the already-selected predictive cells, then copies those cells to `previous_active_cells` and `previous_winners`. `finally` restores the transient snapshot. |
| Current active context | `src/seqmem/model.py:SequentialMemory._active_sources` | Uses `previous_active_cells` when `burst_context` is enabled, otherwise `previous_winners`. The mapping values are SSTD spike times. |
| Candidate generation | `src/seqmem/model.py:SequentialMemory.predict_code` | Iterates live incoming segments for current active sources; candidates retain `PredictionCandidate.time`, crossing time, PSP contributions, peak potential, and the exact segment object from the same call. |
| Candidate identity | `experiments/diagnostics/fig9_candidate_context_oracle.py:stable_candidate_event_id` | Hash of trajectory, record, horizon, field, target column/neuron, stable segment provenance, candidate ordinal, and event time. It does not use object identity or RNG state. |
| Actual observation | `src/seqmem/model.py:SequentialMemory.observe_code` | The real proximal code is applied after prediction. At the end, `previous_active_cells` becomes the actual active cells and `previous_winners` becomes the learning winner cells. Scenario 1/2/3 learning can change long-term segments and synapses here. |
| Temporal state update | `experiments/diagnostics/fig9_temporal_context.py:TemporalContextTracker` | The diagnostic records actual winner activation after the observation. It freezes that history before the next prediction; autonomous cells are tracked in a separate per-rollout map. |

## Answers to the source audit

1. **Source creation time:** a source cell has no independent creation timestamp. Segment creation records are stored on `Segment.creation_transition_index`; source membership is stored in `Segment.synapses` and, when diagnostics are enabled, `creation_source_cell_ids`.
2. **Last actual activation:** not a native model field. The diagnostic derives it causally from actual `previous_winners` after each completed observation and stores only the latest record index per source.
3. **Last predicted activation:** not a native persistent model field. The diagnostic stores it only in the autonomous rollout-local map, keyed by horizon step.
4. **Last autonomous activation:** same rollout-local map; it is reset at each new input record and never merged into actual history.
5. **Actual versus predicted history:** the model itself separates transient `previous_predicted_sources`/`previous_burst_only_sources`; the temporal diagnostic additionally keeps separate actual and autonomous activation maps.
6. **Segment creation time:** `Segment.creation_sentence_index` and `Segment.creation_transition_index` exist. The latter is the record transition used by this Fig.9 runner.
7. **Segment reinforcement time:** no native timestamp list exists. Synapse `age` is reset/incremented by learning, and scenario reinforcement counters are stored, but exact reinforcement record times are not retained.
8. **Source entry time:** no per-source entry timestamp exists beyond the segment creation context snapshot and synapse delay.
9. **Synaptic delay:** stored as `Synapse.delay`. The actual arrival formula is `source_time + synapse.delay`.
10. **Predicted firing time:** `PredictionCandidate.time` is the SSTD target/event time selected by the continuous prediction path; it is not a wall-clock timestamp or a record timestamp.
11. **Continuous contributor time:** each `SynapsePSPContribution` stores `source_time`, `arrival_time`, and the PSP contribution evaluated by the prediction integrator.
12. **Matching time window:** `Segment.timed_overlap` and observation matching use `abs(source_time + delay - target_time) <= timing_tolerance`.
13. **Spike response decay:** yes. `Segment.response_score` and the continuous prediction path apply the double-exponential `spike_response` over elapsed arrival time.
14. **Candidate score recency:** candidate score is the model's overlap/weight or continuous candidate score. It is not an explicit historical record-recency feature; the diagnostic therefore reports correlations rather than assuming independence.
15. **Forgetting:** synapse forgetting uses weight and synapse `age`; active segments are pruned when their synapse set becomes empty. This is learning age, not source activation age.
16. **Pruning age semantics:** `_prune_neuron` and the segment pruning path use synapse forgetting score and age. They do not preserve an ordered source activation history.
17. **Source age used by this audit:** for a source with a prior actual winner activation, `current_record_index - source_last_actual_record_index`; for an autonomous source, `horizon_step - source_last_autonomous_step`. Missing history is `NA`, not zero.

## Representation limitation

The current segment context is an unordered source membership mapping plus
synapse timing parameters. It does not preserve the relative order in which
sources became active. Matching checks current active source identity and
arrival timing, not a historical relative-order template. Therefore the audit
labels this structural fact as:

`SEGMENT_CONTEXT_IS_TEMPORALLY_UNORDERED`

That is a representation fact, not proof that it causes the observed MAPE.
The candidate-level paired analysis is required before proposing an online
mechanism.

## Causal boundary

For prediction at record `t`, the temporal state contains only observations
completed before `t`. The current record is not added until after prediction,
matching, observation, and learning have completed. The current future target
is used only for post-hoc target/false labels in the analyzer.

## Output design

`--temporal-context-diagnostic` is default-off. Its default `summary` level
writes one fixed-width row per candidate to
`temporal_candidate_trace.csv.gz`. It computes O(k) contributor summaries for
each candidate and emits no source-pair, triplet, or full activation-history
trace. `candidate` is an explicit alias for the same bounded candidate row
unit; there is no default source-debug mode.
