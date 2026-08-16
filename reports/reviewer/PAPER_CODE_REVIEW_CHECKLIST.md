# Paper-Code Review Checklist

Review baseline: `d33b02506417c1fda7bf0484ce03c68959046991`. Start with `REVIEWER_GUIDE.md`. A checked item means the reviewer has compared paper text and executable behavior; it does not mean the implementation is paper-equivalent.

| Review item | Paper location | Code location | Current status |
|---|---|---|---|
| [ ] Eq.(1) PSP amplitude / `V0` | Sec. II-B1, Eq.(1); absent from Table I | `src/seqmem/dynamics.py::DSDynamicsParams.kernel_scale`; `spike_response`; `src/seqmem/model.py::MemoryParams.response_scale` | **LOCAL IMPLEMENTATION**: `None` means `1/unscaled_peak`, not “no scale”; current effective `V0` is about 1.87 and PSP peak is exactly 1. |
| [ ] `tau_m` / `tau_s` | Sec. II-B1, Eq.(1); numeric values absent from Table I | `src/seqmem/dynamics.py::DSDynamicsParams`; `src/seqmem/model.py::MemoryParams` | **LOCAL IMPLEMENTATION**: `.10/.02` are repository assumptions, not confirmed Table-I values. |
| [ ] dendritic threshold | Sec. II-B1, Eq.(1); Table I `theta_d=1.0` | `MemoryParams.dendrite_threshold`; `reference_continuous_prediction` | **PARTIAL**: value matches; code also lowers effective threshold by voltage tolerance. |
| [ ] phase precession | Sec. II-B4; Figs. 2(e)-2(f), 5(d)-5(e) | `src/seqmem/dynamics.py::DSNeuronState.oscillation` | **PARTIAL**: mechanism matches; exact crossing-anchored `-cos` trajectory is local. |
| [ ] soma threshold | Sec. II-B5, Eq.(3); Table I `theta=1.0` | `DSDynamicsParams.soma_threshold`; continuous soma search in `model.py` | **PARTIAL**: value matches; voltage epsilon and candidate state handling are local. |
| [ ] `V_inh` | Sec. II-B5, Eq.(3) | `DSNeuronState.membrane_potential`; strict calls in `SequentialMemory` | **IMPLEMENTATION INCOMPLETE / EQUIVALENCE UNPROVEN**: strict lacks a complete continuous trajectory; the public paper does not give its numerical form. |
| [ ] winner semantics | Sec. II-B5: largest soma voltage in mini-column inhibits peers | `SequentialMemory.predict_code`, `winner_by_column` block | **EQUIVALENCE UNPROVEN**: strict selects earliest firing; do not claim equivalence or inevitable scientific error. |
| [ ] timing tolerance | Scenario 2 “same time”; no numeric window published | `MemoryParams.timing_tolerance`; prediction, observe, Scenario matching and contributors | **HIGH-IMPACT LOCAL SEMANTIC**: `.03` affects prediction confirmation, burst classification, matching, and Scenario-1 credit. |
| [ ] integration voltage tolerance | Exact threshold inequalities in Eqs. (1)/(3) | `MemoryParams.integration_voltage_tolerance`; continuous crossing solvers | **NUMERICAL IMPLEMENTATION PARAMETER**: `2e-5` is a numerical threshold epsilon, distinct from semantic timing tolerance. |
| [ ] Scenario 1 contributors | Sec. II-D1, Scenario 1 | `scenario1_contributing_sources`; `_reinforce_segment` | **LOCAL IMPLEMENTATION / OPERATIONALIZATION**: paper says “contributed synapses” but gives no executable criterion. |
| [ ] Scenario 2 matching | Sec. II-D1, Scenario 2, matching segment and `L_match` | `_best_matching_neuron`; `Segment.timed_overlap` | **PARTIAL**: `L_match` values match; same-time eligibility is local. |
| [ ] Scenario 2B segment creation | Sec. II-D1, Scenario 2, no matching segment | `observe_code`; `_grow_segment` | **PARTIAL**: mechanism maps; legacy code counter names this branch `scenario3`. |
| [ ] Scenario 3 punishment | Sec. II-D1, Scenario 3 | `_punish_wrong_predictions` | **PARTIAL**: weight decrement matches; contributor timing rule is local. |
| [ ] synaptic delay | Sec. II-D1, Scenario 2 printed delay rule | `_new_synapse_delay` | **PARTIAL**: strict normalized `current-delay` mapping is not fully derivable from public text. |
| [ ] burst context | Sec. II-A/II-B2: all cells burst when no predictive neuron | `_active_sources`; burst block in `observe_code` | **PARTIAL**: all-neuron firing is explicit; propagation of every burst neuron as the complete next distal context is not fully specified. |
| [ ] forgetting | Sec. II-D; Sec. III-A; Table I | `_reinforce_segment`; `_punish_wrong_predictions`; `_prune_neuron`; `_forgetting_helpers.py` | **PARTIAL**: published weights/ages/thresholds map; exact event ordering is implementation detail. |
| [ ] Fig.9 encoder | Sec. II-C1/2; Sec. III-D: 30/58/482, `K=10` | `build_fig9_encoder`; `SSTDRealValueEncoder` | **LOCAL IMPLEMENTATION**: passenger range, centers, clipping, and sigma are unpublished. |
| [ ] Fig.9 decoder | Sec. III-D: “most likely prediction”; no decoder equation | `SSTDRealValueEncoder.decode_likelihood` | **LOCAL IMPLEMENTATION**: half-spacing grid, overlap ordering, and tie averaging. |
| [ ] Fig.9 metric | Sec. III-D: MAPE citing [58] | `experiments/fig9/metrics.py::mape` | **REFERENCE-SUPPORTED**: [58] uses the implemented ratio of sums; it differs from ordinary pointwise MAPE. |
| [ ] rollout | Sec. III-D: five steps / 2.5 hours, online stream | `rollout_raw_autonomous`; `run_strict_stream` | **PARTIAL**: horizon/raw autonomy match; snapshot, warmup, and evaluation state convention are local. |

## Reviewer Sign-Off

- [ ] I separated paper-explicit behavior from repository strict defaults.
- [ ] I did not use oracle/diagnostic modes as evidence of strict equivalence.
- [ ] I recorded every proposed scientific change as a separate experiment or patch.
- [ ] I checked `reports/paper_code_diff/HIGH_IMPACT_SCIENTIFIC_GAPS.md` before accepting numerical claims.
