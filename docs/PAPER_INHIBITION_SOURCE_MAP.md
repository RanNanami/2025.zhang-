# Paper Inhibition Source Map

Primary source: the supplied Zhang et al. 2025 PDF, `Toward Building Human-Like
Sequential Memory Using Brain-Inspired Spiking Neural Models`, paper pages 3-7,
Sections II-A, II-B, II-C and II-E. This is a semantic audit, not a claim of
numerical reproduction.

| ID | Paper evidence | Mathematical object | Required behavior | Current code | Status |
|---|---|---|---|---|---|
| A | p.3, II-A: a predictive neuron in an active mini-column fires before peers and inhibits them | predictive state -> soma spike | predictive state must lead to an actual ordered spike and same-column suppression | `src/seqmem/model.py:1077-1577`; `reference_continuous_prediction()` at `:2730-2845` | PARTIAL |
| B | p.3, II-A: an active column without a predictive neuron bursts | active column without prediction | every neuron in that active column fires together | `src/seqmem/model.py:2087-2318`, especially `:2310-2318` | MATCH |
| C | p.4, II-B Eq. (2): soma potential combines proximal, apical, distal, inhibitory, oscillatory and refractory terms | `V_pro + V_api + V_dis + V_inh + V_osc + eta` | winner decisions must use the full soma potential | `src/seqmem/dynamics.py:170-181` exposes a simplified state; no candidate-level component trace | UNKNOWN / NOT RECONSTRUCTABLE |
| D | p.4, II-B: inhibitory feedback comes from other mini-columns | `V_inh(t)` | inhibition must be part of the membrane-time calculation | no strict `V_inh(t)` path; diagnostic `competitive_raw` is in `experiments/diagnostics/fig9_competitive_inhibition.py:152-420` | PARTIAL |
| E | p.4, II-B: the highest-potential same-column neuron above threshold fires | same-column WTA | max soma potential plus threshold, with deterministic temporal ordering | strict column selection uses earliest predicted time at `src/seqmem/model.py:1397-1458`; event selection uses score at `:1285-1348` | MISMATCH CANNOT BE QUANTIFIED |
| F | p.4, II-B: firing causes same-column inhibition through the cycle | intracolumn inhibition | after the first actual spike, peers cannot fire in that cycle | strict path keeps one representative per column; no explicit per-neuron suppression state | PARTIAL |
| G | p.4, II-B: inhibition magnitude is only required to be large enough to suppress peers | hard semantic suppression | no free fitted strength is needed for the semantic rule | no paper-core opt-in implementation added | NOT IMPLEMENTED |
| H | p.4, II-B: refractory term prevents a fired neuron from firing again in the same cycle | `eta` | no second spike in the cycle; reset next cycle | `DSNeuronState.is_refractory()` exists, but firing state is local to one candidate solver call | PARTIAL |
| I | p.7, II-E: predictive/depolarized neurons fire in the absence of proximal input and their spikes drive the next context | actual firing propagation | next context must contain actual fired spikes | `advance_prediction()` uses `prediction_active_cells()` at `src/seqmem/model.py:2351-2384`; only selected prediction candidates are advanced | PARTIAL |
| J | p.5, II-C: SSTD selected mini-columns suppress other columns during encoding | intercolumn SSTD sparsity | encoding must preserve top-K selected columns | `src/seqmem/encoding.py:143-153`; no retrieval membrane-level equivalent | MATCH for encoding / UNKNOWN for retrieval |

## Audit rule

The paper explicitly defines the semantic winner, but it does not provide the
numeric time function and amplitude of retrieval `V_inh(t)`. The repository
therefore cannot safely infer a new paper-exact selector from `candidate.score`,
`peak_dendritic_potential`, or firing time. Those are useful diagnostics, not
proofs of full soma-potential equivalence.
