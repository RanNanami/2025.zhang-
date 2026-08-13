# V_inh Citation-Chain Report

## Question-by-question answers

| # | Question | Answer | Evidence class |
|---:|---|---|---|
| 1 | What is the source of each Eq.(3) term? | `V_pro`, `V_api`, `V_dis`, `V_inh`, `V_osc`, and `eta` are defined in Zhang 2025 Eq.(3); only `V_inh` lacks a computable definition. | `PAPER_EXPLICIT` / `STILL_UNKNOWN` |
| 2 | Which terms are directly defined? | The three input zones, oscillation term, refractory term, winner condition, burst condition, and same-column suppression semantics. | `PAPER_EXPLICIT` |
| 3 | Which terms come from references? | References provide motivation or adjacent mechanisms; no citation is explicitly declared as the complete Eq.(3) implementation source. | `REFERENCE_EXPLICIT_BUT_INHERITANCE_UNPROVEN` |
| 4 | Which are biological motivation only? | Neocortex/HTM structure, oscillations, phase precession, dendritic spike biology, synchronization and pyramidal-cell inspiration. | `BIOLOGICAL_MOTIVATION_ONLY` |
| 5 | Is there a concrete `V_inh(t)` formula? | No. | `STILL_UNKNOWN` |
| 6 | Which paper provides it? | None found in the audited Zhang citation chain. | `STILL_UNKNOWN` |
| 7 | Does Zhang explicitly inherit it? | No explicit following/adopting/using sentence was found. | `STILL_UNKNOWN` |
| 8 | Is `V_inh` amplitude given? | No. | `STILL_UNKNOWN` |
| 9 | Is its temporal kernel given? | No. | `STILL_UNKNOWN` |
| 10 | Is its spatial scope given? | Only `from other mini-columns` is stated; neighborhood and neuron/column aggregation are not defined. | `PAPER_EXPLICIT` / `STILL_UNKNOWN` |
| 11 | What is the source of `tau_m`? | Zhang gives its qualitative role in Eq.(1), not a numeric value or inherited source. | `STILL_UNKNOWN` |
| 12 | What is the source of `tau_s`? | Same as `tau_m`. | `STILL_UNKNOWN` |
| 13 | What is the source of `V0`? | It appears in the paper formula, but no numeric value or normalization rule is supplied. | `STILL_UNKNOWN` |
| 14 | Is PSP normalized? | The paper does not say so; current peak normalization is local code behavior. | `STILL_UNKNOWN` / `IMPLEMENTATION_ASSUMPTION` |
| 15 | Why is current code normalized? | `response_scale=None` uses `1 / unscaled_peak` in `src/seqmem/dynamics.py`; this is a local implementation choice from the reproduction work. | `IMPLEMENTATION_ASSUMPTION` |
| 16 | Does author previous work match? | The accessible delay-learning work concerns delay/weight learning, not a proven inheritance of Zhang's three-zone DS-neuron and `V_inh`. | `REFERENCE_EXPLICIT_BUT_INHERITANCE_UNPROVEN` |
| 17 | What is the source of oscillation amplitude? | Table I explicitly gives `A_osc=0.5`; adjacent papers motivate oscillations but do not supply an inherited value. | `PAPER_EXPLICIT` |
| 18 | What is the source of frequency? | Table I gives a layer-dependent bound `f_osc < 1/(2K)` and the paper says frequency decreases up the hierarchy. | `PAPER_EXPLICIT` |
| 19 | What is the numeric phase-precession rule source? | The paper describes a shift to the trough, half-cycle predictive duration and reset, but not a complete numeric state equation. | `PAPER_EXPLICIT` / `STILL_UNKNOWN` |
| 20 | Is refractory fully paper-explicit? | Its current-cycle no-refire semantics and `eta=-theta` behavior are explicit; exact implementation discretization is not. | `PAPER_EXPLICIT` |
| 21 | Does intracolumn inhibition need a concrete strength? | No. Zhang says it is a constant large enough to prevent suppressed firing. | `PAPER_EXPLICIT_REFERENCE_DEFINED` |
| 22 | Can soma winner be fully implemented? | Not from the public chain alone: the winner requires the largest full soma potential, including unresolved `V_inh`. | `UNPROVEN` |
| 23 | Is current event selection paper-proven? | No. Candidate score/event selection is not proven equivalent to full soma WTA. | `UNPROVEN` |
| 24 | Was a clear current-code mismatch found? | Yes as an evidence gap: normalized PSP, local candidate scoring and absent continuous `V_inh` are not paper-proven. A unique corrective formula was not recovered. | `UNKNOWN` |
| 25 | What can be safely repaired? | Documentation, provenance labels, and audit traces. No model equation can be safely repaired from this chain alone. | `IMPLEMENTATION_ASSUMPTION` |
| 26 | What can only be candidate ablation? | Alternative PSP scaling, candidate ranking and inhibition shapes. | `AUTHOR_PREVIOUS_WORK_CANDIDATE` |
| 27 | What must wait for the author? | The numeric `V_inh(t)` function, `V0`, `tau_m`, `tau_s`, normalization and exact soma-WTA implementation. | `STILL_UNKNOWN` |
| 28 | Should the model be changed now? | No. | `DO_NOT_MODIFY_MODEL_WITHOUT_MORE_EVIDENCE` |
| 29 | Primary conclusion? | `CONTINUOUS_VINH_PUBLICLY_UNDERSPECIFIED` | final |
| 30 | Recommended next step? | `DO_NOT_MODIFY_MODEL_WITHOUT_MORE_EVIDENCE` | final |

## Citation-chain buckets

### Bucket A - recovered

- Eq.(1) double-exponential PSP form.
- `v_dep=0.5`, `A_osc=0.5`, `theta=1.0`, `theta_d=1.0`, `w0=0.5`, `delta_w=0.01`, `delta_w_prime=0.01`.
- qualitative half-cycle depolarization, phase shift to trough, current-cycle refractory semantics, and constant-sufficiently-large intracolumn suppression.

### Bucket B - candidate only

- HTM predictive/burst/winner semantics from [33] and [58].
- global inhibitory interneuron semantics in the author-associated 2021 minicolumn SNN work.
- delay-learning details in [20]/[21].
- PSP/refractory equations in other SNN papers.

### Bucket C - unrecoverable

- continuous intercolumn `V_inh(t)` formula and all of its numerical parameters.
- numeric `V0`, `tau_m`, `tau_s`, PSP normalization choice.
- exact population-level soma WTA implementation and simultaneous-spike policy.

## HTM decision

The HTM white paper defines predictive cells, bursting and temporal sequence algorithms, but it is not a source for a continuous soma voltage or `V_inh(t)` equation. Marked as `HTM_SEMANTIC_ONLY_NOT_CONTINUOUS_VINH_SOURCE`.

## No model repair in this round

No implementation was changed. No parameter sweep or Fig.8/Fig.9 formal run was performed.

Sources: [Zhang 2025](https://ira.lib.polyu.edu.hk/bitstream/10397/113864/1/Zhang_Toward_Building_Human-like.pdf), [HTM 2011](https://numenta.com/assets/pdf/whitepapers/hierarchical-temporal-memory-cortical-learning-algorithm-0.2.1-en.pdf), [Cui et al. 2016](https://www.cortical.io/static/downloads/continuous-online-sequence-learning.pdf), [Zhang et al. 2020](https://researchportal.northumbria.ac.uk/files/27753488/A_Belatreche_Elsevier_Neurocomputing.pdf), [Sun et al. 2024](https://backoffice.biblio.ugent.be/download/01JHQKDXSJFR0PG1DZR7MXC4GB/01JHQKKZEHGCADP620DMTB42AD), [Lan et al. 2021](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.650430/full).
