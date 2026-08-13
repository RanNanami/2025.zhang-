# Soma Equation Lineage

## Zhang 2025

Zhang Eq.(3), printed p.10147, defines the soma potential as the sum of `V_pro`, `V_api`, `V_dis`, `V_inh`, `V_osc`, and the refractory term `eta`. The paper states the winner condition and same-column suppression, but does not define a formula for `V_inh^l(t)`.

## Candidate technical sources

| Candidate | What it actually provides | Lineage judgment |
|---|---|---|
| Zhang et al. 2020, synaptic delay-weight plasticity | supervised delay/weight learning for SNNs; no evidence in the accessible metadata that it defines Zhang 2025's three-zone DS-neuron or intercolumn feedback | `REFERENCE_EXPLICIT_BUT_INHERITANCE_UNPROVEN` |
| Sun et al. 2024, delay learning based on temporal coding | a different standard SNN/Slayer-style model with a PSP and refractory kernel; it has no `V_inh` term and is not cited by Zhang 2025 as the Eq.(3) source | `CONCEPTUAL_SIMILARITY` |
| HTM 2011 white paper / Cui et al. 2016 | predictive cells, bursting, sparse temporal memory and sequence prediction semantics; no continuous soma voltage equation corresponding to Zhang Eq.(3) | `HTM_SEMANTIC_ONLY_NOT_CONTINUOUS_VINH_SOURCE` |
| Lan et al. 2021 minicolumn SNN | global inhibitory interneuron and stronger-lateral-input winner semantics, but a different model and not a cited inheritance source in Zhang 2025 | `AUTHOR_PREVIOUS_WORK_CANDIDATE` |

No source met the required standard of “Zhang explicitly follows/adopts this source and that source defines the same `V_inh` function.” Therefore the lineage is `NO_MATCH` for an exact recoverable soma equation.

Sources: [Zhang 2025](https://ira.lib.polyu.edu.hk/bitstream/10397/113864/1/Zhang_Toward_Building_Human-like.pdf), [Zhang et al. 2020](https://researchportal.northumbria.ac.uk/files/27753488/A_Belatreche_Elsevier_Neurocomputing.pdf), [Sun et al. 2024](https://backoffice.biblio.ugent.be/download/01JHQKDXSJFR0PG1DZR7MXC4GB/01JHQKKZEHGCADP620DMTB42AD), [HTM 2011](https://numenta.com/assets/pdf/whitepapers/hierarchical-temporal-memory-cortical-learning-algorithm-0.2.1-en.pdf), [Lan et al. 2021](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.650430/full).
