# PSP Normalization Lineage

| Layer | Formula/value | Evidence class |
|---|---|---|
| Zhang 2025 Eq.(1) | `kappa(t) = V0 [exp(-t/tau_m) - exp(-t/tau_s)]` | `PAPER_EXPLICIT` formula, numeric scale unresolved |
| Zhang 2025 Table I | `V0`, `tau_m`, and `tau_s` are not given as numeric rows | `STILL_UNKNOWN` |
| Zhang 2025 surrounding text | `tau_m` and `tau_s` jointly govern kernel shape | `PAPER_EXPLICIT` qualitative statement |
| Current `src/seqmem/dynamics.py` | when `response_scale is None`, `kernel_scale = 1 / unscaled_peak`, so the peak is normalized to one | `LOCAL_IMPLEMENTATION` |
| Current Fig.9 strict | `response_scale=None` by default | `LOCAL_IMPLEMENTATION` / not paper-proven |
| Zhang et al. 2020 | cited by Zhang 2025 for delay-weight plasticity, not explicitly for Eq.(1) scaling | `REFERENCE_EXPLICIT_BUT_INHERITANCE_UNPROVEN` |
| Sun et al. 2024 | uses a different PSP expression and explicitly describes its own time constants; no Zhang inheritance statement | `CONCEPTUAL_SIMILARITY` |

Conclusion: PSP scaling is **partially recoverable** at the formula level but not at the numeric `V0`/time-constant level. The current normalized kernel is a local implementation assumption, not a defensible Zhang parameter. It must not be promoted to paper truth without author confirmation or an explicit source sentence.

Sources: [Zhang 2025](https://ira.lib.polyu.edu.hk/bitstream/10397/113864/1/Zhang_Toward_Building_Human-like.pdf), [Zhang et al. 2020](https://researchportal.northumbria.ac.uk/files/27753488/A_Belatreche_Elsevier_Neurocomputing.pdf), [Sun et al. 2024](https://backoffice.biblio.ugent.be/download/01JHQKDXSJFR0PG1DZR7MXC4GB/01JHQKKZEHGCADP620DMTB42AD).
