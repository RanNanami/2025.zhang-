# Large Python Modules

Line counts use Python `splitlines()` over tracked files at the frozen snapshot.

| Rank | Lines | Role | File | Classes | Functions |
| ---: | ---: | --- | --- | ---: | ---: |
| 1 | 6089 | experiment | `experiments/fig9_strict_reproduction.py` | 6 | 66 |
| 2 | 4214 | core | `src/seqmem/model.py` | 23 | 65 |
| 3 | 1816 | diagnostic | `experiments/diagnostics/analyze_fig9_match_overlap_decomposition.py` | 0 | 40 |
| 4 | 1654 | diagnostic | `experiments/diagnostics/analyze_fig9_candidate_score_separability.py` | 0 | 46 |
| 5 | 1222 | diagnostic | `experiments/diagnostics/analyze_fig9_reference_neuron_selection.py` | 0 | 21 |
| 6 | 1220 | diagnostic | `experiments/diagnostics/analyze_fig9_pairwise_context.py` | 2 | 49 |
| 7 | 1149 | diagnostic | `experiments/diagnostics/fig9_match_overlap.py` | 2 | 15 |
| 8 | 1120 | diagnostic | `experiments/diagnostics/analyze_fig9_teacher_forced_identity.py` | 0 | 22 |
| 9 | 1042 | test | `tests/test_fig9_teacher_forced_identity.py` | 2 | 47 |
| 10 | 1029 | diagnostic | `experiments/diagnostics/analyze_fig9_readout_dynamics.py` | 0 | 30 |
| 11 | 926 | test | `tests/test_fig9_match_overlap.py` | 2 | 38 |
| 12 | 897 | experiment | `experiments/fig9/parity.py` | 0 | 19 |
| 13 | 880 | diagnostic | `experiments/diagnostics/fig8_temporal_confirmation_semantics.py` | 0 | 31 |
| 14 | 878 | test | `tests/test_fig9_strict_reproduction.py` | 1 | 37 |
| 15 | 843 | diagnostic | `experiments/diagnostics/analyze_fig9_lmatch_stack_2x2.py` | 0 | 23 |
| 16 | 839 | diagnostic | `experiments/diagnostics/fig9_preselection_segments.py` | 0 | 12 |
| 17 | 822 | diagnostic | `experiments/diagnostics/fig9_autonomous_context_provenance.py` | 3 | 33 |
| 18 | 816 | diagnostic | `experiments/diagnostics/analyze_fig9_l2_ambiguity_accumulation.py` | 0 | 33 |
| 19 | 766 | diagnostic | `experiments/diagnostics/fig9_segment_context_composition.py` | 3 | 24 |
| 20 | 763 | diagnostic | `experiments/diagnostics/fig9_branch_provenance.py` | 2 | 20 |

## Risk interpretation

`experiments/fig9_strict_reproduction.py` and `src/seqmem/model.py` are the two
highest-risk modules. Their size reflects multiple responsibilities and long-lived
checkpoint paths. Neither is a suitable first extraction target. The next largest
files are predominantly diagnostic analyzers and their tests; they are easier to
isolate later because they are not checkpoint class owners.
