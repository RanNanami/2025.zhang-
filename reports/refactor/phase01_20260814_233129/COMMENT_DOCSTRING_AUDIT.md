# Comment and Docstring Audit

This is a read-only inventory. No scientific source comments were rewritten.

## Priorities

Retain and strengthen comments that explain paper provenance, state mutation, RNG
sensitivity, floating-point ordering, checkpoint constraints, and nonpaper diagnostic
boundaries. Remove narration and stale performance claims only in the phase that
touches the associated code, so comment edits remain reviewable beside behavior gates.

## Large files with low definition-docstring coverage

| File | Definitions | Documented | Module docstring | Comment lines |
| --- | ---: | ---: | --- | ---: |
| `tests/test_fig9_pairwise_context.py` | 55 | 0 | False | 0 |
| `tests/test_fig9_temporal_context.py` | 54 | 0 | False | 0 |
| `tests/test_paper_alignment.py` | 54 | 0 | False | 0 |
| `tests/test_fig9_anchor_parity.py` | 52 | 0 | False | 0 |
| `experiments/diagnostics/analyze_fig9_pairwise_context.py` | 51 | 7 | True | 13 |
| `tests/test_fig9_teacher_forced_identity.py` | 49 | 0 | False | 0 |
| `experiments/diagnostics/analyze_fig9_match_overlap_decomposition.py` | 40 | 3 | True | 13 |
| `tests/test_fig9_match_overlap.py` | 40 | 0 | False | 2 |
| `tests/test_fig9_paper_code_parity.py` | 39 | 0 | False | 0 |
| `tests/test_fig8_intralayer_plasticity_parity.py` | 38 | 0 | False | 2 |
| `tests/test_fig9_strict_reproduction.py` | 38 | 0 | False | 0 |
| `experiments/diagnostics/fig9_autonomous_context_provenance.py` | 36 | 2 | True | 2 |
| `experiments/diagnostics/analyze_fig9_l2_ambiguity_accumulation.py` | 33 | 3 | True | 0 |
| `experiments/diagnostics/fig8_temporal_confirmation_semantics.py` | 31 | 1 | True | 4 |
| `experiments/diagnostics/analyze_fig9_readout_dynamics.py` | 30 | 1 | True | 2 |
| `experiments/diagnostics/fig9_temporal_context.py` | 30 | 1 | True | 5 |
| `tests/test_fig8_scenario_temporal_credit_parity.py` | 30 | 0 | False | 0 |
| `tests/test_fig9_competitive_inhibition.py` | 30 | 0 | False | 0 |
| `experiments/diagnostics/analyze_fig9_lmatch_ablation.py` | 29 | 1 | True | 3 |
| `tests/test_fig9_candidate_score_separability.py` | 27 | 0 | False | 0 |
| `experiments/diagnostics/analyze_fig9_l2_l4_long_sequence.py` | 26 | 4 | True | 0 |
| `experiments/diagnostics/analyze_fig9_temporal_context.py` | 26 | 1 | True | 0 |
| `experiments/diagnostics/fig7_nonpaper_baselines.py` | 25 | 1 | False | 0 |
| `experiments/diagnostics/analyze_fig9_lmatch_stack_2x2.py` | 23 | 3 | True | 2 |
| `experiments/diagnostics/fig8_scenario_temporal_credit_parity.py` | 23 | 4 | True | 5 |

## Misleading-claim risk

Repository-level reports that say all tests pass or imply complete numerical
reproduction require correction independently of code comments. Diagnostic and
oracle modules should carry an explicit nonpaper marker near their module
docstring and CLI definition in a later documentation-only phase.
