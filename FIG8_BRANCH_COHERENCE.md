# Fig.8 Branch Coherence Diagnostic

This is a nonpaper diagnostic stage. Strict defaults remain unchanged:
raw neural propagation, current-score ranking, current-delay learning,
arrival-window Scenario-1 contribution, paper-scale CBT protocol, seed 11,
100 columns x 10 neurons, K=10, L_match=3, forgetting threshold=500, and
one-pass online learning.

## Modified Files

- `src/seqmem/model.py`
  - Added optional segment provenance fields and predicted/burst active-source
    labels guarded by `capture_branch_diagnostics`.
  - Added `candidate_burst_psp_trace()` for post-hoc PSP source decomposition.
  - Added `select_coherent_prediction_events()` for a read-only diagnostic beam
    selector that uses real PredictionCandidate objects only.
- `experiments/diagnostics/fig8_diagnostic_common.py`
  - Added a diagnostic switch for branch provenance capture.
- `experiments/diagnostics/fig8_branch_coherence.py`
  - Added the branch coherence audit and beam comparison runner.
  - Added `--summary-only` and `--burst-analysis none` for large diagnostic
    grids. Full raw audit still uses all-candidate burst decomposition.
- `tests/test_fig8_branch_coherence.py`
  - Added tests for provenance invariance, burst labels, PSP decomposition,
    candidate-only beam selection, read-only behavior, autonomous A->B->C, and
    strict default preservation.

## Test Result

Command:

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD"
.\.venv\Scripts\python.exe -m compileall -q src experiments tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Result: 96 tests passed.

## Audit Evidence

Raw neural retrieval, full branch/burst audit:

| Sentences | Mean Levenshtein | Expected word presence | Raw columns | Cue burst cells | Correct provenance coherence | Wrong provenance coherence | Correct entropy | Wrong entropy | Historical winner set |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 1.590 | 0.720 | 24.935 | 16.696 | 0.807 | 0.515 | 0.275 | 0.755 | 0.188 |
| 200 | 2.600 | 0.514 | 35.255 | 31.045 | 0.631 | 0.428 | 0.566 | 0.793 | 0.037 |

False prediction PSP-source classification:

| Sentences | Predicted-supported | Burst-assisted | Burst-only |
|---:|---:|---:|---:|
| 100 | 0.833 | 0.098 | 0.070 |
| 200 | 0.728 | 0.120 | 0.152 |

Interpretation: wrong recalls are much less branch-coherent than correct
recalls, and the historical winner-set match collapses from 18.8% at 100
sentences to 3.7% at 200 sentences. Burst contributes more at 200 sentences,
but most false predictions are still predicted-supported, not pure burst-only.

## Selection Comparison

100-sentence grid:

| Selection | Beam | Lambda coherence | Mean Lev | Presence | Early stop | Raw cols | Propagated cols |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 1 | 0.00 | 1.590 | 0.720 | 0.070 | 24.935 | 24.935 |
| local-top1 | 1 | 0.00 | 1.550 | 0.680 | 0.250 | 14.887 | 8.472 |
| coherent-beam | 4 | 0.25 | 1.560 | 0.680 | 0.240 | 14.997 | 8.769 |
| coherent-beam | 4 | 0.50 | 1.500 | 0.688 | 0.240 | 15.096 | 8.801 |
| coherent-beam | 4 | 1.00 | 1.520 | 0.690 | 0.250 | 15.212 | 8.836 |
| coherent-beam | 8 | 0.25 | 1.540 | 0.682 | 0.240 | 14.997 | 8.783 |
| coherent-beam | 8 | 0.50 | 1.530 | 0.682 | 0.230 | 15.084 | 8.793 |
| coherent-beam | 8 | 1.00 | 1.530 | 0.688 | 0.240 | 15.237 | 8.848 |

200-sentence grid:

| Selection | Beam | Lambda coherence | Mean Lev | Presence | Early stop | Raw cols | Propagated cols |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 1 | 0.00 | 2.600 | 0.514 | 0.060 | 35.255 | 35.255 |
| local-top1 | 1 | 0.00 | 2.570 | 0.454 | 0.520 | 20.793 | 7.557 |
| coherent-beam | 4 | 0.25 | 2.540 | 0.460 | 0.515 | 20.995 | 7.844 |
| coherent-beam | 4 | 0.50 | 2.530 | 0.463 | 0.510 | 20.997 | 7.836 |
| coherent-beam | 4 | 1.00 | 2.475 | 0.471 | 0.480 | 20.735 | 7.769 |
| coherent-beam | 8 | 0.25 | 2.525 | 0.461 | 0.515 | 20.989 | 7.846 |
| coherent-beam | 8 | 0.50 | 2.530 | 0.463 | 0.515 | 21.056 | 7.864 |
| coherent-beam | 8 | 1.00 | 2.515 | 0.470 | 0.480 | 20.790 | 7.807 |

The best 200-sentence beam setting improves Levenshtein by 0.125 against raw
but lowers expected-word presence and increases early-stop rate sharply. This
does not satisfy the stage criterion for a 500-sentence run.

## Root Cause Judgment

1. Branch incoherence is real. Wrong recalls have lower provenance coherence,
   lower source coherence, higher entropy, and much lower historical winner-set
   recovery.
2. Burst pollution is present and grows with capacity pressure, but it is not
   the only cause. Most false predictions are still predicted-supported.
3. Coherent beam selection changes the retrieval statistic slightly, so it is
   not just a reporting artifact. However, the improvement is small and does
   not restore expected-word presence at 200 sentences.
4. The 200-sentence capacity collapse is therefore more likely driven by dense
   shared-context ambiguity, neuron-identity branch mixing, and missing stronger
   continuous competition/inhibition than by a single local selection defect.
5. No result here should be described as complete Fig.8 reproduction.

## Outputs

- `results/fig8_diagnostics/branch_coherence/20_audit`
- `results/fig8_diagnostics/branch_coherence/100_audit`
- `results/fig8_diagnostics/branch_coherence/200_audit`
- `results/fig8_diagnostics/branch_coherence/100_beam_grid`
- `results/fig8_diagnostics/branch_coherence/200_beam_grid`
- `results/fig8_diagnostics/branch_coherence/200_beam_best`
