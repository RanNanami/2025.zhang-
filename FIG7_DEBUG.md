# Fig.7 Failure Analysis and Correction

## Symptom

The previous strict ten-trial run ended at moving accuracies of `0.664` for
single-ending adaptation, `0.919` for two endings, and `0.844` for four
endings. In particular, none of the single-ending trials recovered to perfect
moving accuracy after the ending swap.

## Causes Found

1. Bursting cells and learning winners were represented by the same state.
   The paper activates every neuron in an unpredicted mini-column, then chooses
   one neuron to learn. Collapsing both sets discarded the alternate contexts
   needed by later distal matching.
2. Multi-prediction decoding allowed a mini-column only once across an entire
   symbol. SSTD symbols can reuse one column at different spike times, so this
   incorrectly removed valid branches.
3. Equal least-used neurons were always resolved to the first index. Repeated
   deterministic choices made unrelated high-order prefixes collide on the
   same neurons and segments.
4. Scenario-2 learning aged all unrelated segments on the selected neuron.
   The paper assigns that weakening/aging behavior to a correct prediction
   (scenario 1), not to growth after a missing prediction (scenario 2).

An offline prefix audit exposed the practical effect: after training, some
valid branch endings were absent while unrelated symbols occupied the top
prediction slots.

## Retained Corrections

- `previous_active_cells` stores the complete burst or predicted activity;
  `previous_winners` stores only cells selected for learning.
- Predictions and best-segment matching use active cells. New synapses grow
  only from learning winners.
- Prediction inhibition now enforces distinct mini-columns separately at each
  SSTD event time rather than once for the whole symbol.
- Equal least-used choices use a reproducible seeded random tie break.
- Scenario-2 growth leaves unrelated segments unchanged.
- `fig7_context_audit.py` reports incomplete contexts and their ranked spurious
  alternatives without changing the paper experiment.

## Corrected Ten-Trial Run

The current output is `results/fig7_strict_corrected/`.

| Task | Previous final mean | Corrected final mean | Corrected std |
| --- | ---: | ---: | ---: |
| Single ending after change | 0.664 | 0.938 | 0.114 |
| Two endings | 0.919 | 0.955 | 0.056 |
| Four endings | 0.844 | 0.965 | 0.022 |

Seven of ten single-ending trials recovered to a 100-sequence moving accuracy
of 1.0. Their mean recovery time was `3143 +/- 1692` elements after the swap.

## Remaining Difference

The result is substantially closer but is not a complete numerical match.
The paper is approximately perfect after 1000, 3600, and 4400 elements for the
single-, two-, and four-ending tasks. The local ten-trial means are `0.745` at
1000, `0.911` at 3750, and `0.936` at 4500 elements.

The largest remaining modeling difference is the event-driven DS-neuron
approximation. It evaluates spike events and segment thresholds directly
rather than integrating persistent soma and dendritic membrane potentials on
a fine time grid. A tested literal removal of the `L_match` reinforcement gate
made weak event-level false predictions self-reinforcing and reduced accuracy,
so that change was rejected. Other rejected ablations included a smaller noise
pool, larger forgetting thresholds, uncapped weights, relaxed decoding
overlaps, and broader event budgets; none consistently closed the paper gap.

## Run and Inspect

```powershell
# Run from the cloned repository root.
$env:PYTHONPATH = "$PWD\src;$PWD"
.\.venv\Scripts\python.exe experiments\fig7_repeated_trials.py `
  --output-dir results\fig7_strict_corrected
.\.venv\Scripts\python.exe experiments\fig7_reference_comparison.py
.\.venv\Scripts\python.exe experiments\diagnostics\fig7_context_audit.py
```

`curves.csv` contains every saved mean/std point, `summary.csv` contains the
three endpoint summaries and recovery statistics, and `curves.png` is the
ten-trial plot.
