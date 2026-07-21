# Reproduction Result Status

This file is the index of results produced after the paper-alignment audit.
Folders not named here are historical or diagnostic outputs.

## Fig. 7

Current corrected strict output: `results/fig7_strict_corrected/`.

| Task | Trials | Final moving accuracy (mean +/- population std) |
| --- | ---: | ---: |
| single ending, after modification | 10 | 0.938 +/- 0.114 |
| two endings | 10 | 0.955 +/- 0.056 |
| four endings | 10 | 0.965 +/- 0.022 |

Seven of ten single-ending trials recovered to a moving accuracy of 1.0 after
the change. Among those seven, recovery took `3143 +/- 1692` additional
elements; one trial recovered after 2824 elements, close to the roughly 2500
reported in the paper. The ten-trial mean is still slower and less reliable.

The corrected implementation separates bursting active cells from learning
winners, preserves predictions from the same mini-column at distinct SSTD
times, randomizes equal least-used-cell choices, and no longer applies
scenario-1 aging during scenario-2 segment growth. These changes explain the
large improvement over the previous strict output (`0.664`, `0.919`, and
`0.844`).

This remains a protocol-aligned partial numerical reproduction, not full
agreement. At 1000 elements, the single-ending mean is `0.745`; at 3750
elements, the two-ending mean is `0.911`; and at 4500 elements, the
four-ending mean is `0.936`. The paper's curves are approximately perfect at
those points. See `FIG7_DEBUG.md` for the cause analysis and retained fixes.

Official [58] HTM, ELM, TDNN, and LSTM curves plus the local strict curve are
exported to `results/fig7_reference_comparison/`. `curves.csv` preserves every
point and `summary.csv` compares the last saved point before modification with
the final point. The source pickle and its checksum are documented in
`PAPER_ALIGNMENT.md`.

## Fig. 8

The corrected Fig.8(a) retrieval protocol is documented in
`FIG8_NEURAL_RETRIEVAL.md`. Neural retrieval now advances the actual predicted
cell identities and spike times; decoded words are output-only and are never
re-encoded as proximal input. Legacy proximal replay remains available only as
an ablation.

At 1000 CBT sentences, neural retrieval obtained mean Levenshtein distance
`3.981`, versus `3.984` for proximal replay. The paper reports about `0.2` at
10,000 symbols. The protocol correction therefore did not produce numerical
agreement and is not a complete reproduction. Auditable neural outputs are at
`results/fig8_strict_neural_retrieval_1000.csv`, `.png`, and `_details.csv`.
The former `fig8_strict_paperflow_1000` result used decoded-symbol proximal
replay and is historical, not the corrected strict result.

The 180-job resource sweep is complete at
`results/fig8_strict_fixedset_resource_sweep/`. It evaluates every stored
sentence, keeps the CBT sample fixed across all network sizes, and varies the
network seed over ten trials. `trials.csv` contains 180 unique jobs and
`summary.csv` contains all 18 network conditions with ten trials each.

Mean Levenshtein distance improved with both resources: for `M=4`, it changed
from `3.605` at 50 columns to `2.644` at 500 columns; for `M=8`, from `3.096`
to `1.264`; and for `M=12`, from `2.411` to `0.592`. The qualitative resource
trend agrees with Fig.8(b), but the absolute errors remain higher than the
paper and are not claimed as numerical agreement.

Fig.8(c) now has a runnable three-layer approximation at
`results/fig8c_poem_memory.csv` and `.png`, with per-poem scores in
`results/fig8c_poem_details.csv`. It uses 200 deterministic five-character
quatrains from the public `chinese-poetry` corpus. Character and line layers
use the DS `SequentialMemory`; explicit SSTD feedback synapses implement
title-to-line and line-to-character retrieval. Cross-layer source events retain
one of ten context-neuron identities instead of aliasing every event in a
mini-column onto one feedback key.

| Poems | Goal-based | Three-layer contextual | Single-layer contextual |
| ---: | ---: | ---: | ---: |
| 40 | 0.000 | 0.000 | 0.000 |
| 80 | 0.000 | 0.000 | 0.062 |
| 120 | 0.000 | 0.000 | 0.150 |
| 160 | 0.000 | 0.000 | 0.331 |
| 200 | 0.000 | 0.000 | 0.670 |

This is not a numerical reproduction. The paper's plot is approximately
`0.8`, `2`, and `12` at 200 poems. The authors do not publish their poem list,
layer sizes, neuron counts, or cross-layer implementation details. The local
interlayer association remains an event-wise approximation rather than a full
proximal/apical membrane simulation.

## Fig. 9

The existing full-year runs at
`results/fig9_corrected_original.*` and
`results/fig9_corrected_perturbed.*` use the former winner-context/eventwise
compensation. They evaluate 11,611 predictions with 100% coverage and remain
useful adaptation diagnostics, but are no longer labeled strict-default runs.

| Stream | Overall MAPE | Post-April-1 MAPE | Final rolling-400 MAPE |
| --- | ---: | ---: | ---: |
| original | 0.131 | 0.089 | 0.066 |
| perturbed | 0.131 | 0.088 | 0.074 |

The perturbation raises rolling error from about `0.089` to a peak near `0.133`,
then it falls to `0.065` by April 29. This closely reproduces the rise and
rapid adaptation in Fig.9(d); the paper's curve peaks around `0.13` and falls
to roughly `0.05`. The modified-data comparison now correctly uses the
post-April-1 value `0.088`, rather than the full-stream perturbed value. The
formal overlay and checkpoint tables are at
`results/fig9_corrected_comparison/`.

The unified model comparison is at `results/fig9_reference_comparison/`.
`bars.csv` separates approximate paper bar heights from the local full-year
result, and `comparison.png` plots original and modified data side by side.
The cited author repository contains the experiment scripts but not trained
Fig.9 arrays, so these paper bars are explicitly digitized approximations.

The current code defaults to the paper-explicit all-cell burst, raw neural
retrieval, direct scenario-1 reinforcement, and fine-grid integration of
dendritic threshold crossing, phase-precessed soma firing, and first-spike
intracolumn competition. A full-year run under these corrected defaults has
not yet completed, so the historical compensated CSVs must not be reported as
its numerical result.

The rolling curve uses the exact reference-[58] plotting formula: mean
absolute error over the last 400 predictions divided by the global mean
absolute target. `experiments/fig9_recompute_rolling.py` recovers that scale
from each row's stored normalized error.

The historical 1200-record `0.098` result used growth-only learning and
proximal replay during retrieval. It is retained as a diagnostic and is not a
valid corrected paper-protocol result.

## Verification

The current implementation passes 63 unit tests. Run
`check_project.ps1` to repeat compilation, tests, dataset checks, and smoke
experiments.
