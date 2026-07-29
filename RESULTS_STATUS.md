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

The second-stage false-positive analysis is documented in
`FIG8_FALSE_POSITIVE_DIAGNOSTIC.md`. It confirms that the strict normalized
kernel permits two synchronized initial synapses to cross threshold and that
this contributes strongly to early prediction density. A nonpaper `V0=1.0`
diagnostic reduced 200-sentence distance from `3.635` to `2.600` and mean raw
columns from `91.026` to `35.255`, but at 500 sentences it degraded to `3.688`
and `68.218` columns. Eventwise inhibition also caused substantial rollout
termination. These diagnostics do not replace strict defaults and do not
constitute numerical reproduction.

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

The former full-year Fig.9 outputs have been isolated under
`results/fig9_historical_compensated/`. They used winner-context/eventwise or
other compensated diagnostics and are not valid strict Fig.9 reproductions.
Their historical values remain useful background only and must not be used in
the strict main result table.

The current strict runner is `experiments/fig9_strict_reproduction.py`. It
enforces all-cell burst, raw neural propagation, prediction-before-observe,
no decoded-value replay, no rollout learning, no future-covariate clamping,
and no eventwise/coherent/top-k projection. The machine-readable fingerprint is
at `results/fig9_strict/protocol.json`.

Strict stage A completed for 250 original and 250 perturbed records with
`warmup=200`. Both streams are before April 1 and matched exactly over 45
evaluated rows. The original smoke MAPE is `0.505450`, coverage is `1.000`,
mean raw predicted columns are `291.538`, peak raw predicted columns are `358`,
and runtime is `605.327` seconds (`0.413 records/s`). The perturbed smoke has
the same MAPE and density with runtime `592.518` seconds.

The 2,000-record strict stage B and full-year strict runs have not completed.
The 250-record runtime implies roughly 80 minutes per 2,000-record stream by a
linear estimate, before nonlinear growth in segments and raw prediction
density. Therefore no strict Fig.9(b,c,d) numerical reproduction is claimed.

The rolling curve uses the exact reference-[58] plotting formula: mean
absolute error over the last 400 predictions divided by the global mean
absolute target. `experiments/fig9_recompute_rolling.py` recovers that scale
from each row's stored normalized error.

The historical 1200-record `0.098` result used growth-only learning and
proximal replay during retrieval. It is retained as a diagnostic and is not a
valid corrected paper-protocol result.

Candidate-score separability and branch provenance are nonpaper diagnostics on
`experiment/fig9-competitive-inhibition`. The 250-record score diagnostic found
that passenger `original_score` PR-AUC is already near prevalence at Step 1.
The provenance audit further established that one saved
`PredictionCandidate` references one winning segment; its score is not a
multi-segment sum. The new read-only provenance trace therefore tests source
context mixing within that segment, while explicitly marking multi-segment
candidate chimera as structurally unavailable. Formal 250-record provenance
results have not yet been run.

The preselection segment diagnostic moves the audit before
`PredictionCandidate` construction. It records the actual upper-bound,
continuous-response, threshold-crossing, same-column/same-time score, and
earliest-event-per-column funnel without changing selection. It is a nonpaper,
offline, oracle-labelled diagnostic and is disabled by default. Only 20/50
record validation runs are performed by Codex; the formal 250-record run
remains manual through `scripts/fig9_preselection_segments_250.ps1`.
The allowed 50-record batched validation inspected 58,619 segments; 54,039
crossed threshold, 21,737 became saved candidates, and 13,388 were emitted.
Passenger target-column crossing/candidate/emitted recall was
`0.5672/0.5352/0.3648`. Of 1,250 passenger target-column events, 532 had no
inspected target-column segment. Internal groups contain one column and the
encoder defines no target neuron, so target-vs-false replacement inside one
group is not identifiable. Formal 250-record preselection results have not
been run.

The future-observed teacher-forced winner diagnostic records the neuron and
segment selected when each target record later enters the normal online
observe path, then aligns that reference to earlier autonomous candidates
offline. The reference never changes prediction or competition. Codex
validation is limited to 20/50 records; formal 250-record results must be
generated manually with `scripts/fig9_teacher_forced_identity_250.ps1`. No
complete Fig.9 reproduction is claimed.

The allowed 50-record batched check found passenger target-column
raw/candidate recall `0.543`, but future-observed winner-neuron candidate and
emitted recall only `0.062/0.041`. Among emitted passenger target columns,
`89.4%` used a different neuron from the future-observed reference. This is
preliminary evidence that column-level recall overstates branch identity; the
formal 250-record diagnostic has not been run.

## Verification

The current implementation passes the full unit-test suite. Run
`check_project.ps1` to repeat compilation, tests, dataset checks, and smoke
experiments.
