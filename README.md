# Zhang et al. 2025 Sequential Memory Reproduction

This workspace reproduces the experiments in "Toward Building Human-Like
Sequential Memory Using Brain-Inspired Spiking Neural Models".

## Current Scope

- Fig.7: online high-order sequence prediction, 50,000-symbol noise pool,
  random sequence sampling, ending change at element 10,000, and ten trials.
- Fig.8(a,b): real Children's Book Test sentences and the network-size sweep.
- Fig.8(c): a clearly labeled nonauthor-data approximation of three-layer
  five-character-quatrain retrieval.
- Fig.9: the full 17,520-record NYC taxi series and its official changed copy.
- Event-driven SSTD/DS-neuron memory with delayed dendritic integration,
  intralayer plasticity, synaptic aging, and active forgetting.

Read `PAPER_ALIGNMENT.md` before interpreting results. It distinguishes exact
paper settings, settings recovered from reference [58], implementation choices,
and remaining gaps. Only outputs named by `RESULTS_STATUS.md` are current;
other folders under `results/` are historical diagnostics.

Fig.9 follows reference [58]'s reported MAPE normalization,
`sum(abs(prediction - target)) / sum(abs(target))`, and reports it as a ratio
(for example, `0.09`), matching the paper's axis rather than as a percentage.

## Quick Check

```powershell
# Run this after opening PowerShell in the cloned repository.
.\check_project.ps1
```

## Strict Experiments

```powershell
# Fig.7(a-c), 20,000 elements, one trial
.\.venv\Scripts\python.exe experiments\fig7_sequence_prediction.py

# Fig.7(a,d,e), ten trials and mean/std curves
.\.venv\Scripts\python.exe experiments\fig7_repeated_trials.py

# Combine the new DS-memory curve with official [58] baseline curves
.\.venv\Scripts\python.exe experiments\fig7_reference_comparison.py

# Fig.8(a), neural retrieval is the default; decoded words are not replayed
.\.venv\Scripts\python.exe experiments\fig8_sentence_memory.py --retrieval-mode neural --details-csv results\fig8_details.csv

# Same trained model comparison of neural and legacy proximal replay
.\.venv\Scripts\python.exe experiments\fig8_retrieval_ablation.py --num-sentences 100 --output-prefix results\fig8_ablation

# Fig.8(b), 10 trials for M={4,8,12} and N={500,...,50}
.\.venv\Scripts\python.exe experiments\fig8_resource_sweep.py

# Fig.8(c) approximation, 40 to 200 five-character quatrains
.\.venv\Scripts\python.exe experiments\fig8c_poem_memory.py

# Fig.9(b), full original year, with auditable point predictions
.\.venv\Scripts\python.exe experiments\fig9_paper_snn.py --output-csv results/fig9_original.csv --output-plot results/fig9_original.png

# Fig.9(c,d), official change beginning 2015-04-01
.\.venv\Scripts\python.exe experiments\fig9_paper_snn.py --data data/paper_nyc_taxi_perturb.csv --output-csv results/fig9_perturbed.csv --output-plot results/fig9_perturbed.png

# Recompute the exact rolling-400 metric from saved full predictions
.\.venv\Scripts\python.exe experiments\fig9_recompute_rolling.py results/fig9_original.csv results/fig9_perturbed.csv

# Compare local Fig.9 with digitized paper baseline bars
.\.venv\Scripts\python.exe experiments\fig9_reference_comparison.py

# Short strict-default diagnostic
.\.venv\Scripts\python.exe experiments\fig9_paper_snn.py --limit 2000 --warmup 1500

# Historical event-compensation mode, retained only for comparison
.\.venv\Scripts\python.exe experiments\fig9_paper_snn.py --no-burst-context --no-intracolumn-inhibition --projection-mode eventwise --limit 2000 --warmup 1500
```

`run_all.ps1` runs the full strict set. It is intentionally long-running.

See `FIG8_NEURAL_RETRIEVAL.md` for the autonomous-retrieval protocol audit,
tests, 100/200/500/1000-sentence ablation, and failure analysis.
See `FIG8_FALSE_POSITIVE_DIAGNOSTIC.md` for the second-stage response-kernel,
false-positive segment, cue-prefix, and eventwise-inhibition diagnostics. All
numeric response scales in that report are nonpaper sensitivity settings.
See `FIG8_SCENARIO1_CONTRIBUTION.md` for the third-stage audit of whether
Scenario 1 reinforces the same synapses that caused a continuous threshold
crossing. The two continuous contribution modes remain nonpaper diagnostics.
See `FIG8_PREDICTION_COMPETITION.md` for the fourth-stage candidate-score,
event-timing, read-only ranking, and peak-aligned-delay diagnostics. None of
those diagnostic modes changes the strict Fig.8 defaults.

## Data Provenance

- `data/CBTest/`: official CBT archive downloaded through the Facebook/ParlAI
  definition and verified against SHA-256
  `932df0cadc1337b2a12b4c696b1041c1d1c6d4b6bd319874c6288f02e4a61e92`.
- `data/paper_nyc_taxi.csv`: 17,520 half-hour records from 2014-07-01 through
  2015-06-30, from the reference [58] repository; SHA-256
  `092d957f5bb0d2cd62f85098ed2268114a47b4738a5f4b29ea6be4be7349fc4d`.
- `data/paper_nyc_taxi_perturb.csv`: the same series with the published change
  beginning at record 13,152 / 2015-04-01; SHA-256
  `47eb800405d3574c0ba3b820b3c11d8fdd96c4510d5f490932d4748e6d420bd9`.
- `data/fig8c_poems.json`: deterministic 200-poem sample prepared from the
  public `chinese-poetry` Tang JSON corpus; SHA-256
  `2713d1544fd9a06245930dc52218348fccda830b6bbec282bbe6a07fffa8b22c`.
  Run `experiments/prepare_fig8c_poems.py` to rebuild it from the stored source
  shards. This is not the unpublished author poem list.
- `data/fig8c_stress_poems.json`: deterministic 1,000-poem synthetic
  high-conflict diagnostic; SHA-256
  `97619dfc61350d3df59f4b405688aab6cf993e93864c7624c56ade98d50307b2`.
  It is not paper data. See `FIG8C_STRESS_DEBUG.md` for its construction and
  the one-versus-ten source-neuron comparison.

## Diagnostics

The following are useful engineering tests but are not Zhang-paper results:

- `experiments/diagnostics/fig7_nonpaper_baselines.py`
- `experiments/diagnostics/fig7_context_audit.py`
- `experiments/diagnostics/benchmark_real_timeseries.py`
- `experiments/diagnostics/etth1_paper_snn.py`
- `experiments/diagnostics/weather_paper_snn.py`
- `experiments/diagnostics/multivariate_fig9_transfer.py`
- `experiments/diagnostics/multihorizon_fig9_transfer.py`
- `experiments/diagnostics/fig9_taxi_prediction.py`
- `experiments/diagnostics/fig8_response_scale_sweep.py`
- `experiments/diagnostics/fig8_false_positive_ablation.py`
- `experiments/diagnostics/fig8_scenario1_diagnostic.py`
- `experiments/generate_fig8c_stress_dataset.py`
- `experiments/plot_fig8c_stress_comparison.py`
- `run_real_world_benchmarks.ps1`

They include locally retrained baselines, ETTh1/Weather stress tests, KNN, and
context-table methods. They are excluded from `run_all.ps1`.
Install `requirements-diagnostics.txt` only when running these nonpaper tools.
The full Fig. 9-topology ETTh1 transfer and its negative result are documented
in `ETTH1_FIG9_TRANSFER.md`.
The corresponding Weather transfer is documented in
`WEATHER_FIG9_TRANSFER.md`.
The nonpaper multivariate DS extension is documented in
`MULTIVARIATE_FIG9_EXTENSION.md`.
The high-conflict hierarchical-memory test and debugging results are documented
in `FIG8C_STRESS_DEBUG.md`.
The Fig.7 failure analysis, retained fixes, and corrected ten-trial results are
documented in `FIG7_DEBUG.md`.
The Fig.9 protocol audit, corrected full-year results, and remaining numerical
and runtime differences are documented in `FIG9_DEBUG.md`.

## Known Gap

The strict predictor now integrates the published double-exponential distal
response, dendritic threshold crossing, depolarization, phase-precessed soma
firing, and first-spike intracolumn competition on a configurable fine time
grid. Paper-explicit bursting, raw neural retrieval, and scenario-1 learning
are also strict defaults. The paper does not publish `V0`, `tau_m`, or `tau_s`,
so the numerical trajectory remains an approximation and a corrected
full-year strict run is still pending.
