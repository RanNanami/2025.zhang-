# Multivariate Fig. 9 Extension

This is a nonpaper extension. It preserves the Fig. 9 DS sequential-memory
logic while allowing several continuous input fields to contribute to the
online memory state.

## Preserved Logic

- SSTD periodic encoding for weekday and time phase.
- SSTD real-value encoding for every continuous field.
- DS-neuron sequential memory, delayed distal integration, online plasticity,
  synaptic aging, and active forgetting.
- The paper Fig. 9 target field retains 482 mini-columns, ten active columns,
  32 neurons per mini-column, `L_match=4`, and forgetting threshold 65.
- Prediction happens before each actual record is learned.

The one-step readout projects only the target field. All selected fields still
participate in learning the DS state; target-only projection avoids the large
cost of decoding complete future vectors that are not used by the metric.

## Field Layout

- Weekday: 30 columns.
- Time phase: 58 columns.
- Each auxiliary continuous field: 58 columns.
- Target (`OT` or `T (degC)`): 482 columns.
- Ten active columns per field.

ETTh1 uses all six auxiliary variables plus `OT`, producing 918 columns across
nine fields. Weather selects seven auxiliary variables by absolute correlation
with the next target on the warmup interval, then appends `T (degC)`, producing
976 columns across ten fields. Selection and encoder ranges use warmup data
only.

## Matched Medium Results

These are development-scale runs with 1,000 warmup points and 200 test points.
They are not replacements for the 11,520/2,880 standard transfer results.

| Dataset | Input | Model MAE | RMSE | Persistence MAE |
| --- | --- | ---: | ---: | ---: |
| ETTh1 | Three fields | 3.588 | 4.761 | 0.704 |
| ETTh1 | All seven numeric fields | 4.244 | 5.510 | 0.704 |
| Weather | Three fields | 2.160 | 3.690 | 0.092 |
| Weather | Eight numeric fields | 0.383 | 0.517 | 0.092 |

Adding fields substantially improves Weather on the matched interval but does
not beat persistence. Adding every ETTh1 field makes the result worse, which is
consistent with context fragmentation from combining several continuous
encodings. Field count and allocation must therefore be validated rather than
assuming that more fields always help.

Artifacts:

- `results/multivariate_etth1_medium_h1.csv` and `.png`
- `results/multivariate_etth1_f1_medium_h1.csv`
- `results/multivariate_weather_medium_h1.csv` and `.png`
- `results/multivariate_weather_f1_medium_h1.csv`

## Run

```powershell
.\run_multivariate_fig9.ps1
```

For the much longer 11,520/2,880 run:

```powershell
python -u experiments/diagnostics/multivariate_fig9_transfer.py `
  --dataset etth1 --warmup 11520 --test-size 2880

## Recursive multihorizon diagnostic

`experiments/diagnostics/multihorizon_fig9_transfer.py` trains one multivariate
DS memory and follows one greedy recursive rollout. It records the same rollout
at 12, 24, 48, and 96 steps, so no future observations are injected and the
four horizons share the same model and selected fields.

The verified small diagnostic uses 500 warmup records and 20 forecast origins:

| Horizon | Fig. 9 DS MAE | Persistence MAE | Daily seasonal MAE | Coverage |
| ---: | ---: | ---: | ---: | ---: |
| 12 | 2.255 | 0.698 | 1.709 | 1.000 |
| 24 | 2.427 | 0.476 | 1.870 | 1.000 |
| 48 | 4.745 | 2.689 | 5.538 | 1.000 |
| 96 | 4.706 | 2.663 | 1.558 | 1.000 |

The DS model keeps full prediction coverage but does not beat persistence at
any tested horizon. It beats the daily seasonal baseline only at horizon 48.
These 20-origin results are a diagnostic rather than a final benchmark.

Run the verified diagnostic with:

```powershell
.\run_weather_multihorizon_fig9.ps1
```

Outputs are `results/multivariate_weather_multihorizon_w500.csv` and
`results/multivariate_weather_multihorizon_w500_summary.csv` for the recorded
run. The runner writes the corresponding non-suffixed result names.
```

The long run is computationally much more expensive than the three-field
transfer and should only be used after selecting a useful field configuration.
