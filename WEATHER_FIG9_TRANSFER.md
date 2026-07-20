# Weather Transfer of the Fig. 9 DS Memory

This is a nonpaper transfer experiment. Zhang et al. do not evaluate the
Weather dataset.

## Mapping

- Taxi weekday (30 columns) -> Weather weekday (30 columns).
- Taxi half-hour slot (58 columns) -> Weather ten-minute slot (58 columns).
- Taxi passenger count (482 columns) -> `T (degC)` (482 columns).
- Ten active columns per field and 32 neurons per mini-column.
- `L_match=4`, forgetting threshold 65, DS response, online plasticity, and
  global likelihood decoding are unchanged from the Fig. 9 runner.
- The temperature encoder range is fitted only on the warmup interval.
- Other Weather variables are excluded to preserve the three-field topology.

The task is one-step-ahead online temperature prediction, equivalent to a
ten-minute horizon. Persistence and the latest available daily-phase value are
leakage-free baselines.

## Standard Result

The run uses 11,520 warmup points followed by 2,880 evaluated points.

| Model | MAE | RMSE | WAPE |
| --- | ---: | ---: | ---: |
| Fig. 9 DS transfer | 1.867 | 3.737 | 0.301 |
| Persistence | 0.194 | 0.269 | 0.031 |
| Daily seasonal | 2.817 | 3.421 | 0.454 |

Coverage is 2,880/2,880. The DS transfer outperforms the daily-seasonal
baseline but not persistence. Its prediction follows many smooth intervals,
while frequent wrong-level jumps dominate the error. Values above the warmup
maximum of 17.57 degrees C are also clipped by the fixed-range value encoder.

Artifacts:

- `results/weather_fig9_ds_standard_h1.csv`
- `results/weather_fig9_ds_standard_h1.png`

Run command:

```powershell
python -u experiments/diagnostics/weather_paper_snn.py `
  --warmup 11520 --test-size 2880 `
  --output-csv results/weather_fig9_ds_standard_h1.csv `
  --output-plot results/weather_fig9_ds_standard_h1.png
```

## Interpretation

The direct Fig. 9 mapping is not a competitive one-step Weather forecaster in
the current reproduction. This does not establish that the unpublished author
implementation would fail. A practical Weather model needs a multivariate
encoder, a value readout that handles range shift, and comparison on standard
forecast horizons.
