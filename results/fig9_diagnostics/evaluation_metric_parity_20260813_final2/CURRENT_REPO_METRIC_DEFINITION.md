# Current repository metric definition

The historical `mape` field is computed by `experiments/fig9/metrics.py:mape`, called from `experiments/fig9_strict_reproduction.py` when the final summary is assembled.

```text
absolute_error = sum(abs(prediction - target) for paired rows)
repo_metric = absolute_error / sum(abs(target) for paired rows)
```

This is an aggregate ratio-of-sums, equivalent to WAPE for the paired rows. It is not the pointwise mean MAPE. It is not multiplied by 100. The implementation uses `zip`, so unmatched tail rows are ignored. Missing predictions are removed upstream and reported through coverage; they are not inserted as error terms. Zero targets remain in the numerator and denominator uses their absolute value, so a zero target contributes an absolute error but no denominator scale. The strict summary aggregates all valid prediction rows, not only the final horizon step. The five-step horizon describes the forecast distance.

The legacy field and historical values are intentionally preserved. This audit adds standard MAPE, WAPE, MAE, RMSE, median APE, and p90 APE as offline calculations only.
