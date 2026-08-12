# Anchor Parity Metric Definitions

## STANDARD_MAPE

`mean(abs(prediction - target) / abs(target))` over nonzero targets.  A zero
target is omitted from this pointwise mean and counted in `zero_target_count`.

## CURRENT_REPO_METRIC

The historical repository function is `experiments/fig9/metrics.py:mape`:

`sum(abs(prediction - target)) / sum(abs(target))`

It is a globally target-weighted absolute percentage error, commonly called
WAPE, rather than standard pointwise MAPE.  Zero targets contribute error to
the numerator but no scale to the denominator; an all-zero denominator returns
0.0.  Both metrics are reported on the identical common-anchor set.
