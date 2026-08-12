# Paper and reference metric audit

## Zhang et al. 2025

Fig.9 says that mean absolute percentage error (MAPE) [58] is used to compare the most likely prediction with the actual passenger value. The Fig.9 text does not state the denominator, averaging convention, zero-target handling, percentage scaling, evaluation start/end, warmup, cumulative versus rolling aggregation, or whether the reported curve is a stable-phase value. Therefore the Zhang evaluation range is `PAPER_EVALUATION_RANGE_UNSPECIFIED`.

The paper states that the taxi stream has 17,520 half-hour records and that the forecast horizon is five steps (2.5 hours). It does not state that only 250 or 500 records should be evaluated.

## Reference [58]

Reference [58] is Yuwei Cui, Subutai Ahmad, and Jeff Hawkins, *Continuous online sequence learning with an unsupervised neural network model*, Neural Computation 28(11), 2016. Its Appendix metric is `sum(abs(y - y_hat)) / sum(abs(y))`, which is a ratio-of-sums/WAPE-like metric despite being called MAPE in that source. This supports the legacy repository formula, but it does not prove that every detail of Zhang Fig.9 used the same evaluation window.

Source used for the reference audit: https://arxiv.org/abs/1512.05463

## Warmup provenance

The strict configuration default is 5904 records in `experiments/fig9_strict_reproduction.py`. The 200-record value is explicitly supplied by local short-run scripts and written into their protocol JSON, so it is classified as `DIAGNOSTIC_CHOICE / LOCAL_IMPLEMENTATION`, not `PAPER_EXPLICIT`.
