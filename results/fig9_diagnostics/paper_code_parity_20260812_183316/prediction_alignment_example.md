# Prediction Alignment Example

For zero-based loop index 200, the CSV labels
`2014-07-05 04:00:00` as the input and compares against
`2014-07-05 06:30:00`, exactly +2.5 hours.

However, prediction executes before record 200 is observed. The model state
contains actual observations only through index 199 at
`2014-07-05 03:30:00`. Raw autonomous steps
therefore have this causal interpretation:

- step 1: causal continuation index 200 at 2014-07-05 04:00:00
- step 2: causal continuation index 201 at 2014-07-05 04:30:00
- step 3: causal continuation index 202 at 2014-07-05 05:00:00
- step 4: causal continuation index 203 at 2014-07-05 05:30:00
- step 5: causal continuation index 204 at 2014-07-05 06:00:00

The fifth raw continuation corresponds to index
204, while evaluation uses
index 205. Relative to the last
actually observed record, the compared target is
3.0 hours ahead, not 2.5.

Verdict: `PREDICTION_HORIZON_SEMANTICS_MISMATCH` is detected in the current
loop. This audit does not alter it. A separate isolated correction and A/B is
required before another expensive run.
