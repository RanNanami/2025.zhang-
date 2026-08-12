# Fig.9 evaluation metric parity report

## Scope

This is a read-only audit of saved prediction artifacts. No model, encoder, decoder, learning rule, RNG path, strict default, or checkpoint was changed. Historical strict values remain intact.

## Core numbers

- Strict raw L4, 250 records, warmup 200: 45 valid predictions; legacy repository metric = 0.505450072193; standard pointwise MAPE = 0.691410538950.
- The metric-definition gap on the same rows is 0.185960466757; this does not move the result to the paper's approximate figure scale of 0.10.
- Same-family 500-record artifact: 295 valid predictions; legacy repository metric = 0.539279578774; standard pointwise MAPE = 1.087284674664.
- For limit 250, the count is `250 - 200 - 5 = 45`; for limit 500 it is `500 - 200 - 5 = 295`. The loop has no second off-by-one in these counts.

## Window and long-term findings

Fixed first/second/third/final quarters, halves, and complete blocks are written to `evaluation_window_comparison.csv`; cumulative and rolling 25/50/100 curves are written to `evaluation_learning_curve.csv`. Windows were predefined by position and were not selected by score.

The 500-record result is descriptive evidence of a short extension, not a longitudinal convergence result. No compatible 800/1000/2000-or-longer raw L4 artifact was accepted by the conservative filter, so `LONG_TERM_CONVERGENCE_NOT_ESTABLISHED`.

## Allowed conclusion

`CURRENT_REPO_METRIC_DIFFERS_FROM_STANDARD_MAPE`

The repository's historical metric is consistent with the formula in reference [58], but the Zhang paper does not document enough evaluation-window detail to claim numerical parity with Fig.9.

## Recommended next step

`WAIT_FOR_AUTHOR_PROTOCOL_CLARIFICATION`

Do not run another large experiment solely to chase the approximate 0.10 figure until the evaluation range and the intended use of reference [58] are clarified.
