# Fig.9 Strict Reproduction Audit

This report records the strict Fig.9 protocol audit started from commit
`ab84634ff80b15f4f5659da7dcd3a43d263aaf78`.

## 1. Paper-Explicit Settings

- Dataset: NYC taxi passenger data, half-hour records.
- Input fields: weekday, time of day, passenger quantity.
- Encoder sizes: weekday 30, time 58, passenger 482, total 570 mini-columns.
- Active mini-columns per field: K=10.
- Neurons per mini-column: 32.
- L_match: 4.
- Forgetting threshold: 65.
- Prediction horizon: 5 steps, equal to 2.5 hours.
- Learning: one-pass online learning, actual record observed once.
- Modified stream: from 2015-04-01, weekday morning decrease and weekday
  evening increase in passenger quantity.

## 2. Unpublished / Local Choices

- Continuous-time constants remain local choices: `tau_m=0.1`, `tau_s=0.02`.
- Default response scale remains the normalized kernel peak implementation
  (`response_scale=None`, fingerprint `V0=null`).
- Integration step is `0.005` normalized cycles.
- Passenger range is 0 to 40000, matching the local reference-[58] setup.
- The 250-record smoke uses `warmup=200` to keep the strict run finite while
  still exercising five-step autonomous rollout.

## 3. Protocol Ambiguities

- The paper does not explicitly state whether future weekday/time covariates
  may be clamped during five-step prediction. Strict mode sets
  `uses_future_covariates=false`; clamping is a nonpaper diagnostic only.
- The supplied perturbed reference file changes weekday evening records from
  21:00 through 23:30. The paper text says evening 9-11 P.M.; this endpoint is
  recorded as a local data-file interpretation.

## 4. Strict Implementation Verification

The new strict runner is `experiments/fig9_strict_reproduction.py`.

Strict checks:

- Prediction happens before observing the actual record.
- Online learning then observes the actual record once.
- Rollout uses raw predictive neuron identities and firing times.
- Decoded passenger values are output-only and are not re-encoded.
- Rollout does not learn.
- Transient state is restored after rollout.
- Burst context is all-cell, not winner-only.
- No eventwise, coherent beam, top-k projection, direct lag-5 shortcut, or
  known-future covariate clamping is allowed in strict fingerprint validation.

## 5. Historical Compensated Results

Historical Fig.9 outputs were moved under:

- `results/fig9_historical_compensated/`

The directory contains metadata marking them as:

- `nonpaper diagnostic`
- `historical compensated result`
- `not valid as strict Fig.9 reproduction`

These include previous winner-context/eventwise and other compensated outputs.
They are retained for audit history but excluded from the strict main result.

## 6. Strict Original Result

Stage A original smoke:

- Output: `results/fig9_strict/stage_a_250/`
- Records: 250
- Warmup: 200
- Attempted predictions: 45
- Coverage: 1.000
- MAPE: 0.505450
- Final rolling MAPE: 0.449526
- Mean raw predicted columns: 291.538
- Peak raw predicted columns: 358
- Runtime: 605.327 seconds
- Throughput: 0.413 records/second

This result is strict but it is only a 250-record smoke, not the paper's
full-year Fig.9(b).

## 7. Strict Perturbed Result

Stage A perturbed smoke:

- Output: `results/fig9_strict/stage_a_250_perturbed/`
- Records: 250
- Warmup: 200
- Attempted predictions: 45
- Coverage: 1.000
- MAPE: 0.505450
- Final rolling MAPE: 0.449526
- Mean raw predicted columns: 291.538
- Peak raw predicted columns: 358
- Runtime: 592.518 seconds
- Throughput: 0.422 records/second

The first 250 records are before 2015-04-01, so original and perturbed
predictions are identical. A manual comparison over 45 evaluated rows found
zero pre-change mismatches.

## 8. Adaptation Curve

The adaptation curve is not available yet because the strict full-year
perturbed run did not reach records after 2015-04-01. Therefore this audit
cannot claim the Fig.9(d) adaptation behavior under strict settings.

## 9. Runtime and Real-Time Feasibility

The stage A strict run is not real-time in the current implementation.
Throughput is about `0.41 records/s` on 250 records. A linear estimate for
2,000 records is roughly 80 minutes per stream, and full-year original plus
perturbed would exceed 23 hours even before accounting for nonlinear growth in
segments and raw prediction density.

The raw prediction density is already high at 250 records:

- mean raw predicted columns: 291.538 out of 570
- peak raw predicted columns: 358 out of 570

This is an abnormal density for an SSTD sparse code and is consistent with the
shared-context / branch ambiguity found in Fig.8 diagnostics.

## 10. Difference From Paper

The paper Fig.9(b,c) reports full-year results. This strict audit has not
completed full-year original or perturbed runs, so numerical distance from the
paper's Fig.9 bars cannot be honestly computed.

The 250-record smoke MAPE of `0.505` is not comparable to the paper's full-year
MAPE because it uses a short early slice and a local smoke warmup.

## 11. Final Reproduction Status

Current strict result has no compensation, but Fig.9 strict reproduction is not
complete.

Completed:

- strict runner
- machine-readable fingerprint
- historical compensated isolation
- tests
- 250-record original smoke
- 250-record perturbed smoke

Not completed:

- 2,000-record original/perturbed strict stage B
- 17,520-record original strict full-year
- 17,520-record perturbed strict full-year
- strict adaptation curve after 2015-04-01
- numerical comparison to paper Fig.9(b,c,d)

It is not valid to claim complete Fig.9 reproduction yet.
