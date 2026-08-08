# Fig.9 Context Composition x Oracle Candidate Discrimination

This report describes a read-only diagnostic. The oracle target label is
created after prediction and is never used by matching, candidate selection,
competition, reinforcement, learning, or RNG decisions.

All candidate rows use `row_unit=CANDIDATE_SEGMENT`. The formal traces are in
`results/fig9_diagnostics/context_oracle_join_20260808/`.

## Formal Runs

| run | MAPE | coverage | predictions | final segments | final synapses | candidate rows | lossless join |
|---|---:|---:|---:|---:|---:|---:|---|
| L2 | 0.3927571082 | 1.0 | 45 | 1,945 | 115,473 | 73,292 | yes |
| L4 | 0.4086558112 | 1.0 | 45 | 2,377 | 130,270 | 94,932 | yes |

The stable identity audit found zero duplicate IDs, zero orphan rows, and no
many-to-many joins in either formal run.

## Paired Target Versus False Candidates

Pairs are formed within the same autonomous rollout event, horizon step, and
field. The selected target and selected false candidate are compared offline.
The bootstrap intervals below are 95% intervals for target minus false.

| run | paired events | score difference | score 95% CI | IDF difference | IDF 95% CI | very-common fraction difference |
|---|---:|---:|---:|---:|---:|---:|
| L2 | 665 | 0.252647 | [0.108683, 0.408047] | 0.040838 | [-0.011522, 0.088771] | -0.026774 |
| L4 | 672 | 0.044041 | [-0.121739, 0.224458] | -0.074179 | [-0.133764, -0.014928] | -0.024323 |

The field effects are not stable:

| run | Passenger score difference | Time score difference | Weekday score difference |
|---|---:|---:|---:|
| L2 | -0.935346 | 0.329500 | 1.310988 |
| L4 | -0.885268 | -0.099328 | 1.104328 |

Passenger therefore has higher scores for false candidates in both runs,
Weekday favors targets, and Time is weak or changes sign. The context-quality
features do not provide a stable global target-versus-false separator.

## Interpretation

- The lossless join is valid, so the previous identity/data-join concern is
  not the remaining explanation.
- The result is not evidence that IDF, common-source downweighting, or a field
  gate should be added to the strict model.
- Autonomous `overlap_count` means the count of positive continuous
  contributors. It is not an exact timed `L_match` overlap. Exact-2 and
  exact-3 labels are only used for actual-observation rows.
- `l2_relaxed_good_bad.csv` is a posthoc actual-observation table. Error
  categories are defined for generated records only; earlier warmup rows have
  no forecast error by construction.
- No online mechanism was enabled and no optional 500-record run was
  justified by these mixed results.

## Decision

Primary conclusion:

`CONTEXT_QUALITY_DOES_NOT_SEPARATE_TARGET_FROM_FALSE_CANDIDATES`

Recommended next step:

`INVESTIGATE_PAIRWISE_CONTEXT_COHERENCE`

The next diagnostic should examine structured pairwise relationships between
context sources. It should remain a separate nonpaper experiment and must not
alter strict defaults.

## Validation

- `compileall`: passed.
- Full unittest suite: 364 tests passed.
- Formal L2 and L4: exit code 0, coverage 1.0.
- Candidate identity joins: lossless for both formal runs.
- Relevant GZIP traces: read to EOF successfully.
- 20-record diagnostic on/off: predictions SHA256, MAPE, model/RNG
  fingerprints, and segment/synapse counts identical.
