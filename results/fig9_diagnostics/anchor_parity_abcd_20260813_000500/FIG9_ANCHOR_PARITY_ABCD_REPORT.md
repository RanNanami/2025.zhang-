# Fig.9 Five-Step Anchor Parity A/B/C/D

This is an isolated causal-semantics diagnostic. It does not change the strict default.

## Cell Results

| Cell | Predictions | Coverage | Standard MAPE | Repository metric | MAE | Relative improvement vs A |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A CURRENT_BUGGED | 45 | 1.000000 | 0.691410539 | 0.505450072 | 6117.024 | 0.00% |
| B OBSERVE_CURRENT_THEN_5 | 45 | 1.000000 | 0.650780418 | 0.480441066 | 5814.362 | 5.88% |
| C REALIGN_TARGET_ONLY | 45 | 1.000000 | 0.728443402 | 0.492162669 | 5931.555 | -5.36% |
| D SIX_STEP_SAME_TARGET | 45 | 1.000000 | 0.671552453 | 0.518433641 | 6274.153 | 2.87% |

## Direct Answers

1. Cell A historical equivalence passed: `True`.
2. A standard MAPE: `0.691410538950`.
3. A repository metric: `0.505450072193`.
4. B standard MAPE: `0.650780417950`.
5. B repository metric: `0.480441066299`.
6. C standard MAPE: `0.728443401654`.
7. C repository metric: `0.492162668802`.
8. D standard MAPE: `0.671552453380`.
9. D repository metric: `0.518433640792`.
10. B relative Standard-MAPE improvement vs A: `5.88%`.
11. C relative Standard-MAPE improvement vs A: `-5.36%`.
12. D relative Standard-MAPE improvement vs A: `2.87%`.
13. Numerically best cell: `B OBSERVE_CURRENT_THEN_5`.
14. Paper-semantics candidate: `B OBSERVE_CURRENT_THEN_5`, because the current record is available and the five autonomous steps end exactly 2.5 hours later.
15. Numerically best and paper-semantic cells are the same: `True`.
16. A causal endpoint: `t+4`, 30 minutes before its scored target.
17. A evaluation target: `t+5`.
18. B last observed timestamp is the labelled current record `t`.
19. B target offset from its last observation: `150 minutes`.
20. C target offset from the labelled t: `120 minutes`; from last observed t-1 it is `150 minutes`.
21. D autonomous rollout steps: `6`.
22. Future leakage: `False`; target values are attached only after prediction.
23. B's observe(t) uses the repository's normal `learn_actual_code` online path exactly once: `True`.
24. B/C/D endpoint-target timelines all have zero offset: `True`.
25. Local absolute change vs A excess over B: Pearson `0.159523`, Spearman `0.066735`; the association is weak.
26. Q4 A-minus-B MAPE effect `0.402132` is greater than Q1 `-0.130055`: `True`. Q1 actually favors A, so the effect is heterogeneous.
27. Morning/evening B improvements are `16.15%`/`6.97%` versus midday `1.52%`; the amplification is clearest in the morning band, with a smaller evening effect.
28. B vs C: B Standard MAPE is lower by `0.077663`; observing current t contributes useful state beyond merely relabelling A's target.
29. B vs D: B Standard MAPE is lower by `0.020772`; current observation helps more than adding one autonomous step alone.
30. Off-by-one effect class: `SMALL`.
31. Using the historical repository metric and the approximate 0.10 paper bar only as a scale reference, B closes about `6.17%` of the A-to-paper numerical gap; metric/window uncertainty prevents a direct paper comparison.
32. The anchor mismatch cannot explain the full paper gap: `False` for full-gap explanation.
33. Evaluation-window and MAPE audit is still required: `True`.
34. Decoder auditing remains relevant after the evaluation-window/metric audit, but it is not the immediate next step.
35. Corrected 500: mechanical prerequisites `True`, recommended now `False`. No 500 run was executed.
36. Primary conclusion: `ANCHOR_MISMATCH_HAS_SMALL_NUMERICAL_EFFECT`.
37. Recommended next step: `AUDIT_EVALUATION_WINDOW_AND_MAPE`.
38. Tests: `49` anchor-specific tests and `590` full-suite tests passed; compileall and diff checks also passed.
39. Commit SHA is recorded in the task completion report after this result bundle is committed.

## Causal Integrity

- Common anchors: `45/45`.
- Same pre-state and RNG checks: `True`.
- Future ground truth enters only post-hoc scoring and analysis.
- Autonomous rollout does not learn or replay decoded values.
- A is five steps from state through t-1 but scored at t+5; B/C/D endpoints match their targets.

## Local Change

Pearson absolute-change vs A-minus-B APE: `0.159523`; Spearman: `0.066735`.

Quartile and fixed time-of-day tables are in `anchor_change_quartiles.csv` and `anchor_timeofday_analysis.csv`.

The paper's approximately 0.10 bar is figure-digitized, so protocol selection is based on causal semantics rather than proximity to that value. This diagnostic is not a claim of complete Fig.9 reproduction.
