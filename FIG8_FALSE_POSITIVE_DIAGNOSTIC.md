# Fig.8(a) False-Positive Prediction Diagnostic

> **Status: nonpaper diagnostic.** Numeric `response_scale` values and
> `eventwise-inhibited` propagation are sensitivity/implementation diagnostics,
> not published Zhang et al. settings. The strict defaults remain
> `response_scale=None` (normalized kernel) and raw neural propagation.

## Scope and protocol safeguards

This stage started from commit `b77db57b983c440db115887a4ca281bb2729d202`.
The Phase 1 autonomous retrieval correction remains intact:

- each suffix step calls `predict_code()` once;
- decoding does not call `predict_code()`, `observe()`, or learning code;
- neural rollout advances the actual predicted neuron identities and times;
- decoded words are never re-encoded as proximal input;
- evaluation snapshots and restores transient state plus decode/learning RNG;
- strict defaults, CBT data, seed 11, 100x10 network, K=10, `L_match=3`, and
  forgetting threshold 500 are unchanged.

All numeric V0 runs use one online pass over each sentence. There is no
multi-epoch training, expected-word candidate restriction, ground-truth replay,
or fallback word insertion.

## Code changes

| File | Change |
| --- | --- |
| `src/seqmem/dynamics.py` | Testable unscaled/actual kernel peak and minimum synchronous-synapse helpers. |
| `src/seqmem/model.py` | Optional read-only segment/reinforcement traces and real-neuron eventwise selection. Normal prediction remains trace-free by default. |
| `experiments/fig8_sentence_memory.py` | Diagnostic `raw`/`eventwise-inhibited` neural propagation; default remains `raw`. |
| `experiments/diagnostics/fig8_diagnostic_common.py` | Single-pass model construction, cue/suffix diagnostics, transient restoration, contributor and scenario-1 summaries. |
| `experiments/diagnostics/fig8_response_scale_sweep.py` | 20/100/200 response-scale runner, CSV, plot, and report output. |
| `experiments/diagnostics/fig8_false_positive_ablation.py` | Same-trained-model raw/eventwise comparison. |
| `experiments/diagnostics/fig8_scenario1_diagnostic.py` | Training-only full scenario-1 contributor distribution. |
| `tests/test_fig8_false_positive_diagnostics.py` | Ten focused diagnostic tests; total suite is now 73 tests. |

Key behavior is deliberately separated:

```python
raw = model.predict_code(trace=trace)
propagated = model.select_prediction_events(raw, trace=trace)
decoded = model.decode_symbol_from_prediction(propagated)
model.advance_prediction(propagated)
```

`select_prediction_events()` reads only the same-call
`last_prediction_candidates`. It keeps the candidate's real column,
`neuron_index`, score, and firing time. It has no expected-word argument and
does not construct a legal symbol code by re-encoding a decoded word.

## Kernel mechanism

For `tau_m=0.10`, `tau_s=0.02`, the unscaled double-exponential response peaks
at `t=0.04023595` with amplitude `0.534992244`.

| response scale | kernel peak | one w0=0.5 contribution | minimum synchronized initial synapses |
| --- | ---: | ---: | ---: |
| normalized | 1.000000 | 0.500000 | 2 |
| 0.8 | 0.427994 | 0.213997 | 5 |
| 1.0 | 0.534992 | 0.267496 | 4 |
| 1.2 | 0.641991 | 0.320995 | 4 |
| 1.4 | 0.748989 | 0.374495 | 3 |
| 1.6 | 0.855988 | 0.427994 | 3 |

Therefore the strict normalized default is exactly the hypothesized
two-synchronized-initial-synapse threshold case (within the model's numerical
tolerance). The paper publishes the kernel formula but not V0/tau values, so
none of the numeric scales can be called a paper setting.

## Response-scale results

### 100 CBT sentences

| scale | peak | min syn | Lev. | step1 | step2 | step3 | step4 | absent | mean raw cols | p90 cols | mean active | 2-contrib | 4+-contrib | no prediction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| normalized | 1.000 | 2 | 2.570 | .300 | .660 | .750 | .860 | .358 | 67.833 | 93.0 | 67.833 | .163 | .509 | .000 |
| 0.8 | .428 | 5 | 2.050 | .150 | .290 | .620 | .990 | .430 | 17.834 | 42.1 | 17.834 | .000 | .994 | .050 |
| 1.0 | .535 | 4 | 1.590 | .170 | .330 | .410 | .680 | .280 | 24.935 | 59.0 | 24.935 | .001 | .980 | .040 |
| 1.2 | .642 | 4 | 1.410 | .160 | .280 | .430 | .540 | .218 | 29.552 | 69.0 | 29.552 | .007 | .942 | .040 |
| 1.4 | .749 | 3 | 1.820 | .210 | .450 | .540 | .620 | .245 | 39.546 | 81.0 | 39.546 | .016 | .795 | .015 |
| 1.6 | .856 | 3 | 1.940 | .240 | .470 | .590 | .640 | .263 | 46.927 | 86.0 | 46.927 | .060 | .703 | .003 |

`V0=1.0` and `V0=1.2` were selected for 200 sentences because they jointly
reduced distance, raw density, expected-word absence, and two-contributor
crossings without large no-prediction rates.

### 200 CBT sentences

| scale | peak | min syn | Lev. | step1 | step2 | step3 | step4 | absent | mean raw cols | p90 cols | mean active | 2-contrib | 4+-contrib | no prediction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| normalized | 1.000 | 2 | 3.635 | .700 | .945 | .995 | .995 | .694 | 91.026 | 99 | 91.026 | .223 | .359 | .000 |
| 1.0 | .535 | 4 | 2.600 | .370 | .555 | .745 | .930 | .486 | 35.255 | 73 | 35.255 | .002 | .966 | .026 |
| 1.2 | .642 | 4 | 2.735 | .455 | .630 | .765 | .885 | .460 | 47.038 | 84 | 47.038 | .010 | .898 | .018 |

The normalized model has already reached 91 of 100 columns on average at 200
sentences. Literal `V0=1.0` sharply suppresses two-contributor crossings and
improves raw presence and distance, supporting hypothesis A at this scale.

### Conditional 500-sentence check

Only `V0=1.0/raw` satisfied the 200-sentence gate. At 500 sentences it produced:

| sentences | scale/mode | Lev. | step1 | step2 | step3 | step4 | absent | mean raw cols | p90 | 2-contrib | 4+-contrib | early stop |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 1.0/raw | 3.688 | .788 | .936 | .970 | .994 | .728 | 68.218 | 92 | .004 | .967 | .000 |

The benefit did not persist with capacity. Two-contributor crossings remained
rare, but many 4+-contributor segments crossed threshold and prediction density
grew again. This shows that kernel normalization is an important early cause,
not a complete explanation of the capacity failure. The 1000-sentence
diagnostic was not run because the specified 500-sentence gate failed.

## Contributor distributions

The following suffix distributions come from the streamed trace sample of the
first 20 evaluation sentences. Percentages are over accepted raw segments.

| setting | accepted | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10+ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 200 normalized | 7,314 | .23% | 23.42% | 42.82% | 18.59% | 5.96% | 3.32% | 1.41% | .93% | 1.79% | 1.52% |
| 200 V0=1.0 | 2,858 | 0% | .17% | 4.13% | 22.25% | 26.10% | 16.34% | 11.72% | 7.31% | 5.56% | 6.40% |
| 500 V0=1.0 | 5,557 | 0% | .45% | 3.04% | 14.14% | 16.93% | 15.75% | 14.14% | 10.64% | 7.92% | 16.99% |

The normalized 200-sentence suffix is dominated by 2-3 contributor crossings
(`66.24%` in this sample). At `V0=1.0`, crossings shift to larger contributor
counts, but their number and contextual ambiguity still grow with storage.

## Cue-prefix collapse and bursting

The cue CSVs store the first 20 sentences; each row predicts the next word
before that true word is observed.

| setting | transition | raw columns | burst cells after true input | expected present | selected correct |
| --- | --- | ---: | ---: | ---: | ---: |
| 200 normalized | 1->2 | 95.40 | 64.05 | .55 | .10 |
| 200 normalized | 2->3 | 84.45 | 62.85 | .60 | .20 |
| 200 normalized | 6->7 | 87.40 | 0.00 | .65 | .20 |
| 200 V0=1.0 | 1->2 | 66.40 | 53.30 | .55 | .20 |
| 200 V0=1.0 | 2->3 | 47.95 | 38.30 | .80 | .50 |
| 200 V0=1.0 | 6->7 | 46.15 | 0.00 | .90 | .55 |
| 500 V0=1.0 | 1->2 | 86.25 | 70.40 | .40 | .00 |
| 500 V0=1.0 | 6->7 | 79.90 | 0.00 | .35 | .00 |

The normalized collapse is present at the first measurable transition, not
created only by autonomous suffix rollout. Across the sampled within-sentence
lags, previous burst-cell count correlates with next-transition raw density at
`r=0.536` (normalized 200), `0.521` (V0=1.0 at 200), and `0.358` (V0=1.0 at
500). This supports burst-context contamination as a contributor, but it does
not prove bursting is the sole prerequisite because density is already high at
`1->2` and the data are observational. Word frequency also correlates with
density (`r=0.402`, `0.544`, and `0.504` respectively), consistent with shared
high-frequency contexts consuming capacity.

## Raw versus eventwise neural propagation

Both modes below use the same trained long-term state for each response scale.

| scale | propagation | sentences | Lev. | raw events | propagated events | expected present raw | expected present after | no prediction | early stop |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| normalized | raw | 200 | 3.635 | 91.026 | 91.026 | .306 | .306 | .000 | .000 |
| normalized | eventwise-inhibited | 200 | 3.235 | 38.509 | 8.346 | .376 | .374 | .045 | .105 |
| 1.0 | raw | 200 | 2.600 | 35.255 | 35.255 | .514 | .514 | .026 | .060 |
| 1.0 | eventwise-inhibited | 200 | 2.570 | 20.793 | 7.819 | .454 | .454 | .295 | .520 |

Eventwise inhibition really changes neural context: subsequent raw density is
lower after sparse propagation, and the propagated code contains real predicted
neurons rather than decoded-symbol encodings. However, at `V0=1.0` its distance
gain is only `0.030`, while 52% of sentences stop early and 29.5% of requested
steps have no prediction. It therefore does not provide stable autonomous
retrieval and must remain an implementation diagnostic. Correct words are
usually absent from the later raw state, rather than merely removed by the
same-step projection (`present raw` and `present after` are equal for V0=1.0).

## Scenario-1 reinforcement

This is a separate training-only diagnostic over the same 200-sentence sample.
Contributor counts use the learning rule's timing-match definition; zero is
shown because that definition can disagree with the continuous peak trace.

| scale | events | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10+ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| normalized | 2,068 | 16.44% | 3.77% | 16.39% | 5.75% | 2.90% | 1.45% | 1.69% | 3.05% | 6.00% | 10.69% | 31.87% |
| V0=1.0 | 1,842 | 3.15% | .16% | 1.52% | 2.06% | 3.09% | 3.20% | 2.93% | 1.57% | 3.85% | 8.36% | 70.09% |

For normalized/V0=1.0, mean segment weight changed
`0.4768->0.4801` / `0.4781->0.5382`; mean reinforcement number was
`2.73` / `3.31`, and the observed maximum was `36` / `44`. Thus strict
normalized scenario 1 does reinforce a material fraction of low-contributor
segments, which can create positive feedback. Lowering V0 largely removes the
two-contributor subgroup but does not prevent repeated reinforcement of larger,
possibly ambiguous segments. A future third-stage ablation may test
`scenario1-require-min-contributors` or `scenario1-require-l-match`; neither is
implemented or mixed into the present results.

## Answers to the diagnostic questions

1. **Does the default equal two-synapse triggering?** Yes. Its normalized peak
   is 1, so two synchronized `w0=0.5` inputs meet threshold 1.
2. **How many accepted predictions use two contributors?** At 200 sentences,
   full-evaluation summary is `22.32%` for normalized versus `0.24%` for V0=1.0.
3. **Where does density first collapse?** At `1->2`; sampled normalized density
   is already `95.4` columns.
4. **Is bursting related?** Prior burst count has moderate positive lagged
   correlation with next density, but the observational trace cannot establish
   that bursting is necessary.
5. **Does lower V0 help?** Strongly at 100/200 sentences, with little silence,
   but the effect decays by 500 sentences as 4+ contributor crossings grow.
6. **Does eventwise inhibition preserve autonomous neural retrieval?** It uses
   real cells and no replay, but it is not stable: V0=1.0 early-stop rate is 52%.
7. **Is its gain genuine or cosmetic?** It genuinely changes later context,
   but most of the sparsity gain does not translate to accuracy and coincides
   with early termination.
8. **Where is the correct word lost?** Mainly in the raw prediction after one or
   more sparse rollout steps; same-step inhibition adds little extra loss.
9. **Does scenario 1 reinforce low-evidence segments?** Yes under normalized
   dynamics (16.39% exactly-two and 16.44% zero by the learning timing metric),
   though lower V0 shifts reinforcement toward larger segments.
10. **Most supported explanation?** Unpublished V0/tau scaling is a major early
    mismatch; burst/shared-word context and repeated reinforcement explain why
    lowering V0 alone fails with capacity. Missing continuous intercolumn
    inhibition may matter, but the tested eventwise approximation is not the
    missing complete solution. No evidence here supports a decoder-only bug.

## Verification

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD"
.\.venv\Scripts\python.exe -m compileall -q src experiments tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Result: `Ran 73 tests ... OK`. The ten focused tests cover response-scale
semantics, trace non-interference/RNG identity, real-neuron eventwise selection,
sparsity, A->B->C autonomous recall, same-state/order-independent ablation,
read-only prefix diagnostics, no ground-truth selector input, and scenario-1
distribution accounting, plus the strict raw CLI default.

Experiment commands:

```powershell
# 100-sentence six-value sweep
.\.venv\Scripts\python.exe experiments\diagnostics\fig8_response_scale_sweep.py `
  --num-sentences 100 --response-scales normalized 0.8 1.0 1.2 1.4 1.6 `
  --output-dir results\fig8_diagnostics\response_scale_100 `
  --details-sample-sentences 20 --trace-segments

# 200-sentence confirmation
.\.venv\Scripts\python.exe experiments\diagnostics\fig8_response_scale_sweep.py `
  --num-sentences 200 --response-scales normalized 1.0 1.2 `
  --output-dir results\fig8_diagnostics\response_scale_200 `
  --details-sample-sentences 20 --trace-segments

# Same-trained-state propagation comparisons (run once per scale)
.\.venv\Scripts\python.exe experiments\diagnostics\fig8_false_positive_ablation.py `
  --num-sentences 200 --response-scale normalized `
  --neural-propagations raw eventwise-inhibited `
  --output-dir results\fig8_diagnostics\propagation_normalized_200 `
  --details-sample-sentences 20 --trace-segments

.\.venv\Scripts\python.exe experiments\diagnostics\fig8_false_positive_ablation.py `
  --num-sentences 200 --response-scale 1.0 `
  --neural-propagations raw eventwise-inhibited `
  --output-dir results\fig8_diagnostics\propagation_v0_1_200 `
  --details-sample-sentences 20 --trace-segments

# Conditional 500-sentence check and scenario-1 training trace
.\.venv\Scripts\python.exe experiments\diagnostics\fig8_false_positive_ablation.py `
  --num-sentences 500 --response-scale 1.0 --neural-propagations raw `
  --output-dir results\fig8_diagnostics\response_v0_1_raw_500 `
  --details-sample-sentences 20 --trace-segments

.\.venv\Scripts\python.exe experiments\diagnostics\fig8_scenario1_diagnostic.py `
  --num-sentences 200 --response-scales normalized 1.0 `
  --output-csv results\fig8_diagnostics\scenario1_reinforcement_200.csv
```

Compact summaries, cue samples, and the comparison plot are committed under
`results/fig8_diagnostics/`. Streamed `.jsonl.gz` traces total about 381 MB and
remain local regenerable artifacts, excluded from Git by `.gitignore`.

## Conclusion

This diagnostic supports the claim that normalized kernel scaling makes the
current strict implementation over-excitable and explains a substantial part
of the 100/200-sentence false-positive gap. It also demonstrates that selecting
a lower unpublished V0 does not solve capacity scaling, and that the tested
eventwise inhibition cannot sustain autonomous rollout. The repository remains
a protocol-aligned approximation with a diagnosed numerical gap, **not a
complete numerical reproduction of Fig.8(a)**.
