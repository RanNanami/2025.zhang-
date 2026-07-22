# Fig.8 Scenario-1 Contribution Diagnostic

## Scope

This stage tests one hypothesis: Scenario 1 predicted a neuron using continuous
PSP integration, but selected synapses for learning with a separate fixed
arrival window. It does not change Scenario 2/3, V0, tau, thresholds, network
size, data, seed, epoch count, or raw neural retrieval.

The strict default remains `arrival-window`. `continuous-positive` and
`continuous-causal` are labeled **nonpaper diagnostic** and are not used by the
strict Fig.8 command.

## Implementation

Each selected `PredictionCandidate` can now retain its actual dendritic crossing
time, every active synapse's PSP contribution at that crossing, peak potential,
and threshold margin. Scenario 1 receives that exact candidate object from the
prediction that caused the learning event; it does not predict again and does
not use the expected word to select synapses.

The contribution modes are:

- `arrival-window`: existing target-time plus/minus `timing_tolerance` rule.
- `continuous-positive`: every saved active synapse with positive PSP at the
  actual crossing.
- `continuous-causal`: largest saved contributions until their cumulative PSP
  reaches the effective dendritic threshold.

Diagnostic collection is off by default. With it off, strict prediction keeps
its previous execution path.

## Verification

`compileall` passed. The full test suite passed: **80/80 tests**, including the
73 pre-existing tests and 7 new Scenario-1 tests. The new tests cover trace
invariance, active-source membership, causal-subset minimality, exact candidate
identity, read-only evaluation, absence of ground-truth inputs, and unchanged
strict defaults.

## Results

All runs use CBT, one-pass online learning, seed 11, V0/`response_scale=1.0`,
raw neural propagation, 100 columns x 10 neurons, K=10, L_match=3, and forgetting
threshold 500.

| Sentences | Mode | Mean Levenshtein | Raw predicted columns | Expected present | No prediction | Early stop |
|---:|---|---:|---:|---:|---:|---:|
| 100 | arrival-window | 1.590 | 24.935 | 0.720 | 0.040 | 0.070 |
| 100 | continuous-positive | 1.690 | 25.658 | 0.715 | 0.035 | 0.070 |
| 100 | continuous-causal | 1.690 | 25.658 | 0.715 | 0.035 | 0.070 |
| 200 | arrival-window | 2.600 | 35.255 | 0.514 | 0.026 | 0.060 |
| 200 | continuous-positive | 2.625 | 36.762 | 0.528 | 0.040 | 0.080 |
| 200 | continuous-causal | 2.605 | 36.623 | 0.521 | 0.039 | 0.075 |

Suffix error rates:

| Sentences | Mode | Step 1 | Step 2 | Step 3 | Step 4 |
|---:|---|---:|---:|---:|---:|
| 100 | arrival-window | 0.170 | 0.330 | 0.410 | 0.680 |
| 100 | continuous-positive | 0.170 | 0.380 | 0.410 | 0.730 |
| 100 | continuous-causal | 0.170 | 0.380 | 0.410 | 0.730 |
| 200 | arrival-window | 0.370 | 0.555 | 0.745 | 0.930 |
| 200 | continuous-positive | 0.395 | 0.600 | 0.740 | 0.890 |
| 200 | continuous-causal | 0.385 | 0.595 | 0.740 | 0.885 |

Scenario-1 learning statistics:

| Sentences | Mode | Events | Contributors 0/1/2/3/4/5+ | Positive PSP weakened | Mean strengthened/weakened | Repeat mean/max |
|---:|---|---:|---|---:|---:|---:|
| 100 | arrival-window | 725 | .0166/.0000/.0083/.0124/.0166/.9462 | .0108 | 10.352/3.491 | 1.646/12 |
| 100 | continuous-positive | 714 | .0000/.0000/.0000/.0168/.0308/.9524 | .0000 | 10.291/3.395 | 1.618/11 |
| 100 | continuous-causal | 714 | .0000/.0000/.0000/.0168/.0308/.9524 | .0000 | 10.291/3.395 | 1.618/11 |
| 200 | arrival-window | 1842 | .0315/.0016/.0152/.0206/.0309/.9001 | .0290 | 10.277/7.748 | 2.190/22 |
| 200 | continuous-positive | 1789 | .0000/.0000/.0006/.0173/.0425/.9396 | .0000 | 10.505/6.853 | 2.169/23 |
| 200 | continuous-causal | 1789 | .0000/.0000/.0006/.0173/.0425/.9396 | .0001 | 10.504/6.853 | 2.169/23 |

The 20-sentence smoke test completed normally in all three modes with mean
Levenshtein 0.0, ten raw columns per generated step, and no early stops.

The 500-sentence run was not started. The protocol required a clear improvement
at 100/200 first, but neither continuous mode improved Mean Levenshtein or raw
prediction density.

## Root-Cause Judgment

1. **Are arrival-window zero-contribution events caused by mismatched
   definitions?** Yes. At 100 sentences all 12 zero-window events had positive
   PSP contributors at the real crossing and weakened them; at 200 sentences
   the same was true for all 58 events. Their mean actual contributor counts
   were 3.83 and 4.79, respectively.
2. **Do continuous modes reduce mistaken weakening after a prediction?** Yes.
   The measured fraction falls from 1.08% to 0% at 100 sentences and from 2.90%
   to 0% (`continuous-positive`) or 0.011% (`continuous-causal`) at 200.
3. **Is this only a statistical relabeling?** No. The selected synapses actually
   change, so the training rule changes. However, the corrected updates do not
   improve recall in these runs.
4. **Does it delay the 200/500 capacity collapse?** No evidence at 200; the
   continuous modes are slightly worse or effectively tied. Therefore 500 was
   intentionally skipped rather than presenting an unsupported capacity claim.
5. **Most likely remaining cause:** the contribution mismatch is real but not
   dominant. The increasing raw-column density and poor cue-stage predictions
   point more strongly to burst/shared-context interference, ambiguous reused
   segments, and/or missing continuous inhibition.

These are diagnostic results, not a complete numerical reproduction of Fig.8.
