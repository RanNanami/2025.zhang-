# Fig.9 Sequential vs Batched Competition Diagnostic

## Scope

This is a nonpaper diagnostic on
`experiment/fig9-competitive-inhibition`. It does not modify the strict/raw
path, `predict_code()`, learning, RNG order, checkpoint v1, or the sequential
default.

Both policies consume the candidates saved by the same `predict_code()` call:

- `sequential`: each emitted candidate immediately inhibits later candidates.
- `batched`: candidates are assigned to
  `round_half_even(predicted_time / 0.005)` buckets. Candidates in one bucket
  are decided together and do not inhibit each other. Emitted candidates
  inhibit only later buckets.

All competition outputs retain:

```text
diagnostic_only = true
competition_is_local_choice = true
```

## Verification

- `python -m compileall -q src experiments tests`: passed.
- `python -m unittest discover -s tests -v`: 177 tests passed.
- The 50-record run used the same taxi data, warmup 20, horizon 5,
  `continuous_impl=reference`, strength 0.1, tau 0.02, and oracle analysis.
- Final model fingerprint was identical:
  `8e86c6fac808a07c845cf8919ac1c0f1b6da96e811a65eb26842c91212718130`.
- Final RNG fingerprint was identical:
  `146a05aaca1b7e5f27e4787e9a67d50ec003af268c832bdd254132d8a40e5e00`.

## 50-Record Result

| Metric | Sequential | Batched |
|---|---:|---:|
| Final MAPE | 0.3199 | 0.2802 |
| Coverage | 1.0000 | 1.0000 |
| Runtime (s) | 20.74 | 22.26 |
| Step 1 emitted columns | 60.32 | 112.24 |
| Step 5 emitted columns | 70.00 | 132.32 |
| Step 1 passenger target recall | 0.268 | 0.392 |
| Step 5 passenger target recall | 0.192 | 0.428 |
| Step 1 target suppression | 0.614 | 0.499 |
| Step 5 target suppression | 0.249 | 0.134 |
| Step 1 false emitted ratio | 0.894 | 0.925 |
| Step 5 false emitted ratio | 0.817 | 0.884 |

Prediction hashes differ because the rollout policy intentionally differs:

- sequential:
  `EB34EB343E57C267BA49AD4742CA9887DA78794B95D8EDD70EF5FE01F432A241`
- batched:
  `50036D50769C8C22F16AB0465F58BBCE3F21DE8B6BE9E0AE5B0BC5695D373257`

Across 125 aligned rollout steps, batched emitted 412 target columns absent
from the corresponding sequential output, but also emitted 7,259 additional
false columns. After step 1 the two recurrent trajectories have different raw
candidate pools, so these counts describe policy-level trajectory differences,
not a controlled re-selection of one fixed candidate set.

## Interpretation

The result matches interpretation **B**. Removing same-batch order effects
protects some correct candidates, so sequential ordering bias is real.
However, batching protects far more false candidates and nearly doubles the
emitted-column density in several steps. The score/inhibition rule still
cannot reliably distinguish correct from false branches.

Step 5 passenger recall improves while false density remains very high. This
also supports interpretation **F**: recurrent context mixing remains a major
problem. The 50-record MAPE improvement is encouraging but is not formal
evidence, and batched competition is not adopted as a strict default.

Candidate-to-segment provenance remains unavailable:

```text
provenance_available = false
```

The existing user-generated sequential 250 result remains untouched. A formal
batched 250 run and offline comparison must be started manually with the
scripts in `scripts/`.
