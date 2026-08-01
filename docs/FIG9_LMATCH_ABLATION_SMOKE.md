# Fig.9 real L_match ablation: 50-record smoke

This is a semantic and non-interference smoke test, not the formal 250-record
mechanism result. All runs used the same 50 records, warmup 20, seed 0, raw
autonomous rollout, reference continuous implementation, batched competition,
and `max_candidate_score`. The only model variable was real `L_match`.

Protocol comparison passed: all recorded protocol fields except `L_match` were
identical. Strict defaults remain `L_match=4`; no run used future covariates,
ground truth for selection, compensation, or rollout learning.

## Prediction results

| L_match | MAPE | Step1 | Step2 | Step3 | Step4 | Step5 | coverage |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.429397 | 0.168281 | 0.239372 | 0.302664 | 0.387290 | 0.429397 | 1.0 |
| 3 | 0.289677 | 0.184756 | 0.188002 | 0.211797 | 0.347182 | 0.289677 | 1.0 |
| 2 | 0.283294 | 0.197367 | 0.170399 | 0.302016 | 0.272968 | 0.283294 | 1.0 |

Step 2 onward follows recurrently diverged trajectories. These values cannot
be interpreted as a fixed-state local effect of the matching threshold.

## Mechanism smoke

| L_match | Scenario 2 rate | Scenario 3 rate | final segments | multi-neuron ambiguity | mean emitted columns |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.4074 | 0.5356 | 693 | 0.0154 | 123.48 |
| 3 | 0.4459 | 0.4985 | 643 | 0.0336 | 108.64 |
| 2 | 0.4867 | 0.4593 | 590 | 0.0500 | 100.63 |

At 50 records, lowering `L_match` increases Scenario-2 reuse and decreases
Scenario-3 creation as intended. Ambiguity also rises, especially for L=2,
while emitted density and final segment count fall. The sample is too short to
determine whether error reinforcement or ambiguity becomes unstable later.

## Non-interference

The L4 diagnostic-on and diagnostic-off runs matched exactly:

- predictions SHA256:
  `8a64489fc4def9e75e6a2b6e3baa497f183780416a84d0e413ee98af6889455d`
- MAPE: `0.4293969349409696`
- coverage: `1.0`
- final segments: `693`
- final model fingerprint:
  `8e86c6fac808a07c845cf8919ac1c0f1b6da96e811a65eb26842c91212718130`
- final RNG fingerprint:
  `146a05aaca1b7e5f27e4787e9a67d50ec003af268c832bdd254132d8a40e5e00`
- checkpoint model and RNG fingerprints also matched after reload.

The L4 predictions SHA and MAPE also match the previous 50-record L4 path.

## Status

The implementation qualifies for a user-run formal 250-record grid. No formal
250, 500, 1000, or full-year experiment was run by Codex in this stage.
