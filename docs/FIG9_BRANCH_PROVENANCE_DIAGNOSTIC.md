# Fig.9 Branch Provenance Diagnostic

This is a nonpaper diagnostic. It does not change strict defaults,
competition, candidate ranking, learning, or autonomous propagation.

## CLI

Enable capture with:

```powershell
--oracle-candidate-diagnostic `
--branch-provenance-diagnostic `
--branch-provenance-level candidate
```

Levels:

- `summary`: candidate rows and compact step summary;
- `candidate`: candidate and candidate-segment rows;
- `full`: candidate, segment, and positive contributing-source rows.

`--branch-source-top-n N` is optional. The default `0` records every positive
source contribution. Any explicit cap is written to the protocol.

## Outputs

- `branch_candidate_trace.csv`
- `branch_segment_trace.csv`
- `branch_source_trace.csv` in full mode
- `branch_step_summary.json`
- `branch_provenance_protocol.json`

Every output declares that it is diagnostic-only, ground truth is analysis-only,
and branch metrics do not affect prediction.

The offline analyzer writes:

- `branch_candidate_summary.csv`
- `branch_field_summary.csv`
- `branch_horizon_summary.csv`
- `branch_target_vs_false.csv`
- `branch_separability_metrics.csv`
- `branch_threshold_curves.csv`
- `branch_chimera_summary.csv`
- `branch_bootstrap_ci.csv`
- `branch_provenance_summary.json`
- `FIG9_BRANCH_PROVENANCE_REPORT.md`

## Interpretation

`segment_provenance_id` denotes exact creation identity.
Source-set Jaccard denotes similarity only and is never called a branch ID.

Because one `PredictionCandidate` currently references one segment,
`unique_branch_count`, branch entropy, and contribution HHI are structurally
`1`, `0`, and `1` when provenance is available. They are retained in the schema
to make that structural result auditable, not to imply a multi-segment
candidate.

The informative measurements are:

- creation source fingerprint versus current segment source fingerprint;
- current-context overlap and Jaccard;
- predicted/burst contribution split;
- target versus false passenger candidate separation;
- degradation across rollout Steps 1-5.

Formal 250-record runs must be started manually with
`scripts/fig9_branch_provenance_250.ps1`.

## 50-Record Smoke Result

This smoke test uses `limit=50`, `warmup=20`, the reference continuous
implementation, raw propagation, and the fixed competition settings. It is not
a formal 250-record conclusion.

- Sequential candidate rows: 21,473.
- Batched candidate rows: 21,737.
- Provenance available rate: 1.0 for both policies.
- Prediction SHA, stable competition trace, oracle trace, final model
  fingerprint, and RNG fingerprint are identical with capture on and off.
- Every saved candidate has one segment, so exact branch-share PR-AUC equals
  prevalence by construction.

Passenger current-context Jaccard has more signal than original score in some
steps. Sequential Step 1 PR-AUC changes from `0.080` for score to `0.111` for
Jaccard; Steps 2-5 Jaccard PR-AUC is `0.102, 0.090, 0.069, 0.063`. Batched
Step 1 is also `0.111`, but Steps 2-4 are much weaker; Step 5 reaches `0.068`.
The product `score * current-context Jaccard` does not consistently improve on
Jaccard alone.

The stronger structural finding is at segment creation:

- Passenger segments were created from about 907 active source cells on
  average.
- At evaluation they retain about 31 synapses.
- Only about 0.4-0.7 current synapses were added after creation.
- Creation-to-current source Jaccard is about 0.033 for both target and false
  candidates and has PR-AUC near prevalence.
- At Step 1, about 91% of passenger candidates are `burst-only` supported;
  target and false rates are both above 91%.

This does not support the proposed multi-segment branch-chimera mechanism.
Instead, the smoke evidence points to extremely broad all-cell burst context at
segment creation, followed by heavy pruning. Target and false segments are
similarly affected, so retrieval-only branch reranking is unlikely to recover
a clean branch that was never distinctly stored.
