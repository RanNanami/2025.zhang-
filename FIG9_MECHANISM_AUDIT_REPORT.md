# Fig.9 Mechanism Audit: L2 versus L4

This report is diagnostic-only. Strict prediction, learning, competition,
RNG, checkpoint format, and default protocol were not changed.

## Runs

| run | source | MAPE | coverage | predictions | final segments | final synapses | runtime | prediction SHA256 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| L4 | `actual_branch_provenance_250_20260808_073849_L4` | 0.4086558112 | 1.0 | 45 | 2377 | 130270 | 1123.70 s | `5be58c849c1299e2f531c4d7ef7c2ddc1146c8737d0e69b2f83810f118b5c668` |
| L2 | `actual_branch_lineage_250_20260808_083200_L2` | 0.3927571082 | 1.0 | 45 | 1945 | 115473 | 568.29 s* | `3cc2dfa2f4840ac1d7c70e1f028b57aa85ce5accb483ddb8b90d7a001581873a` |

The L2 runtime is resume-only from checkpoint `next_index=170`; it is not a
fair wall-time comparison with L4 from index zero. The L2 MAPE, prediction
hash, model fingerprint, and RNG fingerprint match the previous L2 formal
result exactly.

Model fingerprints:

- L4: `19cd26f434d69a099bc2c4b5819d4a255f756c5cf515f9572f26a9ee8a15a2b8`
- L2: `e069baa10a77abc3957ed6808432a51f798c1369cccfdebc1df5b003034a1e82`

The MAPE gap is L4 - L2 = `0.0158987530062102`.

## Stepwise comparison

| horizon step | L4 MAPE | L2 MAPE | L4 mean raw columns | L2 mean raw columns |
|---:|---:|---:|---:|---:|
| 1 | 0.3292936643 | 0.3992400263 | 262.76 | 230.47 |
| 2 | 0.4370430861 | 0.3408843282 | 274.58 | 237.31 |
| 3 | 0.4639933524 | 0.3905636904 | 297.04 | 227.69 |
| 4 | 0.4239008639 | 0.4015402438 | 312.96 | 228.13 |
| 5 | 0.4086558112 | 0.3927571082 | 334.27 | 215.89 |

All five steps have coverage 1.0 and zero burst-source count in the competition
trace. L4 has substantially denser raw columns, especially at steps 4-5, while
its error is not uniformly better. This is evidence for a denser shared-context
and candidate-ambiguity regime, not proof that a particular selector is the
cause.

## Branch identity and lineage

The legacy branch identifier is an anchor alias in both formal traces:

| run | anchors | legacy branches | P(branch=anchor) | P(new reinforcement creates branch) | depth-zero branches |
|---|---:|---:|---:|---:|---:|
| L4 | 4943 | 4943 | 1.0 | 1.0 | 4943 |
| L2 | 5375 | 5375 | 1.0 | 1.0 | 5375 |

The v2 pre-match lineage trace partially recovers ancestry without affecting
the model:

| run | lineage roots | unique continuations | unresolved events | merges |
|---|---:|---:|---:|---:|
| L4 | 740 | 471 | 3732 | 0 |
| L2 | 690 | 418 | 4267 | 0 |

The unresolved majority means that the trace does not establish a causal L2/L4
difference in historical lineage. It does establish that the old branch ID
must not be interpreted as a biological or contextual lineage ID.

## Actual observation status

| status | L4 | L2 |
|---|---:|---:|
| `ACTUAL_BRANCH_MIXED_HISTORY` | 3732 | 4267 |
| `ACTUAL_BRANCH_NOT_FOUND` | 3117 | 2635 |
| `MULTIPLE_ACTUAL_BRANCH_COMPATIBLE` | 406 | 324 |
| `UNIQUE_ACTUAL_BRANCH_CONTINUATION` | 65 | 94 |
| `POST_OBSERVATION_BRANCH_INVALID` | 30 | 30 |

The historical L2-only Scenario-2 summary contains 424 field-event rows in
total, including 275 passenger rows. The new lineage run intentionally does
not fabricate a new L2-only join because its actual-only trace does not carry
the old comparison fields.

## Root-cause judgment

1. The old branch implementation was a representation artifact: every branch
   was created directly from the current anchor, with no parent branch and
   depth zero.
2. The new diagnostic removes that ambiguity only partially. It finds unique
   continuation relations, but most parent relations remain unresolved and no
   true merge is demonstrated.
3. L2 beats L4 numerically in this protocol, but the branch/lineage trace does
   not explain the gap. The strongest current model-level signal is lower L2
   raw candidate density and lower final segment capacity, which is consistent
   with less shared-context ambiguity.
4. No learning-rule, L_match, inhibition, timing, or state-propagation change
   is justified by this audit. A formal mechanism experiment would currently
   confound unresolved lineage representation with model behavior.

## Recommendation

`SEGMENT_CONTEXT_REPRESENTATION_FIX_REQUIRED`

The next controlled task should make segment context identity explicit and
auditable, then repeat the same L2/L4 comparison. It must remain a separately
labelled diagnostic or ablation until it passes fingerprint and no-ground-truth
checks. Strict defaults remain unchanged.

## Artifacts

- Source audit: `docs/FIG9_ACTUAL_BRANCH_IDENTITY_GRANULARITY_AUDIT.md`
- Joint lineage analysis: `results/fig9_diagnostics/branch_identity_joint_20260808/`
- L4 run: `results/fig9_diagnostics/actual_branch_provenance_250_20260808_073849_L4/`
- L2 lineage run: `results/fig9_diagnostics/actual_branch_lineage_250_20260808_083200_L2/`
