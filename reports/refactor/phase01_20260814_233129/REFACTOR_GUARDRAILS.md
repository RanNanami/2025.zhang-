# Refactor Guardrails

## Immutable scientific constraints

- No parameter or default changes.
- No RNG draws added, removed, or reordered.
- No list/dict/set traversal order changes on scientific paths.
- No floating-point accumulation reordering or approximation.
- No prediction, matching, Scenario learning, punishment, forgetting, burst,
  selection, propagation, decoding, or rollout semantic changes.
- No ground truth, teacher reference, oracle label, or future covariate may
  influence strict prediction or learning.
- No checkpoint-visible class path changes without explicit migration.

## Required exact gate for every behavior-sensitive phase

Rerun all Fig.8 and Fig.9 10/20/50 fixtures and require exact equality for:

- prediction CSV SHA-256;
- science model SHA-256;
- full state SHA-256;
- learning/decode/all RNG SHA-256;
- previous active/winner/candidate SHA-256;
- segment, active segment, synapse, and Scenario summaries;
- metric and coverage.

Any unexplained mismatch is `REFACTOR FAILURE` and requires rollback to the
phase boundary. A numerically close metric is not sufficient.

## Test reporting

Run compileall and single-process discovery. If Windows exits with
`0xC0000005`, record the crash and run the isolated module suite. Never report
an isolated pass as a full single-process pass. Native crash remediation is a
separate task and is not part of architecture cleanup.

## Artifact safety

- No historical results or checkpoints are deleted during extraction phases.
- Generated caches and duplicates may only be removed after manifest review.
- Canonical reports/checkpoints must have an external backup before cleanup.
- No force push, main merge, or checkpoint migration occurs automatically.

## Review scope

Do not run repository-wide Black, isort, or auto-fix. Format only new files and
the smallest touched region. Comments must explain provenance, mutation, RNG,
floating-point, checkpoint, or diagnostic boundaries rather than narrate code.

