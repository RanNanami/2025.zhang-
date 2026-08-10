# Fig.9 L_match x Diagnostic-Stack Protocol Audit

This audit is generated from the strict configuration source and the explicit cell definitions.
It is a protocol document, not an experimental result.

## Source Path

`experiments/fig9_strict_reproduction.py::Fig9StrictConfig` is the source of the common strict values.
The strict/raw path uses `competition_mode=off`, `simultaneous_policy=sequential`, and `intracolumn_selection_policy=existing`.
The diagnostic stack is enabled only by explicit CLI settings: `competitive_raw`, `batched`, and `max_candidate_score`.

## Matrix

| Cell | L_match | Stack | Competition | Simultaneous | Selector | Strict reproduction |
|---|---:|---|---|---|---|---|
| STRICT_RAW_L4 | 4 | STRICT_RAW | off | sequential | existing | True |
| STRICT_RAW_L2 | 2 | STRICT_RAW | off | sequential | existing | False |
| DIAGNOSTIC_STACK_L4 | 4 | DIAGNOSTIC_STACK | competitive_raw | batched | max_candidate_score | False |
| DIAGNOSTIC_STACK_L2 | 2 | DIAGNOSTIC_STACK | competitive_raw | batched | max_candidate_score | False |

## Isolation Checks

- STRICT_RAW_L4 vs STRICT_RAW_L2 differs in model mechanism only by `L_match: 4 -> 2`; the L2 row is explicitly non-strict.
- DIAGNOSTIC_STACK_L4 vs DIAGNOSTIC_STACK_L2 differs in model mechanism only by `L_match: 4 -> 2`.
- STRICT_RAW_L4 vs DIAGNOSTIC_STACK_L4 differs by the explicitly labeled experimental competition/selection stack.
- STRICT_RAW_L2 vs DIAGNOSTIC_STACK_L2 has the same stack distinction and keeps the same L2 ablation label.
- All cells use raw propagation, continuous reference dynamics, all-cell burst context, K=10, 32 neurons/column, forgetting threshold 65, seed 0, original stream, and no rollout learning.
- No cell uses compensation, future covariates, ground-truth selection, decoded replay, or target correction.

## Naming Rule

`STRICT_RAW_L4` is the strict reference. `STRICT_RAW_L2` is a single-factor nonpaper ablation, never a strict paper reproduction.
