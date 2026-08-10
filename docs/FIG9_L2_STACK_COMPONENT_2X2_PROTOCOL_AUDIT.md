# Fig.9 L2 Stack Component 2x2 Protocol Audit

All four cells use L_match=2, raw propagation, reference continuous dynamics, seed=0, warmup=200, K=10, 32 neurons/column, forgetting threshold=65, no compensation, no future covariates, no ground-truth selection, no rollout learning, and no decoded replay.

| Cell | Competition | Simultaneous | Selector | Label |
|---|---|---|---|---|
| P0_RAW_L2 | off | sequential | existing | nonpaper raw L_match=2 baseline |
| P1_SELECTOR_ONLY_L2 | off | sequential | max_candidate_score | selector-only diagnostic |
| P2_COMPETITION_ONLY_L2 | competitive_raw | batched | existing | competition-only diagnostic |
| P3_FULL_STACK_L2 | competitive_raw | batched | max_candidate_score | full diagnostic stack |

P0/P1 differ only by selector. P0/P2 differ only by competition stack. P1/P3 differ only by competition stack. P2/P3 differ only by selector. P0 and P3 are reused from the completed L2 raw/full-stack 250 runs.
