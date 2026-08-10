# Existing Fig.9 Run Inventory

Scanned `results\fig9_diagnostics`. Found 129 run artifacts and 34 complete reference-dynamics candidates.
Only rows with expected prediction count and coverage=1 are marked `reuse_eligible`; protocol stack still requires the explicit audit below.

| L_match | limit | stack | MAPE | coverage | predictions | reusable | run directory |
|---:|---:|---|---:|---:|---:|---|---|
| 2 | 250 | DIAGNOSTIC_STACK | 0.3927571082122346 | 1.0 | 45/45 | True | `results\fig9_diagnostics\actual_branch_lineage_250_20260808_083200_L2` |
| 4 | 250 | DIAGNOSTIC_STACK | 0.4086558112184448 | 1.0 | 45/45 | True | `results\fig9_diagnostics\actual_branch_provenance_250_20260806_102928_L4` |
| 2 | 250 | DIAGNOSTIC_STACK | 0.3927571082122346 | 1.0 | 45/45 | True | `results\fig9_diagnostics\actual_branch_provenance_250_20260807_092438_L2` |
| 4 | 250 | DIAGNOSTIC_STACK | 0.4086558112184448 | 1.0 | 45/45 | True | `results\fig9_diagnostics\actual_branch_provenance_250_20260808_073849_L4` |
| 2 | 250 | DIAGNOSTIC_STACK | 0.3927571082122346 | 1.0 | 45/45 | True | `results\fig9_diagnostics\autonomous_context_provenance_250_20260809_L2` |
| 2 | 250 | DIAGNOSTIC_STACK | 0.3927571082122346 | 1.0 | 45/45 | True | `results\fig9_diagnostics\autonomous_context_provenance_250_20260809_L2_v2` |
| 4 | 250 | DIAGNOSTIC_STACK | 0.4086558112184448 | 1.0 | 45/45 | True | `results\fig9_diagnostics\autonomous_context_provenance_250_20260809_L4` |
| 4 | 250 | DIAGNOSTIC_STACK | 0.4086558112184448 | 1.0 | 45/45 | True | `results\fig9_diagnostics\autonomous_context_provenance_250_20260809_L4_v3_single_thread` |
| 2 | 250 | DIAGNOSTIC_STACK | 0.3927571082122346 | 1.0 | 45/45 | True | `results\fig9_diagnostics\context_oracle_join_20260808\formal_l2` |
| 4 | 250 | DIAGNOSTIC_STACK | 0.4086558112184448 | 1.0 | 45/45 | True | `results\fig9_diagnostics\context_oracle_join_20260808\formal_l4` |
| 2 | 250 | DIAGNOSTIC_STACK | 0.3927571082122346 | 1.0 | 45/45 | True | `results\fig9_diagnostics\context_representation_audit_20260808\L2_stream` |
| 4 | 250 | DIAGNOSTIC_STACK | 0.4086558112184448 | 1.0 | 45/45 | True | `results\fig9_diagnostics\context_representation_audit_20260808\L4_stream` |
| 4 | 250 | DIAGNOSTIC_STACK | 0.4086558112184448 | 1.0 | 45/45 | True | `results\fig9_diagnostics\independent_reference_250_20260805_150639_L4` |
| 2 | 250 | DIAGNOSTIC_STACK | 0.3927571082122346 | 1.0 | 45/45 | True | `results\fig9_diagnostics\independent_reference_250_20260806_091943_L2` |
| 2 | 500 | DIAGNOSTIC_STACK | 0.44441587347438005 | 1.0 | 295/295 | True | `results\fig9_diagnostics\l2_ambiguity_accumulation_20260810_101500\L2_ambiguity_500` |
| 4 | 500 | DIAGNOSTIC_STACK | 0.45442785109530176 | 1.0 | 295/295 | True | `results\fig9_diagnostics\l2_ambiguity_accumulation_20260810_101500\L4_ambiguity_500` |
| 2 | 500 | DIAGNOSTIC_STACK | 0.44441587347438005 | 1.0 | 295/295 | True | `results\fig9_diagnostics\l2_l4_long_sequence_20260810_025042\L2_recordwise_500` |
| 4 | 500 | DIAGNOSTIC_STACK | 0.45442785109530176 | 1.0 | 295/295 | True | `results\fig9_diagnostics\l2_l4_long_sequence_20260810_025042\L4_recordwise_500` |
| 2 | 250 | STRICT_RAW | 0.5405777765582098 | 1.0 | 45/45 | True | `results\fig9_diagnostics\l2_stack_component_250_20260810\250\P1_SELECTOR_ONLY_L2` |
| 2 | 250 | DIAGNOSTIC_STACK | 0.5352289527972052 | 1.0 | 45/45 | True | `results\fig9_diagnostics\l2_stack_component_250_20260810\250\P2_COMPETITION_ONLY_L2` |
| 2 | 250 | DIAGNOSTIC_STACK | 0.3927571082122346 | 1.0 | 45/45 | True | `results\fig9_diagnostics\lmatch_ablation_250_20260804_084050\L2_20260804_084050_135` |
| 2 | 250 | DIAGNOSTIC_STACK | 0.3927571082122346 | 1.0 | 45/45 | True | `results\fig9_diagnostics\lmatch_ablation_250_20260804_084050\L2_assembled_full_200_244` |
| 2 | 250 | DIAGNOSTIC_STACK | 0.3927571082122346 | 1.0 | 45/45 | True | `results\fig9_diagnostics\lmatch_ablation_250_20260804_084050\L2_assembled_full_200_244_failed_context_only` |
| 2 | 250 | DIAGNOSTIC_STACK | 0.3927571082122346 | 1.0 | 45/45 | True | `results\fig9_diagnostics\lmatch_ablation_250_20260804_084050\L2_assembled_full_200_244_runtime_unassembled` |
| 2 | 250 | DIAGNOSTIC_STACK | 0.3927571082122346 | 1.0 | 45/45 | True | `results\fig9_diagnostics\lmatch_ablation_250_20260804_084050\L2_assembled_full_200_244_unfiltered_warmup` |
| 4 | 250 | DIAGNOSTIC_STACK | 0.4086558112184448 | 1.0 | 45/45 | True | `results\fig9_diagnostics\lmatch_l4_native_recovery_formal_20260802_132357` |
| 2 | 250 | STRICT_RAW | 0.5902013451316054 | 1.0 | 45/45 | True | `results\fig9_diagnostics\lmatch_stack_2x2_20260810\250\STRICT_RAW_L2` |
| 4 | 250 | STRICT_RAW | 0.5054500721931258 | 1.0 | 45/45 | True | `results\fig9_diagnostics\lmatch_stack_2x2_20260810\250\STRICT_RAW_L4` |
| 2 | 500 | STRICT_RAW | 0.6206594554436795 | 1.0 | 295/295 | True | `results\fig9_diagnostics\lmatch_stack_500_completion_20260810\500\STRICT_RAW_L2` |
| 4 | 500 | STRICT_RAW | 0.5392795787736315 | 1.0 | 295/295 | True | `results\fig9_diagnostics\lmatch_stack_500_completion_20260810_resume2\500\STRICT_RAW_L4` |
| 2 | 250 | DIAGNOSTIC_STACK | 0.3927571082122346 | 1.0 | 45/45 | True | `results\fig9_diagnostics\temporal_context_structure_250_20260809_L2` |
| 4 | 250 | DIAGNOSTIC_STACK | 0.4086558112184448 | 1.0 | 45/45 | True | `results\fig9_diagnostics\temporal_context_structure_250_20260809_L4` |
| 2 | 250 | DIAGNOSTIC_STACK | 0.3927571082122346 | 1.0 | 45/45 | True | `results\fig9_diagnostics\wrong_segment_reinforcement_20260804_152649\L2` |
| 4 | 250 | DIAGNOSTIC_STACK | 0.4086558112184448 | 1.0 | 45/45 | True | `results\fig9_diagnostics\wrong_segment_reinforcement_20260804_152649\L4` |
