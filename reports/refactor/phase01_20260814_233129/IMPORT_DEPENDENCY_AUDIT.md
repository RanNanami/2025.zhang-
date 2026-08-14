# Import Dependency Audit

- Tracked Python files parsed: `152`
- Parse errors: `0`
- Local import edges: `382`
- Core to non-core violations: `0`
- Fig.9 strict to diagnostics edges: `18`
- Active to historical edges: `1`

## Findings

The `src/seqmem` core has no import edge into `experiments`, diagnostics, or
historical code. This boundary is already healthy and should become an enforced
architecture test.

The Fig.9 strict runner directly imports many diagnostic modules. Even though
their CLI flags default off, this makes strict startup, auditability, and failure
isolation depend on nonpaper code. Future runner decomposition should invert this
relationship after golden baselines exist; this phase only records it.

## Strict diagnostic dependencies

- `experiments/diagnostics/fig9_competitive_inhibition.py` (line 50)
- `experiments/diagnostics/fig9_oracle_candidate.py` (line 58)
- `experiments/diagnostics/fig9_candidate_score_trace.py` (line 63)
- `experiments/diagnostics/fig9_candidate_context_oracle.py` (line 67)
- `experiments/diagnostics/fig9_branch_provenance.py` (line 72)
- `experiments/diagnostics/fig9_preselection_segments.py` (line 79)
- `experiments/diagnostics/fig9_teacher_forced_identity.py` (line 84)
- `experiments/diagnostics/fig9_intracolumn_selector.py` (line 93)
- `experiments/diagnostics/fig9_readout_dynamics.py` (line 101)
- `experiments/diagnostics/fig9_match_overlap.py` (line 106)
- `experiments/diagnostics/fig9_segment_context_composition.py` (line 114)
- `experiments/diagnostics/fig9_context_trajectory.py` (line 120)
- `experiments/diagnostics/fig9_temporal_context.py` (line 125)
- `experiments/diagnostics/fig9_autonomous_context_provenance.py` (line 129)
- `experiments/diagnostics/fig9_segment_reinforcement.py` (line 133)
- `experiments/diagnostics/fig9_ambiguity.py` (line 139)
- `experiments/diagnostics/fig9_independent_reference.py` (line 145)
- `experiments/diagnostics/fig9_actual_branch_provenance.py` (line 153)

## Parse errors

None.
