# Fig.9 Import-Time Diagnostic Audit

## Scope

This audit covers the 18 direct imports from
`experiments/fig9_strict_reproduction.py` into `experiments/diagnostics` at
commit `fe80e91`. It distinguishes import-time coupling from scientific
runtime dependence. No implementation changed during this audit.

## Baseline

- Strict runner LOC: 6,040.
- Direct strict-to-diagnostic module dependencies: 18.
- Diagnostic modules present after importing the strict runner: 19, including
  the `experiments.diagnostics` package.
- All 18 direct modules are loaded even when no diagnostic CLI flag is enabled.
- AST inspection found no top-level callback registration or model mutation in
  these 18 modules. Their imports still load formatting, analysis, gzip, and
  provenance code that the default invocation does not request.

The exact baseline module list is in
`STRICT_DEFAULT_IMPORTED_MODULES_BEFORE.txt`. Symbol-level classification is
in `FIG9_DIAGNOSTIC_DEPENDENCY_MAP.csv`.

## Default-Path Coupling

Four families need extra care because the current runner touches them even
with their diagnostic flag off:

1. `fig9_competitive_inhibition` owns `CompetitionSettings`, validates the
   default disabled competition configuration, and owns a nonpaper prediction
   transform when enabled. It is scientific-risk code, not pure output code.
2. `fig9_ambiguity` constructs an empty `AmbiguityReuseState` and serializes
   its diagnostic payload into checkpoints.
3. `fig9_independent_reference` constructs an empty tracker and serializes its
   diagnostic payload into checkpoints.
4. `fig9_actual_branch_provenance` constructs an empty tracker and serializes
   its diagnostic payload into checkpoints.

`fig9_intracolumn_selector.mark_competition_outcomes()` is also called on the
default path, but only with an empty diagnostic row list when the trace is off.
That call can be guarded only if exact state and output parity proves it inert.

These facts mean that moving all 18 imports at once would be unsafe. In
particular, replacing the three empty trackers with `None` could change the
checkpoint payload even if predictions remained identical.

## Safe Initial Lazy-Load Candidates

The first candidates are modules whose calls are already dominated by an
explicit false-by-default CLI flag: oracle candidate reporting, candidate
score trace, context-oracle joins, branch provenance, preselection trace,
teacher-forced reporting, readout trace, match-overlap trace, context
trajectory, temporal context, autonomous context provenance, segment
reinforcement, and segment-context composition.

The loader must be explicit and small. It must not register callbacks, retain
model references, inspect ground truth, or create output handles merely by
loading a family.

## Ground-Truth Boundary

Oracle and teacher-forced modules use expected/future observations only to
label or analyze captured rows. They are default-off. Importing them currently
does not register a hook, but unconditional imports make that boundary hard to
audit. Lazy loading will make the enabled flag the visible boundary. Ground
truth must never be passed into candidate selection, prediction, learning, or
RNG decisions.

## Deferred Items

- Competition remains `DEFERRED_SCIENTIFIC_RISK` until its configuration and
  prediction-transform ownership can be separated without changing behavior.
- Ambiguity, independent-reference, and actual-branch tracker initialization
  remain `DEFERRED_CHECKPOINT_COUPLING` until their disabled-state checkpoint
  bytes are characterized.
- No posthoc analyzer is directly imported by the strict runner. Analyzer
  scripts are already invoked after artifact production and do not need to be
  moved in this phase.

## Acceptance Evidence Required

Every import migration must preserve all six Phase 01 golden fixtures exactly.
Representative diagnostic-on fixtures must additionally prove scientific
ON/OFF parity and diagnostic artifact compatibility. A mismatch in prediction,
science state, RNG state, Scenario counts, or segment/synapse counts stops the
phase.
