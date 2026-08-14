# Strict Default Diagnostic Import Diff

## Measurement

The measurement imports `experiments.fig9_strict_reproduction` in a fresh
interpreter and lists `sys.modules` entries under `experiments.diagnostics`.
It does not enable a diagnostic flag.

| Measure | Before | After |
| --- | ---: | ---: |
| Direct strict-to-diagnostic imports | 18 | 1 |
| Loaded Fig.9 diagnostic implementation modules | 18 | 1 |
| Loaded entries including package | 19 | 2 |

## Removed From Default Import Time

- actual branch provenance
- ambiguity
- autonomous context provenance
- branch provenance
- candidate context oracle
- candidate score trace
- context trajectory
- independent reference
- intracolumn selector trace
- match overlap
- oracle candidate
- preselection segments
- readout dynamics
- segment context composition
- segment reinforcement
- teacher-forced identity
- temporal context

The modules were not deleted. Their existing implementation is loaded through
small cached functions in `experiments/fig9/diagnostic_loader.py` only when the
corresponding flag or compatibility state requires it.

## Deliberately Retained

`fig9_competitive_inhibition` remains a direct dependency. It owns both the
disabled `CompetitionSettings` used by the strict runner and a nonpaper
prediction transform used when `--competition-mode competitive_raw` is
explicitly selected. Classifying it as pure diagnostic IO would be false, so
Phase 03 does not move or rewrite it.

## Disabled Tracker Compatibility

Ambiguity, independent-reference, and actual-branch modules originally created
empty tracker objects even while disabled. Before deferring those imports, the
exact disabled payloads were measured. The runner now writes the same ordered
empty payload dictionaries without importing tracker implementations. Enabled
runs still instantiate the original classes through the loader. All six golden
fixtures and the relevant checkpoint tests remained exact.

## Scientific Boundary

No CLI name or default changed. The loader has no model reference, callback,
ground truth, future data, writer handle, or RNG. Importing a diagnostic family
does not run it. Prediction, learning, rollout, Scenario logic, and checkpoint
class paths remain in their original owners.
