# Pre-refactor Worktree Audit

## Freeze point

- Repository: `C:\Users\苒苒\Documents\TorF_reproduction`
- Original branch: `experiment/fig9-competitive-inhibition`
- Frozen commit: `1864f46aee5cc0aee814fd4e6154b06f8bfeb08f`
- Frozen subject: `Fix temporal confirmation audit counters`
- Refactor branch: `refactor/architecture-cleanup-v2`
- Audit timestamp: `2026-08-14T23:31:29+08:00`

The refactor branch was created directly from the frozen commit. No scientific
source file was changed before the freeze point was recorded.

## Worktree before Phase 0 files

- Tracked modifications: `0`
- Untracked paths: `9,839`
- Untracked paths below `results/`: `9,835`
- Untracked root log files: `3`
- Other untracked paths below `tools/`: `1`

The untracked set is dominated by experiment outputs and diagnostic traces.
Its complete path-level classification is recorded in
`PRE_REFACTOR_ARTIFACT_MANIFEST.csv`; this document records the immutable count
observed before any Phase 0 report file was created.

## Preservation decision

No untracked artifact was deleted, renamed, or overwritten. The Git bundle
protects all committed history. A separate critical-artifact archive protects
the selected canonical checkpoints, protocols, predictions, and the latest
temporal-confirmation conclusion package that were not in Git.

The remaining untracked outputs stay in place and are classified as
`IMPORTANT`, `REGENERABLE`, or `TEMPORARY` in the artifact manifest. The archive
is intentionally selective because the untracked tree occupies about 18 GB and
contains many repeated raw traces and debug runs.

## Safety assertions

- Scientific behavior changed: `NO`
- Core refactor started: `NO`
- Historical results deleted: `NO`
- Main merged: `NO`
- Native crash declared fixed: `NO`
- Formal experiment continued: `NO`

