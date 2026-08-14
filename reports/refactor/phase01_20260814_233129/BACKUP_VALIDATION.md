# Backup Validation

## Git backups

- Backup branch: `backup/pre-refactor-20260814`
- Backup branch target: `1864f46aee5cc0aee814fd4e6154b06f8bfeb08f`
- Annotated tag: `pre-refactor-architecture-cleanup-v2`
- Tag object: `53e1c39dfee30797ac48402bf6a53cc8ad8b435d`
- Tag peeled commit: `1864f46aee5cc0aee814fd4e6154b06f8bfeb08f`
- Git bundle: `C:\Users\苒苒\Documents\repo_backups\2025.zhang-_pre_refactor_20260814_1864f46.bundle`
- Bundle verification: `PASS`
- Bundle scope: complete history, nine refs, SHA-1 object format

`git bundle verify` reported that the bundle records a complete history and is
valid. Both the backup branch and tag resolve to the frozen scientific commit.

## Critical untracked-artifact backup

- Staging directory: `C:\Users\苒苒\Documents\repo_backups\pre_refactor_critical_artifacts_20260814_233129`
- ZIP archive: `C:\Users\苒苒\Documents\repo_backups\pre_refactor_critical_artifacts_20260814_233129.zip`
- ZIP SHA256: `37125B33CBC3FC239F45E417CDC96A597061E4A903FA90A275DC579A5C819713`
- ZIP entries: `63`
- Uncompressed source bytes: `28,820,041`
- ZIP bytes: `3,050,141`
- ZIP read validation: `PASS`

Protected selections:

1. Fig.9 250-record strict raw L_match=4 checkpoint, protocol, predictions,
   summaries, runtime metadata, and logs.
2. Fig.9 250-record strict raw L_match=2 checkpoint, protocol, predictions,
   summaries, runtime metadata, and logs.
3. The final 20-sentence temporal-confirmation semantic audit, traces, reports,
   and conclusion archive.

The complete untracked tree remains untouched in the repository workspace.

## Recovery commands

Restore committed history from the local branch:

```powershell
git switch backup/pre-refactor-20260814
```

Restore the exact frozen commit without switching branches:

```powershell
git worktree add ..\TorF_reproduction_frozen 1864f46aee5cc0aee814fd4e6154b06f8bfeb08f
```

Clone from the external bundle:

```powershell
git clone C:\Users\苒苒\Documents\repo_backups\2025.zhang-_pre_refactor_20260814_1864f46.bundle ..\TorF_reproduction_bundle_restore
```

Extract critical untracked artifacts:

```powershell
Expand-Archive -LiteralPath C:\Users\苒苒\Documents\repo_backups\pre_refactor_critical_artifacts_20260814_233129.zip -DestinationPath ..\TorF_reproduction_critical_restore
```

