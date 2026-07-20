# How to Run

## Open the Folder

```powershell
# Open PowerShell in the cloned repository.
code .
```

If `code` is unavailable, open VS Code and select this folder with
`File -> Open Folder`.

## Python Environment

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Use the project interpreter for every command:

```text
.\.venv\Scripts\python.exe
```

In VS Code, choose the same path with `Python: Select Interpreter`.

## Verify First

```powershell
.\check_project.ps1
```

This compiles the source, runs the paper-alignment unit tests, checks the exact
dataset sizes, and performs short Fig.7/8/9 smoke runs.

## Full Reproduction

```powershell
.\run_all.ps1
```

This is a multi-hour command. Results are written under `results/latest_strict`
and a timestamped `results/strict_YYYYMMDD_HHMMSS` directory.

Individual commands are listed in `README.md`. The strict result interpretation
and unresolved assumptions are in `PAPER_ALIGNMENT.md`.
