param(
    [ValidateSet(2, 4)] [int]$LMatch = 4,
    [switch]$RunFormal250
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
Write-Host "This formal 250 run must be started manually by the user."
if (-not $RunFormal250) {
    Write-Host "Preview only. Re-run with -RunFormal250 to start the requested run."
    exit 0
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$out = Join-Path "results/fig9_diagnostics" "independent_reference_250_$stamp`_L$LMatch"
New-Item -ItemType Directory -Force $out | Out-Null
$env:PYTHONFAULTHANDLER = "1"
$env:PYTHONUNBUFFERED = "1"
$PythonArgs = @(
    "experiments/fig9_strict_reproduction.py",
    "--limit", "250", "--warmup", "200", "--streams", "original",
    "--l-match", "$LMatch", "--lmatch-real-ablation",
    "--prediction-horizon", "5", "--tie-break-seed", "0",
    "--continuous-impl", "reference", "--competition-mode", "competitive_raw",
    "--inhibition-strength", "0.1", "--inhibition-tau", "0.02",
    "--simultaneous-policy", "batched", "--simultaneous-bin-width", "0.005",
    "--intracolumn-selection-policy", "max_candidate_score",
    "--segment-reinforcement-diagnostic", "--segment-reinforcement-level", "event",
    "--independent-reference-diagnostic", "--independent-reference-level", "segment",
    "--independent-reference-compress", "--observe-scenario-diagnostic",
    "--observe-scenario-level", "full", "--density-trace", "--interval-every", "25",
    "--progress-every", "10", "--checkpoint-every", "10", "--stream-diagnostic-traces",
    "--output-dir", $out
)
& .\.venv\Scripts\python.exe @PythonArgs 2>&1 | Tee-Object (Join-Path $out "run.log")
if ($LASTEXITCODE -ne 0) { throw "Fig.9 independent reference run failed with exit code $LASTEXITCODE" }
Write-Host "Output directory: $out"
