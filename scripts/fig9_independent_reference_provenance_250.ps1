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
    "--oracle-candidate-diagnostic",
    "--branch-provenance-diagnostic", "--branch-provenance-level", "candidate",
    "--preselection-segment-diagnostic", "--preselection-segment-level", "crossing",
    "--teacher-forced-winner-diagnostic", "--teacher-forced-winner-level", "segment",
    "--segment-reinforcement-diagnostic", "--segment-reinforcement-level", "event",
    "--independent-reference-diagnostic", "--independent-reference-level", "segment",
    "--independent-reference-compress", "--observe-scenario-diagnostic",
    "--observe-scenario-level", "full", "--density-trace", "--interval-every", "25",
    "--progress-every", "10", "--checkpoint-every", "10", "--stream-diagnostic-traces",
    "--output-dir", $out
)
$python = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$runner = (Resolve-Path "scripts\run_logged_process.py").Path
$stdoutLog = Join-Path $out "stdout.log"
$stderrLog = Join-Path $out "stderr.log"
$combinedLog = Join-Path $out "combined.log"
$resultJson = Join-Path $out "result.json"
$failureJson = Join-Path $out "failure.json"
$checkpoint = Join-Path $out "checkpoint.pkl"
$command = @($python) + $PythonArgs
@{
    command = $command
    python_args = $PythonArgs
    working_directory = (Get-Location).Path
    environment = @{
        PYTHONFAULTHANDLER = $env:PYTHONFAULTHANDLER
        PYTHONUNBUFFERED = $env:PYTHONUNBUFFERED
    }
} | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $out "command.json") -Encoding UTF8

# Do not pipe a native process through PowerShell. The Python supervisor owns
# both pipes and preserves the real child exit code and complete traceback.
& $python $runner `
    --stdout-log $stdoutLog `
    --stderr-log $stderrLog `
    --combined-log $combinedLog `
    --result-json $resultJson `
    --failure-json $failureJson `
    --checkpoint $checkpoint `
    -- $command
$runnerExitCode = $LASTEXITCODE
if ($runnerExitCode -ne 0) {
    $failure = if (Test-Path $failureJson) { Get-Content $failureJson -Raw | ConvertFrom-Json } else { $null }
    if ($failure) { Write-Host ("Child exit code: " + $failure.exit_code) }
    throw "Fig.9 independent reference runner failed; see $stderrLog and $failureJson"
}
Write-Host "Output directory: $out"
