param(
    [switch]$RunFormal250
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $RepoRoot

Write-Host "This is the manual 250-record batched competition diagnostic."
Write-Host "It must be started manually by the user."
if (-not $RunFormal250) {
    Write-Host ""
    Write-Host "No experiment was started."
    Write-Host "Run intentionally with:"
    Write-Host "  .\scripts\fig9_competitive_batched_250.ps1 -RunFormal250"
    return
}

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python venv not found: $Python"
}

$env:PYTHONPATH = "$RepoRoot\src;$RepoRoot"
$env:PYTHONFAULTHANDLER = "1"
$env:PYTHONUNBUFFERED = "1"

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutDir = Join-Path $RepoRoot "results\fig9_diagnostics\competitive_batched_250_$Stamp"
$CheckpointPath = Join-Path $OutDir "checkpoint.pkl"
$LogPath = Join-Path $OutDir "run.log"
$ComparisonPath = Join-Path $OutDir "competition_policy_comparison.json"

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
New-Item -ItemType File -Path $LogPath -Force | Out-Null

$PythonArgs = @(
    "-X", "faulthandler",
    "-u",
    "experiments\fig9_strict_reproduction.py",
    "--limit", "250",
    "--warmup", "200",
    "--streams", "original",
    "--output-dir", $OutDir,
    "--continuous-impl", "reference",
    "--competition-mode", "competitive_raw",
    "--inhibition-strength", "0.1",
    "--inhibition-tau", "0.02",
    "--simultaneous-policy", "batched",
    "--simultaneous-bin-width", "0.005",
    "--oracle-candidate-diagnostic",
    "--density-trace",
    "--interval-every", "25",
    "--checkpoint-path", $CheckpointPath,
    "--checkpoint-every", "50"
)

Write-Host "Output directory: $OutDir"
Write-Host "Log: $LogPath"

& $Python @PythonArgs 2>&1 | Tee-Object -FilePath $LogPath
$ProcessExitCode = $LASTEXITCODE
if ($ProcessExitCode -ne 0) {
    Write-Error "Batched diagnostic failed with exit code $ProcessExitCode. See $LogPath"
    exit $ProcessExitCode
}

$Required = @(
    (Join-Path $OutDir "original_summary.json"),
    (Join-Path $OutDir "competition_summary.json"),
    (Join-Path $OutDir "oracle_candidate_summary.json")
)
foreach ($Path in $Required) {
    if (-not (Test-Path -LiteralPath $Path)) {
        Write-Error "Run completed without expected output: $Path"
        exit 2
    }
}

Write-Host ""
Write-Host "Manual batched 250 diagnostic completed."
Write-Host "Original summary: $(Join-Path $OutDir 'original_summary.json')"
Write-Host "Competition summary: $(Join-Path $OutDir 'competition_summary.json')"
Write-Host "Oracle summary: $(Join-Path $OutDir 'oracle_candidate_summary.json')"
Write-Host "Comparison path after running the comparison script: $ComparisonPath"
Write-Host "Checkpoint: $CheckpointPath"
Write-Host "Log: $LogPath"
exit 0
