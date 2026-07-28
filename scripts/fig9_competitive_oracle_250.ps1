$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $RepoRoot

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python venv not found: $Python"
}

$env:PYTHONPATH = "$RepoRoot\src;$RepoRoot"
$env:PYTHONFAULTHANDLER = "1"
$env:PYTHONUNBUFFERED = "1"

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutDir = Join-Path $RepoRoot "results\fig9_diagnostics\competitive_oracle_250_$Stamp"
$CheckpointPath = Join-Path $OutDir "checkpoint.pkl"
$LogPath = Join-Path $OutDir "run.log"
$OracleSummaryPath = Join-Path $OutDir "oracle_candidate_summary.json"

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
New-Item -ItemType File -Path $LogPath -Force | Out-Null

$PythonArgs = @(
    "experiments\fig9_strict_reproduction.py",
    "--limit", "250",
    "--warmup", "200",
    "--streams", "original",
    "--output-dir", $OutDir,
    "--competition-mode", "competitive_raw",
    "--inhibition-strength", "0.1",
    "--inhibition-tau", "0.02",
    "--simultaneous-tolerance", "0.0",
    "--oracle-candidate-diagnostic",
    "--density-trace",
    "--interval-every", "25",
    "--checkpoint-path", $CheckpointPath,
    "--checkpoint-every", "50",
    "--continuous-impl", "reference"
)

Write-Host "Running the manual 250-record competitive oracle diagnostic."
Write-Host "Output directory: $OutDir"
Write-Host "Log: $LogPath"

& $Python @PythonArgs 2>&1 | Tee-Object -FilePath $LogPath
$ProcessExitCode = $LASTEXITCODE
if ($ProcessExitCode -ne 0) {
    Write-Error "Fig.9 oracle diagnostic failed with exit code $ProcessExitCode. See $LogPath"
    exit $ProcessExitCode
}

if (-not (Test-Path -LiteralPath $OracleSummaryPath)) {
    Write-Error "Run completed without the expected oracle summary: $OracleSummaryPath"
    exit 2
}

Write-Host ""
Write-Host "Fig.9 competitive oracle diagnostic completed."
Write-Host "Oracle summary: $OracleSummaryPath"
Write-Host "Oracle trace: $(Join-Path $OutDir 'oracle_candidate_trace.csv')"
Write-Host "Checkpoint: $CheckpointPath"
Write-Host "Log: $LogPath"
exit 0
