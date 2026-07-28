param(
    [switch]$RunFormal250
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $RepoRoot

Write-Host "This is the formal 250-record candidate separability diagnostic."
Write-Host "It must be started manually by the user."
if (-not $RunFormal250) {
    Write-Host ""
    Write-Host "No experiment was started."
    Write-Host "Run intentionally with:"
    Write-Host "  .\scripts\fig9_candidate_separability_250.ps1 -RunFormal250"
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
$Root = Join-Path $RepoRoot "results\fig9_diagnostics\candidate_score_separability_250_$Stamp"
$SequentialDir = Join-Path $Root "sequential"
$BatchedDir = Join-Path $Root "batched"
$AnalysisDir = Join-Path $Root "analysis"
New-Item -ItemType Directory -Path $Root -Force | Out-Null

function Invoke-PolicyRun {
    param(
        [string]$Policy,
        [string]$OutputDirectory
    )

    New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
    $LogPath = Join-Path $OutputDirectory "run.log"
    $CheckpointPath = Join-Path $OutputDirectory "checkpoint.pkl"
    New-Item -ItemType File -Path $LogPath -Force | Out-Null

    $PythonArgs = @(
        "-X", "faulthandler",
        "-u",
        "experiments\fig9_strict_reproduction.py",
        "--limit", "250",
        "--warmup", "200",
        "--streams", "original",
        "--output-dir", $OutputDirectory,
        "--continuous-impl", "reference",
        "--competition-mode", "competitive_raw",
        "--inhibition-strength", "0.1",
        "--inhibition-tau", "0.02",
        "--simultaneous-policy", $Policy,
        "--simultaneous-bin-width", "0.005",
        "--simultaneous-tolerance", "0.0",
        "--oracle-candidate-diagnostic",
        "--candidate-separability-trace",
        "--density-trace",
        "--interval-every", "25",
        "--checkpoint-path", $CheckpointPath,
        "--checkpoint-every", "50"
    )

    Write-Host ""
    Write-Host "Running $Policy candidate trace..."
    & $Python @PythonArgs 2>&1 | Tee-Object -FilePath $LogPath
    $ProcessExitCode = $LASTEXITCODE
    if ($ProcessExitCode -ne 0) {
        throw "$Policy run failed with exit code $ProcessExitCode. See $LogPath"
    }

    foreach ($Name in @(
        "candidate_separability_trace.csv",
        "competition_trace.csv",
        "oracle_candidate_trace.csv",
        "original_predictions.csv",
        "original_protocol.json"
    )) {
        $Path = Join-Path $OutputDirectory $Name
        if (-not (Test-Path -LiteralPath $Path)) {
            throw "$Policy run did not produce $Path"
        }
    }
}

Invoke-PolicyRun -Policy "sequential" -OutputDirectory $SequentialDir
Invoke-PolicyRun -Policy "batched" -OutputDirectory $BatchedDir

New-Item -ItemType Directory -Path $AnalysisDir -Force | Out-Null
$AnalysisLog = Join-Path $AnalysisDir "analysis.log"
New-Item -ItemType File -Path $AnalysisLog -Force | Out-Null
$AnalysisArgs = @(
    "-X", "faulthandler",
    "-u",
    "experiments\diagnostics\analyze_fig9_candidate_score_separability.py",
    "--sequential-dir", $SequentialDir,
    "--batched-dir", $BatchedDir,
    "--output-dir", $AnalysisDir,
    "--bootstrap-samples", "1000",
    "--bootstrap-seed", "0"
)

Write-Host ""
Write-Host "Running offline separability analysis..."
& $Python @AnalysisArgs 2>&1 | Tee-Object -FilePath $AnalysisLog
$ProcessExitCode = $LASTEXITCODE
if ($ProcessExitCode -ne 0) {
    throw "Offline analysis failed with exit code $ProcessExitCode. See $AnalysisLog"
}

$Report = Join-Path $AnalysisDir "CANDIDATE_SCORE_SEPARABILITY_REPORT.md"
if (-not (Test-Path -LiteralPath $Report)) {
    throw "Offline analysis did not produce $Report"
}

Write-Host ""
Write-Host "Formal candidate separability diagnostic completed."
Write-Host "Root: $Root"
Write-Host "Sequential trace: $(Join-Path $SequentialDir 'candidate_separability_trace.csv')"
Write-Host "Batched trace: $(Join-Path $BatchedDir 'candidate_separability_trace.csv')"
Write-Host "Summary: $(Join-Path $AnalysisDir 'candidate_score_separability_summary.json')"
Write-Host "Report: $Report"
Write-Host "Analysis log: $AnalysisLog"
exit 0
