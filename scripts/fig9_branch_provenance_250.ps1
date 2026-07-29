param(
    [ValidateSet("sequential", "batched")]
    [string]$Policy = "batched",

    [ValidateSet("summary", "candidate", "full")]
    [string]$Level = "candidate",

    [switch]$RunFormal250
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $RepoRoot

Write-Host "This is the formal 250-record Fig.9 branch provenance diagnostic."
Write-Host "It must be started manually by the user."
if ($Level -eq "full") {
    Write-Host "Estimated full source trace size: roughly 100-250 MB."
    Write-Host "The default records every positive contributor without truncation."
}
if (-not $RunFormal250) {
    Write-Host ""
    Write-Host "No experiment was started."
    Write-Host "Run intentionally with:"
    Write-Host "  .\scripts\fig9_branch_provenance_250.ps1 -RunFormal250 -Policy $Policy -Level $Level"
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
$Root = Join-Path $RepoRoot "results\fig9_diagnostics\branch_provenance_250_${Policy}_${Level}_$Stamp"
$RunDir = Join-Path $Root "run"
$AnalysisDir = Join-Path $Root "analysis"
$RunLog = Join-Path $RunDir "run.log"
$AnalysisLog = Join-Path $AnalysisDir "analysis.log"
$Checkpoint = Join-Path $RunDir "checkpoint.pkl"
New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
New-Item -ItemType Directory -Path $AnalysisDir -Force | Out-Null
New-Item -ItemType File -Path $RunLog -Force | Out-Null
New-Item -ItemType File -Path $AnalysisLog -Force | Out-Null

$PythonArgs = @(
    "-X", "faulthandler",
    "-u",
    "experiments\fig9_strict_reproduction.py",
    "--limit", "250",
    "--warmup", "200",
    "--streams", "original",
    "--output-dir", $RunDir,
    "--continuous-impl", "reference",
    "--competition-mode", "competitive_raw",
    "--inhibition-strength", "0.1",
    "--inhibition-tau", "0.02",
    "--simultaneous-policy", $Policy,
    "--simultaneous-bin-width", "0.005",
    "--simultaneous-tolerance", "0.0",
    "--oracle-candidate-diagnostic",
    "--branch-provenance-diagnostic",
    "--branch-provenance-level", $Level,
    "--density-trace",
    "--interval-every", "25",
    "--checkpoint-path", $Checkpoint,
    "--checkpoint-every", "50"
)

Write-Host ""
Write-Host "Running $Policy provenance capture at level $Level..."
& $Python @PythonArgs 2>&1 | Tee-Object -FilePath $RunLog
$ProcessExitCode = $LASTEXITCODE
if ($ProcessExitCode -ne 0) {
    throw "Provenance run failed with exit code $ProcessExitCode. See $RunLog"
}

$AnalysisArgs = @(
    "-X", "faulthandler",
    "-u",
    "experiments\diagnostics\analyze_fig9_branch_provenance.py",
    "--run-dir", $RunDir,
    "--output-dir", $AnalysisDir,
    "--policy-label", $Policy,
    "--bootstrap-samples", "1000",
    "--bootstrap-seed", "0"
)

Write-Host ""
Write-Host "Running offline branch provenance analysis..."
& $Python @AnalysisArgs 2>&1 | Tee-Object -FilePath $AnalysisLog
$ProcessExitCode = $LASTEXITCODE
if ($ProcessExitCode -ne 0) {
    throw "Offline analysis failed with exit code $ProcessExitCode. See $AnalysisLog"
}

Write-Host ""
Write-Host "Output file sizes:"
Get-ChildItem -LiteralPath $RunDir -File |
    Sort-Object Name |
    Select-Object Name, Length |
    Format-Table -AutoSize

$Report = Join-Path $AnalysisDir "FIG9_BRANCH_PROVENANCE_REPORT.md"
if (-not (Test-Path -LiteralPath $Report)) {
    throw "Offline analysis did not produce $Report"
}

Write-Host ""
Write-Host "Formal branch provenance diagnostic completed."
Write-Host "Root: $Root"
Write-Host "Run log: $RunLog"
Write-Host "Analysis log: $AnalysisLog"
Write-Host "Report: $Report"
exit 0
