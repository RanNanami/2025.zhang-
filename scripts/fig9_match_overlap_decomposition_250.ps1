param(
    [ValidateSet("max_candidate_score", "existing")]
    [string]$Selector = "max_candidate_score",
    [ValidateSet("segment", "source")]
    [string]$Level = "segment"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $RepoRoot

Write-Host "This is the formal 250-record match-overlap diagnostic."
Write-Host "It must be started manually by the user."
Write-Host "It does not modify L_match or run a selector grid."

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python venv not found: $Python"
}

$env:PYTHONPATH = "$RepoRoot\src;$RepoRoot"
$env:PYTHONFAULTHANDLER = "1"
$env:PYTHONUNBUFFERED = "1"

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Root = Join-Path $RepoRoot (
    "results\fig9_diagnostics\match_overlap_250_${Selector}_${Level}_$Stamp"
)
$RunDir = Join-Path $Root "run"
$AnalysisDir = Join-Path $Root "analysis"
New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
New-Item -ItemType Directory -Path $AnalysisDir -Force | Out-Null

$RunLog = Join-Path $RunDir "run.log"
$AnalysisLog = Join-Path $AnalysisDir "analysis.log"
$Checkpoint = Join-Path $RunDir "checkpoint.pkl"
New-Item -ItemType File -Path $RunLog -Force | Out-Null
New-Item -ItemType File -Path $AnalysisLog -Force | Out-Null

# Teacher/Scenario traces require their existing provenance and preselection
# dependencies. These flags only add diagnostic outputs.
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
    "--simultaneous-policy", "batched",
    "--simultaneous-bin-width", "0.005",
    "--intracolumn-selection-policy", $Selector,
    "--oracle-candidate-diagnostic",
    "--branch-provenance-diagnostic",
    "--branch-provenance-level", "candidate",
    "--preselection-segment-diagnostic",
    "--preselection-segment-level", "crossing",
    "--preselection-segment-compress",
    "--teacher-forced-winner-diagnostic",
    "--teacher-forced-winner-level", "segment",
    "--observe-scenario-diagnostic",
    "--observe-scenario-level", "full",
    "--match-overlap-diagnostic",
    "--match-overlap-level", $Level,
    "--match-overlap-compress",
    "--density-trace",
    "--interval-every", "25",
    "--checkpoint-path", $Checkpoint,
    "--checkpoint-every", "50"
)

Write-Host ""
Write-Host "Output directory: $Root"
Write-Host "Selector: $Selector"
Write-Host "Trace level: $Level"
& $Python @PythonArgs 2>&1 | Tee-Object -FilePath $RunLog
$RunExitCode = $LASTEXITCODE
if ($RunExitCode -ne 0) {
    throw "Fig.9 run failed with exit code $RunExitCode. See $RunLog"
}

$AnalysisArgs = @(
    "-X", "faulthandler",
    "-u",
    "experiments\diagnostics\analyze_fig9_match_overlap_decomposition.py",
    "--run-dir", $RunDir,
    "--output-dir", $AnalysisDir,
    "--bootstrap-samples", "1000",
    "--bootstrap-seed", "0"
)
& $Python @AnalysisArgs 2>&1 | Tee-Object -FilePath $AnalysisLog
$AnalysisExitCode = $LASTEXITCODE
if ($AnalysisExitCode -ne 0) {
    throw "Offline analysis failed with exit code $AnalysisExitCode. See $AnalysisLog"
}

$Files = Get-ChildItem -LiteralPath $Root -File -Recurse |
    Select-Object FullName, Length
$Files | Export-Csv (
    Join-Path $Root "output_file_sizes.csv"
) -NoTypeInformation

$SummaryPath = Join-Path $AnalysisDir "match_overlap_summary.json"
$ReportPath = Join-Path (
    $AnalysisDir
) "FIG9_MATCH_OVERLAP_DECOMPOSITION_REPORT.md"

Write-Host ""
Write-Host "Formal diagnostic finished."
Write-Host "Run directory: $RunDir"
Write-Host "Analysis directory: $AnalysisDir"
Write-Host "Summary: $SummaryPath"
Write-Host "Report: $ReportPath"
Write-Host "File sizes: $(Join-Path $Root 'output_file_sizes.csv')"
exit 0
