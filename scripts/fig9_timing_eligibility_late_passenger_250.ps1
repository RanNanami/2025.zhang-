$ErrorActionPreference = "Stop"
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepositoryRoot

Write-Host "This is the manual 250-record late-passenger source diagnostic."
Write-Host "It does not change L_match or strict defaults."

$env:PYTHONUNBUFFERED = "1"
$env:PYTHONFAULTHANDLER = "1"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputDirectory = Join-Path `
    "results/fig9_diagnostics/timing_eligibility" `
    "late_passenger_250_$Timestamp"
$AnalysisDirectory = Join-Path $OutputDirectory "analysis"
$LogPath = Join-Path $OutputDirectory "run.log"

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
New-Item -ItemType File -Force -Path $LogPath | Out-Null

$PythonArgs = @(
    "-X", "faulthandler", "-u",
    "experiments/fig9_strict_reproduction.py",
    "--limit", "250",
    "--warmup", "200",
    "--streams", "original",
    "--continuous-impl", "reference",
    "--competition-mode", "competitive_raw",
    "--inhibition-strength", "0.1",
    "--inhibition-tau", "0.02",
    "--simultaneous-policy", "batched",
    "--simultaneous-bin-width", "0.005",
    "--intracolumn-selection-policy", "max_candidate_score",
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
    "--match-overlap-level", "source",
    "--match-overlap-compress",
    "--timing-eligibility-decomposition",
    "--source-trace-filter-field", "passenger",
    "--source-trace-filter-scenario", "scenario3",
    "--source-trace-filter-record-start", "200",
    "--source-trace-filter-record-end", "244",
    "--source-trace-filter-segments-existing-only",
    "--density-trace",
    "--interval-every", "25",
    "--checkpoint-every", "50",
    "--checkpoint-path", (Join-Path $OutputDirectory "checkpoint.pkl"),
    "--output-dir", $OutputDirectory
)

& .\.venv\Scripts\python.exe @PythonArgs 2>&1 |
    Tee-Object -FilePath $LogPath
$RunExitCode = $LASTEXITCODE
if ($RunExitCode -ne 0) {
    throw "Fig.9 diagnostic failed with exit code $RunExitCode. Log: $LogPath"
}

& .\.venv\Scripts\python.exe `
    experiments/diagnostics/analyze_fig9_timing_eligibility_decomposition.py `
    --run-dir $OutputDirectory `
    --output-dir $AnalysisDirectory `
    --bootstrap-samples 1000 `
    --bootstrap-seed 0
$AnalysisExitCode = $LASTEXITCODE
if ($AnalysisExitCode -ne 0) {
    throw "Analysis failed with exit code $AnalysisExitCode."
}

$SummaryPath = Join-Path $AnalysisDirectory "timing_eligibility_summary.json"
$Summary = Get-Content $SummaryPath -Raw | ConvertFrom-Json
$TracePath = Join-Path $OutputDirectory "match_overlap_source_trace.csv.gz"

Write-Host "Observation coverage: $($Summary.unique_observation_count) / 329"
Write-Host "Segment coverage: $($Summary.unique_segment_count) / 2437"
Write-Host "Source rows: $($Summary.source_rows)"
Write-Host "Unknown rate: $($Summary.unknown_reason_rate)"
Write-Host "Dominant reason: $($Summary.dominant_primary_reason)"
Write-Host "Recommended next step: $($Summary.recommended_next_step)"
Write-Host "Source trace bytes: $((Get-Item $TracePath).Length)"
Write-Host "Report: $(Join-Path $AnalysisDirectory 'FIG9_TIMING_ELIGIBILITY_DECOMPOSITION_REPORT.md')"
Write-Host "Output directory: $OutputDirectory"
