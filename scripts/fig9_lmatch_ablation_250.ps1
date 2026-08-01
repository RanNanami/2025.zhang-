param(
    [ValidateSet(4, 3, 2)]
    [int]$LMatch = 4,
    [switch]$RunGrid,
    [switch]$RunFormal250
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $RepoRoot

Write-Host "This is the formal 250-record real L_match ablation."
Write-Host "It must be started manually by the user."
Write-Host "Strict defaults remain L_match=4. L_match=1 is not supported."
if (-not $RunFormal250) {
    Write-Host ""
    Write-Host "No experiment was started."
    Write-Host "Single group: .\scripts\fig9_lmatch_ablation_250.ps1 -LMatch 4 -RunFormal250"
    Write-Host "Full grid:    .\scripts\fig9_lmatch_ablation_250.ps1 -RunGrid -RunFormal250"
    return
}

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python venv not found: $Python"
}
$env:PYTHONPATH = "$RepoRoot\src;$RepoRoot"
$env:PYTHONFAULTHANDLER = "1"
$env:PYTHONUNBUFFERED = "1"

$RootStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Root = Join-Path $RepoRoot "results\fig9_diagnostics\lmatch_ablation_250_$RootStamp"
New-Item -ItemType Directory -Path $Root -Force | Out-Null
$Requested = if ($RunGrid) { @(4, 3, 2) } else { @($LMatch) }
$Completed = [System.Collections.Generic.List[int]]::new()
$RunDirs = @{}
$RunSummaries = @()

function Invoke-LMatchRun {
    param([int]$Value)

    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
    $RunDir = Join-Path $Root "L${Value}_$Stamp"
    New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
    $RunLog = Join-Path $RunDir "run.log"
    $Checkpoint = Join-Path $RunDir "checkpoint.pkl"
    New-Item -ItemType File -Path $RunLog -Force | Out-Null

    # Scenario/full and trajectory traces depend on the existing read-only
    # oracle/provenance/preselection hooks. None of them changes selection.
    $PythonArgs = @(
        "-X", "faulthandler", "-u",
        "experiments\fig9_strict_reproduction.py",
        "--limit", "250", "--warmup", "200", "--streams", "original",
        "--prediction-horizon", "5", "--tie-break-seed", "0",
        "--l-match", "$Value", "--lmatch-real-ablation",
        "--continuous-impl", "reference",
        "--competition-mode", "competitive_raw",
        "--inhibition-strength", "0.1", "--inhibition-tau", "0.02",
        "--simultaneous-policy", "batched",
        "--simultaneous-bin-width", "0.005",
        "--intracolumn-selection-policy", "max_candidate_score",
        "--oracle-candidate-diagnostic",
        "--branch-provenance-diagnostic", "--branch-provenance-level", "candidate",
        "--preselection-segment-diagnostic", "--preselection-segment-level", "crossing",
        "--preselection-segment-compress",
        "--teacher-forced-winner-diagnostic", "--teacher-forced-winner-level", "segment",
        "--observe-scenario-diagnostic", "--observe-scenario-level", "full",
        "--match-overlap-diagnostic", "--match-overlap-level", "segment",
        "--match-overlap-compress",
        "--context-trajectory-diagnostic", "--context-trajectory-level", "column",
        "--density-trace", "--interval-every", "25",
        "--checkpoint-path", $Checkpoint, "--checkpoint-every", "50",
        "--output-dir", $RunDir
    )

    Write-Host ""
    Write-Host "Starting diagnostic L_match=$Value"
    Write-Host "Output: $RunDir"
    & $Python @PythonArgs 2>&1 | Tee-Object -FilePath $RunLog
    $RunExitCode = $LASTEXITCODE
    if ($RunExitCode -ne 0) {
        throw "L_match=$Value failed with exit code $RunExitCode. See $RunLog"
    }

    $Summary = Get-Content (Join-Path $RunDir "original_summary.json") -Raw |
        ConvertFrom-Json
    $CompetitionSummary = Get-Content (
        Join-Path $RunDir "competition_summary.json"
    ) -Raw | ConvertFrom-Json
    $StepMape = @{}
    foreach ($Step in 1..5) {
        $StepSummary = $CompetitionSummary.step_summary | Where-Object {
            [int]$_.horizon_step -eq $Step
        }
        $StepMape["step$Step"] = if ($null -ne $StepSummary) {
            [double]$StepSummary.mape
        } else { $null }
    }
    $PredictionHash = (Get-FileHash (
        Join-Path $RunDir "original_predictions.csv"
    ) -Algorithm SHA256).Hash.ToLowerInvariant()
    $script:RunSummaries += [pscustomobject]@{
        L_match = $Value
        output_dir = $RunDir
        MAPE = $Summary.mape
        per_step_MAPE = $StepMape
        coverage = $Summary.coverage
        runtime_seconds = $Summary.runtime_seconds
        predictions_SHA256 = $PredictionHash
        model_fingerprint = $Summary.final_model_fingerprint
        RNG_fingerprint = $Summary.final_rng_fingerprint
    }
    $RunDirs[$Value] = $RunDir
    $Completed.Add($Value)
    Write-Host "L=$Value MAPE=$($Summary.mape) coverage=$($Summary.coverage) runtime=$($Summary.runtime_seconds)s"
    Write-Host "Step MAPE: $($StepMape | ConvertTo-Json -Compress)"
}

try {
    foreach ($Value in $Requested) {
        Invoke-LMatchRun -Value $Value
    }
}
catch {
    Write-Error "Failed group: L_match=$Value"
    Write-Host "Completed groups: $($Completed -join ', ')"
    Write-Host "Root: $Root"
    throw
}

$SummaryPath = Join-Path $Root "run_grid_summary.json"
$RunSummaries | ConvertTo-Json -Depth 6 | Set-Content $SummaryPath -Encoding UTF8

if ($RunGrid) {
    $AnalysisDir = Join-Path $Root "analysis"
    New-Item -ItemType Directory -Path $AnalysisDir -Force | Out-Null
    $AnalysisLog = Join-Path $AnalysisDir "analysis.log"
    New-Item -ItemType File -Path $AnalysisLog -Force | Out-Null
    $AnalysisArgs = @(
        "-X", "faulthandler", "-u",
        "experiments\diagnostics\analyze_fig9_lmatch_ablation.py",
        "--run-dir-l4", $RunDirs[4],
        "--run-dir-l3", $RunDirs[3],
        "--run-dir-l2", $RunDirs[2],
        "--output-dir", $AnalysisDir,
        "--bootstrap-samples", "1000", "--bootstrap-seed", "0"
    )
    & $Python @AnalysisArgs 2>&1 | Tee-Object -FilePath $AnalysisLog
    $AnalysisExitCode = $LASTEXITCODE
    if ($AnalysisExitCode -ne 0) {
        throw "L_match offline comparison failed. See $AnalysisLog"
    }
    Write-Host "Analysis report: $(Join-Path $AnalysisDir 'FIG9_LMATCH_ABLATION_REPORT.md')"
}

Write-Host ""
Write-Host "Completed groups: $($Completed -join ', ')"
Write-Host "Run summary: $SummaryPath"
Write-Host "Root output: $Root"
exit 0
