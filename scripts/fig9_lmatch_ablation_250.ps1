param(
    [ValidateSet(4, 3, 2)]
    [int]$LMatch = 4,
    [switch]$RunGrid,
    [switch]$RunFormal250,
    [string]$ResumeRunDirectory = "",
    [switch]$EnableProcDump,
    [string]$ProcDumpPath = ""
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $RepoRoot

Write-Host "This is the formal 250-record real L_match ablation."
Write-Host "It must be started manually by the user."
Write-Host "Strict defaults remain L_match=4. L_match=1 is not supported."
if (-not $RunFormal250) {
    Write-Host "No experiment was started. Add -RunFormal250 explicitly."
    return
}
if ($ResumeRunDirectory -and $RunGrid) {
    throw "ResumeRunDirectory resumes exactly one group; do not combine it with RunGrid."
}
if ($EnableProcDump -and -not $ProcDumpPath) {
    throw "EnableProcDump requires -ProcDumpPath pointing to procdump.exe."
}

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { throw "Python venv not found: $Python" }
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

function New-CurrentRunArchive {
    param([string]$RunDir)
    $Archive = "$RunDir.zip"
    if (Test-Path -LiteralPath $Archive) { Remove-Item -LiteralPath $Archive -Force }
    Compress-Archive -Path (Join-Path $RunDir "*") -DestinationPath $Archive
    return $Archive
}

function Invoke-LMatchRun {
    param([int]$Value)
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
    $RunDir = Join-Path $Root "L${Value}_$Stamp"
    New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
    $RunLog = Join-Path $RunDir "run.log"
    $Checkpoint = Join-Path $RunDir "checkpoint.pkl"
    $FailureJson = Join-Path $RunDir "failure.json"
    $ProcessResult = Join-Path $RunDir "process_result.json"
    New-Item -ItemType File -Path $RunLog -Force | Out-Null

    $ResumeArgs = @()
    if ($ResumeRunDirectory) {
        $SourceRun = (Resolve-Path $ResumeRunDirectory).Path
        $SourceCheckpoint = Join-Path $SourceRun "checkpoint.pkl"
        if (-not (Test-Path -LiteralPath $SourceCheckpoint)) {
            throw "Resume checkpoint not found: $SourceCheckpoint"
        }
        $Inspection = & $Python "experiments\diagnostics\inspect_fig9_checkpoint.py" $SourceCheckpoint |
            ConvertFrom-Json
        if ([int]$Inspection.L_match -ne $Value) {
            throw "Checkpoint L_match=$($Inspection.L_match), requested L_match=$Value"
        }
        $SourceFailure = Join-Path $SourceRun "failure.json"
        if (Test-Path -LiteralPath $SourceFailure) {
            $FailureState = Get-Content $SourceFailure -Raw | ConvertFrom-Json
            if ([int]$FailureState.last_completed_index -gt [int]$Inspection.next_index) {
                throw "Refusing to replay records $($Inspection.next_index)..$([int]$FailureState.last_completed_index - 1). A checkpoint at the last completed index is required."
            }
        }
        Copy-Item -LiteralPath $SourceCheckpoint -Destination $Checkpoint
        $SourceSidecar = "$SourceCheckpoint.branch_provenance.json"
        if (Test-Path -LiteralPath $SourceSidecar) {
            Copy-Item -LiteralPath $SourceSidecar -Destination "$Checkpoint.branch_provenance.json"
        }
        $Inspection | ConvertTo-Json -Depth 6 |
            Set-Content (Join-Path $RunDir "resume_source.json") -Encoding UTF8
        $ResumeArgs = @("--resume-from", $Checkpoint)
    }

    $PythonArgs = @(
        "-X", "faulthandler", "-u", "experiments\fig9_strict_reproduction.py",
        "--limit", "250", "--warmup", "200", "--streams", "original",
        "--prediction-horizon", "5", "--tie-break-seed", "0",
        "--l-match", "$Value", "--lmatch-real-ablation",
        "--continuous-impl", "reference", "--competition-mode", "competitive_raw",
        "--inhibition-strength", "0.1", "--inhibition-tau", "0.02",
        "--simultaneous-policy", "batched", "--simultaneous-bin-width", "0.005",
        "--intracolumn-selection-policy", "max_candidate_score",
        "--oracle-candidate-diagnostic",
        "--branch-provenance-diagnostic", "--branch-provenance-level", "candidate",
        "--preselection-segment-diagnostic", "--preselection-segment-level", "crossing",
        "--preselection-segment-compress", "--teacher-forced-winner-diagnostic",
        "--teacher-forced-winner-level", "segment", "--observe-scenario-diagnostic",
        "--observe-scenario-level", "full", "--match-overlap-diagnostic",
        "--match-overlap-level", "segment", "--match-overlap-compress",
        "--context-trajectory-diagnostic", "--context-trajectory-level", "column",
        "--density-trace", "--interval-every", "25", "--progress-every", "10",
        "--stream-diagnostic-traces", "--checkpoint-path", $Checkpoint,
        "--checkpoint-every", "10", "--output-dir", $RunDir
    ) + $ResumeArgs

    Write-Host "Starting diagnostic L_match=$Value"
    Write-Host "Output: $RunDir"
    $SupervisorArgs = @(
        "scripts\run_logged_process.py", "--log", $RunLog,
        "--result-json", $ProcessResult, "--failure-json", $FailureJson,
        "--checkpoint", $Checkpoint
    )
    if ($EnableProcDump) {
        $SupervisorArgs += @(
            "--procdump", (Resolve-Path $ProcDumpPath).Path,
            "--dump-dir", (Join-Path $RunDir "dumps")
        )
    }
    $SupervisorArgs += @("--", $Python) + $PythonArgs
    & $Python @SupervisorArgs
    $SupervisorExitCode = $LASTEXITCODE
    $NativeResult = Get-Content $ProcessResult -Raw | ConvertFrom-Json
    if ($SupervisorExitCode -ne 0 -or [int64]$NativeResult.exit_code -ne 0) {
        $Archive = New-CurrentRunArchive -RunDir $RunDir
        Write-Error (Get-Content $FailureJson -Raw)
        throw "L_match=$Value native exit code $($NativeResult.exit_code). Run=$RunDir Archive=$Archive"
    }

    $Summary = Get-Content (Join-Path $RunDir "original_summary.json") -Raw | ConvertFrom-Json
    $CompetitionSummary = Get-Content (Join-Path $RunDir "competition_summary.json") -Raw | ConvertFrom-Json
    $StepMape = @{}
    foreach ($Step in 1..5) {
        $StepSummary = $CompetitionSummary.step_summary | Where-Object { [int]$_.horizon_step -eq $Step }
        $StepMape["step$Step"] = if ($null -ne $StepSummary) { [double]$StepSummary.mape } else { $null }
    }
    $PredictionHash = (Get-FileHash (Join-Path $RunDir "original_predictions.csv") -Algorithm SHA256).Hash.ToLowerInvariant()
    $script:RunSummaries += [pscustomobject]@{
        L_match=$Value; output_dir=$RunDir; MAPE=$Summary.mape; per_step_MAPE=$StepMape
        coverage=$Summary.coverage; runtime_seconds=$Summary.runtime_seconds
        predictions_SHA256=$PredictionHash; model_fingerprint=$Summary.final_model_fingerprint
        RNG_fingerprint=$Summary.final_rng_fingerprint
    }
    $RunDirs[$Value] = $RunDir
    $Completed.Add($Value)
    New-CurrentRunArchive -RunDir $RunDir | Out-Null
}

try {
    foreach ($Value in $Requested) { Invoke-LMatchRun -Value $Value }
} catch {
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
    $AnalysisArgs = @(
        "-X", "faulthandler", "-u", "experiments\diagnostics\analyze_fig9_lmatch_ablation.py",
        "--run-dir-l4", $RunDirs[4], "--run-dir-l3", $RunDirs[3],
        "--run-dir-l2", $RunDirs[2], "--output-dir", $AnalysisDir,
        "--bootstrap-samples", "1000", "--bootstrap-seed", "0"
    )
    & $Python "scripts\run_logged_process.py" --log $AnalysisLog `
        --result-json (Join-Path $AnalysisDir "process_result.json") `
        --failure-json (Join-Path $AnalysisDir "failure.json") `
        --checkpoint (Join-Path $RunDirs[4] "checkpoint.pkl") -- $Python @AnalysisArgs
    if ($LASTEXITCODE -ne 0) { throw "L_match offline comparison failed: $AnalysisDir" }
}
Write-Host "Completed groups: $($Completed -join ', ')"
Write-Host "Run summary: $SummaryPath"
Write-Host "Root output: $Root"
