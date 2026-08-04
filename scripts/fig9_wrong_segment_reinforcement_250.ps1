param(
    [switch]$RunFormal250,
    [int]$MaxResumeAttempts = 5
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
$Python = ".\.venv\Scripts\python.exe"

Write-Host "Fig.9 read-only wrong-segment reinforcement diagnostic (L4/L2 only)."
if (-not $RunFormal250) {
    Write-Host "No formal experiment was started. Add -RunFormal250 explicitly."
    exit 0
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Root = "results\fig9_diagnostics\wrong_segment_reinforcement_$Stamp"
New-Item -ItemType Directory -Force -Path $Root | Out-Null

foreach ($LMatch in @(4, 2)) {
    $RunDir = Join-Path $Root "L$LMatch"
    New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
    $Checkpoint = Join-Path $RunDir "checkpoint.pkl"
    $BasePythonArgs = @(
        "-X", "faulthandler", "-u", "experiments\fig9_strict_reproduction.py",
        "--limit", "250", "--warmup", "200", "--streams", "original",
        "--prediction-horizon", "5", "--tie-break-seed", "0",
        "--l-match", "$LMatch", "--lmatch-real-ablation",
        "--continuous-impl", "reference", "--competition-mode", "competitive_raw",
        "--inhibition-strength", "0.1", "--inhibition-tau", "0.02",
        "--simultaneous-policy", "batched", "--simultaneous-bin-width", "0.005",
        "--intracolumn-selection-policy", "max_candidate_score",
        "--segment-reinforcement-diagnostic", "--segment-reinforcement-level", "segment",
        "--density-trace", "--progress-every", "10",
        "--checkpoint-path", $Checkpoint,
        "--checkpoint-every", "10", "--output-dir", $RunDir
    )
    $Attempt = 0
    while ($true) {
        $Attempt += 1
        $ResumeArgs = @()
        if ($Attempt -gt 1) {
            if (-not (Test-Path $Checkpoint)) {
                throw "L_match=$LMatch failed without a resumable checkpoint in $RunDir"
            }
            $ResumeArgs = @("--resume-from", $Checkpoint)
            Write-Host "Resuming L_match=$LMatch from $Checkpoint (attempt $Attempt)."
        }
        $Log = Join-Path $RunDir "run_attempt_$Attempt.log"
        & $Python @BasePythonArgs @ResumeArgs 2>&1 | Tee-Object -FilePath $Log
        $NativeExitCode = $LASTEXITCODE
        if ($NativeExitCode -eq 0) { break }
        if ($Attempt -ge $MaxResumeAttempts) {
            throw "L_match=$LMatch failed after $Attempt attempts; last exit code $NativeExitCode in $RunDir"
        }
    }
}

$Analysis = Join-Path $Root "analysis"
& $Python -m experiments.diagnostics.analyze_fig9_wrong_segment_reinforcement `
    --l2-run (Join-Path $Root "L2") `
    --l4-run (Join-Path $Root "L4") `
    --output-dir $Analysis
if ($LASTEXITCODE -ne 0) { throw "Offline reinforcement analysis failed" }
Write-Host "Completed: $Root"
Write-Host "Report: $(Join-Path $Analysis 'FIG9_WRONG_SEGMENT_REINFORCEMENT_REPORT.md')"
