param(
    [Parameter(Mandatory = $true)]
    [string]$SourceRunDirectory,
    [ValidateSet("all", "core-only", "observe", "match", "branch", "preselection", "teacher", "context", "density", "full-no-gzip", "full-gzip")]
    [string]$Profile = "all",
    [ValidateRange(1, 250)]
    [int]$DebugEndIndex = 202
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
Set-Location $RepoRoot

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$SourceRun = (Resolve-Path $SourceRunDirectory).Path
$SourceCheckpoint = Join-Path $SourceRun "checkpoint.pkl"
if (-not (Test-Path -LiteralPath $SourceCheckpoint)) {
    throw "Checkpoint not found: $SourceCheckpoint"
}
$Inspection = & $Python "experiments\diagnostics\inspect_fig9_checkpoint.py" $SourceCheckpoint | ConvertFrom-Json
$CheckpointLMatch = [int]$Inspection.L_match
$CheckpointNextIndex = [int]$Inspection.next_index
if ($CheckpointLMatch -notin @(2, 3, 4)) {
    throw "Expected a real L_match ablation checkpoint. Found L_match=$CheckpointLMatch."
}
if ($DebugEndIndex -le $CheckpointNextIndex -or $DebugEndIndex -gt ($CheckpointNextIndex + 10)) {
    throw "DebugEndIndex must be 1..10 records after checkpoint next_index=$CheckpointNextIndex."
}

$env:PYTHONPATH = "$RepoRoot\src;$RepoRoot"
$env:PYTHONFAULTHANDLER = "1"
$env:PYTHONUNBUFFERED = "1"
$Root = Join-Path $RepoRoot ("results\fig9_diagnostics\lmatch_l${CheckpointLMatch}_native_isolation_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
New-Item -ItemType Directory -Path $Root -Force | Out-Null

$BaseDiagnostics = @("--oracle-candidate-diagnostic")
$BranchDiagnostics = $BaseDiagnostics + @("--branch-provenance-diagnostic", "--branch-provenance-level", "candidate")
$PreselectionDiagnostics = $BranchDiagnostics + @("--preselection-segment-diagnostic", "--preselection-segment-level", "crossing")
$TeacherDiagnostics = $PreselectionDiagnostics + @("--teacher-forced-winner-diagnostic", "--teacher-forced-winner-level", "segment")
$ObserveDiagnostics = $TeacherDiagnostics + @("--observe-scenario-diagnostic", "--observe-scenario-level", "full")
$MatchDiagnostics = $ObserveDiagnostics + @("--match-overlap-diagnostic", "--match-overlap-level", "segment")
$ContextDiagnostics = $MatchDiagnostics + @("--context-trajectory-diagnostic", "--context-trajectory-level", "column")
$FullBase = $ContextDiagnostics + @("--density-trace", "--stream-diagnostic-traces")

$Profiles = @(
    [pscustomobject]@{ Name="core-only"; Args=@() },
    [pscustomobject]@{ Name="observe"; Args=$ObserveDiagnostics },
    [pscustomobject]@{ Name="match"; Args=$MatchDiagnostics },
    [pscustomobject]@{ Name="branch"; Args=$BranchDiagnostics },
    [pscustomobject]@{ Name="preselection"; Args=$PreselectionDiagnostics },
    [pscustomobject]@{ Name="teacher"; Args=$TeacherDiagnostics },
    [pscustomobject]@{ Name="context"; Args=$ContextDiagnostics },
    [pscustomobject]@{ Name="density"; Args=@("--density-trace") },
    [pscustomobject]@{ Name="full-no-gzip"; Args=$FullBase + @("--context-trajectory-uncompressed") },
    [pscustomobject]@{ Name="full-gzip"; Args=$FullBase + @("--preselection-segment-compress", "--match-overlap-compress") }
)
if ($Profile -ne "all") {
    $Profiles = @($Profiles | Where-Object Name -eq $Profile)
}

$Common = @(
    "-X", "faulthandler", "-u", "experiments\fig9_strict_reproduction.py",
    "--limit", "250", "--warmup", "200", "--streams", "original",
    "--prediction-horizon", "5", "--tie-break-seed", "0",
    "--l-match", "$CheckpointLMatch", "--lmatch-real-ablation",
    "--continuous-impl", "reference", "--competition-mode", "competitive_raw",
    "--inhibition-strength", "0.1", "--inhibition-tau", "0.02",
    "--simultaneous-policy", "batched", "--simultaneous-bin-width", "0.005",
    "--intracolumn-selection-policy", "max_candidate_score",
    "--progress-every", "1", "--debug-end-index", "$DebugEndIndex",
    "--checkpoint-every", "1"
)

$Results = @()
foreach ($Item in $Profiles) {
    $RunDir = Join-Path $Root $Item.Name
    New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
    $Checkpoint = Join-Path $RunDir "checkpoint.pkl"
    Copy-Item -LiteralPath $SourceCheckpoint -Destination $Checkpoint
    if (Test-Path -LiteralPath "$SourceCheckpoint.branch_provenance.json") {
        Copy-Item -LiteralPath "$SourceCheckpoint.branch_provenance.json" -Destination "$Checkpoint.branch_provenance.json"
    }
    $ChildArgs = $Common + $Item.Args + @(
        "--checkpoint-path", $Checkpoint,
        "--resume-from", $Checkpoint,
        "--output-dir", $RunDir
    )
    & $Python "scripts\run_logged_process.py" `
        --log (Join-Path $RunDir "run.log") `
        --result-json (Join-Path $RunDir "process_result.json") `
        --failure-json (Join-Path $RunDir "failure.json") `
        --checkpoint $Checkpoint -- $Python @ChildArgs
    $SupervisorExit = $LASTEXITCODE
    $Native = Get-Content (Join-Path $RunDir "process_result.json") -Raw | ConvertFrom-Json
    $LastPhase = Get-Content (Join-Path $RunDir "native_crash_faulthandler.log") |
        Where-Object { $_ -like "FIG9_NATIVE_PHASE*" } | Select-Object -Last 1
    $Results += [pscustomobject]@{
        profile=$Item.Name
        supervisor_exit=$SupervisorExit
        native_exit=[int64]$Native.exit_code
        reached_end=([int64]$Native.exit_code -eq 0 -and [int]$Native.checkpoint_next_index -ge $DebugEndIndex)
        checkpoint_next_index=$Native.checkpoint_next_index
        last_phase=$LastPhase
        output_dir=$RunDir
    }
}

$Summary = Join-Path $Root "isolation_summary.json"
$Results | ConvertTo-Json -Depth 8 | Set-Content $Summary -Encoding utf8
Write-Host "L$CheckpointLMatch isolation summary: $Summary"
if ($Results | Where-Object { -not $_.reached_end }) { exit 1 }
