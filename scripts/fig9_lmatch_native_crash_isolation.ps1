param(
    [Parameter(Mandatory = $true)]
    [string]$SourceRunDirectory,
    [int]$DebugEndIndex = 220,
    [ValidateSet("all", "core-only", "diagnostics-no-compression", "compression-plain", "compression-gzip", "branch", "preselection", "teacher", "observe", "match", "context", "density", "full-single-thread")]
    [string]$Profile = "all",
    [switch]$EnableProcDump,
    [string]$ProcDumpPath = ""
)

$ErrorActionPreference = "Stop"
if (Test-Path variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
Set-Location $RepoRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$SourceRun = (Resolve-Path $SourceRunDirectory).Path
$SourceCheckpoint = Join-Path $SourceRun "checkpoint.pkl"
if (-not (Test-Path -LiteralPath $SourceCheckpoint)) {
    throw "Checkpoint not found: $SourceCheckpoint"
}
if ($EnableProcDump -and -not $ProcDumpPath) {
    throw "EnableProcDump requires -ProcDumpPath."
}

$env:PYTHONPATH = "$RepoRoot\src;$RepoRoot"
$env:PYTHONFAULTHANDLER = "1"
$env:PYTHONUNBUFFERED = "1"
$Inspection = & $Python "experiments\diagnostics\inspect_fig9_checkpoint.py" $SourceCheckpoint |
    ConvertFrom-Json
if ([int]$Inspection.next_index -lt 210) {
    throw "Refusing to recompute records $($Inspection.next_index)..209. A real next_index=210 checkpoint is required."
}
if ([int]$Inspection.next_index -ne 210) {
    throw "Isolation matrix requires next_index=210; found $($Inspection.next_index)."
}
if ($DebugEndIndex -le 210 -or $DebugEndIndex -gt 220) {
    throw "DebugEndIndex must be in 211..220."
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Root = Join-Path $RepoRoot "results\fig9_diagnostics\lmatch_native_crash_isolation_$Stamp"
New-Item -ItemType Directory -Path $Root -Force | Out-Null
$Common = @(
    "-X", "faulthandler", "-u", "experiments\fig9_strict_reproduction.py",
    "--limit", "250", "--warmup", "200", "--streams", "original",
    "--prediction-horizon", "5", "--tie-break-seed", "0",
    "--l-match", "4", "--lmatch-real-ablation", "--continuous-impl", "reference",
    "--competition-mode", "competitive_raw", "--inhibition-strength", "0.1",
    "--inhibition-tau", "0.02", "--simultaneous-policy", "batched",
    "--simultaneous-bin-width", "0.005",
    "--intracolumn-selection-policy", "max_candidate_score",
    "--progress-every", "1", "--debug-end-index", "$DebugEndIndex"
)
$OracleBranch = @(
    "--oracle-candidate-diagnostic", "--branch-provenance-diagnostic",
    "--branch-provenance-level", "candidate"
)
$Preselection = $OracleBranch + @(
    "--preselection-segment-diagnostic", "--preselection-segment-level", "crossing"
)
$Teacher = $Preselection + @(
    "--teacher-forced-winner-diagnostic", "--teacher-forced-winner-level", "segment"
)
$Observe = $Teacher + @(
    "--observe-scenario-diagnostic", "--observe-scenario-level", "full"
)
$Match = @("--match-overlap-diagnostic", "--match-overlap-level", "segment")
$Context = @(
    "--context-trajectory-diagnostic", "--context-trajectory-level", "column"
)
$FullPlain = $Observe + $Match + $Context + @(
    "--oracle-candidate-diagnostic", "--density-trace",
    "--context-trajectory-uncompressed"
)
$Profiles = @(
    [pscustomobject]@{Name="core-only"; Args=@(); Single=$false},
    [pscustomobject]@{Name="diagnostics-no-compression"; Args=$FullPlain; Single=$false},
    [pscustomobject]@{Name="compression-plain"; Args=$Match; Single=$false},
    [pscustomobject]@{Name="compression-gzip"; Args=$Match + @("--match-overlap-compress", "--stream-diagnostic-traces"); Single=$false},
    [pscustomobject]@{Name="branch"; Args=$OracleBranch; Single=$false},
    [pscustomobject]@{Name="preselection"; Args=$Preselection; Single=$false},
    [pscustomobject]@{Name="teacher"; Args=$Teacher; Single=$false},
    [pscustomobject]@{Name="observe"; Args=$Observe; Single=$false},
    [pscustomobject]@{Name="match"; Args=$Match; Single=$false},
    [pscustomobject]@{Name="context"; Args=$Context + @("--context-trajectory-uncompressed"); Single=$false},
    [pscustomobject]@{Name="density"; Args=@("--density-trace"); Single=$false},
    [pscustomobject]@{Name="full-single-thread"; Args=$FullPlain; Single=$true}
)
if ($Profile -ne "all") { $Profiles = $Profiles | Where-Object Name -eq $Profile }

$OriginalThreads = @{}
foreach ($Key in "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS") {
    $OriginalThreads[$Key] = [Environment]::GetEnvironmentVariable($Key, "Process")
}
$Results = @()
foreach ($Item in $Profiles) {
    $RunDir = Join-Path $Root $Item.Name
    New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
    $Checkpoint = Join-Path $RunDir "checkpoint.pkl"
    Copy-Item -LiteralPath $SourceCheckpoint -Destination $Checkpoint
    $Sidecar = "$SourceCheckpoint.branch_provenance.json"
    if (Test-Path -LiteralPath $Sidecar) {
        Copy-Item -LiteralPath $Sidecar -Destination "$Checkpoint.branch_provenance.json"
    }
    foreach ($Key in $OriginalThreads.Keys) {
        [Environment]::SetEnvironmentVariable(
            $Key,
            $(if ($Item.Single) { "1" } else { $OriginalThreads[$Key] }),
            "Process"
        )
    }
    $ChildArgs = $Common + $Item.Args + @(
        "--resume-from", $Checkpoint, "--output-dir", $RunDir
    )
    $Supervisor = @(
        "scripts\run_logged_process.py", "--log", (Join-Path $RunDir "run.log"),
        "--result-json", (Join-Path $RunDir "process_result.json"),
        "--failure-json", (Join-Path $RunDir "failure.json"),
        "--checkpoint", $Checkpoint
    )
    if ($EnableProcDump) {
        $Supervisor += @(
            "--procdump", (Resolve-Path $ProcDumpPath).Path,
            "--dump-dir", (Join-Path $RunDir "dumps")
        )
    }
    $Supervisor += @("--", $Python) + $ChildArgs
    & $Python @Supervisor
    $Result = Get-Content (Join-Path $RunDir "process_result.json") -Raw | ConvertFrom-Json
    $LastPhase = Get-Content (Join-Path $RunDir "native_crash_faulthandler.log") |
        Select-String "FIG9_NATIVE_PHASE" | Select-Object -Last 1
    $Files = Get-ChildItem -LiteralPath $RunDir -File | Select-Object -ExpandProperty Name
    $Results += [pscustomobject]@{
        profile=$Item.Name
        # Windows native failures such as 0xC0000005 are reported as unsigned
        # 32-bit values (3221225477), which do not fit in System.Int32.
        reached_end=([int64]$Result.exit_code -eq 0 -and [int]$Result.last_completed_index -ge $DebugEndIndex)
        exit_code=$Result.exit_code
        last_index=$Result.last_completed_index
        last_phase=$(if ($LastPhase) { $LastPhase.Line } else { "" })
        single_thread=$Item.Single
        files=$Files
        log_path=(Join-Path $RunDir "run.log")
        native_log_path=(Join-Path $RunDir "native_crash_faulthandler.log")
    }
}
foreach ($Key in $OriginalThreads.Keys) {
    [Environment]::SetEnvironmentVariable($Key, $OriginalThreads[$Key], "Process")
}
$Summary = Join-Path $Root "isolation_summary.json"
$Results | ConvertTo-Json -Depth 6 | Set-Content $Summary -Encoding UTF8
Write-Host "Isolation summary: $Summary"
