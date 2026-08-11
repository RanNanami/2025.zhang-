param(
    [Parameter(Mandatory = $true)]
    [string]$SourceCheckpoint,
    [int]$DebugEndIndex = 225,
    [int]$Attempts = 3,
    [switch]$ReadoutTrace,
    [switch]$SingleThread,
    [switch]$DisableRuntimeDebug,
    [string]$OutputRoot = ""
)

$ErrorActionPreference = "Stop"
if (Test-Path variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
Set-Location $RepoRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$SourceCheckpoint = (Resolve-Path $SourceCheckpoint).Path
$Metadata = [System.IO.Path]::ChangeExtension($SourceCheckpoint, ".metadata.json")
if (-not (Test-Path -LiteralPath $Metadata)) {
    throw "Atomic checkpoint metadata not found: $Metadata"
}
if ($DebugEndIndex -le 0) {
    throw "DebugEndIndex must be positive."
}
if ($Attempts -lt 1 -or $Attempts -gt 5) {
    throw "Attempts must be between 1 and 5."
}

$Inspection = Get-Content -LiteralPath $Metadata -Raw | ConvertFrom-Json
$StartIndex = [int]$Inspection.next_index
if ($DebugEndIndex -le $StartIndex -or $DebugEndIndex -gt ($StartIndex + 10)) {
    throw "DebugEndIndex must be in ($StartIndex)..$($StartIndex + 10)."
}

$env:PYTHONPATH = "$RepoRoot\src;$RepoRoot"
$env:PYTHONFAULTHANDLER = "1"
$env:PYTHONUNBUFFERED = "1"
if ($SingleThread) {
    $env:OMP_NUM_THREADS = "1"
    $env:MKL_NUM_THREADS = "1"
    $env:OPENBLAS_NUM_THREADS = "1"
    $env:NUMEXPR_NUM_THREADS = "1"
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
if (-not $OutputRoot) {
    $OutputRoot = Join-Path $RepoRoot "results\fig9_diagnostics\native_runtime_repro_$Stamp"
}
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$Common = @(
    "-X", "faulthandler", "-u", "experiments\fig9_strict_reproduction.py",
    "--limit", "500", "--warmup", "200", "--streams", "original",
    "--prediction-horizon", "5", "--tie-break-seed", "0",
    "--l-match", "2", "--lmatch-real-ablation", "--continuous-impl", "reference",
    "--competition-mode", "competitive_raw", "--inhibition-strength", "0.1",
    "--inhibition-tau", "0.02", "--simultaneous-policy", "batched",
    "--simultaneous-bin-width", "0.005",
    "--intracolumn-selection-policy", "existing",
    "--progress-every", "1", "--checkpoint-every", "1",
    "--debug-end-index", "$DebugEndIndex"
)
if ($ReadoutTrace) {
    $Common += @("--readout-dynamics-trace", "--stream-diagnostic-traces")
}
if (-not $DisableRuntimeDebug) {
    $Common += @("--native-runtime-debug")
}

$Results = @()
for ($Attempt = 1; $Attempt -le $Attempts; $Attempt++) {
    $RunDir = Join-Path $OutputRoot ("attempt_{0:D2}" -f $Attempt)
    New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
    $Checkpoint = Join-Path $RunDir "checkpoint.pkl"
    Copy-Item -LiteralPath $SourceCheckpoint -Destination $Checkpoint
    $Sidecar = "$SourceCheckpoint.branch_provenance.json"
    if (Test-Path -LiteralPath $Sidecar) {
        Copy-Item -LiteralPath $Sidecar -Destination "$Checkpoint.branch_provenance.json"
    }
    $ChildArgs = $Common
    if (-not $DisableRuntimeDebug) {
        $ChildArgs += @("--native-runtime-debug-dir", $RunDir)
    }
    $ChildArgs += @(
        "--resume-from", $Checkpoint,
        "--checkpoint-path", $Checkpoint,
        "--output-dir", $RunDir
    )
    $Supervisor = @(
        "scripts\run_logged_process.py",
        "--log", (Join-Path $RunDir "run.log"),
        "--result-json", (Join-Path $RunDir "process_result.json"),
        "--failure-json", (Join-Path $RunDir "failure.json"),
        "--checkpoint", $Checkpoint,
        "--", $Python
    ) + $ChildArgs
    & $Python @Supervisor
    $SupervisorExit = $LASTEXITCODE
    $ResultPath = Join-Path $RunDir "process_result.json"
    $Result = if (Test-Path -LiteralPath $ResultPath) {
        Get-Content -LiteralPath $ResultPath -Raw | ConvertFrom-Json
    } else {
        [pscustomobject]@{ exit_code = $SupervisorExit; last_completed_index = $null }
    }
    $RuntimeDebugLog = $null
    if (-not $DisableRuntimeDebug) {
        $RuntimeDebugLog = Join-Path $RunDir "native_runtime_debug.jsonl"
    }
    $Results += [pscustomobject]@{
        attempt = $Attempt
        source_checkpoint = $SourceCheckpoint
        start_index = $StartIndex
        debug_end_index = $DebugEndIndex
        exit_code = $Result.exit_code
        last_completed_index = $Result.last_completed_index
        last_native_phase = $Result.last_native_phase
        last_native_phase_index = $Result.last_native_phase_index
        run_directory = $RunDir
        runtime_debug_log = $RuntimeDebugLog
    }
}

$Summary = Join-Path $OutputRoot "reproduction_summary.json"
$Results | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $Summary -Encoding UTF8
Write-Host "Native runtime reproduction summary: $Summary"
