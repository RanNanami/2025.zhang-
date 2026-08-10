param(
    [ValidateSet("STRICT_RAW_L4", "STRICT_RAW_L2")]
    [string]$Cell = "STRICT_RAW_L4",
    [int]$Limit = 250,
    [int]$Warmup = 200,
    [string]$OutputRoot = "",
    [string]$ResumeCheckpoint = "",
    [switch]$Run,
    [switch]$ReuseDiagnosticCells
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
Set-Location $RepoRoot

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Supervisor = Join-Path $RepoRoot "scripts\run_logged_process.py"
if (-not (Test-Path -LiteralPath $Python)) { throw "Python venv not found: $Python" }
if (-not (Test-Path -LiteralPath $Supervisor)) { throw "Process supervisor not found: $Supervisor" }
if ($Limit -notin @(50, 250, 500)) { throw "Limit must be 50, 250, or 500." }

Write-Host "Fig.9 2x2 decomposition runner"
Write-Host "Cell: $Cell | limit=$Limit | warmup=$Warmup"
Write-Host "STRICT_RAW_L2 is a nonpaper single-factor ablation; strict default remains L4."

function Get-CellPath([string]$Root, [string]$CellName) {
    $path = Join-Path $Root "$Limit\$CellName"
    New-Item -ItemType Directory -Path $path -Force | Out-Null
    return (Resolve-Path $path).Path
}

function Read-ChildResult([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    try { return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json }
    catch { return $null }
}

function Invoke-RawCell([string]$CellName) {
    $root = if ($OutputRoot) {
        New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
        (Resolve-Path $OutputRoot).Path
    } else {
        $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $path = Join-Path $RepoRoot "results\fig9_diagnostics\lmatch_stack_2x2_$stamp"
        New-Item -ItemType Directory -Path $path -Force | Out-Null
        (Resolve-Path $path).Path
    }
    $runDirectory = Get-CellPath $root $CellName
    $checkpoint = Join-Path $runDirectory "checkpoint.pkl"
    if ($ResumeCheckpoint) {
        $sourceCheckpoint = (Resolve-Path -LiteralPath $ResumeCheckpoint).Path
        if (Test-Path -LiteralPath $checkpoint) {
            throw "Refusing to overwrite destination checkpoint: $checkpoint"
        }
        Copy-Item -LiteralPath $sourceCheckpoint -Destination $checkpoint
        $sourceMetadata = "$sourceCheckpoint.metadata.json"
        if (Test-Path -LiteralPath $sourceMetadata) {
            Copy-Item -LiteralPath $sourceMetadata -Destination "$checkpoint.metadata.json"
        }
        $sourceSidecar = "$sourceCheckpoint.branch_provenance.json"
        if (Test-Path -LiteralPath $sourceSidecar) {
            Copy-Item -LiteralPath $sourceSidecar -Destination "$checkpoint.branch_provenance.json"
        }
    }
    $audit = [ordered]@{
        cell = $CellName
        stack = "STRICT_RAW"
        L_match = if ($CellName -eq "STRICT_RAW_L2") { 2 } else { 4 }
        limit = $Limit
        warmup = $Warmup
        stream = "original"
        prediction_horizon = 5
        tie_break_seed = 0
        continuous_impl = "reference"
        competition_mode = "off"
        simultaneous_policy = "sequential"
        simultaneous_bin_width = 0.005
        inhibition_strength = "inactive"
        inhibition_tau = "inactive"
        intracolumn_selector = "existing"
        propagation_mode = "raw"
        burst_context = "all-cell"
        forgetting_threshold = 65.0
        K = 10
        neurons_per_column = 32
        lmatch_real_ablation = ($CellName -eq "STRICT_RAW_L2")
        strict_reproduction = ($CellName -eq "STRICT_RAW_L4" -and $Limit -ne 50)
        diagnostic_stack_active = $false
        uses_compensation = $false
        uses_future_covariates = $false
        uses_ground_truth_for_selection = $false
        learns_during_rollout = $false
        reencodes_decoded_value = $false
        generated_by = "scripts/fig9_lmatch_stack_2x2.ps1"
    }
    $audit | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $runDirectory "strict_raw_path_audit.json") -Encoding UTF8

    $lmatch = if ($CellName -eq "STRICT_RAW_L2") { "2" } else { "4" }
    $pythonArgs = @(
        "-X", "faulthandler", "-u",
        "experiments\fig9_strict_reproduction.py",
        "--limit", "$Limit", "--warmup", "$Warmup", "--streams", "original",
        "--prediction-horizon", "5", "--tie-break-seed", "0",
        "--l-match", $lmatch, "--continuous-impl", "reference",
        "--competition-mode", "off", "--simultaneous-policy", "sequential",
        "--simultaneous-bin-width", "0.005",
        "--intracolumn-selection-policy", "existing",
        "--density-trace", "--long-sequence-ledger",
        "--progress-every", "10", "--checkpoint-path", $checkpoint,
        "--checkpoint-every", "1", "--output-dir", $runDirectory
    )
    if ($CellName -eq "STRICT_RAW_L2") { $pythonArgs += "--lmatch-real-ablation" }
    $env:PYTHONPATH = "$RepoRoot\src;$RepoRoot"
    $env:PYTHONFAULTHANDLER = "1"
    $env:PYTHONUNBUFFERED = "1"
    $attempt = 0
    while ($true) {
        $attempt++
        $resumeFromCheckpoint = Test-Path -LiteralPath $checkpoint
        $suffix = "attempt_{0:D2}" -f $attempt
        $resultJson = Join-Path $runDirectory "process_$suffix.json"
        $failureJson = Join-Path $runDirectory "failure_$suffix.json"
        $stdoutLog = Join-Path $runDirectory "stdout_$suffix.log"
        $stderrLog = Join-Path $runDirectory "stderr_$suffix.log"
        $combinedLog = Join-Path $runDirectory "combined_$suffix.log"
        $attemptArgs = @($pythonArgs)
        if ($resumeFromCheckpoint) {
            $attemptArgs += @("--resume-from", $checkpoint)
        }
        $command = @($Python) + $attemptArgs
        $command | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $runDirectory "command_$suffix.json") -Encoding UTF8
        $supervisorOutput = & $Python $Supervisor `
            --stdout-log $stdoutLog `
            --stderr-log $stderrLog `
            --combined-log $combinedLog `
            --result-json $resultJson `
            --failure-json $failureJson `
            --checkpoint $checkpoint `
            -- $command
        $supervisorExit = $LASTEXITCODE
        if ($supervisorOutput) {
            Write-Host ($supervisorOutput -join [Environment]::NewLine)
        }
        $child = Read-ChildResult $resultJson
        $childExit = if ($null -ne $child -and $null -ne $child.exit_code) { [int64]$child.exit_code } else { [int64]$supervisorExit }
        if ($childExit -eq 0) { break }
        $checkpointMeta = Read-ChildResult (Join-Path $runDirectory "checkpoint.metadata.json")
        $next = if ($null -ne $checkpointMeta) { [int]$checkpointMeta.next_index } else { -1 }
        if ($attempt -ge 5 -or ($childExit -ne 3221225477 -and $childExit -ne -1073741819)) {
            throw "$CellName failed with child exit code $childExit; see $failureJson"
        }
        Write-Host "Recoverable native failure at next_index=$next; retry $($attempt + 1)/5"
    }
    $summaryPath = Join-Path $runDirectory "original_summary.json"
    if (-not (Test-Path -LiteralPath $summaryPath)) { throw "Completed process has no original_summary.json" }
    Write-Host "Completed ${CellName}: $runDirectory"
    return $runDirectory
}

if (-not $Run) {
    Write-Host "No process started. Add -Run to execute one raw cell."
    exit 0
}

$runDirectoryResult = Invoke-RawCell $Cell
Write-Host "Run directory: $runDirectoryResult"
