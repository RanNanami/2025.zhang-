param(
    [ValidateSet("P1_SELECTOR_ONLY_L2", "P2_COMPETITION_ONLY_L2")]
    [string]$Cell = "P1_SELECTOR_ONLY_L2",
    [string]$OutputRoot = "",
    [switch]$Run
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
Set-Location $RepoRoot

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Supervisor = Join-Path $RepoRoot "scripts\run_logged_process.py"
if (-not (Test-Path -LiteralPath $Python)) { throw "Python venv not found: $Python" }
if (-not (Test-Path -LiteralPath $Supervisor)) { throw "Process supervisor not found: $Supervisor" }

function Read-Json([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    try { return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json }
    catch { return $null }
}

function Invoke-ComponentCell([string]$CellName) {
    $root = if ($OutputRoot) {
        New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
        (Resolve-Path $OutputRoot).Path
    } else {
        $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $path = Join-Path $RepoRoot "results\fig9_diagnostics\l2_stack_component_250_$stamp"
        New-Item -ItemType Directory -Path $path -Force | Out-Null
        (Resolve-Path $path).Path
    }
    $runDirectory = Join-Path $root "250\$CellName"
    New-Item -ItemType Directory -Path $runDirectory -Force | Out-Null
    $runDirectory = (Resolve-Path $runDirectory).Path
    $checkpoint = Join-Path $runDirectory "checkpoint.pkl"
    $isP1 = $CellName -eq "P1_SELECTOR_ONLY_L2"
    $audit = [ordered]@{
        cell = $CellName
        L_match = 2
        limit = 250
        warmup = 200
        stream = "original"
        competition_mode = if ($isP1) { "off" } else { "competitive_raw" }
        simultaneous_policy = if ($isP1) { "sequential" } else { "batched" }
        inhibition_strength = if ($isP1) { "inactive" } else { 0.1 }
        inhibition_tau = if ($isP1) { "inactive" } else { 0.02 }
        simultaneous_bin_width = if ($isP1) { "inactive" } else { 0.005 }
        intracolumn_selector = if ($isP1) { "max_candidate_score" } else { "existing" }
        propagation_mode = "raw"
        continuous_impl = "reference"
        K = 10
        neurons_per_column = 32
        forgetting_threshold = 65.0
        lmatch_real_ablation = $true
        strict_reproduction = $false
        diagnostic_only = $true
        uses_compensation = $false
        uses_future_covariates = $false
        uses_ground_truth_for_selection = $false
        learns_during_rollout = $false
        reencodes_decoded_value = $false
        generated_by = "scripts/fig9_l2_stack_component_250.ps1"
    }
    $audit | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $runDirectory "component_path_audit.json") -Encoding UTF8

    $pythonArgs = @(
        "-X", "faulthandler", "-u",
        "experiments\fig9_strict_reproduction.py",
        "--limit", "250", "--warmup", "200", "--streams", "original",
        "--prediction-horizon", "5", "--tie-break-seed", "0",
        "--l-match", "2", "--lmatch-real-ablation",
        "--continuous-impl", "reference",
        "--density-trace", "--long-sequence-ledger",
        "--progress-every", "10", "--checkpoint-path", $checkpoint,
        "--checkpoint-every", "1", "--output-dir", $runDirectory
    )
    if ($isP1) {
        $pythonArgs += @("--competition-mode", "off", "--simultaneous-policy", "sequential", "--intracolumn-selection-policy", "max_candidate_score")
    } else {
        $pythonArgs += @("--competition-mode", "competitive_raw", "--inhibition-strength", "0.1", "--inhibition-tau", "0.02", "--simultaneous-policy", "batched", "--simultaneous-bin-width", "0.005", "--intracolumn-selection-policy", "existing")
    }

    $env:PYTHONPATH = "$RepoRoot\src;$RepoRoot"
    $env:PYTHONFAULTHANDLER = "1"
    $env:PYTHONUNBUFFERED = "1"
    $attempt = 0
    while ($true) {
        $attempt++
        $suffix = "attempt_{0:D2}" -f $attempt
        $resultJson = Join-Path $runDirectory "process_$suffix.json"
        $failureJson = Join-Path $runDirectory "failure_$suffix.json"
        $stdoutLog = Join-Path $runDirectory "stdout_$suffix.log"
        $stderrLog = Join-Path $runDirectory "stderr_$suffix.log"
        $combinedLog = Join-Path $runDirectory "combined_$suffix.log"
        $attemptArgs = @($pythonArgs)
        if (Test-Path -LiteralPath $checkpoint) { $attemptArgs += @("--resume-from", $checkpoint) }
        $command = @($Python) + $attemptArgs
        $command | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $runDirectory "command_$suffix.json") -Encoding UTF8
        $output = & $Python $Supervisor --stdout-log $stdoutLog --stderr-log $stderrLog --combined-log $combinedLog --result-json $resultJson --failure-json $failureJson --checkpoint $checkpoint -- $command
        $supervisorExit = $LASTEXITCODE
        if ($output) { Write-Host ($output -join [Environment]::NewLine) }
        $child = Read-Json $resultJson
        $exitCode = if ($child -and $null -ne $child.exit_code) { [int64]$child.exit_code } else { [int64]$supervisorExit }
        if ($exitCode -eq 0) { break }
        if ($attempt -ge 5 -or ($exitCode -ne 3221225477 -and $exitCode -ne -1073741819)) { throw "$CellName failed with exit code $exitCode; see $failureJson" }
        Write-Host "Recoverable native failure; retry $($attempt + 1)/5"
    }
    if (-not (Test-Path (Join-Path $runDirectory "original_summary.json"))) { throw "Missing original_summary.json" }
    Write-Host "Completed ${CellName}: $runDirectory"
    return $runDirectory
}

if (-not $Run) {
    Write-Host "No process started. Add -Run to execute one component cell."
    exit 0
}

$componentResult = Invoke-ComponentCell $Cell
Write-Host "Run directory: $componentResult"
