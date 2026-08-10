param(
    [ValidateSet(4, 2)]
    [int]$LMatch = 4,
    [switch]$RunGrid,
    [switch]$RunFormal500,
    [switch]$AnalyzeOnly,
    [switch]$AmbiguityDiagnostic,
    [string]$RunDirectory = "",
    [string]$ResumeRunDirectory = ""
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
Set-Location $RepoRoot

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Supervisor = Join-Path $RepoRoot "scripts\run_logged_process.py"
if (-not (Test-Path -LiteralPath $Python)) { throw "Python venv not found: $Python" }
if (-not (Test-Path -LiteralPath $Supervisor)) { throw "Process supervisor not found: $Supervisor" }

Write-Host "This is the formal Fig.9 500-record L2/L4 robustness experiment."
Write-Host "It must be started manually by the user."
Write-Host "No new model mechanism is enabled; strict defaults remain unchanged."

function Invoke-LongSequenceAnalysis([string]$L2Directory, [string]$L4Directory, [string]$AnalysisDirectory) {
    New-Item -ItemType Directory -Path $AnalysisDirectory -Force | Out-Null
    $analysisLog = Join-Path $AnalysisDirectory "analysis.combined.log"
    $resultJson = Join-Path $AnalysisDirectory "analysis.result.json"
    $failureJson = Join-Path $AnalysisDirectory "analysis.failure.json"
    $checkpoint = Join-Path $L4Directory "checkpoint.pkl"
    $analysisArgs = @(
        "-X", "faulthandler", "-u",
        "experiments\diagnostics\analyze_fig9_l2_l4_long_sequence.py",
        "--run-dir-l2", $L2Directory,
        "--run-dir-l4", $L4Directory,
        "--output-dir", $AnalysisDirectory,
        "--bootstrap-samples", "1000",
        "--bootstrap-seed", "0"
    )
    & $Python $Supervisor `
        --stdout-log (Join-Path $AnalysisDirectory "analysis.stdout.log") `
        --stderr-log (Join-Path $AnalysisDirectory "analysis.stderr.log") `
        --combined-log $analysisLog `
        --result-json $resultJson `
        --failure-json $failureJson `
        --checkpoint $checkpoint `
        -- (@($Python) + $analysisArgs)
    if ($LASTEXITCODE -ne 0) {
        throw "Long-sequence analyzer failed; see $AnalysisDirectory"
    }
}

function New-RunArchive([string]$RunDirectory) {
    $archive = "$RunDirectory.zip"
    $items = Get-ChildItem -LiteralPath $RunDirectory -Force |
        Where-Object { $_.Name -ne (Split-Path -Leaf $archive) } |
        Select-Object -ExpandProperty FullName
    if (Test-Path -LiteralPath $archive) { Remove-Item -LiteralPath $archive -Force }
    Compress-Archive -Path $items -DestinationPath $archive -Force
    return $archive
}

function Invoke-LongSequenceRun([int]$Value, [string]$RootDirectory, [string]$ResumeDirectory) {
    $runDirectory = if ($ResumeDirectory) {
        (Resolve-Path $ResumeDirectory).Path
    } else {
        $stamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
        $path = Join-Path $RootDirectory "L${Value}_$stamp"
        New-Item -ItemType Directory -Path $path -Force | Out-Null
        (Resolve-Path $path).Path
    }
    $checkpoint = Join-Path $runDirectory "checkpoint.pkl"
    $pythonArgs = @(
        "-X", "faulthandler", "-u",
        "experiments\fig9_strict_reproduction.py",
        "--limit", "500", "--warmup", "200", "--streams", "original",
        "--prediction-horizon", "5", "--tie-break-seed", "0",
        "--l-match", "$Value",
        "--continuous-impl", "reference",
        "--competition-mode", "competitive_raw",
        "--inhibition-strength", "0.1", "--inhibition-tau", "0.02",
        "--simultaneous-policy", "batched", "--simultaneous-bin-width", "0.005",
        "--intracolumn-selection-policy", "max_candidate_score",
        "--density-trace", "--long-sequence-ledger",
        "--interval-every", "25", "--progress-every", "10",
        "--checkpoint-path", $checkpoint, "--checkpoint-every", "10",
        "--output-dir", $runDirectory
    )
    if ($Value -eq 2) { $pythonArgs += @("--lmatch-real-ablation") }
    if ($AmbiguityDiagnostic) { $pythonArgs += @("--ambiguity-diagnostic") }
    if ($ResumeDirectory) { $pythonArgs += @("--resume-from", $checkpoint) }

    $commandMetadata = @{
        command = @($Python) + $pythonArgs
        python_args = $pythonArgs
        working_directory = (Get-Location).Path
        environment = @{
            PYTHONFAULTHANDLER = "1"
            PYTHONUNBUFFERED = "1"
            PYTHONPATH = "$RepoRoot\src;$RepoRoot"
        }
    }
    $commandMetadata | ConvertTo-Json -Depth 6 |
        Set-Content (Join-Path $runDirectory "command.json") -Encoding UTF8

    Write-Host "Starting L_match=$Value"
    Write-Host "Output: $runDirectory"
    & $Python $Supervisor `
        --stdout-log (Join-Path $runDirectory "stdout.log") `
        --stderr-log (Join-Path $runDirectory "stderr.log") `
        --combined-log (Join-Path $runDirectory "combined.log") `
        --result-json (Join-Path $runDirectory "process_result.json") `
        --failure-json (Join-Path $runDirectory "failure.json") `
        --checkpoint $checkpoint `
        -- (@($Python) + $pythonArgs)
    if ($LASTEXITCODE -ne 0) {
        throw "L_match=$Value failed; see $runDirectory\failure.json"
    }
    $summaryPath = Join-Path $runDirectory "original_summary.json"
    if (-not (Test-Path -LiteralPath $summaryPath)) {
        throw "L_match=$Value finished without original_summary.json"
    }
    return $runDirectory
}

if ($AnalyzeOnly) {
    if (-not $RunDirectory) { throw "-AnalyzeOnly requires -RunDirectory pointing to the 500-run root" }
    $root = (Resolve-Path $RunDirectory).Path
    $l2 = Get-ChildItem -LiteralPath $root -Directory | Where-Object Name -like "L2_*" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $l4 = Get-ChildItem -LiteralPath $root -Directory | Where-Object Name -like "L4_*" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $l2 -or -not $l4) { throw "AnalyzeOnly requires both L2_* and L4_* run directories" }
    Invoke-LongSequenceAnalysis $l2.FullName $l4.FullName (Join-Path $root "analysis")
    Write-Host "Analysis complete: $(Join-Path $root 'analysis')"
    exit 0
}

if (-not $RunFormal500 -and -not $ResumeRunDirectory) {
    Write-Host "No experiment was started. Add -RunFormal500 explicitly."
    return
}
if ($RunGrid -and $ResumeRunDirectory) {
    throw "ResumeRunDirectory resumes one group; do not combine it with RunGrid."
}

$env:PYTHONPATH = "$RepoRoot\src;$RepoRoot"
$env:PYTHONFAULTHANDLER = "1"
$env:PYTHONUNBUFFERED = "1"
$root = if ($ResumeRunDirectory) {
    (Resolve-Path $ResumeRunDirectory).Path | Split-Path -Parent
} else {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $path = Join-Path $RepoRoot "results\fig9_diagnostics\l2_l4_long_sequence_$stamp"
    New-Item -ItemType Directory -Path $path -Force | Out-Null
    (Resolve-Path $path).Path
}

$requested = if ($RunGrid) { @(4, 2) } else { @($LMatch) }
$runDirectories = @{}
foreach ($value in $requested) {
    $resume = if ($ResumeRunDirectory) { $ResumeRunDirectory } else { "" }
    $runDirectories[$value] = Invoke-LongSequenceRun $value $root $resume
    if (-not $resume) { New-RunArchive $runDirectories[$value] | Out-Null }
}

if ($RunGrid) {
    Invoke-LongSequenceAnalysis $runDirectories[2] $runDirectories[4] (Join-Path $root "analysis")
    Write-Host "Analysis complete: $(Join-Path $root 'analysis')"
}
Write-Host "Root output: $root"
