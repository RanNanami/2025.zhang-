param(
    [ValidateSet(2, 4)] [int]$LMatch = 4,
    [switch]$RunFormal250,
    [switch]$AnalyzeOnly,
    [string]$RunDirectory = "",
    [string]$ResumeRunDirectory = ""
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$python = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$runner = (Resolve-Path "scripts\run_logged_process.py").Path

function Invoke-Analysis([string]$Directory) {
    $analysis = Join-Path $Directory "analysis"
    New-Item -ItemType Directory -Force $analysis | Out-Null
    $jobs = @(
        @("experiments/diagnostics/analyze_fig9_independent_reference_provenance.py", "independent_reference_analysis"),
        @("experiments/diagnostics/analyze_fig9_actual_branch_provenance.py", "actual_branch_analysis")
    )
    foreach ($job in $jobs) {
        $tool = Join-Path $PSScriptRoot "..\$($job[0])"
        if (-not (Test-Path $tool)) { throw "Missing analyzer: $tool" }
        $prefix = Join-Path $Directory $job[1]
        $analysisArgs = @($Directory, "--output-dir", $analysis)
        if ($job[1] -eq "actual_branch_analysis") {
            $compare = $null
            if ((Split-Path $Directory -Leaf) -match "L4") {
                $candidate = Join-Path (Split-Path $Directory -Parent) "independent_reference_250_20260806_091943_L2"
                if (Test-Path $candidate) { $compare = $candidate }
            } elseif ((Split-Path $Directory -Leaf) -match "L2") {
                $candidate = Join-Path (Split-Path $Directory -Parent) "actual_branch_provenance_250_20260806_102928_L4"
                if (Test-Path $candidate) { $compare = $candidate }
            }
            if ($compare) { $analysisArgs += @("--compare-run-dir", $compare) }
        }
        & $python $runner `
            --stdout-log "$prefix.stdout.log" `
            --stderr-log "$prefix.stderr.log" `
            --combined-log "$prefix.combined.log" `
            --result-json "$prefix.result.json" `
            --failure-json "$prefix.failure.json" `
            --checkpoint (Join-Path $Directory "checkpoint.pkl") `
            -- $python $tool @analysisArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Analyzer failed; see $prefix.stderr.log"
        }
    }
    $zip = "$Directory.zip"
    $archiveItems = Get-ChildItem -LiteralPath $Directory -Force | Where-Object { $_.Extension -ne ".zip" } | Select-Object -ExpandProperty FullName
    Compress-Archive -Path $archiveItems -DestinationPath $zip -Force
    Write-Host "Analysis directory: $analysis"
    Write-Host "Archive: $zip"
}

if ($AnalyzeOnly) {
    if (-not $RunDirectory) { throw "-AnalyzeOnly requires -RunDirectory" }
    $resolved = (Resolve-Path $RunDirectory).Path
    Write-Host "AnalyzeOnly mode: no model process will be started."
    Invoke-Analysis $resolved
    exit 0
}

if (-not $RunFormal250 -and -not $ResumeRunDirectory) {
    Write-Host "This formal 250 run must be started manually by the user."
    Write-Host "Use -RunFormal250 or -AnalyzeOnly -RunDirectory <path>."
    exit 0
}

Write-Host "This formal 250 run must be started manually by the user."
$out = if ($ResumeRunDirectory) {
    (Resolve-Path $ResumeRunDirectory).Path
} else {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $path = Join-Path "results/fig9_diagnostics" "actual_branch_provenance_250_${stamp}_L$LMatch"
    New-Item -ItemType Directory -Force $path | Out-Null
    (Resolve-Path $path).Path
}
$env:PYTHONFAULTHANDLER = "1"
$env:PYTHONUNBUFFERED = "1"
$checkpoint = Join-Path $out "checkpoint.pkl"
$PythonArgs = @(
    "experiments/fig9_strict_reproduction.py",
    "--limit", "250", "--warmup", "200", "--streams", "original",
    "--l-match", "$LMatch", "--lmatch-real-ablation",
    "--prediction-horizon", "5", "--tie-break-seed", "0",
    "--continuous-impl", "reference", "--competition-mode", "competitive_raw",
    "--inhibition-strength", "0.1", "--inhibition-tau", "0.02",
    "--simultaneous-policy", "batched", "--simultaneous-bin-width", "0.005",
    "--intracolumn-selection-policy", "max_candidate_score",
    "--oracle-candidate-diagnostic",
    "--branch-provenance-diagnostic", "--branch-provenance-level", "candidate",
    "--preselection-segment-diagnostic", "--preselection-segment-level", "crossing",
    "--teacher-forced-winner-diagnostic", "--teacher-forced-winner-level", "segment",
    "--segment-reinforcement-diagnostic", "--segment-reinforcement-level", "event",
    "--independent-reference-diagnostic", "--independent-reference-level", "segment",
    "--independent-reference-compress",
    "--actual-branch-provenance-diagnostic", "--actual-branch-provenance-level", "segment",
    "--actual-branch-provenance-compress",
    "--observe-scenario-diagnostic", "--observe-scenario-level", "full",
    "--density-trace", "--interval-every", "25", "--progress-every", "10",
    "--checkpoint-every", "10", "--stream-diagnostic-traces",
    "--output-dir", $out, "--checkpoint-path", $checkpoint
)
if ($ResumeRunDirectory) { $PythonArgs += @("--resume-from", $checkpoint) }
@{
    command = @($python) + $PythonArgs
    python_args = $PythonArgs
    working_directory = (Get-Location).Path
    environment = @{ PYTHONFAULTHANDLER = $env:PYTHONFAULTHANDLER; PYTHONUNBUFFERED = $env:PYTHONUNBUFFERED }
} | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $out "command.json") -Encoding UTF8

& $python $runner `
    --stdout-log (Join-Path $out "stdout.log") `
    --stderr-log (Join-Path $out "stderr.log") `
    --combined-log (Join-Path $out "combined.log") `
    --result-json (Join-Path $out "result.json") `
    --failure-json (Join-Path $out "failure.json") `
    --checkpoint $checkpoint `
    -- (@($python) + $PythonArgs)
if ($LASTEXITCODE -ne 0) { throw "Model runner failed; see $out\stderr.log and $out\failure.json" }

Invoke-Analysis $out
Write-Host "Output directory: $out"
