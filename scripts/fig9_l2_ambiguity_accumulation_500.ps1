param(
    [switch]$RunGrid,
    [ValidateSet(2, 4)]
    [int]$LMatch = 2,
    [string]$OutputRoot = "",
    [switch]$AnalyzeOnly,
    [string]$RunRoot = ""
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
Set-Location $RepoRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Supervisor = Join-Path $RepoRoot "scripts\run_logged_process.py"
if (-not (Test-Path -LiteralPath $Python)) { throw "Python venv not found: $Python" }
if (-not (Test-Path -LiteralPath $Supervisor)) { throw "Process supervisor not found: $Supervisor" }

function Get-NextIndex([string]$RunDirectory) {
    $metadata = Join-Path $RunDirectory "checkpoint.metadata.json"
    if (-not (Test-Path -LiteralPath $metadata)) { return 0 }
    return [int]((Get-Content $metadata -Raw | ConvertFrom-Json).next_index)
}

function Invoke-LMatchRun([int]$Value, [string]$Root) {
    $run = Join-Path $Root "L${Value}_ambiguity_500"
    New-Item -ItemType Directory -Force -Path $run | Out-Null
    $checkpoint = Join-Path $run "checkpoint.pkl"
    $attempt = 0
    while ((Get-NextIndex $run) -lt 495) {
        $start = Get-NextIndex $run
        $end = [Math]::Min($start + 5, 495)
        $attempt++
        $suffix = "{0:D4}_{1:D4}_a{2:D3}" -f $start, $end, $attempt
        $pythonArgs = @(
            "-X", "faulthandler", "-u",
            "-c",
            "import sys,runpy; from seqmem.model import SequentialMemory; SequentialMemory.SPIKE_RESPONSE_CACHE_LIMIT=50000; sys.argv=['experiments/fig9_strict_reproduction.py']+sys.argv[1:]; runpy.run_path('experiments/fig9_strict_reproduction.py',run_name='__main__')",
            "--limit", "500", "--warmup", "200", "--streams", "original",
            "--prediction-horizon", "5", "--tie-break-seed", "0",
            "--l-match", "$Value", "--continuous-impl", "reference",
            "--competition-mode", "competitive_raw", "--inhibition-strength", "0.1",
            "--inhibition-tau", "0.02", "--simultaneous-policy", "batched",
            "--simultaneous-bin-width", "0.005", "--intracolumn-selection-policy",
            "max_candidate_score", "--density-trace", "--long-sequence-ledger",
            "--ambiguity-diagnostic", "--progress-every", "1",
            "--checkpoint-path", $checkpoint, "--checkpoint-every", "1",
            "--debug-end-index", "$end", "--output-dir", $run
        )
        if ($Value -eq 2) { $pythonArgs += @("--lmatch-real-ablation") }
        if ($start -gt 0) { $pythonArgs += @("--resume-from", $checkpoint) }
        $command = @($Python) + $pythonArgs
        [ordered]@{
            command = $command
            python_args = $pythonArgs
            working_directory = (Get-Location).Path
            start_index = $start
            debug_end_index = $end
            attempt = $attempt
            ambiguity_diagnostic = $true
        } | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $run "command_$suffix.json") -Encoding UTF8
        Write-Host "L_match=$Value attempt=$attempt index=$start->$end"
        & $Python $Supervisor `
            --stdout-log (Join-Path $run "stdout_$suffix.log") `
            --stderr-log (Join-Path $run "stderr_$suffix.log") `
            --combined-log (Join-Path $run "combined_$suffix.log") `
            --result-json (Join-Path $run "process_$suffix.json") `
            --failure-json (Join-Path $run "failure_$suffix.json") `
            --checkpoint $checkpoint `
            -- (@($Python) + $pythonArgs)
        $supervisorExitCode = $LASTEXITCODE
        $processResultPath = Join-Path $run "process_$suffix.json"
        $processResult = $null
        if (Test-Path -LiteralPath $processResultPath) {
            $processResult = Get-Content -LiteralPath $processResultPath -Raw |
                ConvertFrom-Json
        }
        # run_logged_process.py returns 1 for a child failure so it can print
        # the stderr tail. The durable result JSON retains the native code.
        $exitCode = if ($processResult -and $null -ne $processResult.exit_code) {
            [int64]$processResult.exit_code
        } else {
            [int64]$supervisorExitCode
        }
        $next = Get-NextIndex $run
        if ($exitCode -eq 0 -and $next -gt $start) { continue }
        if ($exitCode -eq 3221225477 -and $next -gt $start) { continue }
        throw "L_match=$Value did not advance from $start; exit_code=$exitCode; see $run"
    }
    return $run
}

Write-Host "This is the Fig.9 L2 ambiguity accumulation 500-record diagnostic."
Write-Host "It is diagnostic-only, uses no new model mechanism, and does not run 1000 records."

if ($AnalyzeOnly) {
    if (-not $RunRoot) { throw "-AnalyzeOnly requires -RunRoot" }
    $root = (Resolve-Path $RunRoot).Path
    $analysis = Join-Path $root "analysis"
    & $Python "experiments\diagnostics\analyze_fig9_l2_ambiguity_accumulation.py" `
        --l2-dir (Join-Path $root "L2_ambiguity_500") `
        --l4-dir (Join-Path $root "L4_ambiguity_500") `
        --output-dir $analysis
    if ($LASTEXITCODE -ne 0) { throw "Ambiguity analyzer failed" }
    Write-Host "Analysis complete: $analysis"
    exit 0
}

$env:PYTHONPATH = "$RepoRoot\src;$RepoRoot"
$env:PYTHONFAULTHANDLER = "1"
$env:PYTHONUNBUFFERED = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$root = if ($OutputRoot) {
    New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
    (Resolve-Path $OutputRoot).Path
} else {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $path = Join-Path $RepoRoot "results\fig9_diagnostics\l2_ambiguity_accumulation_$stamp"
    New-Item -ItemType Directory -Force -Path $path | Out-Null
    (Resolve-Path $path).Path
}
$requested = if ($RunGrid) { @(2, 4) } else { @($LMatch) }
foreach ($value in $requested) { Invoke-LMatchRun $value $root | Out-Null }
if ($RunGrid) {
    & $Python "experiments\diagnostics\analyze_fig9_l2_ambiguity_accumulation.py" `
        --l2-dir (Join-Path $root "L2_ambiguity_500") `
        --l4-dir (Join-Path $root "L4_ambiguity_500") `
        --output-dir (Join-Path $root "analysis")
    if ($LASTEXITCODE -ne 0) { throw "Ambiguity analyzer failed" }
}
Write-Host "Root output: $root"
