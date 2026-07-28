param(
    [Parameter(Mandatory = $true)]
    [string]$SequentialDir,

    [Parameter(Mandatory = $true)]
    [string]$BatchedDir,

    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $RepoRoot

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python venv not found: $Python"
}

$SequentialDir = (Resolve-Path $SequentialDir).Path
$BatchedDir = (Resolve-Path $BatchedDir).Path
if (-not $OutputDir) {
    $OutputDir = $BatchedDir
}
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
$OutputDir = (Resolve-Path $OutputDir).Path

$SequentialTrace = Join-Path $SequentialDir "oracle_candidate_trace.csv"
$BatchedTrace = Join-Path $BatchedDir "oracle_candidate_trace.csv"
$SequentialSummary = Join-Path $SequentialDir "original_summary.json"
$BatchedSummary = Join-Path $BatchedDir "original_summary.json"
foreach ($Path in @(
    $SequentialTrace,
    $BatchedTrace,
    $SequentialSummary,
    $BatchedSummary
)) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required comparison input not found: $Path"
    }
}

$env:PYTHONPATH = "$RepoRoot\src;$RepoRoot"
$env:PYTHONFAULTHANDLER = "1"
$env:PYTHONUNBUFFERED = "1"

$LogPath = Join-Path $OutputDir "competition_policy_comparison.log"
New-Item -ItemType File -Path $LogPath -Force | Out-Null
$PythonArgs = @(
    "-X", "faulthandler",
    "-u",
    "experiments\diagnostics\compare_fig9_competition_policies.py",
    "--sequential-trace", $SequentialTrace,
    "--batched-trace", $BatchedTrace,
    "--sequential-summary", $SequentialSummary,
    "--batched-summary", $BatchedSummary,
    "--output-dir", $OutputDir
)

& $Python @PythonArgs 2>&1 | Tee-Object -FilePath $LogPath
$ProcessExitCode = $LASTEXITCODE
if ($ProcessExitCode -ne 0) {
    Write-Error "Policy comparison failed with exit code $ProcessExitCode. See $LogPath"
    exit $ProcessExitCode
}

$ComparisonPath = Join-Path $OutputDir "competition_policy_comparison.json"
if (-not (Test-Path -LiteralPath $ComparisonPath)) {
    Write-Error "Comparison completed without expected output: $ComparisonPath"
    exit 2
}

Write-Host ""
Write-Host "Sequential/batched comparison completed."
Write-Host "Comparison JSON: $ComparisonPath"
Write-Host "Comparison CSV: $(Join-Path $OutputDir 'competition_policy_comparison.csv')"
Write-Host "Log: $LogPath"
exit 0
