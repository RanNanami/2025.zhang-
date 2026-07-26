param(
    [switch]$RunFormal250
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $RepoRoot

Write-Host "This is the formal 250-record A/B benchmark."
Write-Host "It must be started manually by the user."
if (-not $RunFormal250) {
    Write-Host ""
    Write-Host "This script no longer runs the formal 250-record benchmark by default."
    Write-Host "To run it intentionally, execute:"
    Write-Host "  .\scripts\fig9_strict_250_ab.ps1 -RunFormal250"
    return
}

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Python venv not found: $Python"
}

$env:PYTHONFAULTHANDLER = "1"
$env:PYTHONUNBUFFERED = "1"

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$BenchmarkRoot = Join-Path $RepoRoot "results\fig9_strict\formal_250_ab_$Stamp"
New-Item -ItemType Directory -Path $BenchmarkRoot -Force | Out-Null

$Modes = @("reference", "optimized_v1", "optimized_v2")
$Results = @()

foreach ($Mode in $Modes) {
    $OutDir = Join-Path $BenchmarkRoot "${Mode}_$Stamp"
    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
    $LogPath = Join-Path $OutDir "run.log"
    New-Item -ItemType File -Path $LogPath -Force | Out-Null

    $PythonArgs = @(
        "experiments\fig9_strict_reproduction.py",
        "--limit", "250",
        "--warmup", "200",
        "--streams", "original",
        "--output-dir", $OutDir,
        "--continuous-impl", $Mode
    )

    Write-Host ""
    Write-Host "Running $Mode..."
    $WallStart = Get-Date
    & $Python @PythonArgs 2>&1 | Tee-Object -FilePath $LogPath
    if ($LASTEXITCODE -ne 0) {
        throw "$Mode failed with exit code $LASTEXITCODE. See $LogPath"
    }
    $WallSeconds = ((Get-Date) - $WallStart).TotalSeconds

    $SummaryPath = Join-Path $OutDir "original_summary.json"
    $PredictionsPath = Join-Path $OutDir "original_predictions.csv"
    if (-not (Test-Path $SummaryPath)) {
        throw "$Mode did not produce $SummaryPath"
    }
    if (-not (Test-Path $PredictionsPath)) {
        throw "$Mode did not produce $PredictionsPath"
    }

    $Summary = Get-Content $SummaryPath -Raw | ConvertFrom-Json
    $PredictionsHash = (Get-FileHash -Algorithm SHA256 $PredictionsPath).Hash.ToLowerInvariant()

    $Results += [PSCustomObject]@{
        mode = $Mode
        output_dir = $OutDir
        log_path = $LogPath
        runtime_seconds = [double]$Summary.runtime_seconds
        wall_time_seconds = [double]$WallSeconds
        mape = [double]$Summary.mape
        coverage = [double]$Summary.coverage
        mean_raw_columns = [double]$Summary.mean_raw_column_count
        peak_raw_columns = [double]$Summary.peak_raw_column_count
        final_rolling_mape = $Summary.final_rolling_mape
        predictions_csv_sha256 = $PredictionsHash
        final_model_fingerprint = $Summary.final_model_fingerprint
        final_rng_fingerprint = $Summary.final_rng_fingerprint
    }
}

$Reference = $Results | Where-Object { $_.mode -eq "reference" } | Select-Object -First 1
$ReferenceSha = $Reference.predictions_csv_sha256
$ReferenceModelFingerprint = $Reference.final_model_fingerprint
$ReferenceRngFingerprint = $Reference.final_rng_fingerprint

foreach ($Result in $Results) {
    if ($Result.predictions_csv_sha256 -ne $ReferenceSha) {
        throw "Predictions SHA256 mismatch for $($Result.mode): $($Result.predictions_csv_sha256) != $ReferenceSha"
    }
    if ($Result.final_model_fingerprint -ne $ReferenceModelFingerprint) {
        throw "Final model fingerprint mismatch for $($Result.mode)"
    }
    if ($Result.final_rng_fingerprint -ne $ReferenceRngFingerprint) {
        throw "Final RNG fingerprint mismatch for $($Result.mode)"
    }
}

$Comparison = foreach ($Result in $Results) {
    $Speedup = if ($Result.runtime_seconds -gt 0) {
        [double]$Reference.runtime_seconds / [double]$Result.runtime_seconds
    } else {
        $null
    }
    [PSCustomObject]@{
        mode = $Result.mode
        output_dir = $Result.output_dir
        runtime_seconds = $Result.runtime_seconds
        wall_time_seconds = $Result.wall_time_seconds
        mape = $Result.mape
        coverage = $Result.coverage
        mean_raw_columns = $Result.mean_raw_columns
        peak_raw_columns = $Result.peak_raw_columns
        final_rolling_mape = $Result.final_rolling_mape
        predictions_csv_sha256 = $Result.predictions_csv_sha256
        speedup_vs_reference = $Speedup
    }
}

$JsonPath = Join-Path $BenchmarkRoot "fig9_strict_250_ab_comparison.json"
$TextPath = Join-Path $BenchmarkRoot "fig9_strict_250_ab_comparison.txt"
$Comparison | ConvertTo-Json -Depth 5 | Set-Content -Path $JsonPath -Encoding UTF8
$Comparison | Format-Table -AutoSize | Out-String | Set-Content -Path $TextPath -Encoding UTF8

Write-Host ""
Write-Host "Formal 250-record A/B benchmark finished."
Write-Host "Comparison JSON: $JsonPath"
Write-Host "Comparison text: $TextPath"
