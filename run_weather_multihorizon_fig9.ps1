param(
    [int]$Warmup = 500,
    [int]$TestSize = 20
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$BundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$Python = if (Test-Path -LiteralPath $BundledPython) {
    $BundledPython
} elseif (Test-Path -LiteralPath $VenvPython) {
    $VenvPython
} else {
    "python"
}

New-Item -ItemType Directory -Force -Path "results" | Out-Null
& $Python -u experiments\diagnostics\multihorizon_fig9_transfer.py `
    --dataset weather --warmup $Warmup --test-size $TestSize `
    --horizons 12,24,48,96 `
    --output-csv results\multivariate_weather_multihorizon.csv `
    --output-summary results\multivariate_weather_multihorizon_summary.csv
