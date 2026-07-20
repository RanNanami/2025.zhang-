param(
    [int]$Warmup = 1000,
    [int]$TestSize = 200
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$Python = if (Test-Path -LiteralPath $VenvPython) { $VenvPython } else { "python" }

New-Item -ItemType Directory -Force -Path "results" | Out-Null

& $Python -u experiments\diagnostics\multivariate_fig9_transfer.py `
    --dataset etth1 --warmup $Warmup --test-size $TestSize `
    --output-csv results\multivariate_etth1_medium_h1.csv `
    --output-plot results\multivariate_etth1_medium_h1.png

& $Python -u experiments\diagnostics\multivariate_fig9_transfer.py `
    --dataset weather --warmup $Warmup --test-size $TestSize `
    --output-csv results\multivariate_weather_medium_h1.csv `
    --output-plot results\multivariate_weather_medium_h1.png

Write-Host "Finished multivariate Fig. 9 extension runs."
