$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$Python = if (Test-Path -LiteralPath $VenvPython) { $VenvPython } else { "python" }

New-Item -ItemType Directory -Force -Path "results" | Out-Null

& $Python experiments\diagnostics\benchmark_real_timeseries.py --dataset etth1 --data data\ETTh1.csv --horizon 1 --lookback 24 --test-size 2000 *> results\benchmark_etth1_h1.txt
& $Python experiments\diagnostics\benchmark_real_timeseries.py --dataset etth1 --data data\ETTh1.csv --horizon 24 --lookback 24 --test-size 2000 *> results\benchmark_etth1_h24.txt
& $Python experiments\diagnostics\benchmark_real_timeseries.py --dataset weather --data data\weather\weather.csv --horizon 1 --lookback 144 --test-size 2000 --train-stride 1 *> results\benchmark_weather_h1_full_memory.txt
& $Python experiments\diagnostics\benchmark_real_timeseries.py --dataset weather --data data\weather\weather.csv --horizon 144 --lookback 144 --test-size 2000 --train-stride 1 *> results\benchmark_weather_h144_full_memory.txt

Write-Host "Finished. Results are in $Root\results"
