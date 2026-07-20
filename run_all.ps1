$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $Python)) { throw "Missing .venv. See RUN.md." }
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Latest = Join-Path $Root "results/latest_strict"
$Archive = Join-Path $Root "results/strict_$Stamp"
New-Item -ItemType Directory -Force -Path $Latest,$Archive | Out-Null
Set-Location -LiteralPath $Root
$env:PYTHONPATH = "$Root/src;$Root"

function Run-Step {
    param([string]$Name, [string[]]$CommandArgs)
    $Log = Join-Path $Archive "$Name.txt"
    & $Python @CommandArgs 2>&1 | Tee-Object -FilePath $Log
    Copy-Item -LiteralPath $Log -Destination (Join-Path $Latest "$Name.txt") -Force
}

Run-Step "00_check" @("-m", "unittest", "discover", "-s", "tests", "-v")
Run-Step "fig7_trials" @("experiments/fig7_repeated_trials.py", "--output-dir", "$Archive/fig7")
Run-Step "fig7_reference" @("experiments/fig7_reference_comparison.py", "--snn-curves", "$Archive/fig7/curves.csv", "--output-dir", "$Archive/fig7_comparison")
Run-Step "fig8_cbt" @("experiments/fig8_sentence_memory.py", "--eval-samples", "0", "--output-csv", "$Archive/fig8_cbt.csv", "--output-plot", "$Archive/fig8_cbt.png")
Run-Step "fig8_resources" @("experiments/fig8_resource_sweep.py", "--output-dir", "$Archive/fig8_resources")
Run-Step "fig9_original" @("experiments/fig9_paper_snn.py", "--output-csv", "$Archive/fig9_original.csv", "--output-plot", "$Archive/fig9_original.png")
Run-Step "fig9_changed" @("experiments/fig9_paper_snn.py", "--data", "data/paper_nyc_taxi_perturb.csv", "--output-csv", "$Archive/fig9_changed.csv", "--output-plot", "$Archive/fig9_changed.png")

Copy-Item -LiteralPath (Join-Path $Archive "fig7") -Destination $Latest -Recurse -Force
Copy-Item -LiteralPath (Join-Path $Archive "fig7_comparison") -Destination $Latest -Recurse -Force
Copy-Item -LiteralPath (Join-Path $Archive "fig8_resources") -Destination $Latest -Recurse -Force
Copy-Item -LiteralPath (Join-Path $Archive "fig8_cbt.csv") -Destination $Latest -Force
Copy-Item -LiteralPath (Join-Path $Archive "fig8_cbt.png") -Destination $Latest -Force
Copy-Item -LiteralPath (Join-Path $Archive "fig9_original.csv") -Destination $Latest -Force
Copy-Item -LiteralPath (Join-Path $Archive "fig9_original.png") -Destination $Latest -Force
Copy-Item -LiteralPath (Join-Path $Archive "fig9_changed.csv") -Destination $Latest -Force
Copy-Item -LiteralPath (Join-Path $Archive "fig9_changed.png") -Destination $Latest -Force
Write-Output "Strict reproduction completed: $Archive"
