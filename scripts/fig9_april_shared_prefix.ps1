$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutDir = Join-Path $RepoRoot "results\fig9_strict\april_shared_prefix_$Stamp"
$Checkpoint = Join-Path $OutDir "april1_shared_prefix.pkl"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$Log = Join-Path $OutDir "run.log"

$env:PYTHONPATH = "$RepoRoot\src;$RepoRoot"
$QuoteArg = { param([string]$Value) '"' + ($Value -replace '"', '\"') + '"' }
$Split = & $Python -c "from pathlib import Path; from experiments.fig9 import read_records; from experiments.fig9_strict_reproduction import find_timestamp_split; print(find_timestamp_split(read_records(Path('data/paper_nyc_taxi.csv'), 0)))"
Write-Host "Timestamp split index: $Split"
$Command = @($Python, "experiments\fig9_strict_reproduction.py", "--streams", "original", "--output-dir", $OutDir, "--checkpoint-path", $Checkpoint, "--checkpoint-at-index", $Split, "--stop-after-record", $Split)
$CommandLine = (($Command | ForEach-Object { & $QuoteArg $_ }) -join " ") + " 2>&1"
& cmd.exe /d /c $CommandLine | Tee-Object -FilePath $Log
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Shared checkpoint: $Checkpoint"
