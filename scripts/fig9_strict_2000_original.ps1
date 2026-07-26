$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutDir = Join-Path $RepoRoot "results\fig9_strict\stage_b_2000_original_$Stamp"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$Log = Join-Path $OutDir "run.log"

$env:PYTHONPATH = "$RepoRoot\src;$RepoRoot"
$QuoteArg = { param([string]$Value) '"' + ($Value -replace '"', '\"') + '"' }
$Command = @(
    $Python, "experiments\fig9_strict_reproduction.py",
    "--limit", "2000",
    "--warmup", "200",
    "--streams", "original",
    "--output-dir", $OutDir,
    "--density-trace",
    "--interval-every", "250"
)
Write-Host "Running: $($Command -join ' ')"
$CommandLine = (($Command | ForEach-Object { & $QuoteArg $_ }) -join " ") + " 2>&1"
& cmd.exe /d /c $CommandLine | Tee-Object -FilePath $Log
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
