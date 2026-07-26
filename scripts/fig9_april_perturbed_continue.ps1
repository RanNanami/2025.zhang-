param(
    [Parameter(Mandatory=$true)]
    [string]$CheckpointPath
)
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutDir = Join-Path $RepoRoot "results\fig9_strict\april_perturbed_continue_$Stamp"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$Log = Join-Path $OutDir "run.log"

$env:PYTHONPATH = "$RepoRoot\src;$RepoRoot"
$QuoteArg = { param([string]$Value) '"' + ($Value -replace '"', '\"') + '"' }
$Command = @($Python, "experiments\fig9_strict_reproduction.py", "--streams", "perturbed", "--output-dir", $OutDir, "--resume-from", $CheckpointPath, "--density-trace", "--interval-every", "250")
$CommandLine = (($Command | ForEach-Object { & $QuoteArg $_ }) -join " ") + " 2>&1"
& cmd.exe /d /c $CommandLine | Tee-Object -FilePath $Log
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
