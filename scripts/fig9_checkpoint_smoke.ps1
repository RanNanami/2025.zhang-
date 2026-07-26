$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutDir = Join-Path $RepoRoot "results\fig9_strict\checkpoint_smoke_$Stamp"
$Checkpoint = Join-Path $OutDir "strict_state.pkl"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$Log = Join-Path $OutDir "run.log"

$env:PYTHONPATH = "$RepoRoot\src;$RepoRoot"
$QuoteArg = { param([string]$Value) '"' + ($Value -replace '"', '\"') + '"' }
Write-Host "Running checkpoint prefix smoke"
$PrefixCommand = @($Python, "experiments\fig9_strict_reproduction.py", "--limit", "24", "--warmup", "8", "--streams", "original", "--output-dir", (Join-Path $OutDir "prefix"), "--checkpoint-path", $Checkpoint, "--checkpoint-every", "12", "--stop-after-record", "12")
$PrefixLine = (($PrefixCommand | ForEach-Object { & $QuoteArg $_ }) -join " ") + " 2>&1"
& cmd.exe /d /c $PrefixLine | Tee-Object -FilePath $Log
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Running checkpoint resume smoke"
$ResumeCommand = @($Python, "experiments\fig9_strict_reproduction.py", "--limit", "24", "--warmup", "8", "--streams", "original", "--output-dir", (Join-Path $OutDir "resumed"), "--resume-from", $Checkpoint, "--density-trace", "--debug-record-index", "12", "--debug-output-json", (Join-Path $OutDir "debug_record.json"), "--profile", "--profile-output", (Join-Path $OutDir "profile.txt"))
$ResumeLine = (($ResumeCommand | ForEach-Object { & $QuoteArg $_ }) -join " ") + " 2>&1"
& cmd.exe /d /c $ResumeLine | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
