$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$LogDir = Join-Path $RepoRoot "results\script_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Log = Join-Path $LogDir "test_all.log"

$env:PYTHONPATH = "$RepoRoot\src;$RepoRoot"
Write-Host "Running compileall and unittest from $RepoRoot"
$CompileCommand = "`"$Python`" -m compileall -q src experiments tests 2>&1"
& cmd.exe /d /c $CompileCommand | Tee-Object -FilePath $Log
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$TestCommand = "`"$Python`" -m unittest discover -s tests -v 2>&1"
& cmd.exe /d /c $TestCommand | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
