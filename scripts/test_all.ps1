$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $RepoRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}

$LogDir = Join-Path $RepoRoot "results\script_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Log = Join-Path $LogDir "test_all.log"
New-Item -ItemType File -Force -Path $Log | Out-Null

$env:PYTHONPATH = "$RepoRoot\src;$RepoRoot"
$env:PYTHONFAULTHANDLER = "1"
$env:PYTHONUNBUFFERED = "1"

Write-Host "Running compileall and unittest from $RepoRoot"

function Invoke-PythonLogged {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Stage,
        [Parameter(Mandatory = $true)]
        [string[]]$PythonArgs
    )

    # unittest writes normal progress to stderr. Separate process streams avoid
    # Windows PowerShell 5.1 mislabeling those lines as NativeCommandError.
    $StdoutLog = Join-Path $LogDir "$Stage.stdout.log"
    $StderrLog = Join-Path $LogDir "$Stage.stderr.log"
    New-Item -ItemType File -Force -Path $StdoutLog, $StderrLog | Out-Null
    $Process = Start-Process `
        -FilePath $Python `
        -ArgumentList $PythonArgs `
        -WorkingDirectory $RepoRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $StdoutLog `
        -RedirectStandardError $StderrLog `
        -Wait `
        -PassThru

    Get-Content -LiteralPath $StdoutLog, $StderrLog |
        Tee-Object -FilePath $Log -Append
    $ProcessExitCode = $Process.ExitCode

    if ($ProcessExitCode -ne 0) {
        throw "$Stage failed with exit code $ProcessExitCode. See $Log"
    }
}

$PythonArgs = @("-m", "unittest", "discover", "-s", "tests", "-v")
Invoke-PythonLogged -Stage "compileall" -PythonArgs @(
    "-m", "compileall", "-q", "src", "experiments", "tests"
)
Invoke-PythonLogged -Stage "unittest" -PythonArgs $PythonArgs

Write-Host "All tests passed. Log: $Log"
