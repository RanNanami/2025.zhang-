param(
    [switch]$RunFormal250
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $RepoRoot

Write-Host "This is the formal 250-record Fig.9 scenario-aware identity diagnostic."
Write-Host "It must be started manually by the user."
if (-not $RunFormal250) {
    Write-Host ""
    Write-Host "No experiment was started."
    Write-Host "Run intentionally with:"
    Write-Host "  .\scripts\fig9_scenario_aware_identity_250.ps1 -RunFormal250"
    return
}

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python venv not found: $Python"
}

$env:PYTHONPATH = "$RepoRoot\src;$RepoRoot"
$env:PYTHONFAULTHANDLER = "1"
$env:PYTHONUNBUFFERED = "1"

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Root = Join-Path $RepoRoot "results\fig9_diagnostics\scenario_aware_identity_250_$Stamp"
$RunDir = Join-Path $Root "run"
$AnalysisDir = Join-Path $Root "analysis"
$RunLog = Join-Path $RunDir "run.log"
$AnalysisLog = Join-Path $AnalysisDir "analysis.log"
$Checkpoint = Join-Path $RunDir "checkpoint.pkl"
New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
New-Item -ItemType Directory -Path $AnalysisDir -Force | Out-Null
New-Item -ItemType File -Path $RunLog -Force | Out-Null
New-Item -ItemType File -Path $AnalysisLog -Force | Out-Null

$PythonArgs = @(
    "-X", "faulthandler",
    "-u",
    "experiments\fig9_strict_reproduction.py",
    "--limit", "250",
    "--warmup", "200",
    "--streams", "original",
    "--output-dir", $RunDir,
    "--continuous-impl", "reference",
    "--competition-mode", "competitive_raw",
    "--inhibition-strength", "0.1",
    "--inhibition-tau", "0.02",
    "--simultaneous-policy", "batched",
    "--simultaneous-bin-width", "0.005",
    "--oracle-candidate-diagnostic",
    "--branch-provenance-diagnostic",
    "--branch-provenance-level", "candidate",
    "--preselection-segment-diagnostic",
    "--preselection-segment-level", "crossing",
    "--preselection-segment-compress",
    "--teacher-forced-winner-diagnostic",
    "--teacher-forced-winner-level", "segment",
    "--observe-scenario-diagnostic",
    "--reference-neuron-selection-diagnostic",
    "--density-trace",
    "--interval-every", "25",
    "--checkpoint-path", $Checkpoint,
    "--checkpoint-every", "50"
)

Write-Host ""
Write-Host "Running formal scenario-aware capture..."
& $Python @PythonArgs 2>&1 | Tee-Object -FilePath $RunLog
$ProcessExitCode = $LASTEXITCODE
if ($ProcessExitCode -ne 0) {
    throw "Scenario-aware run failed with exit code $ProcessExitCode. See $RunLog"
}

$AnalysisArgs = @(
    "-X", "faulthandler",
    "-u",
    "experiments\diagnostics\analyze_fig9_scenario_aware_identity.py",
    "--run-dir", $RunDir,
    "--output-dir", $AnalysisDir,
    "--policy-label", "batched",
    "--bootstrap-samples", "1000",
    "--bootstrap-seed", "0"
)

Write-Host ""
Write-Host "Running offline scenario-aware analysis..."
& $Python @AnalysisArgs 2>&1 | Tee-Object -FilePath $AnalysisLog
$ProcessExitCode = $LASTEXITCODE
if ($ProcessExitCode -ne 0) {
    throw "Offline analysis failed with exit code $ProcessExitCode. See $AnalysisLog"
}

Get-ChildItem -LiteralPath $RunDir -File |
    Where-Object {
        $_.Length -gt 10MB -and
        $_.Extension -eq ".csv" -and
        $_.Name -ne "teacher_forced_observation_trace.csv"
    } |
    ForEach-Object {
        $GzipPath = "$($_.FullName).gz"
        if (-not (Test-Path -LiteralPath $GzipPath)) {
            $Input = [System.IO.File]::OpenRead($_.FullName)
            try {
                $Output = [System.IO.File]::Create($GzipPath)
                try {
                    $Gzip = [System.IO.Compression.GZipStream]::new(
                        $Output,
                        [System.IO.Compression.CompressionMode]::Compress
                    )
                    try {
                        $Input.CopyTo($Gzip)
                    }
                    finally {
                        $Gzip.Dispose()
                    }
                }
                finally {
                    $Output.Dispose()
                }
            }
            finally {
                $Input.Dispose()
            }
        }
    }

$Report = Join-Path $AnalysisDir "FIG9_SCENARIO_AWARE_IDENTITY_REPORT.md"
if (-not (Test-Path -LiteralPath $Report)) {
    throw "Offline analysis did not produce $Report"
}

Write-Host ""
Write-Host "Output file sizes:"
Get-ChildItem -LiteralPath $Root -File -Recurse |
    Sort-Object FullName |
    Select-Object FullName, Length |
    Format-Table -AutoSize

Write-Host ""
Write-Host "Formal scenario-aware diagnostic completed."
Write-Host "Root: $Root"
Write-Host "Run log: $RunLog"
Write-Host "Analysis log: $AnalysisLog"
Write-Host "Report: $Report"
exit 0

