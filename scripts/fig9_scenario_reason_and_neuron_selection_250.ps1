param(
    [switch]$RunFormal250
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $RepoRoot

Write-Host "This is the formal 250-record Fig.9 scenario/reference audit."
Write-Host "It must be started manually by the user."
if (-not $RunFormal250) {
    Write-Host ""
    Write-Host "No experiment was started."
    Write-Host "Run intentionally with:"
    Write-Host "  .\scripts\fig9_scenario_reason_and_neuron_selection_250.ps1 -RunFormal250"
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
$Root = Join-Path $RepoRoot "results\fig9_diagnostics\scenario_reason_neuron_250_$Stamp"
$RunDir = Join-Path $Root "run"
$AnalysisDir = Join-Path $Root "analysis"
$Checkpoint = Join-Path $RunDir "checkpoint.pkl"
New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
New-Item -ItemType Directory -Path $AnalysisDir -Force | Out-Null

$RunLog = Join-Path $RunDir "run.log"
$IdentityLog = Join-Path $AnalysisDir "identity_analysis.log"
$ScenarioLog = Join-Path $AnalysisDir "scenario_analysis.log"
$SelectionLog = Join-Path $AnalysisDir "selection_analysis.log"
foreach ($Log in @($RunLog, $IdentityLog, $ScenarioLog, $SelectionLog)) {
    New-Item -ItemType File -Path $Log -Force | Out-Null
}

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
    "--observe-scenario-level", "full",
    "--reference-neuron-selection-diagnostic",
    "--reference-neuron-selection-level", "crossing",
    "--density-trace",
    "--interval-every", "25",
    "--checkpoint-path", $Checkpoint,
    "--checkpoint-every", "50"
)

Write-Host ""
Write-Host "Running formal 250-record trace capture..."
& $Python @PythonArgs 2>&1 | Tee-Object -FilePath $RunLog
$ProcessExitCode = $LASTEXITCODE
if ($ProcessExitCode -ne 0) {
    throw "Trace capture failed with exit code $ProcessExitCode. See $RunLog"
}

$IdentityArgs = @(
    "-X", "faulthandler",
    "-u",
    "experiments\diagnostics\analyze_fig9_teacher_forced_identity.py",
    "--run-dir", $RunDir,
    "--output-dir", $AnalysisDir,
    "--policy-label", "batched",
    "--bootstrap-samples", "1000",
    "--bootstrap-seed", "0"
)
& $Python @IdentityArgs 2>&1 | Tee-Object -FilePath $IdentityLog
$ProcessExitCode = $LASTEXITCODE
if ($ProcessExitCode -ne 0) {
    throw "Identity analysis failed with exit code $ProcessExitCode."
}

$ScenarioArgs = @(
    "-X", "faulthandler",
    "-u",
    "experiments\diagnostics\analyze_fig9_observe_scenario_reasons.py",
    "--run-dir", $RunDir,
    "--output-dir", $AnalysisDir,
    "--bootstrap-samples", "1000",
    "--bootstrap-seed", "0"
)
& $Python @ScenarioArgs 2>&1 | Tee-Object -FilePath $ScenarioLog
$ProcessExitCode = $LASTEXITCODE
if ($ProcessExitCode -ne 0) {
    throw "Scenario analysis failed with exit code $ProcessExitCode."
}

$SelectionArgs = @(
    "-X", "faulthandler",
    "-u",
    "experiments\diagnostics\analyze_fig9_reference_neuron_selection.py",
    "--run-dir", $RunDir,
    "--identity-dir", $AnalysisDir,
    "--output-dir", $AnalysisDir,
    "--bootstrap-samples", "1000",
    "--bootstrap-seed", "0"
)
& $Python @SelectionArgs 2>&1 | Tee-Object -FilePath $SelectionLog
$ProcessExitCode = $LASTEXITCODE
if ($ProcessExitCode -ne 0) {
    throw "Reference-neuron analysis failed with exit code $ProcessExitCode."
}

Get-ChildItem -LiteralPath $RunDir -File |
    Where-Object {
        $_.Length -gt 10MB -and
        $_.Extension -eq ".csv" -and
        $_.Name -notin @(
            "observe_scenario_trace.csv",
            "teacher_forced_observation_trace.csv"
        )
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
            Remove-Item -LiteralPath $_.FullName
        }
    }

$ScenarioReport = Join-Path $AnalysisDir "OBSERVE_SCENARIO_REASON_REPORT.md"
$SelectionReport = Join-Path $AnalysisDir "REFERENCE_NEURON_SELECTION_REPORT.md"
if (-not (Test-Path -LiteralPath $ScenarioReport)) {
    throw "Missing scenario report: $ScenarioReport"
}
if (-not (Test-Path -LiteralPath $SelectionReport)) {
    throw "Missing selection report: $SelectionReport"
}

Write-Host ""
Write-Host "Output file sizes:"
Get-ChildItem -LiteralPath $Root -File -Recurse |
    Sort-Object FullName |
    Select-Object FullName, Length |
    Format-Table -AutoSize

Write-Host ""
Write-Host "Formal diagnostic completed."
Write-Host "Root: $Root"
Write-Host "Scenario report: $ScenarioReport"
Write-Host "Reference-neuron report: $SelectionReport"
exit 0
