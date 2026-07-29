param(
    [ValidateSet(
        "existing",
        "max_candidate_score",
        "max_response_peak",
        "max_contributor_count",
        "context_then_score"
    )]
    [string]$Policy = "",
    [switch]$RunGrid,
    [string]$BaselineDir = ""
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $RepoRoot

Write-Host "This is the formal 250-record intracolumn selector diagnostic."
Write-Host "It must be started manually by the user."

if (-not $RunGrid -and [string]::IsNullOrWhiteSpace($Policy)) {
    Write-Host ""
    Write-Host "No experiment was started."
    Write-Host "Run the three-policy grid:"
    Write-Host "  .\scripts\fig9_intracolumn_selector_250.ps1 -RunGrid"
    Write-Host "Or one policy:"
    Write-Host "  .\scripts\fig9_intracolumn_selector_250.ps1 -Policy max_candidate_score"
    return
}

if ($RunGrid -and -not [string]::IsNullOrWhiteSpace($Policy)) {
    throw "Use either -RunGrid or -Policy, not both."
}

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python venv not found: $Python"
}

if ([string]::IsNullOrWhiteSpace($BaselineDir)) {
    $BaselineDir = Join-Path $RepoRoot (
        "results\fig9_diagnostics\" +
        "scenario_reason_neuron_250_20260729_195338"
    )
}
$BaselineDir = [System.IO.Path]::GetFullPath($BaselineDir)
if (-not (Test-Path -LiteralPath $BaselineDir)) {
    throw "Baseline directory not found: $BaselineDir"
}

$Policies = if ($RunGrid) {
    @(
        "max_candidate_score",
        "max_response_peak",
        "max_contributor_count"
    )
}
else {
    @($Policy)
}

$env:PYTHONPATH = "$RepoRoot\src;$RepoRoot"
$env:PYTHONFAULTHANDLER = "1"
$env:PYTHONUNBUFFERED = "1"

$GridStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$GridRoot = Join-Path $RepoRoot (
    "results\fig9_diagnostics\intracolumn_selector_250_$GridStamp"
)
New-Item -ItemType Directory -Path $GridRoot -Force | Out-Null
$ComparisonRows = @()

foreach ($CurrentPolicy in $Policies) {
    $PolicyStamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $PolicyRoot = Join-Path $GridRoot "${CurrentPolicy}_$PolicyStamp"
    $RunDir = Join-Path $PolicyRoot "run"
    $AnalysisDir = Join-Path $PolicyRoot "analysis"
    New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
    New-Item -ItemType Directory -Path $AnalysisDir -Force | Out-Null

    $RunLog = Join-Path $RunDir "run.log"
    $AnalysisLog = Join-Path $AnalysisDir "analysis.log"
    $Checkpoint = Join-Path $RunDir "checkpoint.pkl"
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
        "--intracolumn-selection-policy", $CurrentPolicy,
        "--intracolumn-selection-diagnostic",
        "--oracle-candidate-diagnostic",
        "--density-trace",
        "--interval-every", "25",
        "--checkpoint-path", $Checkpoint,
        "--checkpoint-every", "50"
    )

    Write-Host ""
    Write-Host "Running policy: $CurrentPolicy"
    & $Python @PythonArgs 2>&1 | Tee-Object -FilePath $RunLog
    $RunExitCode = $LASTEXITCODE
    if ($RunExitCode -ne 0) {
        Write-Error (
            "Policy $CurrentPolicy failed with exit code $RunExitCode. " +
            "See $RunLog"
        )
        continue
    }

    $AnalysisArgs = @(
        "-X", "faulthandler",
        "-u",
        "experiments\diagnostics\analyze_fig9_intracolumn_selector.py",
        "--run-dir", $RunDir,
        "--baseline-dir", $BaselineDir,
        "--output-dir", $AnalysisDir,
        "--policy-label", $CurrentPolicy,
        "--bootstrap-samples", "1000",
        "--bootstrap-seed", "0"
    )
    & $Python @AnalysisArgs 2>&1 | Tee-Object -FilePath $AnalysisLog
    $AnalysisExitCode = $LASTEXITCODE
    if ($AnalysisExitCode -ne 0) {
        Write-Error (
            "Analysis for $CurrentPolicy failed with exit code " +
            "$AnalysisExitCode. See $AnalysisLog"
        )
        continue
    }

    $Summary = Get-Content (
        Join-Path $RunDir "original_summary.json"
    ) -Raw | ConvertFrom-Json
    $Selection = Get-Content (
        Join-Path $RunDir "intracolumn_selection_summary.json"
    ) -Raw | ConvertFrom-Json
    $StepRows = Import-Csv (
        Join-Path $AnalysisDir "intracolumn_horizon_summary.csv"
    )
    $StepMape = ($StepRows | ForEach-Object {
        "step$($_.horizon_step)=$($_.mape)"
    }) -join ";"
    $ComparisonRows += [pscustomobject]@{
        policy = $CurrentPolicy
        output_directory = $PolicyRoot
        mape = $Summary.mape
        final_rolling_mape = $Summary.final_rolling_mape
        coverage = $Summary.coverage
        runtime_seconds = $Summary.runtime_seconds
        mean_raw_columns = $Summary.mean_raw_column_count
        peak_raw_columns = $Summary.peak_raw_column_count
        fraction_differing_from_existing = (
            $Selection.fraction_differing_from_existing
        )
        candidate_pool_fingerprint = $Selection.candidate_pool_fingerprint
        step1_candidate_pool_parity = "pending"
        per_step_mape = $StepMape
    }

    Write-Host "Policy completed: $CurrentPolicy"
    Write-Host "MAPE: $($Summary.mape)"
    Write-Host "Per-step MAPE: $StepMape"
    Write-Host "Runtime seconds: $($Summary.runtime_seconds)"
}

if ($ComparisonRows.Count -gt 0) {
    $AnchorTrace = Join-Path (
        Join-Path $ComparisonRows[0].output_directory "run"
    ) "intracolumn_selection_trace.csv"
    $AnchorStep1 = @{}
    foreach ($Row in Import-Csv $AnchorTrace) {
        if ([int]$Row.horizon_step -eq 1) {
            $AnchorStep1["$($Row.input_index)|$($Row.column)"] = (
                $Row.candidate_pool_fingerprint
            )
        }
    }
    foreach ($Comparison in $ComparisonRows) {
        $Trace = Join-Path (
            Join-Path $Comparison.output_directory "run"
        ) "intracolumn_selection_trace.csv"
        $CurrentStep1 = @{}
        foreach ($Row in Import-Csv $Trace) {
            if ([int]$Row.horizon_step -eq 1) {
                $CurrentStep1["$($Row.input_index)|$($Row.column)"] = (
                    $Row.candidate_pool_fingerprint
                )
            }
        }
        $Parity = $CurrentStep1.Count -eq $AnchorStep1.Count
        if ($Parity) {
            foreach ($Key in $AnchorStep1.Keys) {
                if (
                    -not $CurrentStep1.ContainsKey($Key) -or
                    $CurrentStep1[$Key] -ne $AnchorStep1[$Key]
                ) {
                    $Parity = $false
                    break
                }
            }
        }
        $Comparison.step1_candidate_pool_parity = $Parity
        if (-not $Parity) {
            throw (
                "Step-1 candidate-pool parity failed for policy " +
                $Comparison.policy
            )
        }
    }
}

$ComparisonCsv = Join-Path $GridRoot "selector_grid_comparison.csv"
$ComparisonJson = Join-Path $GridRoot "selector_grid_comparison.json"
$ComparisonRows | Export-Csv -Path $ComparisonCsv -NoTypeInformation
$ComparisonRows | ConvertTo-Json -Depth 5 |
    Set-Content -Path $ComparisonJson -Encoding UTF8

Write-Host ""
Write-Host "Formal selector diagnostics finished."
Write-Host "Grid root: $GridRoot"
Write-Host "Comparison CSV: $ComparisonCsv"
Write-Host "Comparison JSON: $ComparisonJson"
exit 0
