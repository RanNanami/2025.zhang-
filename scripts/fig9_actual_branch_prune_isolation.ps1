param(
    [ValidateSet("exact-all", "actual-off", "actual-summary", "branch-off", "minimal")]
    [string]$Group = "exact-all",
    [int]$StartIndex = 10,
    [int]$EndIndex = 20,
    [string]$BaseRun = "results\fig9_diagnostics\actual_branch_prune_resume_base",
    [string]$OutputRoot = "results\fig9_diagnostics\actual_branch_prune_isolation"
)

$ErrorActionPreference = "Stop"
if ($EndIndex -le $StartIndex -or $EndIndex - $StartIndex -gt 10) {
    throw "Isolation must advance from the checkpoint by 1..10 records."
}

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repo
$stamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
$runDir = Join-Path $OutputRoot ("{0}_{1}" -f $Group, $stamp)
$sourceDir = (Resolve-Path $BaseRun).Path
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$env:PYTHONFAULTHANDLER = "1"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONMALLOC = "debug"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:PYTHONPATH = "$repo\src;$repo"

& .\.venv\Scripts\python.exe scripts\prepare_resume_from_checkpoint.py `
    --source-run-dir $sourceDir `
    --target-run-dir $runDir
if ($LASTEXITCODE -ne 0) { throw "Could not prepare isolated resume directory." }

$flags = @(
    "--limit", "250", "--warmup", "200", "--streams", "original",
    "--l-match", "2", "--lmatch-real-ablation", "--prediction-horizon", "5",
    "--tie-break-seed", "0", "--continuous-impl", "reference",
    "--competition-mode", "competitive_raw", "--inhibition-strength", "0.1",
    "--inhibition-tau", "0.02", "--simultaneous-policy", "batched",
    "--simultaneous-bin-width", "0.005",
    "--intracolumn-selection-policy", "max_candidate_score",
    "--oracle-candidate-diagnostic", "--branch-provenance-level", "candidate",
    "--preselection-segment-level", "crossing", "--teacher-forced-winner-level", "segment",
    "--observe-scenario-level", "full", "--segment-reinforcement-level", "event",
    "--independent-reference-level", "segment", "--independent-reference-compress",
    "--actual-branch-provenance-level", "segment", "--actual-branch-provenance-compress",
    "--progress-every", "1", "--checkpoint-every", "1", "--stream-diagnostic-traces",
    "--debug-end-index", "$EndIndex", "--output-dir", $runDir,
    "--checkpoint-path", (Join-Path $runDir "checkpoint.pkl"),
    "--resume-checkpoint", (Join-Path $runDir "checkpoint.pkl")
)

switch ($Group) {
    "exact-all" {
        $flags += @(
            "--branch-provenance-diagnostic", "--preselection-segment-diagnostic",
            "--teacher-forced-winner-diagnostic", "--segment-reinforcement-diagnostic",
            "--independent-reference-diagnostic", "--actual-branch-provenance-diagnostic",
            "--observe-scenario-diagnostic", "--density-trace"
        )
    }
    "actual-off" {
        $flags += @(
            "--branch-provenance-diagnostic", "--preselection-segment-diagnostic",
            "--teacher-forced-winner-diagnostic", "--segment-reinforcement-diagnostic",
            "--independent-reference-diagnostic", "--observe-scenario-diagnostic", "--density-trace"
        )
    }
    "actual-summary" {
        $flags += @(
            "--branch-provenance-diagnostic", "--preselection-segment-diagnostic",
            "--teacher-forced-winner-diagnostic", "--segment-reinforcement-diagnostic",
            "--independent-reference-diagnostic", "--actual-branch-provenance-diagnostic",
            "--actual-branch-provenance-level", "summary", "--observe-scenario-diagnostic", "--density-trace"
        )
    }
    "branch-off" {
        $flags = @(
            "--limit", "250", "--warmup", "200", "--streams", "original",
            "--l-match", "2", "--lmatch-real-ablation", "--prediction-horizon", "5",
            "--tie-break-seed", "0", "--continuous-impl", "reference",
            "--competition-mode", "competitive_raw", "--inhibition-strength", "0.1",
            "--inhibition-tau", "0.02", "--simultaneous-policy", "batched",
            "--simultaneous-bin-width", "0.005", "--intracolumn-selection-policy", "max_candidate_score",
            "--progress-every", "1", "--checkpoint-every", "1", "--debug-end-index", "$EndIndex",
            "--output-dir", $runDir, "--checkpoint-path", (Join-Path $runDir "checkpoint.pkl"),
            "--resume-checkpoint", (Join-Path $runDir "checkpoint.pkl")
        )
    }
    "minimal" {
        $flags = @(
            "--limit", "250", "--warmup", "200", "--streams", "original",
            "--l-match", "2", "--lmatch-real-ablation", "--prediction-horizon", "5",
            "--tie-break-seed", "0", "--continuous-impl", "reference",
            "--competition-mode", "competitive_raw", "--inhibition-strength", "0.1",
            "--inhibition-tau", "0.02", "--simultaneous-policy", "batched",
            "--simultaneous-bin-width", "0.005", "--intracolumn-selection-policy", "max_candidate_score",
            "--progress-every", "1", "--checkpoint-every", "1", "--debug-end-index", "$EndIndex",
            "--output-dir", $runDir, "--checkpoint-path", (Join-Path $runDir "checkpoint.pkl"),
            "--resume-checkpoint", (Join-Path $runDir "checkpoint.pkl")
        )
    }
}

$result = Join-Path $runDir "result.json"
$failure = Join-Path $runDir "failure.json"
$checkpoint = Join-Path $runDir "checkpoint.pkl"
$runnerArgs = @(
    "--result-json", $result, "--failure-json", $failure, "--checkpoint", $checkpoint,
    "--stdout-log", (Join-Path $runDir "stdout.log"), "--stderr-log", (Join-Path $runDir "stderr.log"),
    "--combined-log", (Join-Path $runDir "combined.log"), "--",
    ".\.venv\Scripts\python.exe", "-X", "dev", "-u", "experiments\fig9_strict_reproduction.py"
) + $flags

& .\.venv\Scripts\python.exe scripts\run_logged_process.py @runnerArgs
$exit = $LASTEXITCODE
Write-Host ("group={0} exit_code={1} output={2}" -f $Group, $exit, $runDir)
exit $exit
