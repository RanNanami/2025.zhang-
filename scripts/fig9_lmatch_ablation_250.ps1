param(
    [ValidateSet(4, 3, 2)]
    [int]$LMatch = 4
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepositoryRoot

Write-Host "This script is reserved for a manual, nonpaper L_match ablation."
Write-Host "It is never run automatically. The strict default remains L_match=4."

if ($LMatch -ne 4) {
    throw @"
L_match=$LMatch is gated but not enabled in the runner yet.
The current stage produced only offline overlap evidence. Add an explicit
nonpaper runner override in the next stage, review its protocol labeling, and
then run this script manually. No silent strict-configuration override is used.
"@
}

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputDirectory = Join-Path `
    "results/fig9_diagnostics/lmatch_ablation" `
    "L4_reference_$Timestamp"

Write-Host "Running the unchanged L_match=4 reference only."
Write-Host "Output: $OutputDirectory"

& .\.venv\Scripts\python.exe experiments\fig9_strict_reproduction.py `
    --limit 250 `
    --warmup 200 `
    --streams original `
    --continuous-impl reference `
    --competition-mode competitive_raw `
    --inhibition-strength 0.1 `
    --inhibition-tau 0.02 `
    --simultaneous-policy batched `
    --simultaneous-bin-width 0.005 `
    --intracolumn-selection-policy max_candidate_score `
    --output-dir $OutputDirectory

Write-Host "Completed unchanged L_match=4 reference: $OutputDirectory"
