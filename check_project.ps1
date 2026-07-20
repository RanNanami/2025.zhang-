$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $Python)) { throw "Missing .venv. See RUN.md." }
Set-Location -LiteralPath $Root

& $Python -m compileall -q src experiments tests
$env:PYTHONPATH = "$Root/src;$Root"
& $Python -m unittest discover -s tests -v

$CbtHash = (Get-FileHash data/cbt.tar.gz -Algorithm SHA256).Hash.ToLowerInvariant()
if ($CbtHash -ne "932df0cadc1337b2a12b4c696b1041c1d1c6d4b6bd319874c6288f02e4a61e92") {
    throw "CBT archive checksum mismatch."
}

$ExpectedHashes = @{
    "data/paper_nyc_taxi.csv" = "092d957f5bb0d2cd62f85098ed2268114a47b4738a5f4b29ea6be4be7349fc4d"
    "data/paper_nyc_taxi_perturb.csv" = "47eb800405d3574c0ba3b820b3c11d8fdd96c4510d5f490932d4748e6d420bd9"
    "data/fig8c_poems.json" = "2713d1544fd9a06245930dc52218348fccda830b6bbec282bbe6a07fffa8b22c"
    "data/fig8c_stress_poems.json" = "97619dfc61350d3df59f4b405688aab6cf993e93864c7624c56ade98d50307b2"
    "data/chinese_poetry_source/poet.tang.0.json" = "4f88f429938082523cdc72a1612bd7a7c3e3f7a33f8192c61ae51340efc3c92d"
    "data/chinese_poetry_source/poet.tang.1000.json" = "6e703b8d8dcedbd6110f6d48d043665ca2131db2035a5bfcef5e59a19aeb6d75"
    "data/chinese_poetry_source/poet.tang.2000.json" = "ec28efa7e2ae9cdbbba09f0ed186e32ccae47a325ad1e5816b7ad56071c1665a"
    ".deps/ContinuousLearnExperiment.pkl" = "2e2477143f795d927b0a76eca3a670a3e6e15b01ad5a032b9a1e7e9e58659f07"
}
foreach ($Entry in $ExpectedHashes.GetEnumerator()) {
    $Actual = (Get-FileHash -LiteralPath $Entry.Key -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Actual -ne $Entry.Value) { throw "Checksum mismatch: $($Entry.Key)" }
}

$TaxiLines = (Get-Content data/paper_nyc_taxi.csv | Measure-Object -Line).Lines
if ($TaxiLines -ne 17523) { throw "Expected 17,520 taxi records plus three header rows." }

& $Python experiments/fig7_sequence_prediction.py --max-elements 500 --modify-after-elements -1 --report-every 500
& $Python experiments/fig8_sentence_memory.py --num-sentences 20 --report-every 20 --eval-samples 0
& $Python experiments/fig9_paper_snn.py --limit 80 --warmup 50 --report-every 80

Write-Output "Project check completed."
