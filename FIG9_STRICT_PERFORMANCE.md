# Fig.9 Strict Performance Notes

This note tracks implementation-only optimizations for the strict Fig.9 runner.
These changes do not alter the paper protocol, model parameters, encoding,
prediction threshold, PSP formula, rollout logic, or learning rules.

## Continuous Prediction Implementations

`experiments/fig9_strict_reproduction.py` exposes:

```powershell
--continuous-impl reference
--continuous-impl optimized_v1
--continuous-impl optimized_v2
```

- `reference`: original continuous PSP integration path, and the strict default.
- `optimized_v1`: local function/dict bindings plus a per-`potential(time)`
  rounded-delta response memo.
- `optimized_v2`: non-default experimental implementation. It keeps the
  conservative local bindings, removes the no-benefit repeated-`t` cache idea,
  and adds a per-`predict_code()` memo for exactly identical arrivals sequences.

`continuous_impl` and `continuous_impl_version` are written to protocol,
summary, and runtime JSON files so a result can be audited later. They are
implementation labels, not changes to the paper protocol. Equivalence is checked
by predictions CSV SHA256, final model fingerprint, and RNG fingerprint.

## Micro-Diagnostic Summary

Measured on a 50-record original-stream smoke setup:

- repeated `potential(t)` calls: 0.0%
- duplicate rounded delta keys within one `potential(time)`: about 60.0%
- repeated arrivals signatures within a `predict_code()` call: about 28.2%
- global repeated arrivals signatures: about 50.1%
- arrivals already ordered: about 30.0%
- delays off the 0.005 integration grid: about 87.7%
- arrivals off the 0.005 integration grid: about 86.8%
- evaluated potential times off the 0.005 grid: about 93.3%

The micro-diagnostics suggested an exact-arrivals memo might help, but formal
250-record timing showed that the added memo overhead outweighed the savings.
So:

- a `potential(t)` value cache was not retained;
- an exact integer-grid fast path was not retained;
- the per-`predict_code()` exact-arrivals memo is kept only as a non-default
  equivalence-test/diagnostic path.

## Latest 50-Record A/B

Directory:

```text
results/fig9_strict/ab_50_20260726_213506
```

All three modes produced identical:

- predictions CSV SHA256;
- MAPE;
- coverage;
- mean/peak raw columns;
- final rolling MAPE;
- final model fingerprint;
- final RNG fingerprint.

The 50-record run is only a small equivalence and smoke-performance check. It
must not be used as a formal speedup claim.

## Formal 250-Record A/B

The formal manually run 250-record original-stream A/B reported:

| mode | runtime | relative to reference | adopted |
| --- | ---: | ---: | --- |
| `reference` | 276.1203575 s | baseline | yes |
| `optimized_v1` | 283.1157817 s | 2.53% slower | no |
| `optimized_v2` | 287.9934537 s | 4.30% slower | no |

All three paths were exactly equivalent for predictions CSV SHA256, MAPE,
coverage, mean/peak raw columns, final rolling MAPE, model fingerprint, and RNG
fingerprint. Because both optimized paths were slower in the formal 250-record
A/B, the strict default was restored to `reference`.

The script is retained for future manual checks, but it requires an explicit
flag so it is not rerun accidentally:

```powershell
.\scripts\fig9_strict_250_ab.ps1 -RunFormal250
```

The script runs `reference`, `optimized_v1`, and `optimized_v2` with 250 records,
warmup 200, original stream only, profiling disabled, and density trace
disabled. It writes a comparison JSON/text file and fails loudly if predictions
or final fingerprints diverge.
