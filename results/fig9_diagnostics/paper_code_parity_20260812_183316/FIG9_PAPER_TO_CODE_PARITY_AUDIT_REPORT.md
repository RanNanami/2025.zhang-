# Fig.9 Paper-to-Code Parity Audit Report

## Primary Conclusion

`MULTIPLE_CRITICAL_PROTOCOL_MISMATCHES_FOUND`

The current implementation is a serious paper-constrained implementation of
the same task family, but the existing 250/500 MAPE values are not a direct
reproduction of the paper's Fig.9 number. The strongest directly demonstrated
problem is a one-record prediction-anchor mismatch. Evaluation also uses a
local 200-record warmup and a short slice, whereas the paper reports a
one-year experiment and does not state that protocol.

## What Is Aligned

- One-year candidate file with 17,520 contiguous half-hour records.
- Three fields and 30/58/482 columns, K=10, 570 total, 32 neurons/column.
- Periodic weekday/time and real-valued ordered Gaussian SSTD coding.
- Taxi L_match=4 and forgetting threshold=65 are paper-explicit and matched.
- Core Scenario 1/2/3 and weight/age/forgetting equations broadly match.
- Retrieval uses raw predictive neural activity, not decoded proximal replay.

## What Is Not Established

- Current data SHA `092d957f5bb0d2cd62f85098ed2268114a47b4738a5f4b29ea6be4be7349fc4d` is byte-identical to the bundled
  Numenta [58] asset, but no author preprocessing release was found. TLC raw
  records contain per-trip passenger counts; paper aggregation scope/formula
  is unspecified. Current data is a candidate, not proven author data.
- Passenger `l`, sigma, range and real-value decoder are absent from the paper.
- Zhang does not state warmup, exact MAPE equation, evaluation range, rolling
  window, random seed, tau values, timing tolerance, or integration grid.

## High-Impact Findings

1. **Five-step anchor mismatch.** At loop index 200, the state ends at 03:30,
   five autonomous continuations causally reach 06:00, but evaluation compares
   to 06:30. The output label still displays 04:00 -> 06:30 (+2.5 h), hiding
   that the 04:00 record was not observed before rollout.
2. **Short-window evaluation mismatch.** The completed strict 250 result
   (0.505450) evaluates 45 points; the completed raw L4 500 result (0.539280)
   evaluates 295 points. Both use warmup 200, a local diagnostic setting.
3. **Data/readout underspecification.** Exact passenger aggregation and decoder
   are unknown. They remain material reproducibility risks, although ideal
   encode/decode MAPE `0.001413`
   shows current quantization alone does not produce the large gap.

## Paper Value

Fig.9(b) `Ours` is approximately 0.10 +/-
0.01 by visual digitization. It is not an
exact reported table value. Fig.9(c) is modified data; Fig.9(d) is the
post-modification error trajectory and must not be mixed with panel (b).

## Matrix Summary

The matrix has 157 checks: {"MATCH": 70, "MISMATCH": 7, "PARTIAL": 24, "UNKNOWN": 56}.

## Does Protocol Mismatch Plausibly Explain the Gap?

Yes, it can explain a substantial and currently unquantified part of the gap,
especially the causal horizon offset and incomparable evaluation window. The
audit cannot claim all error comes from protocol: dense raw predictions and
continuous/inhibition assumptions remain genuine model-mechanism risks.

## Recommendation

`FIX_FIVE_STEP_PREDICTION_PARITY`

First add an isolated, explicit anchor convention where the current record is
observed before forecasting t+5 (or equivalently align target to the fifth
causal continuation), then run a small A/B. Do not run another 500/full-year
experiment until that parity decision is tested. The Windows native runtime
crash investigation is operationally important but causally separate from
this paper-protocol audit.

# Direct Answers to the 67 Audit Questions

1. **Paper records:** 17,520.
2. **One year:** yes, explicit.
3. **Sampling:** every 30 minutes.
4. **Exact year range:** not stated.
5. **April 1 constraint:** if original/modified use one stream, it must extend past 2015-04-01; it does not prove the start date.
6. **Paper source:** NYC TLC trip-record source [66].
7. **Current source:** a finished Numenta [58] half-hour aggregate bundled in `.deps`; exact author use is unverified.
8. **Current rows:** 17,520 in the strict default; a historical 10,320-row prefix also exists.
9. **Current range:** 2014-07-01 00:00 through 2015-06-30 23:30.
10. **Same dataset:** structurally plausible but not proven identical to the author's processed stream (`PARTIAL`).
11. **Is NAB the paper source:** no. The paper cites TLC; NAB/Numenta is a processed derivative candidate.
12. **NAB/TLC relation:** common TLC origin is plausible, but the repository lacks the raw aggregation chain.
13. **Passenger aggregation:** unspecified by the paper and absent locally.
14. **Exact reconstruction possible:** no, not without author data or aggregation rules.
15. **Weekday encoding:** periodic Gaussian SSTD ring.
16. **Time encoding:** periodic Gaussian SSTD ring.
17. **Passenger encoding:** real-value Gaussian population SSTD.
18. **K:** 10 per field.
19. **Columns:** 30/58/482, 570 total.
20. **Neurons:** 32 per mini-column.
21. **Current topology:** those values match.
22. **Gaussian interval `l`:** no numeric value in paper.
23. **Sigma/receptive width:** no numeric value in paper.
24. **Passenger range:** not stated.
25. **Current scale:** 0..40,000 from [58], 482 evenly spaced centers, sigma=one spacing.
26. **Strict encoder parity:** algorithmically aligned, numerically only partial because scale is unpublished.
27. **Spike order:** preserved in `SymbolCode` and timed-overlap decoding.
28. **Paper decoder:** not fully specified.
29. **Current decoder:** maximize `(timed overlap, column overlap)` over a half-spacing codebook; average ties.
30. **Perfect-code roundtrip:** repository MAPE 0.001412670691; MAE 21.29696.
31. **Does decoder alone create huge error:** ideal quantization does not; dense/partial prediction tie behavior can still matter.
32. **Paper five-step statement:** passenger count after 2.5 h, five steps in advance.
33. **Five autonomous cycles explicit:** no; it is strongly implied by retrieval, not stated as a Fig.9 implementation detail.
34. **Reported target timestamp:** exactly +2.5 h from the CSV input label.
35. **Off-by-one:** yes causally; the labelled current record is observed only after rollout.
36. **Overall current MAPE step:** final rollout step only.
37. **Paper horizon metric:** most reasonably t+5, but exact step aggregation is not described.
38. **Warmup=200:** local short-run diagnostic design.
39. **Paper warmup:** not specified.
40. **Current 500 vs paper:** not directly comparable.
41. **Paper Fig.9(b) Ours:** about 0.10.
42. **Value precision:** visual digitization, approximately +/-0.01, not exact.
43. **Paper L_match:** yes.
44. **L_match=4 source:** PDF page 8 explicitly says taxi task uses 4.
45. **Paper forgetting threshold:** yes, taxi threshold 65 on page 8.
46. **Current forgetting:** formula and threshold match; pruning timing is an implementation detail.
47. **Scenario 1/2/3:** broad rule structure is aligned.
48. **Learning timing:** one online observe per actual record; exact continuous contribution timing is partial.
49. **Network config:** explicit topology/threshold values match; continuous constants/inhibition are partial or unknown.
50. **Dataset match:** `PARTIAL`.
51. **Encoder match:** `PARTIAL`.
52. **Decoder match:** `UNKNOWN`.
53. **Prediction match:** `MISMATCH` because of anchor semantics.
54. **Evaluation match:** `MISMATCH` for completed short runs.
55. **“strict” naming:** too strong for numerical reproduction; “paper-constrained implementation” is more accurate.
56. **Largest MAPE-gap suspect:** five-step anchor mismatch.
57. **Second suspect:** evaluation-window/warmup mismatch.
58. **Third suspect:** unresolved passenger aggregation/provenance.
59. **Direct mismatches:** causal anchor/target alignment and short-run-vs-one-year comparison.
60. **Paper-unspecified items:** aggregation, range, l/sigma, decoder, warmup/formula/range, tau/V0/grid/tolerance, seed.
61. **Runtime crash relation:** unrelated to the scientific parity finding.
62. **Run another 500 now:** no.
63. **Next experiment:** isolated five-step anchor correction A/B on a small range.
64. **Primary conclusion:** `MULTIPLE_CRITICAL_PROTOCOL_MISMATCHES_FOUND`.
65. **Recommended next step:** `FIX_FIVE_STEP_PREDICTION_PARITY`.
66. **Tests:** 37 new parity tests plus the full existing suite.
67. **Commit SHA:** recorded after the final commit; see repository HEAD and final handoff.
