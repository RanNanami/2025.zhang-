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
