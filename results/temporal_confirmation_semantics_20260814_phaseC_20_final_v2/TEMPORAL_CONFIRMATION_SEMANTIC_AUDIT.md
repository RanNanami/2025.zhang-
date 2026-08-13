# Temporal Confirmation Semantic Audit

1. The old 172/49/123 observation summary is mechanically reproduced as any=172, current S1 is not used as the candidate denominator, and timing rejection is 123.
2. Gate-before classification finds unique=172 and multiple=0.
3. Unique timing rejection is 123; exact candidate punishment is 123.
4. Unique rejected fallback counts are current S2A=13 and current S2B=110.
5. Branch replacement/double-credit count is 123.
6. `timing_tolerance=0.03` is local implementation provenance, not paper/reference provenance.
7. The paper supports the existence of a predictive/depolarized early winner in an active mini-column, but does not fully specify a multi-predictive soma winner.
8. The candidate gate is passed; no candidate is run when it is not passed.

This report distinguishes document evidence, mechanistic trace evidence, and any later performance evidence. It does not claim complete numerical reproduction.

## Phase-C candidate A/B (20 sentences)

The identity mode is a nonpaper semantic diagnostic. It does not change the
strict default and it does not relax `timing_tolerance`. The observed metrics
were:

| mode | mean Levenshtein | expected presence | mean raw columns | early-stop rate | S1 | S2A | S2B | S3 | segments |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current | 0.0 | 1.0 | 10.0375 | 0.0 | 49 | 14 | 1937 | 1249 | 1737 |
| unique_predictive_identity | 0.45 | 0.9 | 9.144736842105264 | 0.2 | 160 | 4 | 1836 | 1066 | 1636 |

On this 20-sentence sample, identity confirmation increased direct S1
confirmation and reduced segment/S3 counts, but it did **not** improve the
retrieval score: mean Levenshtein changed from `0.0`
to `0.45` and early-stop rate changed from
`0.0` to `0.2`. This is
mechanism evidence, not a claim of performance improvement or complete paper
reproduction. No 100-, 200-, 500-, or Fig.9 run was performed in this round.

