# Current vs Paper Comparability

| Axis | Status | Reason |
| --- | --- | --- |
| DATASET | PARTIAL | Full 17,520-row structure matches, but author aggregation/provenance is unresolved. |
| ENCODING | PARTIAL | Topology/K/order match; l, sigma and range are unpublished assumptions. |
| NETWORK | PARTIAL | Explicit topology/thresholds match; continuous constants and inhibition solver are unpublished. |
| LEARNING | PARTIAL | Scenario structure and taxi thresholds match; exact timing/contribution implementation remains interpretive. |
| PREDICTION | MISMATCH | Rollout runs before current anchor observation, creating a one-record causal offset. |
| DECODING | UNKNOWN | Paper does not publish the numeric real-value decoder. |
| EVALUATION | MISMATCH | Short 250/500 warmup-200 windows are not the paper's one-year evaluation. |

**COMPARABILITY LEVEL: NOT DIRECTLY COMPARABLE.**

The current label “strict reproduction” is too strong for numerical claims.
`PAPER_CONSTRAINED_IMPLEMENTATION` or `STRICT_TO_CURRENT_INTERPRETATION` is
more accurate until the anchor/evaluation protocol and unpublished data/readout
details are resolved. No code rename is performed by this audit.
