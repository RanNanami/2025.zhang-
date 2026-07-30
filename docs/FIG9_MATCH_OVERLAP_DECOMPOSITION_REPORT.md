# Fig.9 Match Overlap Decomposition Report

Status: diagnostic-only preliminary validation. No formal 250-record run was
executed by Codex. Strict defaults, `L_match=4`, prediction, matching,
learning, and checkpoint v1 remain unchanged.

## Non-interference

The final 50-record `max_candidate_score` on/off comparison produced:

- predictions SHA256:
  `8A64489FC4DEF9E75E6A2B6E3BAA497F183780416A84D0E413EE98AF6889455D`
  in both runs;
- MAPE: `0.4293969349409696` in both runs;
- coverage: `1.0` in both runs;
- mean/peak raw columns: `175.976 / 268` in both runs;
- identical final model and RNG fingerprints.

With every existing diagnostic hook enabled, intracolumn-selection,
observe-scenario, and teacher-forced trace files also had identical raw
SHA256 values. Competition trace runtime columns differed, as expected; after
removing only the declared runtime fields, both stable SHA256 values were:

`70c1342ea6a2a43cf5109a9ca1a6ff1e2970c59606a178227445d1e5be5a9abc`

## Real matching definition

The real match context is `previous_active_cells` under strict all-cell burst
context. Segment creation and Scenario-2 growth use `previous_winners`.

`timed_overlap` counts unique exact source cell IDs whose segment synapse
exists and whose delayed arrival is within `timing_tolerance` of the
dendritic target time. It is not a column count or a weight sum.

This difference matters: a segment is created from a compact winner-only
identity set but is later searched against a much larger all-cell context.

## Preliminary 50-record findings

For 450 observed passenger columns:

- Scenario 3: `399 / 450 = 88.67%`;
- mean best overlap across all passenger columns: `1.1489`;
- columns with any existing segment: `50.89%`;
- existing passenger segments: `420`;
- mean segment overlap: `1.2929`;
- mean creation-source exact active retention: `0.5006`;
- mean creation-context/current-all-cell Jaccard: `0.0164`.

The low Jaccard and moderate exact-source retention are not contradictory:
roughly half of the 30 creation winners remain active, but the current
all-cell context is much larger.

For passenger Scenario-3 columns that already had segments, offline fixed
state counterfactuals were:

| L | any passing segment | ambiguous column |
|---:|---:|---:|
| 1 | 24.72% | 3.37% |
| 2 | 6.18% | 0.00% |
| 3 | 0.56% | 0.00% |
| 4 | 0.00% | 0.00% |

This does not justify changing `L_match`. It shows that lowering to 1 would
recover only part of early Scenario 3 and would already add ambiguity.

For passenger columns with existing segments:

| context | mean best overlap | L4 passing segments |
|---|---:|---:|
| all-cell current | 2.2576 | 0.2183 |
| winner-only | 1.8253 | 0.1747 |
| predicted-only | 0.0044 | 0.0000 |
| predicted plus winner | 1.8253 | 0.1747 |
| without burst-only | 1.8253 | 0.1747 |
| burst-only | 2.2533 | 0.2183 |

In this early window, removing burst-only cells reduces overlap. The trace
therefore does not currently support a real winner-only or predicted-only
context change.

## Source-level smoke

The 20-record source trace contained 30,542 rows and had
`UNKNOWN_SOURCE_LOSS=0`.

Passenger source outcomes:

- timing or eligibility excluded: `61.93%`;
- source column not active: `33.94%`;
- exact timed match: `3.03%`;
- source column active but wrong neuron: `1.09%`.

At this early stage, timing eligibility and missing source columns dominate
exact neuron-identity drift. This is a smoke-test result, not a late-capacity
conclusion.

## Field difference

Mean overlap for existing segments was:

- weekday: `21.2558`;
- time: `1.5004`;
- passenger: `1.2929`.

Weekday segments had a mean of about 374 synapses, versus about 32 for time
and passenger. The high weekday overlap is therefore strongly associated
with segment size and repeated growth, not simply better identity retention.

## Current judgement

The code-level mismatch is now explicit: winner-only creation is followed by
all-cell timed matching. In the short run, passenger overlap loss is driven
mostly by timing/eligibility and source-column inactivity; exact wrong-neuron
identity is measurable but not dominant.

No real `L_match` or burst-context ablation is supported yet. The existing
formal evidence for records 200-244 cannot substitute for this new trace.
The manual 250-record diagnostic must be run before deciding whether the late
capacity failure has the same decomposition.
