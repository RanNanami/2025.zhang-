# Fig.9 Pairwise Context Coherence Audit

This is a read-only, nonpaper diagnostic. Pair statistics are computed from
actual pre-match context records before the candidate record. They are not
used by matching, competition, reinforcement, prediction, RNG, or learning.

## Implementation

- `experiments/diagnostics/analyze_fig9_pairwise_context.py` builds unordered
  source pairs, causal co-occurrence, recent co-occurrence, smoothed PMI-like
  values, support buckets, field/range/reinforcement summaries, offline
  target-vs-false comparisons, and ranking tables.
- `experiments/diagnostics/fig9_candidate_context_oracle.py` records immutable
  source metadata in the existing candidate identity trace.
- `experiments/diagnostics/fig9_segment_context_composition.py` records the
  complete pre-observation context IDs for the actual trajectory.
- `experiments/fig9_strict_reproduction.py` adds the default-off
  `--pairwise-context-diagnostic` capture flag. It does not add an online
  pairwise score.
- Pair trace rows are streamed to gzip; pair registries are aggregated in a
  temporary SQLite file and removed after export. This prevents the previous
  multi-gigabyte in-memory expansion.

## Protocol Boundary

The actual trajectory uses exact pre-match context and causal history from
records `< t`. The autonomous trajectory uses positive continuous contributor
pairs and does not reinterpret that count as `L_match`. Ground-truth labels
are post-hoc only. PMI-like smoothing is alpha `1`; shrinkage uses `k=3`; the
recent window is 20 records.

## Smoke Evidence

The successful L4 20-prediction smoke produced 4,173,604 pair rows and
302,931 unique pairs. Support buckets were:

| support | rows |
|---|---:|
| 0 | 1,246,427 |
| 1 | 359,397 |
| 2-3 | 546,223 |
| 4-7 | 611,713 |
| 8+ | 1,409,844 |

Offline target-vs-false pair mean PMI difference was `+0.10977` with a smoke
bootstrap interval `[0.02199, 0.19399]`. However, the field results were
mixed: Passenger `-0.07020`, Time `-0.14409`, and Weekday `+0.54360`.
Candidate-score target Top-1 was `0.3067`; pair-mean-PMI Top-1 was `0.3900`.
These are offline smoke rankings, not model results.

## Formal Run Status

The new formal L2 run used the strict data, seed, `L_match=2`, raw
propagation, reference continuous implementation, and pairwise diagnostic
capture. It terminated natively near index 211 with no Python traceback; the
last phase was rollout step 5. The earlier attempt with the heavier branch
diagnostic terminated near index 206. Therefore a complete current pairwise
250 L2/L4 comparison was not produced. The previously confirmed strict
baselines remain separate: L2 MAPE `0.3927571082`, L4 MAPE `0.4086558112`.

No formal pairwise result is claimed here, and no 500-record run was started.

## Decision

Primary conclusion: `PAIRWISE_COHERENCE_HAS_SIGNAL_BUT_INSUFFICIENT_SUPPORT`.

Recommended next step: `INVESTIGATE_TEMPORAL_CONTEXT_STRUCTURE`.

The smoke signal is not stable across fields, and there is no current L2/L4
formal pairwise evidence. Consequently no `PAIR_COHERENCE_LOCAL_TIEBREAK` or
other online mechanism was implemented. The strict default and historical
results remain unchanged.

Detailed smoke artifacts are under
`results/fig9_diagnostics/pairwise_context_coherence_smoke20_20260808/analysis_v9/`.
Formal failed-run evidence is under
`results/fig9_diagnostics/pairwise_context_coherence_20260809/formal_l2/` and
`formal_l2_v2/`.
