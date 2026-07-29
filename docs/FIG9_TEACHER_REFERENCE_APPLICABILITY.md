# Fig.9 Teacher Reference Applicability

This document records which teacher-forced winner identities exist before the
target observation. It is a code-semantics audit, not a change to prediction,
learning, or the strict protocol.

## Actual Scenario Semantics

`SequentialMemory.observe_code()` assigns each active mini-column through the
following paths:

- **Scenario 1**: a cell is already represented by the matching
  `PredictionCandidate`. The predicted neuron and reinforced segment both
  existed before observation. See `src/seqmem/model.py:1746` and the
  same-candidate reinforcement guard at `src/seqmem/model.py:2773`.
- **Scenario 2**: no matching predictive cell is accepted, but
  `_best_matching_neuron()` finds an existing segment whose timed overlap
  reaches `L_match`. The selected neuron and segment therefore existed before
  observation. See `src/seqmem/model.py:1769` and
  `src/seqmem/model.py:1966`.
- **Scenario 3**: no eligible predictive or matching segment is available.
  The least-used neuron is selected during observation, with RNG tie-breaking,
  and a segment may then be created. See `src/seqmem/model.py:1713`,
  `src/seqmem/model.py:1789`, and `src/seqmem/model.py:2954`.

Scenario 3's winner is an operational post-observation identity. It is not a
pre-existing predictive branch that autonomous retrieval could have selected.

## Applicability Classes

The observation trace derives applicability from actual branch outcomes,
including whether a segment was reinforced or created:

- `PREEXISTING_PREDICTED_REFERENCE`
- `PREEXISTING_MATCHING_SEGMENT_REFERENCE`
- `POST_OBSERVATION_CREATED_NEURON`
- `POST_OBSERVATION_CREATED_SEGMENT`
- `REFERENCE_UNAVAILABLE_BOUNDARY`
- `REFERENCE_UNAVAILABLE_OTHER`

The mapping is implemented in
`experiments/diagnostics/fig9_teacher_forced_identity.py` by
`reference_applicability_from_trace()`. It does not infer applicability from a
scenario label alone.

## Metric Denominators

`ALL_OPERATIONAL_REFERENCES` describes observed runtime winners. It may report
target-column availability but cannot support a claim that an autonomous
selector chose the wrong pre-existing neuron.

`PREEXISTING_NEURON_REFERENCES` is the denominator for reference-neuron
crossing, candidate, and emitted recall; conditional neuron match; and
correct-column wrong-neuron rate.

`PREEXISTING_SEGMENT_REFERENCES` is the denominator for exact segment match,
same creation transition, and segment-source fingerprint match.

Every scoped metric written by
`experiments/diagnostics/analyze_fig9_teacher_forced_identity.py` includes its
numerator, denominator, rate, excluded count, and exclusion reasons.

## Boundary Censoring

The formal 250 trace has 450 unavailable target-column rows. They all target
the final five records after the last record observed by the main training
loop. They remain explicitly censored in
`teacher_reference_boundary_audit.csv`.

No shadow continuation was added: doing so would require a cloned learned
model and RNG state, while the existing trace is sufficient to identify and
exclude the boundary rows. This keeps predictions, checkpoints, model
fingerprints, and RNG fingerprints unchanged.

## Diagnostic Hooks

`--observe-scenario-diagnostic` copies branch-local values already computed by
`observe_code()`, including matching overlap and the Scenario 3 assignment
reason. It does not repeat prediction, matching, observation, or learning.

`--reference-neuron-selection-diagnostic` enables the trace inputs needed by
the offline within-column analyzer. Both switches default to off and are
nonpaper diagnostics.

