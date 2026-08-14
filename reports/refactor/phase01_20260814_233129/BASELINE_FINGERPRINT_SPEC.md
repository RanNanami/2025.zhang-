# Baseline Fingerprint Specification

## Serialization

All fingerprints use UTF-8 canonical JSON with sorted object keys, compact
separators, and Python's round-trip float representation. Dictionaries are
encoded as key/value pairs sorted by the canonical key bytes; sets are sorted
by canonical member bytes. The digest is SHA-256. Python `hash()` and object
addresses are never serialized.

## Prediction SHA256

`predictions.csv` is written with a stable field order and LF line endings.
The SHA-256 digest covers the exact CSV bytes. Fig.8 rows include sentence,
suffix step, expected/output words, raw activity, active prediction cells, and
stop reason. Expected words are evaluation labels only. Fig.9 uses the strict
runner's existing prediction CSV unchanged.

## Science model SHA256

`SCIENCE_MODEL_SHA256` includes:

- model class path and every `MemoryParams` field;
- encoder structure and learned code maps, excluding RNG state;
- columns and neurons in list order;
- segments in list order;
- segment `active` and `target_time`;
- synapses sorted by source cell, including source, weight, delay, and age.

Diagnostic IDs, creation labels, and reinforcement counters are excluded from
this digest because they do not drive the numerical trajectory. They remain in
the full-state digest.

## Full state SHA256

`FULL_STATE_SHA256` includes the science payload plus:

- diagnostic/provenance segment metadata;
- encoder RNG state;
- previous active cells and winners;
- last candidates, scores, times, PSP metadata, and stable segment coordinates;
- prediction/observation statistics and symbol ranking;
- predicted/burst source labels;
- incoming-index coordinates, dirty-source set, and next diagnostic ID;
- all RNG states.

Runtime callbacks and the reproducible spike-response cache are excluded.
Candidate segments use `(column, neuron, segment_index)` coordinates, never
`id()` values.

## RNG SHA256

`RNG_SHA256` serializes `getstate()` from learning, decoding, and every encoder
`random.Random` instance. It does not draw a value. Separate learning and
decode RNG digests are also recorded.

## Transient component SHA256

Separate digests cover `previous_active_cells`, `previous_winners`, and the
last prediction-candidate collection. This makes a mismatch localizable before
comparing the larger full-state digest.

## Scenario summary

Fig.8 captures all training observations. Legacy observation labels map to
paper terminology as `scenario1 -> S1`, `scenario2 -> S2A`, and the
new-segment legacy `scenario3 -> S2B`. Paper S3 counts candidate predictions
whose column/time is absent from the actual proximal event.

Fig.9's existing strict runner does not persist a cumulative paper-S3 counter.
The baseline records evaluated-observation legacy counts from the read-only
long-sequence ledger and stores S3 as `null` rather than inventing a value.

## Fixture boundary

Fig.8 10/20/50 fixtures use the existing strict local defaults and evaluate all
selected sentences. Fig.9 10/20/50 fixtures use the existing strict runner with
fixed reduced warmups of 2/5/10. Every Fig.9 protocol says
`REGRESSION_FIXTURE`, `paper_result=false`, and `reduced_warmup=true`.
