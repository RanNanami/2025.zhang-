# Fig.9 Paper Protocol Source Audit

This audit uses the supplied English IEEE paper as the primary source. It
separates direct statements from method implications and omissions. Current
code or Numenta reference values never fill a paper-side unknown.

## Direct Fig.9 Protocol

| Fact | Evidence | Level |
| --- | --- | --- |
| Public NYC taxi passenger source, cited as TLC [66] | PDF p.10, Section III-D; reference [66] on p.12 | EXPLICIT_IN_PAPER |
| Half-hour aggregation, 17,520 records, one year | PDF p.10, Section III-D | EXPLICIT_IN_PAPER |
| Record fields: day of week, time of day, passenger number | PDF p.10, Section III-D | EXPLICIT_IN_PAPER |
| Periodic weekday/time and real-value passenger encoding | PDF p.10, Section III-D; algorithm on p.5, Section II-C | EXPLICIT_IN_PAPER |
| K=10 per field; 30/58/482 columns; 570 total; 32 neurons/column | PDF p.10, Section III-D | EXPLICIT_IN_PAPER |
| Predict 2.5 hours, five steps ahead | PDF p.10, Section III-D | EXPLICIT_IN_PAPER |
| MAPE between most-likely prediction and actual data | PDF p.10, Section III-D | EXPLICIT_IN_PAPER |
| L_match=4 and forgetting threshold=65 for taxi | PDF p.8, Section III-A after Table I | EXPLICIT_IN_PAPER |
| w0=.5, delta=.1, incorrect delta=.01, Vdep=.5, Aosc=.5, thresholds=1, Lw=10, La=1 | PDF p.8, Table I | EXPLICIT_IN_PAPER |
| Modified stream after 2015-04-01: weekday 07:00-11:00 -20%, 21:00-23:00 +20% | PDF p.10, Section III-D | EXPLICIT_IN_PAPER |

## Method Evidence

- PDF p.5, Section II-C: real-value encoding uses Gaussian receptive fields
  separated by interval `l`, selects top K outputs, has resolution `l/2`, and
  assigns ordered spike times evenly in the first half-cycle.
- PDF p.5: periodic encoders arrange the Gaussian population on a ring.
- PDF p.6, Section II-D: Scenario 1/2/3, structural growth, delay assignment,
  weight/age updates, and the forgetting equation are explicit.
- PDF p.7, Section II-E/F: a predictive neuron may fire without feedforward
  input and drive further prediction. Using raw autonomous activity is
  strongly implied; an exact statement that one normalized cycle equals one
  30-minute record is absent.

## Explicit Unknowns

`NOT_SPECIFIED_IN_PAPER`: exact year range, timezone, TLC fleet/geographic
scope, pickup/dropoff anchor, passenger aggregation formula, missing-bin and
DST treatment, passenger min/max, numerical `l`, Gaussian sigma, real-value
decoder equation, tau_m, tau_s, response normalization, integration grid,
timing tolerance, random seed, evaluation warmup/range, rolling window, and
the exact MAPE normalization equation.

The Fig.9(b) `Ours` bar is approximately 0.10. This is
`FIGURE_DIGITIZED_APPROXIMATE`, with a conservative visual uncertainty of
about +/-0.01; it is not an exact table value.

Public searches found the official TLC source but did not identify an author
release containing the processed 17,520-row stream or preprocessing code:
`AUTHOR_PROCESSED_DATASET_NOT_PUBLICLY_IDENTIFIED`.
