# Fig.9 Paper Protocol Source Audit

This audit uses the supplied English IEEE paper as the primary source. It
separates direct statements from method implications and omissions. Current
code or Numenta reference values never fill a paper-side unknown.

## Direct Fig.9 Protocol

| Fact | Evidence | Level |
| --- | --- | --- |
| Public NYC taxi passenger source, cited as TLC [66] | PDF p.10, Section III-D; reference [66] on p.12 | `EXPLICIT_IN_PAPER` |
| Half-hour aggregation, 17,520 records, one year | PDF p.10, Section III-D | `EXPLICIT_IN_PAPER` |
| Record fields: day of week, time of day, passenger number | PDF p.10, Section III-D | `EXPLICIT_IN_PAPER` |
| Periodic weekday/time and real-value passenger encoding | PDF p.10, Section III-D; algorithm on p.5, Section II-C | `EXPLICIT_IN_PAPER` |
| K=10 per field; 30/58/482 columns; 570 total; 32 neurons/column | PDF p.10, Section III-D | `EXPLICIT_IN_PAPER` |
| Predict 2.5 hours, five steps ahead | PDF p.10, Section III-D | `EXPLICIT_IN_PAPER` |
| MAPE between most-likely prediction and actual data | PDF p.10, Section III-D | `EXPLICIT_IN_PAPER` |
| L_match=4 and forgetting threshold=65 for taxi | PDF p.8, Section III-A after Table I | `EXPLICIT_IN_PAPER` |
| w0=.5, delta=.1, incorrect delta=.01, Vdep=.5, Aosc=.5, thresholds=1, Lw=10, La=1 | PDF p.8, Table I | `EXPLICIT_IN_PAPER` |
| Modified stream after 2015-04-01: weekday 07:00-11:00 -20%, 21:00-23:00 +20% | PDF p.10, Section III-D | `EXPLICIT_IN_PAPER` |

## Method Evidence

- PDF p.5, Section II-C specifies Gaussian receptive fields separated by
  interval `l`, top-K responses, resolution `l/2`, and ordered spike times
  evenly distributed in the first half-cycle.
- PDF p.5 specifies circular Gaussian populations for periodic values.
- PDF p.6, Section II-D specifies Scenario 1/2/3, structural growth,
  weight/age updates, delay assignment, and forgetting.
- PDF p.7, Section II-E/F supports raw autonomous predictive activity. It does
  not explicitly equate one normalized cycle with one 30-minute record.

## Explicit Unknowns

`NOT_SPECIFIED_IN_PAPER`: exact year range, timezone, fleet/geographic scope,
pickup/dropoff anchor, passenger aggregation formula, missing-bin and DST
treatment, passenger range, numerical `l`, sigma, real-value decoder equation,
tau values, response normalization, integration grid, timing tolerance, seed,
evaluation warmup/range, rolling window, and exact MAPE normalization.

Fig.9(b) `Ours` is approximately 0.10 +/-0.01 by visual digitization:
`FIGURE_DIGITIZED_APPROXIMATE`. Public searches found TLC but no author release
of the processed stream: `AUTHOR_PROCESSED_DATASET_NOT_PUBLICLY_IDENTIFIED`.
