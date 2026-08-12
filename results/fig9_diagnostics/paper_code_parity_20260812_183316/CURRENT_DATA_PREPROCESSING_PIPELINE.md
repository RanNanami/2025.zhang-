# Current Data Preprocessing Pipeline

1. Strict CLI default: `data/paper_nyc_taxi.csv`.
2. `read_records()` skips the two NuPIC metadata rows because they are not
   parseable timestamps/numbers, then reads `timestamp` and `passenger_count`.
3. No resampling, normalization, interpolation, zero fill, outlier removal,
   timezone conversion, or train/test scaling occurs in the loader.
4. `record_values()` computes Python weekday (`Monday=0`) and half-hour slot
   (`hour*2 + minute//30`); it forwards passenger count unchanged.
5. The passenger encoder clips only at its configured range 0..40,000.

Current file SHA256: `092d957f5bb0d2cd62f85098ed2268114a47b4738a5f4b29ea6be4be7349fc4d`. It has 17520
records from 2014-07-01 00:00:00 through 2015-06-30 23:30:00,
with no missing half-hours or duplicate timestamps.

The file is byte-identical to `.deps/reference_nyc_taxi.csv`, the bundled
Numenta [58] reference asset. It was committed as a finished aggregate; no
raw TLC-to-half-hour generation script exists in this repository. Therefore
its exact aggregation semantics remain unresolved.
