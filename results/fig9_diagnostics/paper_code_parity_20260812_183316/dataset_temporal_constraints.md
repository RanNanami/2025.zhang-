# Dataset Temporal Constraints

## Direct Facts

- The paper states one year, 17,520 records, and half-hour aggregation.
- `17,520 = 365 * 48`; this is exactly a non-leap 365-day half-hour grid.
- The modified experiment changes records after April 1, 2015.

## Logical Constraints

If original and modified experiments operate on the same one-year stream, the
stream must include records after 2015-04-01. This does not establish a
specific first date.

## Unknown

`PAPER_YEAR_RANGE_NOT_EXPLICITLY_SPECIFIED`: the paper does not state the
first timestamp, last timestamp, timezone, or whether the year is July-June,
January-December, or another 365-day interval.
