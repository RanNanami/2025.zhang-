# ETTh1 Transfer of the Fig. 9 DS Memory

This is a nonpaper transfer experiment. Zhang et al. do not evaluate ETTh1.
The experiment reuses the Fig. 9 network rather than the earlier SSTD nearest-
neighbor diagnostic.

## Mapping

- Taxi weekday (30 columns) -> ETTh1 weekday (30 columns).
- Taxi half-hour slot (58 columns) -> ETTh1 hour (58 columns).
- Taxi passenger count (482 columns) -> ETTh1 `OT` (482 columns).
- Ten active columns per field and 32 neurons per mini-column.
- `L_match=4`, forgetting threshold 65, DS response, online plasticity, and
  global likelihood decoding are unchanged from the strict Fig. 9 runner.
- The `OT` encoder range is fitted only on the warmup interval.
- The other six ETTh1 variables are intentionally excluded so the Fig. 9
  three-field topology remains unchanged.

The task is one-hour-ahead online prediction. The model observes each current
record, learns once, and predicts the next record. Persistence and the latest
available daily-seasonal value are leakage-free baselines.

## Standard-Boundary Result

Warmup ends at index 11,520 (12 training months plus four validation months),
followed by 2,880 evaluated points.

| Model | MAE | RMSE | WAPE |
| --- | ---: | ---: | ---: |
| Fig. 9 DS transfer | 2.094 | 4.644 | 0.420 |
| Persistence | 0.420 | 0.593 | 0.084 |
| Daily seasonal | 1.527 | 1.964 | 0.306 |

Coverage is 2,880/2,880. The updated DS prediction follows the target during
many ordinary intervals, but still contains frequent high spikes up to about
35. Online learning during evaluation does not remove these errors, and both
simple leakage-free baselines remain more accurate.

Artifacts:

- `results/etth1_fig9_ds_standard_h1.csv`
- `results/etth1_fig9_ds_standard_h1.png`

## Historical Additional 70/30 Check

Before the current Fig. 9 event-response correction, a 70% warmup and 2,000
evaluated points produced DS-transfer MAE 9.897, RMSE 12.407, and WAPE 2.547.
Persistence obtained MAE 0.410 and RMSE 0.571. These historical numbers are not
directly comparable with the updated standard-boundary result above and have
not yet been rerun with the current core.

## Interpretation

The event-response correction substantially improves the transfer, but the
current implementation and direct Fig. 9 mapping are still not competitive
ETTh1 forecasters. This does not establish that the unpublished author
implementation would fail. ETTh1 requires a new, explicitly nonpaper
multivariate design and a more stable continuous-value readout before testing
the standard 96/192/336/720 horizons.
