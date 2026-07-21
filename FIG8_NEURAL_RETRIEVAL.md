# Fig.8(a) Autonomous Neural Retrieval Audit

## Protocol correction

The former suffix loop decoded a neural prediction and then called
`observe(decoded_word, learn=False)`. That rebuilt the complete SSTD code as
proximal input. Any unpredicted mini-column then burst all of its cells, so the
decoded symbol, rather than the cells that actually predicted, became the next
context. This violates the paper's contextual retrieval protocol.

The default `neural` mode now performs exactly one `predict_code()` per suffix
step, decodes that returned code for reporting only, and passes the same code
and `last_prediction_candidates` to `advance_prediction()`. It stops on no raw
prediction, decode failure, or no prediction-active cells. It never calls
`observe(decoded_word, learn=False)` during autonomous suffix generation.
`proximal-replay` retains the former behavior for controlled comparison.

`decode_symbol_from_prediction()` and `decode_symbols_from_prediction()` are
pure state decoders: they do not predict, observe, advance, learn, age, or
modify segments. `predict_symbols()` calls `predict_code()` once and delegates
to the decoder. Evaluation snapshots and restores active cells, winners,
prediction candidates, ranking, decode RNG, and learning RNG, so checkpoint
reporting cannot alter later training.

## Verification

PowerShell commands:

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD"
.\.venv\Scripts\python.exe -m compileall -q src experiments tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

All 63 tests pass. Nine focused Fig.8 tests cover: no second prediction during
decode; autonomous A-B-C propagation without observing B; complete synapse and
segment read-only recall; exact prediction-active-cell advancement; preserved
legacy replay; one prediction per public prediction API; checkpoint evaluation
and RNG isolation; bounded response caching; and cache-free model checkpoints.

## Controlled ablation

All runs use the same CBT file and seed 11, ten words per sentence, six cue
words, four recalled words, 100 mini-columns, 10 neurons per column, K=10,
L_match=3, forgetting threshold 500, and continuous dynamics. For each size,
both modes evaluate the same deterministically trained network state.

| Sentences | Proximal replay | Neural | Neural - replay |
| ---: | ---: | ---: | ---: |
| 100 | 2.070 | 2.570 | +0.500 |
| 200 | 3.510 | 3.635 | +0.125 |
| 500 | 3.950 | 3.964 | +0.014 |
| 1000 | 3.984 | 3.981 | -0.003 |

The neural mode is not significantly better. At 1000 sentences the 0.003
difference is negligible relative to the four-word suffix and the gap from the
paper's approximately 0.2 result.

## Neural diagnostics

| Sentences | Step 1 error | Step 2 error | Step 3 error | Step 4 error |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 30.0% | 66.0% | 75.0% | 86.0% |
| 200 | 70.0% | 94.5% | 99.5% | 99.5% |
| 500 | 96.6% | 100.0% | 100.0% | 99.8% |
| 1000 | 98.5% | 99.6% | 100.0% | 100.0% |

| Sentences | Mean raw events | Mean prediction-active cells | Mean ranked candidates | Early stop |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 67.833 | 67.833 | 43.768 | 0.0% |
| 200 | 91.026 | 91.026 | 65.636 | 0.0% |
| 500 | 98.730 | 98.730 | 100.326 | 0.0% |
| 1000 | 99.726 | 99.726 | 115.168 | 0.0% |

Neural burst count is zero at every size. At 1000 sentences the expected word
is absent from the raw ranking for 85.7%, 95.1%, 93.8%, and 95.4% of suffix
steps 1-4. Prediction-active-cell count equals raw event count, and no rollout
terminates early. Thus `advance_prediction()` is not dropping predicted cells.
The dominant failure is already present in raw prediction: nearly every one of
the 100 columns predicts, many false-positive segments cross threshold, and the
correct continuation is usually absent before decoding. Decoder-only errors
are secondary (12.8%, 4.5%, 6.2%, and 4.6%).

## Outputs

- `results/fig8_ablation_{proximal,neural}_{100,200,500,1000}.csv`
- Matching `_details.csv` files with per-step raw-state diagnostics
- `results/fig8_strict_neural_retrieval_1000.csv`
- `results/fig8_strict_neural_retrieval_1000.png`
- `results/fig8_strict_neural_retrieval_1000_details.csv`

This is a protocol-aligned implementation result, not a complete numerical
reproduction. Parameter sensitivity belongs to a separately labeled second
diagnostic phase; no tau, timing tolerance, network size, threshold, data, or
ground-truth fallback was changed to obtain these numbers.
