# Fig. 8(c) High-Conflict Stress Test

This is an engineering diagnostic for the local Fig. 8(c) approximation. It
is not a Zhang-paper dataset or result.

## Dataset

`data/fig8c_stress_poems.json` contains 1,000 synthetic four-line,
five-character poems generated with seed 2026. It deliberately creates:

- 80 distinct first lines, each shared by 12 or 13 poems;
- groups of eight near-duplicate lines with one-character changes;
- 1,000 unique titles and 1,000 unique complete poem texts.

This design separates two questions. A unique title should still retrieve one
poem, while a first line is intentionally insufficient to select one of its
12 or 13 continuations. The latter is an irreducible ambiguity for a
single-answer contextual query, not a fair paper-accuracy benchmark.

Rebuild the data with:

```powershell
.\.venv\Scripts\python.exe experiments\generate_fig8c_stress_dataset.py
```

## Debugging Findings

1. Candidate symbols were iterated from a Python `set`, so tied predictions
   could change across processes. Candidate order is now sorted before the
   seeded tie shuffle.
2. Cross-layer feedback originally keyed only the source mini-column and spike
   time. It omitted the source neuron identity, causing unrelated concepts to
   share feedback synapses. `InterLayerSequenceMemory` now allocates up to ten
   context cells per source event, matching the layer's ten neurons per column.
3. Cross-layer decoding scanned the full learned vocabulary. It now obtains
   candidates from the encoder's event-to-symbol inverted index.
4. Sequential prediction now skips a dendritic segment when the sum of all
   active synaptic weights is provably below threshold. This is an exact upper
   bound and does not change accepted predictions.

The runner also reports three diagnostic stages: title-to-line distance,
first-line-to-suffix distance, and true-line-to-character feedback distance.

## Results

The control uses one source neuron per event, reproducing the old cross-layer
aliasing under deterministic decoding. The fixed run uses ten.

| Poems | Goal control | Goal fixed | Context control | Context fixed | Feedback control | Feedback fixed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.280 | 0.000 | 3.170 | 2.990 | 0.280 | 0.000 |
| 200 | 1.445 | 0.000 | 9.310 | 8.750 | 1.445 | 0.000 |
| 400 | 2.470 | 0.005 | 12.000 | 11.680 | 2.470 | 0.005 |
| 600 | 2.707 | 0.015 | 12.970 | 12.635 | 2.707 | 0.015 |
| 800 | 2.471 | 0.059 | 13.274 | 13.104 | 2.471 | 0.059 |
| 1000 | 2.332 | 0.021 | 13.603 | 13.447 | 2.332 | 0.021 |

At 1,000 poems, the source-cell fix reduces goal-based and true-line feedback
distance by about 99.1%. Title-to-line distance remains exactly zero. The
first-line-to-suffix distance is 2.76 out of 3, explaining why contextual
character retrieval stays poor on the deliberately ambiguous first lines.

## Reproduce

```powershell
# Deterministic one-source-neuron control
.\.venv\Scripts\python.exe experiments\fig8c_poem_memory.py `
  --data data\fig8c_stress_poems.json `
  --checkpoints 100 200 400 600 800 1000 `
  --interlayer-source-neurons 1 `
  --output-csv results\fig8c_stress_control.csv

# Fixed ten-source-neuron run; the cache avoids repeating the unaffected
# single-layer character recall.
.\.venv\Scripts\python.exe experiments\fig8c_poem_memory.py `
  --data data\fig8c_stress_poems.json `
  --checkpoints 100 200 400 600 800 1000 `
  --interlayer-source-neurons 10 `
  --reuse-single-summary results\fig8c_stress_control.csv `
  --reuse-single-details results\fig8c_stress_control_details.csv `
  --output-csv results\fig8c_stress_fixed.csv

.\.venv\Scripts\python.exe experiments\plot_fig8c_stress_comparison.py
```

The stress test demonstrates robust title-conditioned storage after the fix,
but it does not establish production readiness. The inter-layer mechanism is
still an event-wise approximation, and real deployment data needs a query with
enough context to identify one continuation.
