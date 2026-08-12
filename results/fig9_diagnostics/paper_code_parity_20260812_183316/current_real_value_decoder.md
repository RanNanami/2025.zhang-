# Current Real-Value Decoder

`SSTDRealValueEncoder.decode_likelihood()` filters raw prediction events to
passenger columns 88..569. It compares them with every half-spacing legal
codebook value. Ranking is lexicographic: timed overlap first, column overlap
second. All values tied at the best score are averaged.

Temporal order is genuinely used: candidate event times must match predicted
times within tolerance 0.03. Column membership remains a secondary fallback.
This is not a set-only decoder, although dense raw predictions can create many
ties and weaken order selectivity.

The paper gives only “most likely prediction results” and no numerical decode
equation, codebook, tolerance, or tie rule. Therefore
`REAL_VALUE_DECODER_NOT_FULLY_SPECIFIED_IN_PAPER`.

Ideal ordered-code audit: 17520 records,
673 unique dataset codes, repository MAPE
0.001412670691, pointwise MAPE
0.002670361700, mean signed bias
17.567573.
