# Passenger Encoder Parity

The current implementation matches the published topology and algorithmic
shape: 482 passenger mini-columns, K=10, Gaussian population ranking, ordered
events, and ten evenly spaced times from 0 through 0.5 cycle.

It is not numerically strict to the paper because the paper omits passenger
range, interval `l`, and sigma. Current code imports the Numenta-style range
0..40,000, spaces centers by `40000/(482-1) = 83.160083...`, and defaults sigma
to one spacing. Thus `ENCODER_SCALE_UNDERSPECIFIED` remains.

The ideal full-dataset code-to-decoder audit gives repository-formula MAPE
`0.001412670691`, MAE
`21.296961`, and maximum absolute error
`83.045738`. Quantization alone is far too small to
explain the observed 0.4-0.6 strict diagnostic MAPE.
