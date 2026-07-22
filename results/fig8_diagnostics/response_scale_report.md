# Fig.8 Response-Scale Sensitivity

This compact table is the 100-sentence sweep. The 200/500-sentence confirmation,
propagation ablation, and interpretation are in `FIG8_FALSE_POSITIVE_DIAGNOSTIC.md`.

All numeric response scales are nonpaper diagnostics. The strict default
remains `normalized` (`response_scale=None`). Each sentence is trained once.

| scale | kernel peak | minimum initial synapses | Levenshtein | raw columns | expected absent | two contributors | no prediction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| normalized | 1.000000 | 2 | 2.570 | 67.832 | 0.357 | 0.163 | 0.000 |
| 0.8 | 0.427994 | 5 | 2.050 | 17.834 | 0.430 | 0.000 | 0.050 |
| 1 | 0.534992 | 4 | 1.590 | 24.935 | 0.280 | 0.001 | 0.040 |
| 1.2 | 0.641991 | 4 | 1.410 | 29.552 | 0.217 | 0.007 | 0.040 |
| 1.4 | 0.748989 | 3 | 1.820 | 39.546 | 0.245 | 0.016 | 0.015 |
| 1.6 | 0.855988 | 3 | 1.940 | 46.927 | 0.263 | 0.060 | 0.003 |
