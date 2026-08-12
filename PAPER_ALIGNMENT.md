# Paper Alignment Audit

This file separates author-specified behavior from implementation choices.
The strict runners are `fig7_sequence_prediction.py`,
`fig7_repeated_trials.py`, `fig8_sentence_memory.py`, and
`fig9_paper_snn.py`.

## Sources

- Zhang et al. (2025), the PDF supplied with this project.
- Cui, Ahmad, and Hawkins (2016), reference [58] in Zhang et al.
- Numenta's original [sequence-prediction code](https://github.com/numenta/htmresearch/tree/master/projects/sequence_prediction).
- The original [high-order dataset generator](https://github.com/numenta/htmresearch/blob/master/htmresearch/support/sequence_prediction_dataset.py).
- Facebook/ParlAI's [CBT download definition](https://github.com/facebookresearch/ParlAI/blob/main/parlai/tasks/cbt/build.py).

## Exact Paper Settings

| Area | Setting | Strict implementation |
| --- | --- | --- |
| SSTD | `K=10`, ordered spikes in first half-cycle | Yes |
| Network | Fig.7: 200 columns x 10 neurons | Yes |
| Network | Fig.8: 100 columns x 10 neurons | Yes |
| Network | Fig.9: 30 + 58 + 482 columns, 32 neurons | Yes |
| Plasticity | `w0=.5`, correct `+/- .1`, incorrect `-.01` | Yes; scenario 1 has no extra `L_match` gate |
| Forgetting | `f=L_weight(1-w)+L_age*age`; weights 10 and 1 | Yes |
| DS dynamics | `V_dep=.5`, `A_osc=.5`, soma/distal thresholds `1.0` | Yes |
| Thresholds | Fig.7 25/50, Fig.8 500, Fig.9 65 | Yes |
| Matching | `L_match=3`, Fig.9 `L_match=4` | Yes |
| Fig.7 stream | random sequence draw, 50,000 noise symbols in stream | Yes |
| Fig.7 change | swap endings after 10,000 elements | Single-ending task |
| Fig.7 report | moving accuracy over 100 sequences, 10 trials/std | Yes |
| Fig.7 decoding | top-N overlap over sequence vocabulary; random tie breaks | Yes, from [58] |
| Fig.8 data | CBT sentences of at least 10 words; first 10 used | Yes |
| Fig.8 recall | first 6 words input, remaining 4 recalled | Yes |
| Fig.8 score | mean Levenshtein distance over all stored sentences | Yes |
| Fig.9 data | 17,520 half-hour records over one year | Yes |
| Fig.9 horizon | five steps / 2.5 hours | Yes |
| Fig.9 evaluation | MAPE on five-step-ahead passenger prediction | Partial; Zhang does not specify warmup, window, or exact formula. The current 5904/400/global-target normalization comes from [58]. |
| Fig.9 MAPE | `sum(abs(error)) / sum(abs(target))` | Yes, from [58] |

## Recovered From Reference [58]

The PDF gives only examples, not every sequence or the CSV. The eight
single-ending sequences, complete two/four-ending sets, full NYC taxi CSV,
and changed taxi CSV are therefore taken from the [58] reference code. Zhang
Fig.9 cites TLC [66] as the taxi source but does not identify the processed
17,520-row asset or its aggregation rule. The local July 2014-June 2015 stream
is consequently a structurally matching candidate, not proven author data.
The changed file begins at record 13,152, corresponding to 2015-04-01.

Locally retrained HTM/TDNN/LSTM/ELM models are diagnostics, not reproductions
of Zhang's reported baseline bars. The processed Fig.7 baseline object
committed by [58] is stored at
`.deps/ContinuousLearnExperiment.pkl` (SHA-256
`2e2477143f795d927b0a76eca3a670a3e6e15b01ad5a032b9a1e7e9e58659f07`)
and exported by `experiments/fig7_reference_comparison.py`. The author
repository does not contain the trained Fig.9 result arrays, so
`fig9_reference_comparison.py` exports approximate bar heights digitized from
Zhang Fig.9(b,c), with the source status recorded in every CSV row.

## Explicit Implementation Choices

The paper does not publish source code or all continuous-time constants.
These choices remain visible and must not be called author-specified:

- Time is normalized to one oscillation cycle. This preserves spike order,
  half-cycle encoding, and the paper's half-cycle distal-to-soma relation. The
  paper constrains oscillation frequency relative to the encoding timing but
  does not provide a physical cycle duration for these experiments.
- The double-exponential constants default to `tau_m=.10` and `tau_s=.02`
  cycles; the paper gives the equation but not these values.
- Real-value receptive-field width defaults to one center spacing. The paper
  specifies Gaussian fields and spacing `l`, but not a numerical `l`.
- Real-valued predictions are decoded over the complete half-spacing codebook:
  the midpoint(s) whose ordered SSTD code has maximum overlap are selected and
  ties are averaged. The paper asks for the most likely prediction but does not
  publish a separate decoding equation.
- Sentence state is reset at sentence boundaries. The paper presents each
  sentence as a separate memory sequence but does not state the reset signal.
- The paper does not publish random seeds or the exact CBT sentence sample.
  Strict runners use fixed seeds for repeatability and draw eligible CBT
  sentences without replacement. The resource sweep keeps that sample fixed
  across network sizes and varies the encoding/network seed over ten trials.
- Fig.7 decoding ranks the complete predefined sequence vocabulary by timed
  overlap and takes the top N. Noise remains continuous network input but is
  not an ending candidate, matching the classifier in reference [58].
- When an input mini-column has no correctly predictive neuron, every neuron
  in that column becomes active but only the best-matching or least-used neuron
  becomes the learning winner. New segments grow from the previous winners;
  distal matching and prediction use the full previous active set.
- Equal least-used neurons are selected with a seeded random tie break. The
  paper specifies the least-used rule but does not publish its tie-breaking
  implementation.
- First-spike intracolumn inhibition is the default. Retaining predictions
  from one mini-column at multiple SSTD times is available only with
  `--no-intracolumn-inhibition` for diagnosis.
- Scenario 1 directly reinforces the segment that made a correct prediction,
  without applying the scenario-2 `L_match` gate. Scenario 2 reinforces or
  grows a matching segment without aging unrelated segments; scenario 3
  weakens synapses that contributed to an incorrect prediction.
- Intercolumn inhibition scheduling is unpublished. Event-wise projection is
  retained only as a nonpaper diagnostic and is forbidden in the strict Fig.9
  runner.
- Fig.9 uses a passenger range of 0 to 40,000, matching the cited [58]
  experiment configuration; Zhang specifies 482 fields but not range bounds.
- Fig.9 strict retrieval advances raw predictive-neuron spikes and applies
  likelihood decoding only to the final passenger pattern. Coherent legal-code
  and eventwise projection remain optional diagnostics outside the strict
  runner.
- Literal all-neuron bursting and first-spike intracolumn inhibition are Fig.9
  strict defaults. The former winner-only/eventwise compensation is isolated
  under `results/fig9_historical_compensated/` and is not a strict result.
- Fig.8(c) uses the first 200 unique-title/text five-character quatrains
  selected from three deterministic `chinese-poetry` Tang-poem JSON shards.
  The paper does not publish its poem list. All three local layers use the
  Fig.8(a) size of 100 columns x 10 neurons because Fig.8(c) layer sizes are
  also unpublished.
- Fig.8(c) cross-layer feedback preserves a bounded source-neuron identity for
  each concept event (ten cells per mini-column by default). The paper uses
  neuron-to-neuron synapses but does not publish its cross-layer allocation
  rule, so this remains an explicit approximation.

## Removed From Strict Runs

- Treating noise as a state-reset separator.
- Alternating through sequences in a fixed order.
- Restricting decoding to only ending symbols or evaluation-subset words.
- Synthetic `w0`, `w1`, ... sentences for Fig.8.
- Extra clean-prefix recall scoring.
- Supervised next-symbol labels in the SNN stream.
- Calling local TDNN/LSTM/ELM retraining a Zhang baseline reproduction.

## Remaining Gaps

- The strict predictor numerically integrates distal double-exponential
  responses, the first dendritic threshold crossing, depolarization,
  phase-precessed soma firing, and within-column first-spike competition at a
  default resolution of `0.005` normalized cycles. The remaining membrane
  terms are still a one-layer approximation rather than a general simulator.
- The paper publishes the response-kernel equation but not `V0`, `tau_m`, or
  `tau_s`; exact threshold-crossing times cannot be recovered from the article
  alone. The current values remain explicit implementation choices.
- Full persistent membrane state and multi-layer proximal/apical learning are
  not complete. Intracolumn winner selection and event-wise intercolumn
  sparsity are represented algorithmically rather than by integrating an
  inhibitory membrane current on a fine time grid.
- Fig.8(c) has a runnable approximation, but not a numerical reproduction.
  Title-to-line and line-to-character feedback are explicit weighted SSTD
  associations; they are not integrated as persistent proximal/apical
  membrane currents. The authors' poem list and layer sizes remain unknown.
- Fig.7 now reaches final ten-trial means of `0.938`, `0.955`, and `0.965`, but
  its early convergence remains slower than the paper: the corresponding means
  are `0.745` at 1000 single-ending elements, `0.911` at 3750 two-ending
  elements, and `0.936` at 4500 four-ending elements. The event-driven model
  retains an `L_match` eligibility check before reinforcement; removing it to
  follow a literal scenario-1 reading reinforces weak event-level false
  matches and causes a large regression. A fine-time membrane simulation is
  needed to determine whether the paper's stated dynamics remove that check.
- Fig.8(a) and Fig.8(b) reproduce the protocols and qualitative resource trend
  but do not numerically match the paper. Older result folders are historical
  and are not valid strict runs.
- The previous full-year Fig.9 outputs used winner-only context, eventwise
  projection, or other compensated diagnostics. They have been moved to
  `results/fig9_historical_compensated/`. A 250-record strict smoke completed
  with MAPE `0.505450`, mean raw predicted columns `291.538`, and runtime
  `605.327` seconds for the original stream. The 2,000-record and full-year
  strict reruns remain pending because the strict raw continuous runner is far
  slower and denser than the compensated historical path.
