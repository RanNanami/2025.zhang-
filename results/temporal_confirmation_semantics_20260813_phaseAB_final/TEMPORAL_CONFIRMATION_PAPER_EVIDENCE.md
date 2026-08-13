# Temporal Confirmation Paper Evidence

The local extraction of Zhang et al. 2025 was reviewed for the four requested questions.

| Question | Local source | Evidence classification | Finding |
|---|---|---|---|
| Does the paper require a numeric absolute-time confirmation tolerance? | `paper_text.txt`, Section II-D and SSTD discussion | NOT_SPECIFIED | The paper describes subsequent proximal activation and ordered spikes, but does not give `0.03` or an equivalent numeric gate. |
| Does the paper support a predictive neuron winning inside an active mini-column? | `paper_text.txt`, lines 236-250, Section II-A | PAPER_EXPLICIT | A predictive/depolarized neuron fires before its same-column counterparts; if none is predictive, the column bursts. |
| Does the paper fully specify multiple predictive-neuron soma competition? | `paper_text.txt`, lines 427-445 | PAPER_PARTIAL | The text gives the winner/inhibition idea, but not enough implementation detail to resolve every multi-candidate tie. |
| Is relative SSTD order part of the representation? | `paper_text.txt`, lines 463-478, Section II-C | PAPER_EXPLICIT | The SSTD description uses ordered spikes and distinguishes temporal patterns, while it does not say that exact local absolute time is irrelevant. |

The candidate tested in Phase C is therefore a narrowly constrained paper-semantic diagnostic, not a claim that the published algorithm has been completely recovered.
