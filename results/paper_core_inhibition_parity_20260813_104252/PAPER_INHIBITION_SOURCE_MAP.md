# Paper Inhibition Source Map

This result is generated from the canonical audit in
`docs/PAPER_INHIBITION_SOURCE_MAP.md`. The supplied Zhang et al. 2025 PDF was
read locally; evidence is from paper pages 3-7, Sections II-A, II-B, II-C and
II-E. No strict model default was changed.

The paper explicitly requires predictive-state to soma firing, same-column
winner suppression, all-neuron burst when no prediction exists, refractory
protection, and actual-spike autonomous propagation. It also places
intercolumn inhibitory feedback in the soma equation. The current code has the
burst branch, one selected event per strict column, and selected-cell raw
propagation, but does not expose full soma component traces or a numeric
retrieval `V_inh(t)`.

See the detailed source map at `docs/PAPER_INHIBITION_SOURCE_MAP.md`.
