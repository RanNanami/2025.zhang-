# Intercolumn Inhibition Parity

## Paper side

Section II-B includes inhibitory feedback from other mini-columns in the soma
potential. Section II-C also describes intercolumn inhibition used by SSTD to
keep only selected mini-columns active. The paper gives the semantic role, but
does not specify a complete retrieval-time `V_inh(t)` amplitude, kernel, or
discretization that can be reproduced without an additional assumption.

## Code side

| Layer | Current implementation | Paper parity |
|---|---|---|
| Encoding | SSTD encoder selects K columns and assigns ordered event times | aligned with the sparse encoding description |
| Continuous segment prediction | integrates distal PSP and checks dendritic threshold, then tests a simplified soma state | no membrane-level cross-column inhibition |
| Strict Fig.8/Fig.9 raw propagation | returns the strict selected raw events and resolves their selected cells | no explicit retrieval `V_inh(t)` |
| `competitive_raw` | post-prediction event competition with configurable strength, tau, tolerance and batching | diagnostic only; not paper implementation |
| Fig.8 `eventwise-inhibited` | post-prediction event filtering | diagnostic only; not paper implementation |

The diagnostic competition path is evidence that suppression can affect the
trajectory. It is not evidence that its free parameters are the parameters of
the paper. It remains unchanged by this audit.

## Decision

Do not rename or promote `competitive_raw`. Do not add a fitted inhibition
strength to strict defaults. The next scientifically valid step is an audit of
continuous retrieval `V_inh(t)` or author implementation details.
