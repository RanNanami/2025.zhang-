# Neural State Transition Map

The canonical state map is in `docs/NEURAL_STATE_TRANSITION_MAP.md`.

Audit result: current raw propagation does not copy every valid segment
candidate. It copies cells resolved from the already selected prediction code.
The unresolved boundary is that those selected candidates are not independently
materialized as full-soma fired neurons.
