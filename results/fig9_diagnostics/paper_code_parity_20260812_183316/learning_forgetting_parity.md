# Learning and Forgetting Parity

The current one-layer implementation broadly follows paper Section II-D:

- Scenario 1 reuses the matching prediction candidate; contributing synapses
  strengthen/rejuvenate and noncontributing synapses weaken/age.
- Scenario 2 requires timed overlap at L_match=4, reinforces the best matching
  segment, and grows missing previous-winner synapses.
- Scenario 3 grows a segment on a least-used neuron when no eligible segment
  exists; wrong predictions are punished by delta'=0.01.
- Forgetting uses `10*(1-weight) + age` and removes a synapse at threshold 65.

L_match=4 and threshold=65 are paper-explicit on PDF page 8. Earlier project
documentation that treated them as inferred was incorrect.

Remaining partial assumptions concern the exact continuous definition of a
“contributing” arrival, delay/time normalization, winner tie-breaking,
burst/context representation, pruning timing, and the numerical continuous
solver. Those details are not sufficient grounds to rewrite the paper rule.
