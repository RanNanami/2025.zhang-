# Paper-Core Inhibition / Winner / Firing Parity Report

## Scope

This audit reads the local Zhang et al. 2025 PDF first and then traces the
current implementation. It does not tune MAPE, change strict defaults, enable
diagnostic competition, use ground truth, or run a new Fig.9 formal experiment.

## Findings

1. Paper-explicit burst behavior is present: an unmatched active proximal
   column activates all neurons in that column.
2. The strict prediction path does not propagate every valid predictive
   candidate. It performs event-level score selection and then keeps one
   earliest-time representative per column.
3. This reduction is not a reconstructed paper soma WTA. The current score is
   the continuous prediction return value based on dendritic peak/threshold;
   no candidate trace contains the full `V_pro`, `V_api`, `V_dis`, `V_inh`,
   `V_osc`, `eta` components required by Eq. (2).
4. The strict path has no membrane-level continuous retrieval `V_inh(t)`. The
   existing `competitive_raw` implementation is post-prediction diagnostic
   competition with non-paper parameters and remains diagnostic.
5. A local `DSNeuronState` refractory test exists, but it is instantiated
   inside candidate prediction rather than maintained as a cycle-wide population
   firing state. Full paper firing/refractory parity therefore cannot be proven
   from current traces.

## Fig.8 and Fig.9 gate

Existing raw artifacts are retained for comparison. The previous raw Fig.8
artifacts report Mean Levenshtein 1.59 at 100 sentences and 2.60 at 200
sentences. The existing raw Fig.9 L4 artifact reports legacy MAPE
0.5054500721931258 over 45 valid predictions. A new repair run was not started,
because there is no paper-exact soma/intercolumn implementation to test without
adding an assumption.

## Conclusion

**Primary conclusion:** `INHIBITION_PARITY_REMAINS_UNRESOLVED`

The evidence is strong enough to reject the claim that the current selector is
already a complete paper firing-WTA implementation, but not strong enough to
justify a new default algorithm. Adding `max_candidate_score`, `max_response_peak`,
or `competitive_raw` would change the model under diagnostic assumptions rather
than restore a uniquely specified paper equation.

**Recommended next step:** `AUDIT_CONTINUOUS_INTERCOLUMN_VINH`

That audit should first obtain or reconstruct the missing retrieval-time
inhibitory function and full soma component trace. Only then should a
paper-constrained firing repair be evaluated on Fig.8 20/100/200.
