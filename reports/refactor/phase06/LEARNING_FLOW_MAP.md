# Phase 06 Learning Flow Map

Audit commit: `6cc5e05a31965fc47e416a1b03d738fc3d762e3e`.

`SequentialMemory.observe_code()` remains the observation and learning
orchestrator. The order below is the actual pre-extraction order.

| Order | Step | Function / lines | Reads | Writes | RNG | Persistent mutation | Float-sensitive | Paper scenario | Current label |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | Resolve prior context | `observe_code` 2139-2148; `_active_sources` | previous active/winners; burst_context | trace copies only | no | no | no | NONE | none |
| 2 | Resolve prediction candidates for each proximal event | `observe_code` 2178-2250 | last candidates; event time; tolerance; candidate active/score | diagnostic metadata; identity-confirmed set | no | no | timing gate and max score | possible S1 evidence | none |
| 3 | Search fallback segment when prediction is absent | `observe_code` 2251-2280; `_best_matching_neuron` 2816-2889 | incoming index; active sources; segment weights/delays | optional trace | learning RNG on exact tie | no | score and timed overlap traversal | S2A/S2B evidence | none |
| 4 | Establish matched identity and prediction confirmation | `observe_code` 2271-2282 | selected candidate and exact causal Segment | local values | no | no | no new float work | S1 evidence | none |
| 5 | Select no-match branch | `observe_code` 2283-2339 | matched presence; candidate evidence; existing segment counts | scenario counter/reason; selected neuron | least-used selection | segment creation only when learn | no | PAPER_S2B | `scenario3` |
| 6 | Select predicted branch | `observe_code` 2340-2367 | learn; was_predicted; same candidate/segment | scenario counter/reason | no | reinforcement follows | no | PAPER_S1 | `scenario1` |
| 7 | Select matching branch | `observe_code` 2368-2385 | existing segment timed overlap; L_match | scenario counter/reason | no | reinforcement/growth follows | exact timed-overlap expression | PAPER_S2A | `scenario2` |
| 8 | Select insufficient-match fallback | `observe_code` 2386-2404 | failed L_match result | scenario counter/reason; selected neuron | least-used selection | segment creation follows | consumes existing comparison only | PAPER_S2B | `scenario3` |
| 9 | Publish per-event winner and predicted/burst activation | `observe_code` 2406-2709 | winner and was_predicted | local active/winner/source sets; traces | no | no | no | NONE | none |
| 10 | Punish failed predictions | `observe_code` 2711-2717; `_punish_wrong_predictions` 3967-4054 | all saved candidates; actual event map; timing tolerance; active sources | synapse weight/age; prune state | no | yes | timing gate and contribution test | PAPER_S3 | `wrong_prediction_punishment` phase |
| 11 | Publish next transient context | `observe_code` 2718-2732 | local active/winner/source sets and counters | eight transient fields/stats | no | no | no | NONE | none |

## Canonical taxonomy

| Canonical paper name | Current operational path |
|---|---|
| `PAPER_S1` | confirmed predictive candidate, historical `scenario1` |
| `PAPER_S2A` | no confirmed prediction plus eligible matching segment, historical `scenario2` |
| `PAPER_S2B` | no eligible match and new-segment path, historical `scenario3` |
| `PAPER_S3` | failed candidate punishment in `_punish_wrong_predictions()` |

Historical strings, counters, CSV fields, and checkpoint fields are compatibility
interfaces and must not be renamed in Phase 06.

## Extraction boundary

The only safe production extraction identified by this audit is classification
from already-computed booleans: learning enabled, matched identity present,
prediction confirmed, and matching eligibility already computed. Timing gates,
candidate traversal, best-match traversal, least-used selection, reinforcement,
growth, punishment, publish, and pruning remain in `SequentialMemory`.
