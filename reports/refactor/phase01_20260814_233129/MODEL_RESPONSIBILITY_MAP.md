# Model Responsibility Map

`src/seqmem/model.py` is both the checkpoint compatibility owner and the main
scientific state machine. Its class module paths must remain stable until an
explicit checkpoint migration exists.

| Responsibility | Current symbols/range | State or sensitivity | First-phase decision |
| --- | --- | --- | --- |
| Persistent graph entities | `Synapse` 27-40; `Segment` 44-98; `Neuron` 102-103; `MiniColumn` 107-120 | Pickle class paths, weights, delay, age, active flags | Do not move |
| Prediction evidence | `PredictionCandidate` 133-148 and trace dataclasses | Candidate identity, crossing time, PSP metadata | Keep colocated initially |
| Transient state | `TransientStateSnapshot` 365-382 | Active cells, winners, candidates, RNG-neutral evaluation | Extract helper functions only after baselines |
| Protocol parameters | `MemoryParams` 386-429 | Defaults directly affect dynamics and checkpoint replay | Do not rename or reorder fields |
| Model facade/state machine | `SequentialMemory` 512-4214 | Prediction, observation, learning, pruning, propagation, RNG | Preserve module/class path |
| Prediction | `predict_code`, continuous prediction helpers, candidate selection | Floating accumulation order and tie-break sensitive | Late extraction |
| Decoding | `decode_*_from_prediction`, `_decode_candidates` | Decode RNG and ranking sensitive | Candidate for helper extraction after golden tests |
| Learning | `observe_code`, matching/growth/reinforcement/punishment | Scenario counts and graph mutation | Late extraction |
| Forgetting | `_prune_neuron`, `_live_incoming` | Segment/synapse identity and age | Medium-to-high checkpoint risk |
| Runtime diagnostics | callbacks and trace emitters | Must be read-only but currently mixed with core flow | Detach only with trajectory parity tests |

## Checkpoint-sensitive class paths

- `seqmem.model.Synapse`
- `seqmem.model.Segment`
- `seqmem.model.Neuron`
- `seqmem.model.MiniColumn`
- `seqmem.model.MemoryParams`
- `seqmem.model.SequentialMemory`

The proposed architecture keeps `model.py` as a compatibility facade. Pure
helpers may later move behind it, while object ownership and pickle-visible
paths remain unchanged.
