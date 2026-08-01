# Fig.9 real L_match call chain

This audit describes the controlled nonpaper `L_match` ablation. The strict
default remains `4`; no prediction, learning, competition, or selector rule is
changed by the ablation plumbing.

## Configuration path

| Stage | File and line | Behavior |
|---|---|---|
| Strict default | `experiments/fig9_strict_reproduction.py:126-136` | `Fig9StrictConfig.l_match` defaults to `4`. |
| CLI | `experiments/fig9_strict_reproduction.py:3027` | `--l-match` accepts only `2`, `3`, or `4`; default is `4`. |
| Safety gate | `experiments/fig9_strict_reproduction.py:3290-3296` | Values `2` and `3` require `--lmatch-real-ablation`. |
| Model construction | `experiments/fig9_strict_reproduction.py:608-619` | `config.l_match` is passed unchanged to `MemoryParams`. |
| Protocol | `experiments/fig9_strict_reproduction.py:742-782` | `protocol_fingerprint()` records `L_match`. |
| Diagnostic labels | `experiments/fig9_strict_reproduction.py:840-852` | The explicit ablation records nonpaper and no-ground-truth markers. |

`MemoryParams` has a library-level default of `3` in
`src/seqmem/model.py:362`. Fig.9 does not rely on that value: its strict runner
always supplies `Fig9StrictConfig.l_match`, whose default is `4`.

## Real matching condition

1. `SequentialMemory.observe_code()` reaches the unpredicted-event path and
   calls `_best_matching_neuron()` at `src/seqmem/model.py:2051-2063`.
2. `_best_matching_neuron()` gathers live segments reachable from the actual
   previous active-cell context and computes each segment's `timed_overlap` at
   `src/seqmem/model.py:2394-2437`.
3. The selected segment is accepted as Scenario 2 only when its real
   `timed_overlap >= self.params.l_match` at
   `src/seqmem/model.py:2132-2136`. Otherwise observation follows the existing
   Scenario-3 segment-creation path. This is the only model variable changed by
   the experiment.

The threshold does not alter candidate scoring, continuous PSP integration,
competition, intracolumn selection, or rollout. Step 2 onward can nevertheless
diverge because Scenario-2 reinforcement versus Scenario-3 creation changes
the learned recurrent trajectory.

## Checkpoint and resume

`save_strict_checkpoint()` records the frozen config, protocol fingerprint,
model, and an explicit top-level `L_match` while retaining checkpoint format
`fig9-strict-v1` (`experiments/fig9_strict_reproduction.py:353-406`).

`validate_checkpoint_l_match()` at
`experiments/fig9_strict_reproduction.py:428-455` reads the value from the
top-level field, older frozen config, protocol fingerprint, or model params.
`run_strict_stream()` calls it before restoring state at line 1839. A mismatch
is rejected rather than silently resuming under another threshold.

## Counterfactual diagnostics

`experiments/diagnostics/fig9_match_overlap.py:520-575` computes the real
`meets_L_match` field plus read-only `passes_L1` through `passes_L4` columns.
Those columns are calculated after capturing the same observation context.
They do not assign `model.params.l_match`, invoke matching, select a segment,
or affect learning. Changing real `L_match` therefore changes the model's true
Scenario path while preserving the meaning of all offline counterfactual
columns.

## Tests that assumed four

`tests/test_fig9_strict_reproduction.py` explicitly checks the Fig.9 default
of `4`; this remains correct. Fig.8 and generic `SequentialMemory` tests use
their own local `MemoryParams` values and are outside this Fig.9 ablation.
