# Phase 05 Checkpoint Compatibility

## Result

`PASS`. Phase 05 did not modify `save_strict_checkpoint()`,
`load_strict_checkpoint()`, `SequentialMemory.__getstate__()`, the checkpoint
format string, or any pickle-visible class location.

## Pre-refactor load

Read-only source:

`results/fig9_diagnostics/independent_reference_resume_smoke_part1/checkpoint.pkl`

The checkpoint predates the architecture refactor and loaded successfully with
the current `_StrictCheckpointUnpickler`.

- format: `fig9-strict-v1`
- `next_index`: 20
- model class: `seqmem.model.SequentialMemory`
- segment count: 325
- long-term fingerprint:
  `ee6aba2c009686defa4140df78ba845f72864b4e4fdb34858ccc27214067575f`
- RNG fingerprint:
  `6dd97b90e065d6926899f70321f8d96185a2e20bfd91fa6609ff8c22799b39dd`

The historical payload has 26 keys. It predates the existing
`long_sequence_rows` addition; the loader already supports its absence. Phase 05
did not add migration or rewrite logic.

## Current save and fresh-process load

`tests.test_phase05_checkpoint_compatibility` used the current formal writer to
create a temporary checkpoint, loaded it in the parent process, then started a
fresh Python interpreter and loaded the same file again.

The following were exact across processes:

- all 27 current payload keys;
- model class module;
- long-term model fingerprint;
- combined RNG fingerprint.

The temporary checkpoint was removed with its temporary directory. No historical
checkpoint or run directory was changed.

## Pickle paths

The following definitions remain in `seqmem.model`: `Synapse`, `Segment`,
`Neuron`, `MiniColumn`, `MemoryParams`, `TransientStateSnapshot`, and
`SequentialMemory`. The helper module contains functions only.
