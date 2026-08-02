# Fig.9 L4 Windows Native Crash Resolution

## Outcome

- Formal output: `results/fig9_diagnostics/lmatch_l4_native_recovery_formal_20260802_132357`
- Native exit code: `0`
- Process wall time: `703.5617202 s`
- Predictions: `45/45`, coverage `1.0`
- MAPE: `0.4086558112184448` (exact historical match)
- Final model fingerprint: `19cd26f434d69a099bc2c4b5819d4a255f756c5cf515f9572f26a9ee8a15a2b8`
- Final RNG fingerprint: `7840c30ab61b7b05198c2194dd2c39deb4c58a5dc61f3430bedcae1c4688e11e`
- Recoverable checkpoint next observation index: `245`

`next_index=245` is intentional: with 250 records and horizon 5, prediction
anchors are indices 200 through 244. All 250 records are used, but indices
245 through 249 are future targets rather than subsequent observation anchors.
Changing this field to 250 would misrepresent the model state on resume.

## Root Cause

The operational trigger was `faulthandler.dump_traceback_later(60,
repeat=True)` on this Windows/Python runtime. Two isolation failures occurred
immediately after the 60-second watchdog message. The sampled model frame moved
from the C `lru_cache` key hash to an ordinary Python dictionary lookup after a
strictly equivalent cache experiment, showing that the sampled model line was
not the stable fault source.

The fix keeps native fault handling enabled but replaces the asynchronous
watchdog with a normal daemon Python thread. Every 60 seconds that thread calls
`faulthandler.dump_traceback()` synchronously while holding the GIL. The final
run completed 11 periodic snapshots without a native crash.

Evidence strength is high for the operational trigger, but the exact native
faulting module remains unproven because ProcDump was unavailable. WER reported
`BEX64`, `StackHash_ac46`, and an ntdll PCH address rather than a definitive
module stack.

## Validation

- Full unit suite: `325/325` passed.
- Compileall and `git diff --check`: passed.
- Core, compression, branch, preselection, teacher, observe, match, context,
  density, and single-thread 210-to-220 isolation paths were exercised.
- All successful isolation paths had identical prediction SHA, MAPE, model
  fingerprint, and RNG fingerprint.
- Predictions SHA256: `5be58c849c1299e2f531c4d7ef7c2ddc1146c8737d0e69b2f83810f118b5c668`.
- Independently recomputed MAPE exactly matched the summary.
- Every gzip diagnostic stream was read to EOF successfully.
- The context trajectory analyzer completed independently with zero unknown
  reasons and trajectory pair coverage `1.0`.

## Model Semantics

No model formula, L_match behavior, learning rule, rollout rule, RNG ordering,
competition parameter, encoder, decoder, or strict default was changed. The
changes are limited to crash logging, process supervision, checkpoint
durability at a non-multiple terminal index, and diagnostic execution safety.
