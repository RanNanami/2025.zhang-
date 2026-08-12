# Fig.9 Windows Runtime Stability Report

## Conclusion

`EXTERNAL_BLOCKER`. The repository-level runtime problem is **not fixed** and the
acceptance condition `RUNTIME_STABILITY_FIXED` has not been reached.

The strongest supported conclusion is process memory/control-flow corruption
outside the model's legal Python state transitions. The exact failing component
(Windows, firmware, CPU, or RAM) is not yet proven. Repository code changes must
not be presented as a fix until the machine is checked and fresh P2/P3 runs pass
the required repeated validation.

## Evidence

- Failures are not record deterministic. Identical checkpoints can pass a window
  once and crash in another attempt at a different index or rollout step.
- CPython 3.12 failed with `0xC0000005` in `python312.dll` at varying offsets.
- Independently installed CPython 3.13 failed with
  `Fatal Python error: _PyEval_EvalFrameDefault: Executing RESERVED instruction`
  and Windows reported `0xC0000409` in `ucrtbase.dll`.
- A core-only P3 checkpoint replay, with optional readout and compressed
  diagnostic traces disabled, still crashed. This excludes those diagnostic
  paths as the root cause.
- Module-state instrumentation observed impossible Python-level symptoms such as
  `_EXP` appearing as a `DSDynamicsParams` instance and a dynamics-parameter
  object appearing as a float. No legal assignment path was found.
- Failed checkpoints load cleanly in a fresh process with valid model/module
  types and fingerprints. Restore itself does not immediately corrupt state.
- Windows Error Reporting retained a readable minidump, but this machine lacks
  the Microsoft debugger executable needed for a native stack analysis.
- The installed ASUS FX607JV BIOS is 314. ASUS publishes BIOS 316 (2025-11-04)
  and recommends the firmware update for system performance optimization:
  https://www.asus.com/us/supportonly/fx607jv/helpdesk_bios/

## Required Questions

1. **Final crash symptom:** intermittent native access violations and control-flow
   corruption, sometimes preceded by impossible Python object-type changes.
2. **Record deterministic:** no.
3. **Correct `_EXP` type:** built-in callable `math.exp`.
4. **Legal `_EXP` rebinding found:** no; the repository defines it once.
5. **Actual incorrect Python assignment found:** no.
6. **Why can `tau_m` params appear as float:** no legal Python path explains it;
   this is evidence of underlying process memory corruption, not a confirmed
   assignment bug.
7. **Checkpoint restore pollutes module state:** not observed. Failed checkpoints
   deserialize cleanly in new processes.
8. **Diagnostic callback participates:** callbacks were audited and guarded, but
   core-only crashes show diagnostics are not required for the failure.
9. **Native dump captured:** Windows WER captured a CPython 3.13 minidump.
10. **Faulting module:** observed modules include `python312.dll` and
    `ucrtbase.dll`; older events also include `ntdll.dll`/unknown invalid execute.
11. **Root cause found:** the repository root cause was not found. External
    runtime corruption is strongly supported; its exact hardware/OS component is
    unresolved.
12. **Evidence level:** high for an external native/runtime fault, low for the
    exact physical component.
13. **Final retained change:** low-cost module-state/native breadcrumbs and
    incremental diagnostic readout writing.
14. **Model mathematics changed:** no.
15. **RNG changed:** no.
16. **Strict defaults changed:** no.
17. **Expected valid outputs before changes:** P2 MAPE about 0.489661 and P3 MAPE
    about 0.444416 from historical complete runs.
18. **Outputs after retained diagnostic change:** successful bounded A/B windows
    preserve predictions, model fingerprint, and RNG fingerprint. No new fresh
    valid 500 result exists.
19. **Prediction SHA preserved:** yes in successful controlled A/B runs.
20. **Model fingerprint preserved:** yes in successful controlled A/B runs.
21. **RNG fingerprint preserved:** yes in successful controlled A/B runs.
22. **P2 500 successes:** three 450-to-495 resume-window completions in the final
    stress sequence; these are not fresh 500 completions.
23. **P2 crashes:** multiple historical native crashes; the count is not treated
    as exhaustive because older result directories predate the unified ledger.
24. **P3 500 successes:** three 450-to-495 resume-window completions after
    streaming readout; these are not fresh 500 completions.
25. **P3 crashes:** at least one fresh run at index 409 plus one core-only replay
    and one CPython 3.13 replay in the final isolation sequence.
26. **Three consecutive accepted successes:** no. Resume windows do not satisfy
    the required fresh/continuous validation.
27. **Fresh-process stable:** no; a fresh P3 run crashed.
28. **Resume stable:** no; identical checkpoint replays are nondeterministic.
29. **Continuous stable:** no.
30. **Debug on/off stable:** no; core-only still crashes.
31. **Tests:** 504 tests passed at commit `a1d8fa1`.
32. **Compileall:** passed.
33. **Diff check:** passed.
34. **Relevant commits:** `03881d1`, `e5ca657`, `091f030`, `a1d8fa1`.
35. **Runtime conclusion:** `EXTERNAL_BLOCKER`.
36. **Reached `RUNTIME_STABILITY_FIXED`:** no.

## Retained Engineering Change

Commit `a1d8fa1` streams readout diagnostic rows incrementally when the explicit
streaming option is enabled. It reduces diagnostic memory pressure and preserves
the successful A/B outputs. It is a diagnostic I/O improvement, not a native
crash fix.

## External Validation Required

1. Update the FX607JV from BIOS 314 to the official BIOS 316 using ASUS guidance.
2. Reboot and run Windows Memory Diagnostic (`mdsched.exe`), then inspect its
   result after restart.
3. In an elevated terminal, run `sfc /scannow` and
   `DISM /Online /Cleanup-Image /RestoreHealth`.
4. Install Microsoft's Debugging Tools for Windows and analyze the retained WER
   minidump to identify the native faulting thread and stack.
5. Replay the same checkpoint on another known-stable machine. This is the most
   direct separation of repository behavior from this host.
6. Only after the machine checks pass, rerun fresh P2 and P3 500 validations. Do
   not call the runtime fixed unless all acceptance repetitions pass.

