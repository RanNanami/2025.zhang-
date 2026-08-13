# Timing Tolerance Provenance

`MemoryParams.timing_tolerance` is currently `0.03`. Git search first finds the introduction in `c195603 Initial paper reproduction package`. The local audit found no paper-explicit or reference-defined numeric value. It is therefore classified as `LOCAL_IMPLEMENTATION / LOCAL_TEMPORAL_CREDIT_ASSUMPTION`. This round does not sweep or change it.

The value is used by `observe_code` to confirm a saved candidate and by `_punish_wrong_predictions` to identify a missing or mismatched proximal event.
