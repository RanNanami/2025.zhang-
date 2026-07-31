# Fig.9 Context Trajectory Decomposition

This diagnostic is read-only and keeps actual-observation and autonomous-rollout trajectories separate.

- Trace rows: 6552
- Unique dependencies: 3276
- Unique observations: 120
- Unique segments: 201
- Trajectory pair coverage: 1.000000
- Unknown rate: 0.000000

## actual_observation

- Dominant first loss: `EMITTED_NOT_PROPAGATED`
- Dominant recent loss: `EMITTED_NOT_PROPAGATED`
- Dominant latest state loss: `EMITTED_NOT_PROPAGATED`
- Dominant pattern: `NEVER_PRESENT_AFTER_CREATION`
- Dominant primary reason: `EMITTED_NOT_PROPAGATED`

## autonomous_rollout

- Dominant first loss: `LOST_INTERCOLUMN`
- Dominant recent loss: `LOST_INTERCOLUMN`
- Dominant latest state loss: `NONE`
- Dominant pattern: `LOST_THEN_RECOVERED`
- Dominant primary reason: `LOST_BEFORE_CURRENT_TRANSITION`

## Recommendation

`recommended_next_step = SEGMENT_CONTEXT_REPRESENTATION_FIX_REQUIRED`

The recommendation is diagnostic, not a strict-default change.
