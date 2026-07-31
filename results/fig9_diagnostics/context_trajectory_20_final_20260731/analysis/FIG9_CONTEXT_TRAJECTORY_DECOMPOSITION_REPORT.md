# Fig.9 Context Trajectory Decomposition

This diagnostic is read-only and keeps actual-observation and autonomous-rollout trajectories separate.

- Trace rows: 1662
- Unique dependencies: 831
- Unique observations: 49
- Unique segments: 78
- Trajectory pair coverage: 1.000000
- Unknown rate: 0.000000

## actual_observation

- Dominant first loss: `EMITTED_NOT_PROPAGATED`
- Dominant recent loss: `EMITTED_NOT_PROPAGATED`
- Dominant latest state loss: `EMITTED_NOT_PROPAGATED`
- Dominant pattern: `NEVER_PRESENT_AFTER_CREATION`
- Dominant primary reason: `NEVER_REACTIVATED_AFTER_CREATION`

## autonomous_rollout

- Dominant first loss: `LOST_BEFORE_CURRENT_TRANSITION`
- Dominant recent loss: `LOST_BEFORE_CURRENT_TRANSITION`
- Dominant latest state loss: `LOST_BEFORE_CURRENT_TRANSITION`
- Dominant pattern: `NO_TRAJECTORY_HISTORY`
- Dominant primary reason: `LOST_BEFORE_CURRENT_TRANSITION`

## Recommendation

`recommended_next_step = AUTONOMOUS_CONTEXT_DRIFT_FIX_REQUIRED`

The recommendation is diagnostic, not a strict-default change.
