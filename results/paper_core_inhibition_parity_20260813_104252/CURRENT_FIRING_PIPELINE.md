# Current Firing Pipeline

The canonical audit is in `docs/CURRENT_FIRING_PIPELINE.md`.

The strict pipeline is: active context -> incoming segments -> dendritic PSP
crossing -> event score selection -> earliest-time column selection -> stored
prediction candidate -> selected-cell propagation. Proximal unmatched columns
use the all-cell burst branch.
