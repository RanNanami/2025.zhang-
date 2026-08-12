# Paper vs Current Protocol Diff

Only mismatches, unknowns, and implementation assumptions are shown.

| Severity | Category | Item | Status | Current | Action |
| --- | --- | --- | --- | --- | --- |
| CRITICAL | DATA | passenger aggregation formula | UNKNOWN | not present in repository | Locate author aggregation or contact authors |
| CRITICAL | DECODING | paper numerical decoder equation | UNKNOWN | custom likelihood codebook | Contact authors or run controlled codebook parity test |
| CRITICAL | ENCODING | real Gaussian sigma | UNKNOWN | 83.16008316008316 | Keep assumption explicit |
| CRITICAL | ENCODING | real center interval l | UNKNOWN | 83.16008316008316 | Keep assumption explicit |
| CRITICAL | EVALUATION | current500 direct comparability | MISMATCH | 500 rows, warmup 200 | Do not compare short diagnostic MAPE as full paper reproduction |
| CRITICAL | EVALUATION | evaluation record range | UNKNOWN | index >= warmup through limit-horizon | Do not compare short diagnostic MAPE as full paper reproduction |
| CRITICAL | EVALUATION | full-year evaluation completed | MISMATCH | False | Do not compare short diagnostic MAPE as full paper reproduction |
| CRITICAL | EVALUATION | metric equation in Zhang | UNKNOWN | sum abs error / sum abs target | Do not compare short diagnostic MAPE as full paper reproduction |
| CRITICAL | EVALUATION | paper original Ours MAPE | MISMATCH | 0.50545 (250) / 0.53928 (500) | Do not compare short diagnostic MAPE as full paper reproduction |
| CRITICAL | EVALUATION | warmup 200 provenance | UNKNOWN | local short diagnostic design | Do not compare short diagnostic MAPE as full paper reproduction |
| CRITICAL | EVALUATION | warmup in Zhang | UNKNOWN | default 5904; diagnostics 200 | Do not compare short diagnostic MAPE as full paper reproduction |
| CRITICAL | LEARNING | Scenario 1 contribution definition | PARTIAL | arrival-window strict default | Keep strict defaults unchanged during audit |
| CRITICAL | LEARNING | burst active cells | PARTIAL | all-cell active; one winner | Keep strict defaults unchanged during audit |
| CRITICAL | NETWORK | intercolumn inhibition | PARTIAL | strict rollout competition off | Mechanism diagnostic only after protocol alignment |
| CRITICAL | NETWORK | intracolumn inhibition | PARTIAL | first-spike selection | Mechanism diagnostic only after protocol alignment |
| CRITICAL | NETWORK | response scale V0 | UNKNOWN | normalized kernel (None) | Mechanism diagnostic only after protocol alignment |
| CRITICAL | NETWORK | tau_m | UNKNOWN | 0.1 | Mechanism diagnostic only after protocol alignment |
| CRITICAL | NETWORK | tau_s | UNKNOWN | 0.02 | Mechanism diagnostic only after protocol alignment |
| CRITICAL | PREDICTION | last observed to target duration | MISMATCH | 3.0 hours | Fix only after isolated parity test and user approval |
| CRITICAL | PREDICTION | prediction state anchor | MISMATCH | state ends at anchor-1 | Fix only after isolated parity test and user approval |
| CRITICAL | PREDICTION | step1 causal meaning | MISMATCH | current unobserved record | Fix only after isolated parity test and user approval |
| CRITICAL | PREDICTION | step5 causal index | MISMATCH | raw step5 corresponds to anchor+4 | Fix only after isolated parity test and user approval |
| HIGH | DATA | author code located | UNKNOWN | False | Contact authors or locate supplement |
| HIGH | DATA | exact end timestamp | UNKNOWN | 2015-06-30 23:30:00 | Do not backfill paper year range |
| HIGH | DATA | exact start timestamp | UNKNOWN | 2014-07-01 00:00:00 | Do not backfill paper year range |
| HIGH | DATA | geographic scope | UNKNOWN | not recorded | Locate generation code |
| HIGH | DATA | normalization before encoding | UNKNOWN | none; clipped to 0..40000 | Treat as implementation assumption |
| HIGH | DATA | passenger semantic unit | PARTIAL | aggregate named passenger_count | Distinguish SUM passenger_count from COUNT trips |
| HIGH | DATA | pickup/dropoff anchor | UNKNOWN | not recoverable from aggregate | Locate generation code |
| HIGH | DATA | raw source granularity | PARTIAL | already aggregated bins | Document transformation provenance |
| HIGH | DATA | source agency | PARTIAL | byte-identical to bundled Numenta [58] reference CSV | Resolve author-processed dataset provenance |
| HIGH | DATA | yellow/green/FHV scope | UNKNOWN | not recorded | Locate generation code |
| HIGH | DECODING | candidate grid | UNKNOWN | half-spacing 963 values | Contact authors or run controlled codebook parity test |
| HIGH | DECODING | most-likely criterion | PARTIAL | max timed then column overlap | Contact authors or run controlled codebook parity test |
| HIGH | DECODING | tie handling | UNKNOWN | mean all tied candidate values | Contact authors or run controlled codebook parity test |
| HIGH | DECODING | timing tolerance | UNKNOWN | 0.03 | Contact authors or run controlled codebook parity test |
| HIGH | ENCODING | 482 derivation | UNKNOWN | hard-coded paper topology; spacing derived from 0..40000 | Keep assumption explicit |
| HIGH | ENCODING | encoding resolution | PARTIAL | 41.58004158004158 | Keep assumption explicit |
| HIGH | ENCODING | passenger maximum bound | UNKNOWN | 40000.0 | Keep assumption explicit |
| HIGH | ENCODING | passenger minimum bound | UNKNOWN | 0.0 | Keep assumption explicit |
| HIGH | EVALUATION | 250 run prediction count | UNKNOWN | 45 | Do not compare short diagnostic MAPE as full paper reproduction |
| HIGH | EVALUATION | 500 run prediction count | UNKNOWN | 295 | Do not compare short diagnostic MAPE as full paper reproduction |
| HIGH | EVALUATION | evaluation window sensitivity | UNKNOWN | 0.499-0.539 WAPE across starts in existing 500 run | Do not compare short diagnostic MAPE as full paper reproduction |
| HIGH | EVALUATION | warmup 5904 provenance | UNKNOWN | Numenta [58] plotting utility | Do not compare short diagnostic MAPE as full paper reproduction |
| HIGH | LEARNING | Scenario 2 L_match eligibility | PARTIAL | timed overlap >= 4 | Keep strict defaults unchanged during audit |
| HIGH | LEARNING | new delay equation | PARTIAL | normalized half-cycle alignment | Keep strict defaults unchanged during audit |
| HIGH | NETWORK | candidate threshold crossing | PARTIAL | grid first crossing | Mechanism diagnostic only after protocol alignment |
| HIGH | NETWORK | continuous implementation | PARTIAL | reference Python integrator | Mechanism diagnostic only after protocol alignment |
| HIGH | NETWORK | cycle duration normalization | UNKNOWN | 1.0 | Mechanism diagnostic only after protocol alignment |
| HIGH | NETWORK | dendrite reset after spike | PARTIAL | first crossing modeled per prediction | Mechanism diagnostic only after protocol alignment |
| HIGH | NETWORK | integration step | UNKNOWN | 0.005 | Mechanism diagnostic only after protocol alignment |
| HIGH | NETWORK | proximal membrane integration | PARTIAL | external code handled algorithmically | Mechanism diagnostic only after protocol alignment |
| HIGH | NETWORK | timing tolerance | UNKNOWN | 0.03 | Mechanism diagnostic only after protocol alignment |
| HIGH | PREDICTION | exact five-cycle mapping | UNKNOWN | five autonomous cycles | Fix only after isolated parity test and user approval |
| HIGH | PREDICTION | no prediction handling | UNKNOWN | excluded from target/prediction MAPE; coverage separate | Fix only after isolated parity test and user approval |
| MEDIUM | DATA | DST treatment | UNKNOWN | fixed 30-minute naive grid | Locate timezone/preprocessing |
| MEDIUM | DATA | NAB 10,320 relationship | UNKNOWN | exact prefix of current full-year file | Do not call seven-month NAB file paper data |
| MEDIUM | DATA | data max | UNKNOWN | 39197 | Do not treat observed max as paper bound |
| MEDIUM | DATA | data min | UNKNOWN | 8 | Do not treat observed min as paper bound |
| MEDIUM | DATA | missing-bin treatment | UNKNOWN | no missing bins in finished CSV | Locate generation code |
| MEDIUM | DATA | modified evening wording | PARTIAL | generator applies 21:00 through 23:30 bins | Audit endpoint interpretation |
| MEDIUM | DATA | outlier treatment | UNKNOWN | clipping only in encoder | Treat as implementation assumption |
| MEDIUM | DATA | timezone | UNKNOWN | naive datetime | Locate author preprocessing |
| MEDIUM | DECODING | unique legal code count | UNKNOWN | 940 | Contact authors or run controlled codebook parity test |
| MEDIUM | ENCODING | Gaussian response normalization | UNKNOWN | standard exp distance/sigma | Keep assumption explicit |
| MEDIUM | ENCODING | passenger clipping | UNKNOWN | clip to 0..40000 | Keep assumption explicit |
| MEDIUM | ENCODING | time unit | PARTIAL | half-hour slot 0..47 | Keep assumption explicit |
| MEDIUM | ENCODING | weekday convention | PARTIAL | Python Monday=0 | Keep assumption explicit |
| MEDIUM | EVALUATION | conventional pointwise MAPE | PARTIAL | not used | Do not compare short diagnostic MAPE as full paper reproduction |
| MEDIUM | EVALUATION | coverage reporting | UNKNOWN | separate coverage field | Do not compare short diagnostic MAPE as full paper reproduction |
| MEDIUM | EVALUATION | rolling window in Zhang | UNKNOWN | 400 | Do not compare short diagnostic MAPE as full paper reproduction |
| MEDIUM | EVALUATION | rolling window provenance | UNKNOWN | local/reference [58] interpretation | Do not compare short diagnostic MAPE as full paper reproduction |
| MEDIUM | LEARNING | winner tie break | UNKNOWN | seeded least-used tie break | Keep strict defaults unchanged during audit |
| MEDIUM | NETWORK | refractory duration | PARTIAL | one normalized cycle | Mechanism diagnostic only after protocol alignment |
| MEDIUM | PREDICTION | future covariates | UNKNOWN | False | Fix only after isolated parity test and user approval |
| MEDIUM | PREDICTION | step1-5 mean used | PARTIAL | False | Fix only after isolated parity test and user approval |
| MEDIUM | PREDICTION | transient state restoration | UNKNOWN | True | Fix only after isolated parity test and user approval |
| LOW | DATA | duplicate treatment | UNKNOWN | no duplicates in finished CSV | None |
| LOW | DATA | modified start index | UNKNOWN | 13152 | Current value comes from [58], not Zhang text |
| LOW | DECODING | decoder lower-bound implication | PARTIAL | far below observed 0.4-0.6 error | Contact authors or run controlled codebook parity test |
| LOW | DECODING | perfect-code max error | UNKNOWN | 83.04573804573738 | Contact authors or run controlled codebook parity test |
| LOW | DECODING | perfect-code roundtrip MAE | UNKNOWN | 21.296961476756007 | Contact authors or run controlled codebook parity test |
| LOW | DECODING | perfect-code roundtrip WAPE | UNKNOWN | 0.0014126706909136887 | Contact authors or run controlled codebook parity test |
| LOW | ENCODING | top-K ranking tie break | UNKNOWN | lower column index | Keep assumption explicit |
| LOW | EVALUATION | paper MAPE precision | PARTIAL | local exact CSV calculation | Do not compare short diagnostic MAPE as full paper reproduction |
| LOW | NETWORK | apical input | PARTIAL | not used in one-layer Fig9 | Mechanism diagnostic only after protocol alignment |
| LOW | NETWORK | tie-break seed | UNKNOWN | 0 | Mechanism diagnostic only after protocol alignment |
