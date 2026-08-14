# Reproduction Credibility Draft

## Scale

- `A`: protocol, mechanism, data provenance, and numerical result are strongly verified.
- `B`: protocol is substantially aligned, but numerical or unpublished-detail gaps remain.
- `C`: nonpaper diagnostic, approximation, ablation, or application transfer.
- `D`: historical output known to violate the corrected protocol or claim boundary.

## Fig.7: B

The corrected stream and sequence protocol are substantially aligned. Final
ten-trial accuracy is high, but convergence is slower and less reliable than
the paper. This supports a partial protocol-aligned reproduction, not complete
numerical agreement. Locally retrained reference models remain grade C.

## Fig.8: B for corrected CBT protocol, C/D for auxiliaries

Autonomous neural retrieval now propagates actual predictive cells and does
not replay decoded words. The 1000-sentence distance remains 3.981, far from
the paper's approximate 0.2 at 10,000 symbols. Capacity, response-scale,
competition, and temporal-confirmation runs are grade C diagnostics. The old
proximal-replay claim is grade D.

## Fig.9: B for strict 250, C/D elsewhere

The strict 250-record L_match=4 result uses raw autonomous propagation and has
MAPE 0.505450 with complete coverage. The paper is near 0.10, the exact author
taxi asset is not verified, and 2000/full-year strict runs are incomplete.
Competition, oracle, selector, real L_match=2, ETTh1, and Weather work are grade
C. Historical compensated 0.098 output is grade D.

## Claim boundary

ETTh1 and Weather demonstrate application-transfer behavior only. They cannot
be used as evidence that Zhang Fig.9 has been reproduced. The strongest
defensible project statement is that key protocols and mechanisms have been
implemented and audited, while important numerical gaps and unpublished paper
details remain.
