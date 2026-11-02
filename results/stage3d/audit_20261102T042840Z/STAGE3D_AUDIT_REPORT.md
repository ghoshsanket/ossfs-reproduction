# Stage 3D evidence audit report

Audit run `20261102T042840Z` · generated 2026-11-02T09:31:15Z · verdict **STAGE3D_AUDIT_PASS**

## A. Audit verdict
`STAGE3D_AUDIT_PASS`. All 460 selections, 1,380 classifier outcomes, and 15 baselines are
present with unique keys. Numerical oracles pass. The batch-pool defect is
demonstrated and repaired in a versioned v2 mode.

## B. Raw evidence inventory
- selections: 460/460, unique keys: 460
- classifiers: 1380/1380, unique keys: 1380
- baselines: 15/15, unique keys: 15
- datasets: ['ARCENE', 'DEXTER', 'GINA', 'GISETTE', 'LEUKEMIA', 'LYMPHOMA', 'MADELON', 'PROSTATE', 'SRBCT']
- arms: ['C1', 'C2', 'C3', 'C4', 'D1', 'D2']

## C. Independent reproduction of hypotheses
- H-BATCH-STABILITY: NOT_SUPPORTED (delta=-0.1354, threshold +0.10)
- H-BATCH-CART: SUPPORTED (mean delta=+0.0035, threshold -0.03)
- H-PVAL-SCORING: descriptive only (no advance claim)
- H-NUMERICAL-CORRECTNESS: SUPPORTED

## D. Fail-before/pass-after evidence
- D1 (batch pool): v1 flushes pool between batches (witness: v1 pool_size_after_batch1=0, v2=2)
- D2 (import): `conditional_fisherz` imported nonexistent `FisherMarginal`; fixed to `FisherResult`

## E. Batch semantics and v1/v2 separation
- v1 (`batch_score_shared_v1`): flushes pool at end of every batch (archived)
- v2 (`batch_score_shared_v2`): pool persists across batches; capacity flush only at 100; terminal flush once at stream end

## F. Broken import / Fisher checks
- Import: fixed (FisherMarginal -> FisherResult)
- Oracles: Fisher r,Z,p, Holm, partial correlation all pass within tolerance

## G. Chronology and preservation
- Freeze chronology: UNVERIFIED (no independent attestation)
- Original records unchanged: True
- No commits made

## H. Test/lint status
- pytest: .                                                                        [100%]
- ruff: returncode 0

## I. Unresolved blockers
- None blocking the audit verdict; the pool bug is repaired in v2

## J. Next experiment
Separate authorization required for any new sweep or algorithm change.

FINAL_VERDICT: STAGE3D_AUDIT_PASS
