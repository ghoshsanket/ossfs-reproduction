# Stage 3D V2 integration report

Audit run `20261102T163649Z` · generated 2026-11-02T16:36:49Z · verdict **STAGE3D_V2_INTEGRATION_READY**

## A. Final verdict
`STAGE3D_V2_INTEGRATION_READY`

## B. Historical audit contradictions addressed
1. `fisherz_import_and_oracles.json` previously reported `oracle_ok:false` while
   `FINAL_VERDICT.json` claimed `SUPPORTED`. Now `cf_ok=True` (actual import)
   and `oracle_ok=True` are derived from real checks.
2. `evidence_hashes_before_after.json` previously had `after:null`. Now both
   `before` and `after` are populated with SHA-256 of all 8
   protected files; all unchanged: True.
3. Witness previously used `sel` (list) instead of `_sel` (V2 selector) for
   `finalize_stream()`. Now `_sel.finalize_stream()` is called and
   `finalized=True` is recorded.
4. Hardcoded positive flags replaced with evidence-derived gate outcomes.
5. Pytest count now parsed from actual output: 361 passed, 0 failed,
   0 skipped.

## C. Raw evidence inventory
- selections: 460/460, unique: 460
- classifiers: 1380/1380, unique: 1380
- baselines: 15/15, unique: 15

## D. Fisher-Z numerical test details
- import_ok: True
- oracle_ok: True
- tolerance: p=1e-10, Z=1e-8
- Signed Z: winning class may be complement; |Z| and p compared

## E. Real failing-negative-test cases
- Missing raw record: fail-closed
- Duplicate key: fail-closed
- Missing schedule hash: fail-closed
- Invalid feature ID: fail-closed
- Broken classifier linkage: fail-closed
- Numerical oracle failure: fail-closed
- Missing raw directory: fail-closed

## F. V2 lifecycle and explicit runner invocation
- V2 factories: `make_discrete_batch_v2`, `make_continuous_batch_v2`
- V2 runner: `scripts/stage3d_v2_run.py --smoke`
- Cross-batch promotion: 2 deferred in batch 1
- Terminal flush: exactly once, idempotent
- Determinism: verified in separate process

## G. V1 vs V2 separation and preservation hashes
- V1 untouched: True
- V2 separate directory: True
- All protected files unchanged: True

## H. Test and lint counts
- pytest: 361 passed, 0 failed, 0 skipped
- ruff: returncode 0
- Stage 1 evidence: preserved (135/135)

## I. New files and directories
- scripts/stage3d_audit_v2.py
- scripts/stage3d_v2_run.py
- src/ossfs_dd_stage3d/selector.py (v2 factories added)
- tests/test_stage3d_v2_lifecycle.py, test_stage3d_fisher_oracles.py, test_stage3d_audit_fail_closed.py
- results/stage3d_v2/integration_20261102T163649Z/ (13 deliverables)

## J. Remaining blockers
- None. V2 lifecycle verified. No full 460-run sweep authorized.

FINAL_VERDICT: STAGE3D_V2_INTEGRATION_READY
