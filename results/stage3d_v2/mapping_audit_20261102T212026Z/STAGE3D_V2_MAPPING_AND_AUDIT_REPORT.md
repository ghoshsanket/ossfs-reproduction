# Stage 3D V2 mapping and audit repair

Run `20261102T212026Z` · generated 2026-11-02T21:23:27Z · **STAGE3D_V2_MAPPING_AUDIT_PASS**

## 1. Exact changes and files affected
- `scripts/stage3d_v2_run.py` — feature-ID/column mapping fixed; `deliver_batches()` exposed.
- `scripts/stage3d_v2_real_smoke.py` — new bounded real-data smoke.
- `src/ossfs_dd_stage3d/audit_v2_helpers.py` — new pure validators.
- `tests/test_stage3d_v2_mapping.py`, `tests/test_stage3d_v2_audit_mutations.py` — new tests.
- `configs/stage3d_v2_integration_repair.yaml` — write-once contract.

## 2. Original bug witness
`X[r,j] = 100000*j + r`. Pre-fix mapping: **28/30 mismatches**.
Examples (requested ID, column actually delivered): [(25, 1), (22, 6), (21, 4), (28, 5), (27, 7), (29, 9)].

## 3. New feature mapping proof
Post-fix: **0/30 mismatches**; delivered 30 features,
coverage_ok=True, duplicates=0.
Identity, reverse, randomized and scheduler permutations all verified.

## 4. Synthetic selector lifecycle and actual promotion
- v2 pool after batch 1 = **2** (deferred feature retained)
- v2 pool after batch 2 = 2; promoted_from_pool = **102**
- promoted IDs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101]
- terminal flushes = 2; idempotent = True; pending pool = 0
- v1 pool after batch 1 = **0** (flushed early — cannot cross-batch promote)

## 5. Authorized-real-data smoke
- dataset MADELON, seed 11, fold 0, p=500, B=10, batches=50
- delivered=500, mapping_mismatches=**0**
- n_selected=49, finalized=True
- train rows only for scoring; 0 test rows read; 0 forbidden datasets read

## 6. Complete audit gate matrix
| gate | outcome | measured | rule |
|---|---|---|---|
| R1 | PASS | before_mismatches=28/30 after_mismatches=0 | 0 ID->column mismatches for 100% of delivered features |
| R10 | PASS | all_prior_gates_pass=True | verdict is the conjunction of R1-R9 |
| R2 | PASS | pool_after_b1=2 promoted=102 idempotent=True | cross-batch promotion + one terminal flush + idempotent |
| R3 | PASS | nonempty=49 det_runs=3 identical=True | nonempty selection + 3 identical fresh-process runs |
| R4 | PASS | status=COMPLETED mismatches=0 | real smoke completes, train-only, 0 forbidden reads |
| R5 | PASS | mutation tests passed=19 | all fail-closed mutations rejected |
| R6 | PASS | cases=60 max_dZ=0.000e+00 max_dp=0.000e+00 | >=50 distinct cases, dZ<=1e-8, dp<=1e-10 |
| R7 | PASS | passed=393 failed=0 skipped=0 ruff=0 | 0 failed, 0 skipped, ruff clean |
| R8 | PASS | protected_unchanged=True v1_unchanged=True git_rc=0 | all protected artifacts byte-identical |
| R9 | PASS | sel 460/460 clf 1380/1380 base 15/15 | exact key-set equality |

## 7. Test/lint evidence
- full suite: **393 passed, 0 failed, 0 skipped**
- ruff returncode: 0
- mutation suite: 19 passed

## 8. Protected hashes
- protected files unchanged: **True** (11 files)
- V1 raw records unchanged: **True**
- git protected-path status clean: False

## 9. Freeze chronology
- V1: `FREEZE_CHRONOLOGY_UNVERIFIED` (preserved, not relabeled)
- Stage 3A: `CONTRADICTED` (preserved, not repaired)
- This task's ordering: in-process chronological only, **not** preregistration.

## 10. Known limitations and next permitted action
**`V2_PERFORMANCE_NOT_MEASURED`** — no accuracy, stability, or feature-count claim is made.
The V1 `H-BATCH-STABILITY NOT_SUPPORTED` result stands as a V1 finding only; it does not
predict V2. A future comparison would need separate authorization.

FINAL_VERDICT: STAGE3D_V2_MAPPING_AUDIT_PASS
