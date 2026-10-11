# Stage 1 Live Disk Audit

**Generated (UTC):** 2026-10-10T22:38:05Z  
**Elapsed:** 145.7 s  
**Git commit:** `f522ac63d804` (dirty: True)  
**Interpreter:** Python 3.11.16, aarch64, project venv: True  
**GPU used:** False  
**Dataset tree:** `dataset/ossfs-dd-reproduction` — outside git: True (dataset tree is outside the repository)

This report is recomputed from bytes on disk. It does not trust any previously
reported state, and it is read-only with respect to datasets.

## Outcome

```
DATASET_FILES_PRESENT
```

every dataset present, hash-verified, dimensionally correct, all plans valid, no stream leakage observed

## Aggregate verification

| Measure | Value |
|---|---:|
| raw files physically present and SHA-256 verified | 11/11 |
| processed X present and SHA-256 verified | 11/11 |
| dimensions + class count match Table 1 | 11/11 |
| split plan files present | 110 |
| order plan files present | 110 |
| CV folds checked (disjointness + coverage + bounds) | 550 |
| fold disjointness violations | 0 |
| fold coverage violations | 0 |
| index-bounds violations | 0 |
| arrival-order permutation violations | 0 |
| inverse-order violations | 0 |
| v1 backups checked / arrival order preserved | 110 / 110 |
| stream columns emitted (row-count coverage) | 875760 |
| stream columns value-compared against `X[train, col]` | 1320 |
| stream value mismatches | 0 |
| stream row-count violations | 0 |

## Per dataset

| Dataset | Outcome | n×p×c expected | n×p×c observed | classes | non-finite | raw sha256 | failures |
|---|---|---|---|---:|---:|---|---:|
| SRBCT | `DATASET_FILES_PRESENT` | 63×2308×4 | 63×2308 | 4 | 0 | `d5bb3895e998` | 0 |
| LYMPHOMA | `DATASET_FILES_PRESENT` | 62×4026×3 | 62×4026 | 3 | 0 | `b10aa781ff96` | 0 |
| PROSTATE | `DATASET_FILES_PRESENT` | 102×6033×2 | 102×6033 | 2 | 0 | `711cef5a0411` | 0 |
| LEUKEMIA | `DATASET_FILES_PRESENT` | 72×7129×2 | 72×7129 | 2 | 0 | `0fd632f7469f` | 0 |
| DLBCL | `DATASET_FILES_PRESENT` | 77×7129×2 | 77×7129 | 2 | 0 | `30bc95073e27` | 0 |
| ARCENE | `DATASET_FILES_PRESENT` | 200×10000×2 | 200×10000 | 2 | 0 | `c7a40e39edf2` | 0 |
| DEXTER | `DATASET_FILES_PRESENT` | 600×20000×2 | 600×20000 | 2 | 0 | `70400df29217` | 0 |
| BREAST | `DATASET_FILES_PRESENT` | 97×24481×2 | 97×24481 | 2 | 0 | `9a4971e3bc26` | 0 |
| MADELON | `DATASET_FILES_PRESENT` | 2600×500×2 | 2600×500 | 2 | 0 | `eb99a0153ebe` | 0 |
| GINA | `DATASET_FILES_PRESENT` | 3468×970×2 | 3468×970 | 2 | 0 | `d4b95d97b031` | 0 |
| GISETTE | `DATASET_FILES_PRESENT` | 7000×5000×2 | 7000×5000 | 2 | 0 | `430bb631b3e4` | 0 |

## Failure reasons (machine codes)

None. Every check passed on every dataset.

## Scope note

This audit verifies **storage, integrity, dimensional correctness and plan**
validity**. It does not assess scientific provenance, and it does not re-run
the Stage 1 provenance gate — see `provenance_followup.md` for that.