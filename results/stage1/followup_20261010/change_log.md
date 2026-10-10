# Stage 1 Follow-up — Change Log

**Recorded (UTC):** 2026-10-10T16:59:49Z

## Test counts

| | before | after |
|---|---:|---:|
| tests passed | 22 | 63 |
| tests failed | 0 | 0 |
| tests skipped | 0 | 0 |

New coverage: real-data loading/streaming (`tests/test_real_data_loading.py`) and
fail-closed evidence (`tests/test_evidence_fail_closed.py`).

## Repairs

| ID | Defect | Repair | Data touched |
|---|---|---|---|
| B1 | `inverse_position_of_feature` stored the FORWARD mapping, duplicating `arrival_order` | now the true inverse `feature_index -> arrival_position`; schema `stream_order/2.0` | none — `arrival_order` byte-identical in all 110 plans, proven against `*.json.v1backup` |
| B2 | Stage 2 handoff loader resolved workspace-relative paths against CWD; doc also called non-existent `stream_orders()` | uses `resolve()`; added `stream_orders_dir`; verified by real-data tests from a foreign CWD | none |
| B3 | `sample_ids.npy` object-dtype, required `allow_pickle=True` | fixed-width Unicode, `allow_pickle=False`; `processed/2.0` | values proven identical to the v1 backups; no predictor/label touched |
| B4 | git audit returned `0` when git was unavailable/failed — absent evidence read as a pass | fail-closed: status `ok`/`unavailable`/`error`, `violations=null` when unmeasured; skipped required tests and config-load failure also block | none |
| B5 | `self_check()` reported `train_indices_disjoint_from_complement`, a tautology | replaced with element-wise `emitted == X[train, col]` plus held-out distinguishability, on real data | none |

## Migrations (explicit, versioned, backed up, deterministic)

| Migration | Files | Backup | Verified |
|---|---:|---|---|
| `stream_order/1.0 -> 2.0` | 110 | `*.json.v1backup` | `arrival_order` identical in 110/110 |
| `processed/1.0` sample_ids object -> Unicode | 11 | `sample_ids.npy.v1backup` | values identical in 11/11 |

121 backup files retained. They are not `*.json`, so gate-visible plan counts stay
110/110.

## Processed artifact hashes — did anything change?

| Artifact | Changed? | Why |
|---|---|---|
| `X` (all 11) | **NO** | no repair touched predictor values |
| `y` (all 11) | **NO** | no repair touched labels |
| `sample_ids.npy` (11) | **YES** | dtype object -> fixed-width Unicode (B3); values identical |
| `stream_orders/*.json` (110) | **YES** | inverse field corrected (B1); `arrival_order` unchanged |
| raw archives (all) | **NO** | raw is immutable; never rewritten |
| split plans (110) | **NO** | CV membership is untouched |

## New finding, not repaired

BREAST carries source-level sentinels (`100.0` x42,026; `-2.0` x4,686),
confirmed present in BOTH the CSV and MATLAB encodings of the source record.
A conservative detector now measures this for every dataset and flags BREAST
only. Nothing was imputed, clipped or replaced, and no status was relabelled.
See `provenance_followup.md` §5.

## Gate impact

**None.** `configs/stage1.yaml` untouched. `exact_verified_provenance` still
requires 11/11 and still fails at 9/11. `STAGE1_PARTIAL` stands.

## Current result artefact hashes (SHA-256)

```
results/stage1/test_results.json: 2bb6d1a10612d5ece7920f602184864c1bd4e76552c422768009134f92b3cd53
results/stage1/readiness_summary.json: 4f5696ddca037b4bedbda72c1d0b744ceabbc1aa5ef17a72a408e309bcaee0d8
results/stage1/STAGE1_REPORT.md: ee2384cedd3396217693d9e9fd1036beb2e17cfdc9f7802d86fed65af233107b
results/stage1/dataset_status.csv: a9c8d45d673e7d5ea52855351e08092030e310babc8c699e1ba344a8f4c0360f
results/stage1/source_registry_ledger.csv: 7758bbc442aec38a634ca8475b84f93bc0020c47eeb8ccae8e9edc80027ced28
```
