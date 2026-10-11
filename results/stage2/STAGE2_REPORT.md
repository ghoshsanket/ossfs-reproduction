# Stage 2 Report — OSSFS-DD Restricted Implementation

**Generated (UTC):** 2026-10-11T05:54:47Z  
**Run id:** 20261011T055002Z  
**Mode:** `matlab_compatible_v1` · seed 11 · fold 0  
**Git HEAD:** `b44b8b01bc7d`  
**Authors' source:** `https://github.com/doodzhou/OSFS` @ `845a8e9d5625616b43e8cee16e8d22081dbe3f62`

> No classifier was fitted and no accuracy was computed. Feature counts and
> runtimes below are engineering diagnostics, not paper results.

## Verdict

```
STAGE2_PARTIAL
```

Failed gates: full_stream_restricted_smoke

## Gates

| Gate | Threshold | Result | Detail |
|---|---|---|---|
| `restricted_dataset_membership` | 9 allowlisted / 2 forbidden / 0 forbidden reads | **PASS** | 9 allowed, 2 excluded, 9 attempted, 0 unhandled |
| `stream_integrity` | 0 duplicate final selected IDs | **PASS** | 1670 selected IDs across 5 completed runs, 0 duplicates |
| `stage2_tests` | 100% pass, 0 failed, 0 skipped | **PASS** | 135 passed, 0 failed, 0 skipped (full Stage 1 + Stage 2 suite) |
| `stage1_regression` | all Stage-1 required tests pass, none skipped | **PASS** | missing=[] skipped=[] |
| `selfcheck` | all mathematical self-checks pass | **PASS** | 22 passed, 0 failed |
| `full_stream_restricted_smoke` | 9 | **FAIL** | 5/9 completed with verified semantics; 4/9 UNSUPPORTED_UNVERIFIED_SEMANTICS (['SRBCT', 'LYMPHOMA', 'PROSTATE', 'LEUKEMIA']) |
| `source_semantic_audit` | every behaviour labelled; no guess presented as confirmed | **PASS** | audit complete against https://github.com/doodzhou/OSFS@845a8e9d5625; 3 items UNRESOLVED (Fisher_Ztest absent) and declared as such |
| `matlab_runtime_parity` | parity claimed only if measured | **PASS** | NOT_MEASURED — not claimed |
| `runtime_resource_safety` | <= 32 GiB RSS, <= 60 min/dataset | **PASS** | all completed runs within budget |

## Restricted dataset smoke (seed 11, fold 0, whole feature stream)

| Dataset | Status | Paper type | Metric family | p | n_train | n_selected | secs | peak RSS GiB |
|---|---|---|---|---:|---:|---:|---:|---:|
| SRBCT | `UNSUPPORTED_UNVERIFIED_SEMANTICS` | Real | gaussian_fisher_z | 2308 | — | — | — | — |
| LYMPHOMA | `UNSUPPORTED_UNVERIFIED_SEMANTICS` | Real | gaussian_fisher_z | 4026 | — | — | — | — |
| PROSTATE | `UNSUPPORTED_UNVERIFIED_SEMANTICS` | Real | gaussian_fisher_z | 6033 | — | — | — | — |
| LEUKEMIA | `UNSUPPORTED_UNVERIFIED_SEMANTICS` | Real | gaussian_fisher_z | 7129 | — | — | — | — |
| ARCENE | `COMPLETED` | Integer | discrete_information | 10000 | 160 | 863 | 1.911 | 0.071 |
| DEXTER | `COMPLETED` | Integer | discrete_information | 20000 | 480 | 470 | 49.807 | 0.071 |
| MADELON | `COMPLETED` | Integer | discrete_information | 500 | 2080 | 57 | 0.29 | 0.088 |
| GINA | `COMPLETED` | Integer | discrete_information | 970 | 2774 | 98 | 2.167 | 0.114 |
| GISETTE | `COMPLETED` | Integer | discrete_information | 5000 | 5600 | 182 | 64.808 | 0.138 |

## Why four datasets did not run

The four paper-`Real` datasets route to `OSFS_DD_z2.m`, which calls
`Fisher_Ztest`. That function **does not exist anywhere in the pinned authors'
repository** (67 blobs, zero matches). Its score, units and conditional test cannot
be determined from source, so the runner raises
`UNSUPPORTED_UNVERIFIED_SEMANTICS` instead of substituting a proxy statistic.
No `1-p`, `|r|`, `p`, correlation magnitude, discretised MI or `mutual_info_classif`
was used. See `docs/stage2_source_audit.md` §2.

## Tests

- Full suite: **135 passed, 0 failed, 0 skipped** (Stage 1 + Stage 2)
- Mathematical self-check: **22 passed, 0 failed**
- Git audit: `ok`, 0 violations

## Stage 1 untouched

- Stage-1 verdict unchanged: `STAGE1_PARTIAL`
- No Stage-1 config, gate, manifest, split or processed array was modified.