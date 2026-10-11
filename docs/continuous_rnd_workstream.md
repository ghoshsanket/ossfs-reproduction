# Workstream B — reconstructed continuous Fisher-Z estimator

Generated 2026-10-11T08:13:32Z · mode `continuous_reconstructed_fisherz_v1` · semantics **RECONSTRUCTED_NOT_SOURCE_VERIFIED** · verdict **CONTINUOUS_RND_DIAGNOSTIC_READY**

## Status: this is research, not reproduction

The authors' `OSFS_DD_z2.m` calls `Fisher_Ztest`, which **does not exist** in their pinned repository (commit `845a8e9d5625616b43e8cee16e8d22081dbe3f62`). Its output, units, sign handling and pairwise semantics are unknown. Everything below is our declared reconstruction from `configs/continuous_rnd_v1.yaml`.

**Stage 2's verdict is unchanged.** The `gaussian_fisher_z` route remains `UNRESOLVED` and fail-closed for these same four datasets. Workstream B does not unlock, replace or reinterpret it.

## What is source-verified vs reconstructed

The *decision structure* is taken from the pinned `z2.m` and is verifiable independently of the missing helper:

| Behaviour | Source | Reproduced |
|---|---|---|
| Warm-up **accepts** the first two features (unlike `mi2.m`, which skips them) | `z2.m:41-46` | yes |
| NaN relevance score is mapped to 0 | `z2.m:35-37` | yes |
| A pair is skipped when the helper reports `CI==1` or a NaN cmi | `z2.m:124`, `z2.m:152` | yes |
| Inner-loop break on pool promotion, outer loop continues | `z2.m:129-131` | yes |

The *scores* are ours and are not the authors' `Fisher_Ztest`:

- **marginal** — one-vs-rest Pearson correlation, Bonferroni-corrected, `tanh`-bounded to `[0,1]`;

- **pair** — an in-sample multiple correlation of `b_c ~ 1 + x + z`. This is a **heuristic** (`heuristic_joint_v1`): a two-predictor Fisher conditional-independence theorem does not transfer to an in-sample R, so the formula is declared heuristic and labelled as such everywhere it is used.

## Diagnostics (seed 11, fold 0, training rows only)

| Dataset | Train rows | Features seen | Selected | Fraction | Pair skips | Seconds |
|---|---|---|---|---|---|---|
| LEUKEMIA | 57 | 7129 | 608 | 0.085 | 0 | 7.8 |
| LYMPHOMA | 49 | 4026 | 343 | 0.085 | 0 | 4.3 |
| PROSTATE | 81 | 6033 | 469 | 0.078 | 0 | 5.5 |
| SRBCT | 50 | 2308 | 211 | 0.091 | 0 | 1.6 |

Leakage contract: every record has `training_rows_only=true`, `test_rows_read=0` and `classifier_evaluated=false` (verified: True). No accuracy was computed for any continuous dataset, because no source-verified continuous selector exists to evaluate.

## Stated limitations

- One-hot class indicators are not Gaussian, so the Fisher-Z approximation is not exact.

- The estimator captures **linear** dependence only; XOR-style synergy is invisible to it.

- With K=2 the one-vs-rest contrasts are reflected duplicates. The declared K=2 Bonferroni correction is **retained** rather than silently changed.

- One seed and one fold per dataset: this is a smoke test of the estimator's behaviour, not a performance measurement.

- Selection fractions (~8-9%) reflect the reconstructed score, not the authors' algorithm. They must not be read as a reproduction of their continuous results.

