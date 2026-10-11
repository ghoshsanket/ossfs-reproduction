# Continuous reconstructed estimator — audited boundaries

Mode `continuous_reconstructed_fisherz_v1` · status **RECONSTRUCTED_NOT_SOURCE_VERIFIED**

These are **not** the authors' `Fisher_Ztest` semantics; that helper is absent from their pinned repository.

**a.** The authors' Fisher_Ztest helper is absent from commit 845a8e9d5625616b43e8cee16e8d22081dbe3f62. These scores are a declared reconstruction, not the authors' semantics.

**b.** One-vs-rest class labels are binary 0/1 indicators. The Fisher-Z approximation assumes bivariate normality, which a Bernoulli indicator does not satisfy; the test is an approximation.

**c.** heuristic_joint_v1 is an in-sample two-predictor linear regression of b_c on (1, x, z). It cannot generally identify XOR-style or otherwise non-linear synergy, and its in-sample R is upward-biased relative to an out-of-sample quantity.

## (d) Does a single pair score ever have to exceed 2*beta?

- pair score range: `[0, 1]`
- actual threshold form: `promote when marginal_of_a + pair_score > 2*beta`
- joint computation adds a marginal component: **True**
- impossibility claim: **NOT_APPLICABLE**

A bare pair score is bounded to [0,1], so a condition requiring the PAIR SCORE ALONE to exceed 2*beta>1 could never hold. But inspection of the actual code shows the comparison uses `a.score + pair` (marginal + pair), which ranges over [0,2]. The impossibility claim therefore does NOT apply to this implementation, and it would be wrong to assert it.

Empirical: promotions observed = **True**; promoted-from-pool per dataset = `{'SRBCT': 161, 'LYMPHOMA': 468, 'PROSTATE': 277, 'LEUKEMIA': 294}`; beta ever exceeded one = **False**.

Pool promotions were observed, so the promotion condition was satisfiable: beta stayed small enough that marginal+pair exceeded 2*beta. Zero pair skips is NOT evidence of 'no interaction'; skips only count pairs the estimator could not evaluate.

Per-step thresholds: **NOT_MEASURED** — The diagnostic records retain counts but not per-event alpha/beta traces, so the exact per-step threshold at each comparison cannot be reconstructed from the saved artifacts. `beta_exceeds_one_events` is a recorded counter and bounds beta at every evaluated step.

## Per-dataset diagnostics

| Dataset | train rows | seen | selected | fraction | pair skips | promoted | beta>1 events | train-only | test rows read | classifier |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SRBCT | 50 | 2308 | 211 | 0.091 | 0 | 161 | 0 | True | 0 | False |
| LYMPHOMA | 49 | 4026 | 343 | 0.085 | 0 | 468 | 0 | True | 0 | False |
| PROSTATE | 81 | 6033 | 469 | 0.078 | 0 | 277 | 0 | True | 0 | False |
| LEUKEMIA | 57 | 7129 | 608 | 0.085 | 0 | 294 | 0 | True | 0 | False |

No classifier was fit and no accuracy computed on any continuous dataset; all four diagnostics record classifier_evaluated=false and test_rows_read=0.
