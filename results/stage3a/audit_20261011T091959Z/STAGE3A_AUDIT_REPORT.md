# Stage 3A scientific audit report

Audit run `20261011T091959Z` · generated 2026-10-11T09:19:59Z · **STAGE3A_AUDIT_PASS**

No selection, no classifier fitting and no continuous classification were performed. No historical result was modified.

## A. Overall verdict

`STAGE3A_AUDIT_PASS`

## B. Ledger reconciliation (M1)

- checks run: **44**, disagreements: **0**
- recompute rule: mean over folds within a seed, then mean over the 10 seeds

All cross-checks agree: raw records ↔ per-run CSV ↔ aggregate.json ↔ committed markdown, with no unexplained discrepancy.

*Note.* results/stage3a/aggregate.json stores per_seed_mean with STRING seed keys ('11', '17', ...) because JSON object keys are always strings, while the in-memory producer uses int keys. Values are unaffected. Cosmetic/interoperability only; no numeric discrepancy. None. The committed aggregate was NOT modified.

## C. Backend / default-cap erratum and convergence (M2)

- `SVC` docstring states libsvm: **True**
- `LinearSVC` docstring states liblinear: **True**
- `SVC` documented `max_iter` default: **-1**
- `LinearSVC` documented `max_iter` default: **1000**
- project-declared cap: **2000000** (user-declared, not a library default)

### Erratum

1. Stage 3A documentation stated the raw SVC cap of 2,000,000 was 'liblinear's own documented default'. That is incorrect on two counts.
2. sklearn.svm.SVC (including kernel='linear') is implemented using LIBSVM, not liblinear. liblinear backs sklearn.svm.LinearSVC/LinearSVR.
3. SVC's documented max_iter default is -1 (no user cap). 2,000,000 was a hard cap introduced by this project, not any library default.
4. Accordingly the accurate description is: 'user-declared hard iteration cap of 2,000,000 on the libsvm-backed SVC'.

### Convergence by dataset

| Dataset | raw converged | raw nonconverged | std converged | std nonconverged |
|---|---|---|---|---|
| ARCENE | 48/50 | 2 | 50/50 | 0 |
| DEXTER | 50/50 | 0 | 50/50 | 0 |
| MADELON | 0/50 | 50 | 50/50 | 0 |
| GINA | 0/50 | 50 | 50/50 | 0 |
| GISETTE | 0/50 | 50 | 50/50 | 0 |

750 primary predictions were produced. That is a fact about prediction generation and is independent of whether the SVM fits reached a converged optimum. A non-converged fit still returns predictions; it does not return a converged optimum.

**Cause:** SOLVER_CONDITIONING_HYPOTHESIS_SUPPORTED_NOT_PROVEN. 
Feature-scale heterogeneity is a supported contributor: MADELON's per-column standard deviations span ~224x, and 0/250 standardized fits failed to converge versus 152/250 raw fits. Standardization also changes conditioning, margin geometry and the libsvm solver's working-set selection, so the disappearance of nonconvergence after scaling is consistent with a conditioning hypothesis but does not by itself prove that scaling is the sole cause. No counterfactual (raw-fit-with-scaling-altered) run was performed, so causation is NOT established.

## D. Freeze-chronology evidence (M3)

**Status: CONTRADICTED**

| event | UTC |
|---|---|
| configs/continuous_rnd_v1.yaml last modified (mtime) | 2026-10-11T07:04:00Z |
| configs/stage3a.yaml last modified (mtime) | 2026-10-11T07:35:19Z |
| first Stage 3A run record written | 2026-10-11T07:37:19Z |
| last Stage 3A run record written | 2026-10-11T08:10:23Z |
| Stage 3A report_manifest generated_utc | 2026-10-11T08:17:33Z |
| frozen configs first committed (git committer date) | 2026-10-11T14:06:47+05:30 |
| DECLARED frozen_utc in configs/stage3a.yaml | 2026-10-11T12:00:00Z |
| DECLARED frozen_utc in configs/continuous_rnd_v1.yaml | 2026-10-11T12:00:00Z |

The self-declared `frozen_utc: 2026-10-11T12:00:00Z` in both configs is LATER than every observed Stage 3A artifact (first run record ~07:37Z, last ~08:10Z, report generated 08:17Z). A freeze timestamp cannot postdate the evaluation it claims to precede. The declared value is therefore not a truthful record of when the configs were frozen and cannot be used to support a pre-registration claim.

Filesystem mtimes place the last config modification (~07:35Z) about two minutes before the first run record (~07:37Z), which is ordering-consistent with 'configured before evaluation'. This is WEAK evidence: mtimes are mutable by ordinary file operations, carry no tamper-evident signature, and can be rewritten.

### What provenance evidence is missing
- snapshot proving the configs predated the runs.
- for comparison therefore have no version-control provenance.
- so the sweep has no self-recorded execution window.
- timestamp in the workspace.

No date was altered, no earlier time was invented, and no retroactive pre-registration is claimed. The original configs and reports are left byte-identical; this file is the only record of the finding.

## E. Selected-count distribution (M4)

| Dataset | min | p25 | median | p75 | max | mean | SD | total feats |
|---|---|---|---|---|---|---|---|---|
| ARCENE | 34 | 614.75 | 1023.5 | 1444.75 | 1561 | 963.64 | 487.29 | 10000 |
| DEXTER | 440 | 453.0 | 461.0 | 466.75 | 482 | 460.18 | 9.50 | 20000 |
| MADELON | 40 | 50.5 | 56.0 | 59.75 | 74 | 56.08 | 7.50 | 500 |
| GINA | 76 | 85.0 | 92.0 | 96.0 | 111 | 91.36 | 7.90 | 970 |
| GISETTE | 152 | 172.25 | 176.0 | 181.0 | 218 | 175.64 | 9.48 | 5000 |

### ARCENE 10 seeds x 5 folds

| seed | fold0 | fold1 | fold2 | fold3 | fold4 | mean |
|---|---|---|---|---|---|---|
| 11 | 863 | 1139 | 1422 | 1457 | 134 | 1003.0 |
| 17 | 1476 | 1482 | 1528 | 204 | 918 | 1121.6 |
| 23 | 1446 | 1032 | 232 | 481 | 922 | 822.6 |
| 31 | 745 | 1048 | 137 | 638 | 497 | 613.0 |
| 47 | 1499 | 189 | 607 | 1432 | 600 | 865.4 |
| 59 | 919 | 1523 | 461 | 824 | 1029 | 951.2 |
| 71 | 1216 | 877 | 1448 | 924 | 781 | 1049.2 |
| 83 | 785 | 1092 | 34 | 1441 | 140 | 698.4 |
| 97 | 1175 | 1536 | 1207 | 1483 | 1548 | 1389.8 |
| 109 | 37 | 1018 | 1561 | 1555 | 1440 | 1122.2 |

### ARCENE selection stability (Jaccard)
- same seed, different fold (shared arrival_order): mean 0.381 over n=100
- different seed, same fold: mean 0.307 over n=225

All five folds of a seed share ONE arrival_order, so within-seed pairs are order-matched but NOT training-set independent; across-seed pairs differ in both order and split. Jaccard is descriptive only.

**Cause of variability: UNRESOLVED.** 
ARCENE's selected count ranges widely (see matrix). Per-seed Jaccard overlap is reported above, but the cause of the spread is NOT diagnosed here. Candidate explanations (pool flush timing interacting with a large 10,000-feature space; the dynamic alpha/beta thresholds adapting to score drift) are NOT established by this audit and are not asserted. No post-hoc selection, tuning or fold choosing was performed.

Published values are from Zhou et al., Table 7, DOI 10.1145/3502737. They are a DIFFERENT RUN with a different implementation and are reference context only. No statistical significance test against a point estimate was performed, and no MATLAB runtime parity is claimed.

| Dataset | ours mean selected | published (Table 7) | difference |
|---|---|---|---|
| ARCENE | 963.64 | 33.6 | +929.99 |
| DEXTER | 460.18 | 56.7 | +403.48 |
| MADELON | 56.08 | 15.3 | +40.78 |
| GINA | 91.36 | 25.2 | +66.16 |
| GISETTE | 175.64 | 120.8 | +54.84 |

## F. Continuous R&D scope and limitations (M5)

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

## G. Pytest side-effect fix (M6)

`tests/conftest.py` wrote `results/stage1/test_results.json` on every pytest session, so any unrelated suite rewrote the Stage 1 gate artifact with its own totals (197 instead of the committed 135).

**Finding 1 — the always-on hook.** The write is now opt-in via `--write-stage1-evidence`. The payload schema and `gates.py` semantics are unchanged, and an ordinary `uv run pytest -q` leaves the Stage 1 file byte-identical. Regression tests live in `tests/test_stage1_evidence_side_effect.py`.

**Finding 2 — destructive tests, previously masked.** Making the hook opt-in exposed a second, previously hidden defect: `tests/test_evidence_fail_closed.py` wrote synthetic payloads into the **real** `results/stage1/test_results.json` and then deleted the file in a `finally` block. That damage was invisible because the always-on hook rewrote the file at the end of every session; it only surfaced once the hook stopped running. Running the full suite with the opt-in hook therefore *deleted the committed Stage 1 artifact*.

Those tests now redirect `gates.results_dir` to a temporary directory via the `isolated_stage1_results` fixture and never touch the committed file. Verified: the Stage 1 artifact is byte-identical after the full suite (219 passed, 0 failed, 0 skipped). Earlier stages silently reverted this file with `git checkout`; that is no longer needed and was never the long-term fix — the destructive tests were the actual bug.

## H. Immutable data and git audit (M7)

- datasets executed (from run records): `['ARCENE', 'DEXTER', 'GINA', 'GISETTE', 'MADELON']`
- forbidden dataset reads: **0**
- unexpected datasets executed: **[]**
- Stage 1 / Stage 2 working trees unchanged: **True**
- `git add -A --dry-run` entries: **7**
- of which data/binary/run-record entries: **0**
- commits made by this audit: **0**

Forbidden names appear in prose only, in statements that those datasets were NOT read. The executed-record check above is the positive evidence that no forbidden dataset was loaded.

### Path discrepancy

- expected `$RESEARCH_OUTPUT_ROOT/ossfs-dd-reproduction/stage3a/`: NOT_FOUND_AT_EXPECTED_PATH; observed `projects/ossfs-dd-reproduction/results/stage3a/runs/records/`. Stage 3A wrote its run records into the project results tree, not into $RESEARCH_OUTPUT_ROOT. The raw records exist and were audited at their real location; no path was invented.

## I. Questions that remain open

1. **Why did raw SVC fits fail to converge?** The conditioning hypothesis is supported but not proven; no counterfactual run was performed.
2. **Why is ARCENE's selected count so variable?** Cause reported `UNRESOLVED`; the audit deliberately did not pick a favourite fold.
3. **Were the configs genuinely frozen before evaluation?** The declared timestamp is CONTRADICTED and no pre-evaluation git commit exists to prove pre-registration.
4. **What were the authors' actual continuous Fisher-Z semantics?** Still unknown and unrecoverable from the published repository.
5. **Why do selected counts exceed the published Table 7 values?** Consistent with faithful `mi2.m` semantics, but not diagnosed here.
6. **What are the per-step beta values in the continuous runs?** Not recoverable from the saved artifacts (`NOT_MEASURED`).

## J. Next recommended experiment

A **separately authorized Stage 3B** is reasonable, and should be scoped to close open questions 1 and 2 rather than to widen coverage:

1. An SVM **conditioning study** on MADELON/GINA/GISETTE: for a fixed selected subset, vary only scaling and `max_iter`, recording `fit_status_`/`n_iter_` at each setting. This isolates the cause without changing the selector, the data, or any reported Stage 3A number.
2. An **ARCENE feature-order sensitivity** study: vary only the arrival order at a fixed seed/fold to separate order effect from split effect.

**Separate researcher approval is required before running either**, since both entail new compute on Stage 3A datasets. A full 250-run re-sweep is NOT recommended: the ledger already reconciles, so re-running it would produce no new scientific evidence.


FINAL_VERDICT: STAGE3A_AUDIT_PASS
