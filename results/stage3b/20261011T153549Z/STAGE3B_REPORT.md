# Stage 3B — SVM conditioning and ARCENE feature-order sensitivity

Run `20261011T153549Z` · generated 2026-10-11T16:01:41Z · config `03749595529f1ad4`

## A. Verdict and scope

**STAGE3B_READY**

- Experiment A: **450/450** SVM fits (234 converged, 216 nonconverged, 0 timeout, 0 error).
- Experiment B: **50/50** ARCENE selections (5 folds x 10 orders, split seed 11 fixed).

This is an operational completion verdict, **not** evidence that the 2022 paper is reproduced. Scientific hypotheses are reported separately.

## B. Data, subset and authorization proof

- Experiment A datasets: `['ARCENE', 'DEXTER', 'MADELON', 'GINA', 'GISETTE']` — all `EXACT_VERIFIED`, stored-matrix SHA-256 re-verified at preflight.
- Experiment B dataset: **ARCENE only**, shape 10000 features, Stage 1 `seed_11.json` folds.
- Stage 3A subsets loaded from the runner-defined path `results/stage3a/runs/records/`      **75/75** verified (unique id-sequences, in-range, n_selected consistent).
- Forbidden reads: **0**. `DLBCL`/`BREAST` and the four continuous datasets were never read.

> **Stage 3A provenance defect carried forward.** All 250 Stage 3A records share one `protocol_digest` (True), but they carry 11 distinct `code_hash` values because Stage 3A recomputed that hash per task from a source tree being edited mid-sweep. `code_hash` is therefore **not** authoritative. The Stage 3B gate uses `protocol_digest` plus a selection-code fingerprint (`be005ea1f3fdd96e`) instead.

## C. Protocol freeze

| event | UTC |
|---|---|
| PRE_FLIGHT_WRITTEN | 2026-10-11T15:35:49Z |
| FIRST_DATA_EVALUATION_STARTED | 2026-10-11T15:36:09Z |
| EXPERIMENT_B_FINISHED | 2026-10-11T15:38:00Z |
| FIRST_DATA_EVALUATION_STARTED | 2026-10-11T15:38:26Z |
| EXPERIMENT_A_STARTED | 2026-10-11T15:38:26Z |
| LAST_EVALUATION_FINISHED | 2026-10-11T15:38:26Z |
| FIRST_DATA_EVALUATION_STARTED | 2026-10-11T15:38:44Z |
| EXPERIMENT_A_STARTED | 2026-10-11T15:38:44Z |
| LAST_EVALUATION_FINISHED | 2026-10-11T15:38:45Z |
| FIRST_DATA_EVALUATION_STARTED | 2026-10-11T15:38:49Z |
| EXPERIMENT_A_STARTED | 2026-10-11T15:38:49Z |
| LAST_EVALUATION_FINISHED | 2026-10-11T15:44:53Z |
| FIRST_DATA_EVALUATION_STARTED | 2026-10-11T16:01:32Z |
| EXPERIMENT_B_FINISHED | 2026-10-11T16:01:32Z |
| FIRST_DATA_EVALUATION_STARTED | 2026-10-11T16:01:33Z |
| EXPERIMENT_A_STARTED | 2026-10-11T16:01:33Z |
| LAST_EVALUATION_FINISHED | 2026-10-11T16:01:33Z |

Config frozen and hashed (`03749595529f1ad4`) **before** the first fit or selection, and the write-once guard refuses to overwrite it. This is an **in-process, hashed, chronologically ordered freeze** — there is no independent timestamp attestation, and it is **not** a preregistration. Stage 3A's contradictory freeze timestamp is preserved unchanged.

## D. SVM conditioning (Experiment A)

| dataset | scaling | cap | converged | rate | acc (converged) | acc (nonconv. diagnostic) |
|---|---|---|---|---|---|---|
| ARCENE | raw | 2,000,000 | 15/15 | 1.0000 | 0.8600 | n/a |
| ARCENE | standardized | 2,000,000 | 15/15 | 1.0000 | 0.8483 | n/a |
| DEXTER | raw | 2,000,000 | 15/15 | 1.0000 | 0.8961 | n/a |
| DEXTER | standardized | 2,000,000 | 15/15 | 1.0000 | 0.8950 | n/a |
| MADELON | raw | 2,000,000 | 0/15 | 0.0000 | n/a | 0.5236 |
| MADELON | standardized | 2,000,000 | 15/15 | 1.0000 | 0.6068 | n/a |
| GINA | raw | 2,000,000 | 0/15 | 0.0000 | n/a | 0.4655 |
| GINA | standardized | 2,000,000 | 15/15 | 1.0000 | 0.8447 | n/a |
| GISETTE | raw | 2,000,000 | 0/15 | 0.0000 | n/a | 0.8501 |
| GISETTE | standardized | 2,000,000 | 15/15 | 1.0000 | 0.9567 | n/a |

**A-H1:** SUPPORTED. Standardized convergence exceeds raw by 1.00 (criterion >= 0.50) and reaches 1.00 (criterion >= 0.80) on MADELON, GINA and GISETTE.

**A-H2:** SUPPORTED. All 10 prespecified cap comparisons hold with zero violations.

**Interpretation limits.** Scaling's effect here is an *association under a controlled intervention*. Standardization also changes margin geometry and the solver's working-set selection, so this does **not** isolate a physical cause. Note the raw non-converged accuracies are near or below chance (e.g. GINA 0.4655 — they are recorded as incomplete-fit diagnostics and are never pooled with converged accuracy.

**Control check vs Stage 3A:** 150 same-subset accuracies compared at cap 2,000,000, tolerance 1e-10      **0 mismatches**.

## E. ARCENE order sensitivity (Experiment B)

| fold | min | q25 | median | q75 | max | mean | SD | range | >=500 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 223 | 601.0 | 733.0 | 1083.5 | 1509 | 828.8 | 383.7 | **1286** | yes |
| 1 | 9 | 987.0 | 1110.5 | 1175.5 | 1253 | 943.3 | 415.6 | **1244** | yes |
| 2 | 42 | 176.5 | 528.0 | 1267.5 | 1488 | 678.0 | 602.5 | **1446** | yes |
| 3 | 180 | 594.2 | 1000.0 | 1436.8 | 1504 | 967.4 | 492.4 | **1324** | yes |
| 4 | 134 | 499.2 | 1221.5 | 1499.0 | 1561 | 1003.2 | 568.7 | **1427** | yes |

**B-H1:** SUPPORTED. 5/5 folds reach a within-order range of >= 500 features with the CV split held **completely fixed** (criterion: >= 3). Order alone moves the count from 9 to 1561.

**Jaccard (225 same-split cross-order pairs):** mean 0.3397, median 0.2962, min 0.0041. Low overlap means order changes *which* features survive, not merely how many.

**Anchor match:** 5/5 anchor cells reproduce the existing Stage 3A ARCENE records exactly.

**B-H2:** ASSOCIATION_OBSERVED_NOT_ISOLATED. `redundancy_removals` anti-correlates with `n_selected` on every fold, but the identity `n_selected = direct_accepts + promoted_from_pool - redundancy_removals` holds exactly for 45/50 runs, so that association is **arithmetically entailed**, not a mechanism. The five exceptions (all high-promotion runs) show the Stage 2 counters are not a perfect ledger. **MECHANISM_UNRESOLVED.**

## F. Reproducibility and failures

- Raw ledger re-read from disk: 450 fit records, 50 selection records, all with unique IDs.
- Traces: 50 files    0 arrival/ID mismatches      0 checkpoint alpha/beta recomputation mismatches at tolerance 1e-10.
- Determinism `MADELON_seed11_fold0__raw__it200000`: original=NONCONVERGED, repeat=NONCONVERGED, identical=True (fresh process          timing/RSS excluded          convergence not forced).
- Determinism `GISETTE_seed11_fold0__standardized__it200000`: original=NONCONVERGED, repeat=NONCONVERGED, identical=True (fresh process          timing/RSS excluded          convergence not forced).
- ARCENE repeat `ARCENE_split11_fold0_order11`: selected IDs identical=True.
- Timeouts: 0    fit errors: 0. No observation was dropped or replaced.

## G. Scientific limitations

- The five ARCENE folds overlap in training rows and each order is reused across folds      the 50 cells are **not** 50 independent samples. This identifies within-split sensitivity to order, not a population generalisation estimate.
- Repeated CV cells are not independent patients    no inferential p-value is claimed.
- Stage 3A's freeze timestamp remains **CONTRADICTED**    Stage 3B's freeze is in-process only.
- Published ARCENE Table 7 (~33.65 features) is contextual, **not** a target or comparator for these controlled runs, and no MATLAB runtime parity is claimed.
- The continuous-feature route remains unverified and was **excluded** here.
- Scaling is one of several changed factors    its effect is association, not proven cause.

## H. Next research decision

Both hypotheses were supported, so the follow-up should target the open questions rather than repeat these runs:

1. **SVM conditioning, narrowly.** At a fixed subset, vary *only* `C` and `tol` under fixed scaling to separate conditioning from margin choice. This is the one factor Stage 3B deliberately did not vary.
2. **ARCENE decision trajectory.** The Stage 2 counters are not an exact ledger (5/50 identity exceptions)      a decision-trajectory diagnosis would explain why pool promotion dominates and why counts swing by >1000 under permutation.

**No Stage 3C and no new algorithm is recommended without explicit permission.**

## Figures

- `/home/iemiedc2026/Documents/Sanket/Research/outputs/ossfs-dd-reproduction/stage3b/20261011T153549Z/figures/svm_convergence_by_cap.png`
- `/home/iemiedc2026/Documents/Sanket/Research/outputs/ossfs-dd-reproduction/stage3b/20261011T153549Z/figures/arcene_counts_heatmap.png`
- `/home/iemiedc2026/Documents/Sanket/Research/outputs/ossfs-dd-reproduction/stage3b/20261011T153549Z/figures/arcene_within_fold_range.png`
- `/home/iemiedc2026/Documents/Sanket/Research/outputs/ossfs-dd-reproduction/stage3b/20261011T153549Z/figures/arcene_jaccard.png`

FINAL_VERDICT: STAGE3B_READY
