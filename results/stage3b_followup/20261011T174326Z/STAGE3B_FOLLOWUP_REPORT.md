# Stage 3B Follow-up — diagnostic ledger repair, SVM C/tol study, ARCENE decision trajectory

Run `20261011T174326Z` · generated 2026-10-11T18:11:30Z · config `ae4db6a27387abb5`

## A. Verdict

**STAGE3B_FOLLOWUP_READY**

| gate | result | detail |
|---|---|---|
| F1_AUTHORIZATION_AND_SOURCE | PASS | forbidden reads=0; hashes verified=True; Stage 3A subsets=45/45 |
| F2_FAIL_BEFORE_PASS_AFTER | PASS | D1 3 red -> 0; D2 4 red -> 0; both repair files now pass the full suite |
| F3_SELECTOR_INVARIANCE | PASS | ordered selected IDs identical 50/50; arrival positions 50/50 |
| F4_TRUE_LEDGER | PASS | ledger reconciles with state in 50/50 runs; prefix/terminal checkpoint structure valid in 50/50 |
| F5_C_TRIAL_COMPLETE | PASS | unique fit IDs=270/270 with terminal status=270; timeouts=0; fit errors=0 |
| F6_ARCENE_CLASSIFICATION_COMPLETE | PASS | subset classifiers=150/150; baselines=15/15; non-converged SVM subset fits remain visible as diagnostics |
| F7_NO_LEAKAGE | PASS | test_rows_read==0 in 50/50 selections; scaler fitted on training rows only; fixed subsets verified |
| F8_CHRONOLOGY_AND_HASH | PASS | journal=['PRE_FLIGHT_WRITTEN', 'FIRST_DATA_EVALUATION_STARTED', 'EXPERIMENT_A_STARTED', 'FIRST_DATA_EVALUATION_STARTED', 'EXPERIMENT_A_STARTED', 'EXPERIMENT_A_FINISHED', 'FIRST_DATA_EVALUATION_STARTED', 'EXPERIMENT_A_STARTED', 'EXPERIMENT_A_FINISHED', 'FIRST_DATA_EVALUATION_STARTED', 'FIRST_DATA_EVALUATION_STARTED', 'EXPERIMENT_B_FINISHED', 'FIRST_DATA_EVALUATION_STARTED', 'EXPERIMENT_B_FINISHED', 'FIRST_DATA_EVALUATION_STARTED', 'EXPERIMENT_A_STARTED', 'EXPERIMENT_A_FINISHED', 'FIRST_DATA_EVALUATION_STARTED', 'EXPERIMENT_B_FINISHED']; every event references the frozen config digest True; write-once enforced |
| F9_REPRODUCIBILITY | PASS | SVM control repeat=True; ARCENE repeat=True; threshold oracle tolerance 1e-10 |
| F10_CODE_AND_IMMUTABILITY | PASS | ruff clean=True; pytest failed=0 skipped=0; protected paths clean=True; forbidden git entries=0; Stage 1 evidence present=True |

## B. What changed and why

Two **diagnostic-only** repairs. Neither changes feature-selection mathematics.

**D1 — `Selector._redundancy_pass` (`src/ossfs_dd/selector.py`).** The scan can queue older candidates in `to_remove` and then set `drop_new` and break; the branch that *applies* `to_remove` never runs, yet the queued IDs were still returned, counted in `state.redundancy_removals`, and written into the accept event's `details['removed']`.

- Fail-before: `AssertionError: D1: accept event reported [3, 9] but only the newcomer was actually deleted` (`assert [3, 9] == [9]`).

- Fix: build the ledger only from removals that actually shrank the candidate list, via `CandidateSet.remove`'s boolean return. Deletion order, pair-search order, tie-breaking and promotions are untouched.

**D2 — `checkpoint_rows` (`src/ossfs_dd_stage3b/trace.py`).** All events were accumulated into the cumulative counters *before* the checkpoint loop, so the checkpoint at arrival 100 reported end-of-run totals.

- Fail-before: `AssertionError: D2: checkpoint 100 still equals the final total` (`assert 400 != 400`).

- Fix: consume events chronologically and snapshot at each checkpoint's own prefix. A `pool_flush` is not an arrival; the post-stream final flush is counted only in the terminal summary.

Counter meanings now fixed and documented in the frozen config: `direct_accepts` = entered the direct-accept path; `promoted_from_pool` = actual candidate insertions; `redundancy_removals` = **applied** deletions only; `n_selected` = candidate-list length, not a de-duplicated set size; the `mi2` warm-up **skips** the first two arrivals and they are not insertions.

## C. Reproduction safety

- Selector invariance: **50/50** ARCENE runs reproduce the saved Stage 3B ordered selected IDs, and 50/50 reproduce the arrival positions. The repairs are diagnostic-only.
- True ledger: **50/50** runs reconcile candidate-list length, direct inserts, pool inserts and applied removals independently of `state`.
- Protected Stage 1/2/3A/3B paths clean: **True** (dirty: none).
- Git staging dry-run contains 0 data/binary/trace entries. Commits made: 0.

## D. Frozen protocol

| event | UTC |
|---|---|
| PRE_FLIGHT_WRITTEN | 2026-10-11T17:43:26Z |
| FIRST_DATA_EVALUATION_STARTED | 2026-10-11T17:43:49Z |
| EXPERIMENT_A_STARTED | 2026-10-11T17:43:49Z |
| FIRST_DATA_EVALUATION_STARTED | 2026-10-11T17:43:55Z |
| EXPERIMENT_A_STARTED | 2026-10-11T17:43:55Z |
| EXPERIMENT_A_FINISHED | 2026-10-11T17:43:57Z |
| FIRST_DATA_EVALUATION_STARTED | 2026-10-11T17:44:07Z |
| EXPERIMENT_A_STARTED | 2026-10-11T17:44:07Z |
| EXPERIMENT_A_FINISHED | 2026-10-11T17:47:10Z |
| FIRST_DATA_EVALUATION_STARTED | 2026-10-11T17:47:54Z |
| FIRST_DATA_EVALUATION_STARTED | 2026-10-11T17:49:54Z |
| EXPERIMENT_B_FINISHED | 2026-10-11T17:50:45Z |
| FIRST_DATA_EVALUATION_STARTED | 2026-10-11T17:50:51Z |
| EXPERIMENT_B_FINISHED | 2026-10-11T17:51:41Z |
| FIRST_DATA_EVALUATION_STARTED | 2026-10-11T18:00:39Z |
| EXPERIMENT_A_STARTED | 2026-10-11T18:00:39Z |
| EXPERIMENT_A_FINISHED | 2026-10-11T18:00:39Z |
| FIRST_DATA_EVALUATION_STARTED | 2026-10-11T18:00:40Z |
| EXPERIMENT_B_FINISHED | 2026-10-11T18:01:30Z |

Config SHA-256 `ae4db6a27387abb542bc7682` was written **before** the first real-data fit or selection and is guarded write-once. This is an **in-process, chronologically ordered freeze** — there is no independent attestation, so it is **not** an independently registered preregistration. The Stage 3A contradicted chronology is preserved unchanged.

Known source-fingerprint limitation, disclosed rather than hidden: Stage 3A's per-run `code_hash` is non-authoritative (11 distinct values, recomputed per task from a moving source tree). Validation here uses `protocol_digest`, subset identity, the stable Stage 3B raw records and the current source fingerprint.

## E. Experiment A — fixed-standardization SVM C/tol

| dataset | C | tol | converged | rate | acc (converged) | acc (nonconv. diag) | median n_iter |
|---|---|---|---|---|---|---|---|
| MADELON | 0.1 | 0.001 | 15/15 | 1.000 | 0.6081 | n/a | 18407 |
| MADELON | 0.1 | 0.0001 | 15/15 | 1.000 | 0.6083 | n/a | 50624 |
| MADELON | 1.0 | 0.001 | 15/15 | 1.000 | 0.6068 | n/a | 219975 |
| MADELON | 1.0 | 0.0001 | 15/15 | 1.000 | 0.6063 | n/a | 651958 |
| MADELON | 10.0 | 0.001 | 1/15 | 0.067 | 0.5904 | 0.6076 | 2000000 |
| MADELON | 10.0 | 0.0001 | 0/15 | 0.000 | n/a | 0.6071 | 2000000 |
| GINA | 0.1 | 0.001 | 15/15 | 1.000 | 0.8465 | n/a | 21048 |
| GINA | 0.1 | 0.0001 | 15/15 | 1.000 | 0.8464 | n/a | 58589 |
| GINA | 1.0 | 0.001 | 15/15 | 1.000 | 0.8447 | n/a | 303499 |
| GINA | 1.0 | 0.0001 | 14/15 | 0.933 | 0.8465 | 0.8213 | 1327639 |
| GINA | 10.0 | 0.001 | 0/15 | 0.000 | n/a | 0.8444 | 2000000 |
| GINA | 10.0 | 0.0001 | 0/15 | 0.000 | n/a | 0.8443 | 2000000 |
| GISETTE | 0.1 | 0.001 | 15/15 | 1.000 | 0.9580 | n/a | 37708 |
| GISETTE | 0.1 | 0.0001 | 15/15 | 1.000 | 0.9580 | n/a | 104674 |
| GISETTE | 1.0 | 0.001 | 15/15 | 1.000 | 0.9567 | n/a | 598868 |
| GISETTE | 1.0 | 0.0001 | 8/15 | 0.533 | 0.9563 | 0.9573 | 1764649 |
| GISETTE | 10.0 | 0.001 | 0/15 | 0.000 | n/a | 0.9573 | 2000000 |
| GISETTE | 10.0 | 0.0001 | 0/15 | 0.000 | n/a | 0.9564 | 2000000 |

**A-H1 (C sensitivity): INCONCLUSIVE.** 0/3 datasets meet the 0.02 criterion; every dataset lacked the 15 complete converged pairs (C=10 rarely converges), so no dataset could be assessed.

**A-H2 (tol sensitivity): SUPPORTED.** 3/3 datasets show a >=10% median n_iter increase when tol tightens from 1e-3 to 1e-4 (criterion: >=2).

**A-H3 (feasible convergence): NOT_SUPPORTED.** 173/270 planned cells converged (rate 0.6407); threshold 0.95 -> NOT_SUPPORTED.

Control cell (C=1, tol=1e-3) vs saved Stage 3B standardized highest-cap: **0/0** match at tolerance 1e-10 (status `NOT_DIRECTLY_COMPARABLE`).

Uncertainty: repeated cross-validation folds are not independent patients, so the 15 subsets per cell are correlated. These are descriptive effect sizes, not inference. A capped or warned solver is `NONCONVERGED` and its accuracy is a diagnostic only; those are never pooled with converged accuracy.

## F. Experiment B — ARCENE trajectory and downstream accuracy

| fold | CART min | CART max | CART range | >= 0.05 |
|---|---|---|---|---|
| 0 | 0.7500 | 0.9000 | **0.1500** | yes |
| 1 | 0.7000 | 0.8750 | **0.1750** | yes |
| 2 | 0.6250 | 0.8750 | **0.2500** | yes |
| 3 | 0.6500 | 0.8250 | **0.1750** | yes |
| 4 | 0.6500 | 0.7750 | **0.1250** | yes |

**B-H2 (downstream order effect, CART): SUPPORTED.** 5/5 folds show a within-fold CART accuracy range >= 0.05 across the ten arrival orders (criterion: >=3).

**B-H3 (mechanism): MECHANISM_UNRESOLVED.** Threshold/pool/accept trajectories were captured for all 50 runs, but the observed count co-varies with counters that are arithmetically coupled to it (insert - remove). No additional independent intervention was performed, so no mechanism is claimed.

The range reflects order-induced subset differences within a single fixed split; the folds overlap in patients, so this is within-fold sensitivity, not an independent-sample test.

All-feature baselines (contextual comparators, same three frozen classifiers on all 10,000 original features, identical fold indices):

| fold | KNN | CART | SVM |
|---|---|---|---|
| 0 | 0.8500 | 0.6250 | 0.9250 |
| 1 | 0.8000 | 0.7000 | 0.9500 |
| 2 | 0.7500 | 0.6750 | 0.9000 |
| 3 | 0.8000 | 0.5750 | 0.8500 |
| 4 | 0.8250 | 0.6500 | 0.8250 |

## G. Mechanism analysis

Directly observed: within a fixed fold, the arrival order changes both **how many** features survive and **which** ones, and the downstream CART accuracy range far exceeds the 0.05 criterion on every fold.

Algebraic, not causal: `n_selected` equals candidate-list length and the ledger identity `direct inserts + pool inserts - applied removals = list length` holds in every run. Because `pool_flushes`, `promoted_from_pool` and `redundancy_removals` are arithmetically coupled to the count, any correlation between them and `n_selected` is partly entailed by that identity and is **not** independent evidence of a mechanism.

Correlation between n_selected and its accounting components is not independent evidence of causation; these components define the count.

No mechanism is claimed. Establishing one would require an additional independent intervention that this prompt does not authorize.

## H. Scientific limitations

- The five ARCENE folds overlap in patients; 50 runs are **not** 50 independent samples and no population-level p-value is reported.
- Accuracy is measured on reused folds, so several exploratory plots reuse the same 40 held-out rows.
- The paper's ARCENE Table 7 value (~33.65 features) is **contextual only** — not a target, and no MATLAB runtime parity is claimed.
- Continuous-feature semantics remain unverified and were excluded here.
- Stage 3A's freeze chronology remains **CONTRADICTED**; this follow-up's freeze is in-process only, and the Stage 3A `code_hash` limitation stands.
- Nonconverged SVM fits (notably at C=10) are reported as diagnostics; their accuracies are not established classifier results.

## I. Failures and resource usage

- Experiment A statuses: `{'CONVERGED': 173, 'NONCONVERGED': 97}` — 0 timeouts, 0 fit errors.
- Subset classifier statuses: `{'COMPLETED': 150}`.
- Baseline statuses: `{'COMPLETED': 15}`.
- Nonconverged (scientific, not a failure): **97/270** Experiment A fits and **2/150** ARCENE SVM fits. All are visible with their diagnostic accuracies.
- Budgets: CPU-only, ≤2 fit workers, BLAS threads 1, 10 min/fit, well inside the 3 h (A) and 4 h (B) ceilings. No partial factorial; no row was removed.

## J. Concrete next research action

1. **Convergent-C study.** C=10 fails to converge on 89/90 standardized cells while C=0.1 and C=1 fully converge. A follow-up over an intermediate C grid would locate the conditioning boundary — without touching the selector.

2. **Order-permutation ablation.** Randomising the arrival order *within* the Stage 3B folds to separate order effects from the fold's training-set composition.

No Stage 3C and no new algorithm work is proposed or performed here.

## Figures

- `/home/iemiedc2026/Documents/Sanket/Research/outputs/ossfs-dd-reproduction/stage3b_followup/20261011T174326Z/figures/a_c_tol_MADELON.png`
- `/home/iemiedc2026/Documents/Sanket/Research/outputs/ossfs-dd-reproduction/stage3b_followup/20261011T174326Z/figures/a_c_tol_GINA.png`
- `/home/iemiedc2026/Documents/Sanket/Research/outputs/ossfs-dd-reproduction/stage3b_followup/20261011T174326Z/figures/a_c_tol_GISETTE.png`
- `/home/iemiedc2026/Documents/Sanket/Research/outputs/ossfs-dd-reproduction/stage3b_followup/20261011T174326Z/figures/b_accuracy_vs_size_by_order.png`
- `/home/iemiedc2026/Documents/Sanket/Research/outputs/ossfs-dd-reproduction/stage3b_followup/20261011T174326Z/figures/b_candidate_trajectory.png`

FINAL_VERDICT: STAGE3B_FOLLOWUP_READY
