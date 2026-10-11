# Stage 3B follow-up — protocol, repairs and causal limits

A **new, separately versioned** follow-up. It does not replace or alter
Stage 3B, Stage 3A, or the unchanged Stage 2 partial gate.

## 1. Two diagnostic-only repairs

Neither changes feature-selection mathematics. Both were proven **fail-before /
pass-after** and are documented in `repair_fail_before_pass_after.json`.

| | file | symbol | defect | fix |
|---|---|---|---|---|
| **D1** | `src/ossfs_dd/selector.py` | `Selector._redundancy_pass` | candidates queued in `to_remove` were reported and counted even when `drop_new` broke the scan and the apply branch never ran | ledger built only from removals that actually shrank the candidate list, via `CandidateSet.remove`'s boolean return |
| **D2** | `src/ossfs_dd_stage3b/trace.py` | `checkpoint_rows` | all events were accumulated *before* the checkpoint loop, so checkpoint 100 showed end-of-run totals | events consumed chronologically; counters snapshotted at each prefix; the post-stream final flush is counted only in the terminal summary |

Preserved exactly: deletion order, pair-search order, tie-breaking, promotions,
selected feature IDs and their order. Verified by **50/50** ARCENE runs
reproducing the saved Stage 3B ordered selected IDs and arrival positions.

### Counter semantics (frozen in the config)

- `direct_accepts` — entered the direct-accept path, even if removed immediately.
- `promoted_from_pool` — actual candidate insertions from a pool flush.
- `redundancy_removals` — **only applied, successful** deletions.
- `n_selected` — candidate-list length, **not** a de-duplicated set size.
- `mi2` warm-up **skips** the first two arrivals; they are not insertions.

If these definitions had conflicted with the live selector, the conflict would
have been disclosed rather than resolved by editing feature selection.

## 2. Experiment A — fixed-standardization SVM C/tol (270 fits)

3 datasets (MADELON, GINA, GISETTE) x 3 seeds x 5 folds x 3 `C` x 2 `tol`.
Standardization is **frozen on** and fitted on selected **training rows only**.
Subsets come from the raw Stage 3A records; nothing is reselected.

`SVC(kernel='linear')` is libsvm-backed; `LinearSVC` is never substituted.
`C` and `tol` change the model/margin and the optimization stopping
respectively, so neither is a clean decomposition of "conditioning".

Hypotheses are fixed before execution and are **not** operational gates:

- **A-H1** |mean acc(C=0.1) - mean acc(C=10)| at tol=1e-3 >= 0.02 on >=1 dataset.
- **A-H2** median `n_iter` at C=1 rises >=10% when tol tightens 1e-3 -> 1e-4, on >=2 datasets. A zero denominator yields `UNDEFINED`, never a division.
- **A-H3** >=95% of 270 planned cells converge.

## 3. Experiment B — ARCENE trajectory and downstream accuracy

5 folds x 10 order seeds = 50 cells, split **always** Stage 1 `seed_11.json`.
`order_seed` selects only the permutation, never the split.

Three frozen classifiers per subset (150) plus 15 all-feature baselines:
KNN(k=5, uniform, minkowski p=2), CART(gini, best, `random_state=11` **fixed
across all orders within a fold**), and SVC(linear, C=1, tol=1e-3, cap 2e6).

- **B-H1** (operational gate): ledger conservation + selected-ID agreement 50/50.
- **B-H2** CART within-fold accuracy range across the ten orders >= 0.05 on >=3 folds.
- **B-H3** no mechanism preregistered; `MECHANISM_UNRESOLVED` unless the traces
  distinguish a non-arithmetic explanation.

## 4. Causal limitations

1. **Scaling, `C` and `tol` are confounded interventions.** Varying `C` also
   changes the model/margin; varying `tol` changes optimization stopping.
2. **`n_selected` is arithmetically coupled to its own components.**
   `inserts - removals = list length` holds by construction, so correlating
   count with its components is not independent evidence of mechanism.
3. **Folds overlap in patients.** 50 runs are not 50 independent samples; no
   population-level p-value is reported and accuracy is measured on reused rows.
4. **Nonconverged SVM fits are diagnostics, not established results.** They are
   never pooled with converged accuracy.
5. **Paper Table 7 (~33.65 features) is contextual only** — not a target, and no
   MATLAB runtime parity is claimed.

## 5. Chronology and provenance

`configs/stage3b_followup_v1.yaml` is frozen and hashed **before** the first
real-data fit or selection, guarded write-once, with a runtime UTC stamp and an
append-only event journal. This is an **in-process, chronologically ordered
freeze** — it is **not** an independently registered or attested
preregistration. Stage 3A's contradicted chronology is preserved unchanged.

Stage 3A's per-run `code_hash` is non-authoritative (11 distinct values, since
it was recomputed per task from a moving source tree). Validation uses
`protocol_digest`, subset identity, the stable Stage 3B raw records and the
current source fingerprint instead.