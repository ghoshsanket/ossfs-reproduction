# Stage 3A audit addendum

Audit run `20261011T091959Z`. This addendum **corrects** statements in
`docs/stage3a_evaluation.md`, `configs/stage3a.yaml` and their commentary. Per
the audit contract the original files are left byte-identical; this file and the
versioned `results/stage3a/audit_20261011T091959Z/` directory carry the corrections.

## Corrections to previously published statements

1. **Solver backend.** `sklearn.svm.SVC(kernel='linear')` is implemented using
   **libsvm**. Stage 3A described the cap in liblinear terms. liblinear backs
   `sklearn.svm.LinearSVC`/`LinearSVR`, which are **not** drop-in replacements:
   different solver, different default loss, different multiclass handling.
2. **Iteration cap.** `SVC`'s documented `max_iter` default is **-1**. The value
   2,000,000 used in Stage 3A is a **user-declared hard cap** declared by this
   project, not a library or solver default. It should be read as
   "user-declared cap of 2,000,000 on the libsvm-backed SVC".
3. **Nonconvergence cause.** Stage 3A stated feature scaling as the cause. That
   is now recorded as a **solver-conditioning hypothesis**: 152/250 raw
   SVM fits failed to converge versus 0/250 standardized fits. Scaling
   is a supported contributor, but standardization also changes margin geometry
   and libsvm working-set selection, so causation is **not** established and no
   counterfactual was run.
4. **Predictions vs convergence.** 750 primary predictions were produced. That
   remains true and is preserved. How many SVM fits converged is a separate
   question (ARCENE 48/50, DEXTER 50/50, MADELON 0/50, GINA 0/50, GISETTE 0/50). A non-converged fit still returns predictions but does
   not return a converged optimum; it must not be relabelled as converged.
5. **Freeze chronology.** The declared `frozen_utc: 2026-10-11T12:00:00Z`
   postdates every Stage 3A artifact. Status: **CONTRADICTED**. No date was
   altered, no earlier time invented, and no pre-registration is claimed
   retroactively.
6. **Path.** Stage 3A wrote run records under
   `projects/ossfs-dd-reproduction/results/stage3a/runs/records/`, not under
   `$RESEARCH_OUTPUT_ROOT/ossfs-dd-reproduction/stage3a/`. Audited at the real
   location; no path was invented.
7. **Provenance gaps.** `configs/paper_reference_discrete.csv` is not
   git-tracked (`.gitignore` line 40, `*.csv`), and `run_manifest.json` records
   no execution timestamp.
8. **A hidden destructive test.** `tests/test_evidence_fail_closed.py` wrote
   synthetic payloads into the **real** `results/stage1/test_results.json` and
   deleted the file in a `finally` block. This was masked because the conftest
   hook rewrote the file at the end of every session; once that hook became
   opt-in, a full test run genuinely deleted the committed Stage 1 artifact.
   Those tests now isolate `gates.results_dir` to a temporary directory. The
   earlier `git checkout` workarounds were treating the symptom.

## What the audit confirmed

- Ledger reconciles end to end: 44 checks,
  **0 disagreements** (raw records ↔ per-run CSV ↔
  `aggregate.json` ↔ committed markdown).
- All 250 selections and 1500 classifier outcomes are present with unique IDs,
  bounded accuracies and explicit denominators.
- ARCENE's selected count spans 34-1561
  (mean 963.64), confirmed from raw records.
- No forbidden dataset (`DLBCL`, `BREAST`) was read at any point; the executed
  records contain exactly the five authorized datasets.
- Stage 1 and Stage 2 working trees are unchanged.

## What remains open

Feature-count variability cause (`UNRESOLVED`), raw-SVM nonconvergence cause
(hypothesis only), config freeze chronology (`CONTRADICTED`), and the
authors' continuous Fisher-Z semantics (unrecoverable).

Verdict: **STAGE3A_AUDIT_PASS**
