# Stage 3A — summary and verdict

**Verdict: `STAGE3A_READY`** (Workstream A) · **Workstream B: `CONTINUOUS_RND_DIAGNOSTIC_READY`**

This stage does not change Stage 1 or Stage 2. `configs/stage1.yaml`,
`configs/stage2.yaml`, the Stage-1 data tree and the Stage-1/2 reports are
byte-identical to their committed state, and the Stage-2 verdict
(`STAGE2_PARTIAL | DISCRETE_CORRECTNESS_PASS`) stands, including its
`UNRESOLVED` continuous semantics.

## What was run

### Workstream A — discrete evaluation

5 `EXACT_VERIFIED` discrete datasets x 10 seeds x 5 folds = **250 selections**,
each evaluated by three fixed classifiers = **750 classifier outcomes**
(raw primary). Plus an equally-sized labelled **standardized sensitivity
analysis** computed from the same selections.

| Artefact | Path |
|---|---|
| Full report | `docs/stage3a_evaluation.md` |
| Per-run table | `results/stage3a/per_run_outcomes.csv` |
| Aggregate + bootstrap | `results/stage3a/aggregate.json` |
| Paper comparison | `results/stage3a/paper_comparison_*.json` |
| Gate manifest | `results/stage3a/report_manifest.json` |
| Figures | `results/stage3a/figures/` (git-ignored by repo policy) |
| Raw run records | `results/stage3a/runs/records/` (250 files, git-ignored) |
| Frozen config | `configs/stage3a.yaml` |
| Published reference ledger | `configs/paper_reference_discrete.csv` |

### Workstream B — reconstructed continuous estimator

Four `Real` datasets, **seed 11 / fold 0 only, training rows only**, no
classifier and no accuracy. See `docs/continuous_rnd_workstream.md`.

## Gates

| Gate | Result |
|---|---|
| Exactly the five frozen datasets, zero forbidden reads | PASS |
| 250 selector evaluations | PASS |
| 750 classifier outcomes | PASS |
| All selections terminal | PASS |
| Accuracy bounded, denominators present | PASS |
| Resource limits respected (peak RSS 0.33 GiB of 32) | PASS |
| Full test suite (197 passed, 0 failed, 0 skipped) | PASS |
| Ruff clean | PASS |
| Determinism repeats (MADELON, GISETTE identical) | PASS |

## Findings worth reading

1. **DEXTER SVM and CART reproduce the published numbers closely** (0.8943 vs
   0.8933; 0.8348 vs 0.8358) — and DEXTER is also the one dataset whose raw SVM
   fully converges, so that agreement is on a sound footing.

2. **The raw linear SVC does not converge on 3 of 5 datasets** (MADELON, GINA,
   GISETTE: 50/50 fits each; ARCENE 2/50). The cause is feature scale, not the
   selector. These fits are flagged `converged=false`, were never re-tuned, and
   are explicitly marked in the results table. **The raw SVM numbers for those
   three datasets are not converged optima and should not be read as results.**
   The standardized column is the interpretable figure there.

3. **Feature counts are far above the published ones** (e.g. ARCENE 964 vs 33.6,
   GISETTE 176 vs 120.8). The selector keeps a far larger subset than the paper
   reports; this is a faithful consequence of the released `mi2.m` semantics,
   not of the evaluation harness.

4. **MADELON and GISETTE fold-0 selections reproduce the Stage-2 diagnostics
   exactly** (57 and 182 features) from an independent recomputation, which
   cross-validates Workstream A against Stage 2.

5. **No joint-state memory optimisation was needed.** Per-column cardinality in
   Stage-1 data is small (max 960 distinct values dataset-wide), so the widest
   three-way count stays far inside the 32 GiB budget; observed peak RSS was
   0.33 GiB. This was verified with brute-force oracles in
   `tests/test_stage3a_numerics.py` rather than asserted.

## What this stage does NOT establish

- It does **not** reproduce the authors' continuous route. Workstream B is a
  declared reconstruction, labelled `RECONSTRUCTED_NOT_SOURCE_VERIFIED`.
- It makes **no** runtime claim: the published timings used different hardware
  and MATLAB, and are shown only as context.
- It runs **no** significance test against published means; a published point
  estimate is not a sample. Differences in the tables are descriptive.
- The bootstrap intervals are **descriptive**, not generalisation confidence
  intervals: repeated CV over one dataset is not an independent population.

## Reproducibility notes

- The sweep is resumable and content-addressed: a record is reused only when the
  frozen protocol digest and source code hash both match, so changing code
  invalidates cached results instead of silently mixing generations.
- `tests/conftest.py` rewrites `results/stage1/test_results.json` on **any**
  pytest run, counting every test in the suite. After running the suite the
  Stage-1 file was restored with `git checkout`, so the Stage-1 results tree is
  byte-identical to its committed state. Be aware of this if you re-run tests.
- Re-run everything with:

  ```bash
  uv run python scripts/stage3a_run.py --workers 4
  uv run python scripts/stage3a_report.py
  uv run python scripts/continuous_rnd_run.py
  uv run python scripts/continuous_rnd_report.py
  ```