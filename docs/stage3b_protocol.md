# Stage 3B — protocol, designs and causal limits

Two controlled, separable experiments. Neither alters Stage 1/2/3A data,
configs, selectors or results.

## Experiment A — SVM conditioning

Reuses the **already selected** Stage 3A feature subsets verbatim. No feature
selection is performed, no subset is recomputed, and no CSV average is ever
turned into a feature-index list.

- **Subsets:** 5 discrete datasets x seeds {11,17,23} x 5 folds = **75**,
  taken from `results/stage3a/runs/records/` and validated (unique IDs,
  in-range, `n_selected` consistent, `protocol_digest` matching).
- **Factors:** scaling {raw, standardized} x `max_iter` {20 000, 200 000,
  2 000 000} = **6** fits per subset = **450** planned fits.
- **Standardized** means `StandardScaler` fitted on the **selected training
  rows only**; test rows are transformed with that scaler and never influence
  it. **Raw** is the identity.
- `SVC(kernel='linear', C=1.0, max_iter=cap)`; every other parameter is an
  installed-version default recorded verbatim per fit. `LinearSVC` is not a
  drop-in substitute and is never used. `max_iter=-1` is excluded because it
  gives no deterministic bound.
- Each fit runs in a killable worker with a **10-minute** hard timeout.

### Hypotheses (fixed before execution)

- **A-H1** — at `max_iter=2 000 000`, standardized convergence exceeds raw by
  >= 0.50 **and** standardized >= 0.80, per dataset on MADELON/GINA/GISETTE.
- **A-H2** — with scaling fixed, convergence at 2 000 000 is no lower than at
  20 000, giving **10** prespecified comparisons (5 datasets x 2 scalings).

Denominators are never reduced: a convergence rate exists only when all 15
fits of a cell reached a terminal observed status.

## Experiment B — ARCENE feature-arrival order

The **CV partition is fixed** to Stage 1 `seed_11.json`. Only the arrival
permutation varies. `order_seed` selects a permutation and never selects a
split.

- **Design:** 5 folds x 10 order seeds {11,17,23,31,47,59,71,83,97,109} = a
  full factorial of **50** runs.
- **Anchor:** `order_seed=11` at each fold must reproduce the existing Stage 3A
  ARCENE `(seed=11, fold)` records exactly — a comparison of existing evidence
  with freshly executed selection, not copied results.
- Traces record every arrival plus pool-flush events, with checkpoints at
  100, 200, ... and independent `ddof=1` threshold recomputation.

### Hypotheses

- **B-H1** — within-fold `range(n_selected)` across the 10 orders reaches
  >= 500 on at least 3 of 5 folds. This is an **effect criterion**, not an
  operational success gate.
- **B-H2** — investigate whether count differences co-occur with
  `redundancy_removals`, `promoted_from_pool`, direct accepts or threshold
  trajectories. **No mechanism is preregistered.**

## Causal limitations (read before interpreting anything)

1. **Scaling is an intervention, not an isolated cause.** Standardization also
   changes margin geometry, conditioning and the solver's working-set
   selection. A-H1 establishes association under a controlled intervention; it
   does **not** prove feature scale is the physical cause of non-convergence.
2. **Repeated CV cells are not independent samples.** The folds overlap in
   training rows and each order is reused across folds, so the 50 cells are
   **not** 50 independent population samples. Experiment B identifies
   within-split sensitivity to order, not a generalization estimate.
3. **Counters are arithmetically coupled.** `n_selected` is essentially
   `direct_accepts + promoted_from_pool - redundancy_removals`, so correlations
   between those counters and the count are partly entailed by arithmetic and
   do not establish a mechanism.
4. **The ARCENE folds are one dataset.** No population-level inference is
   claimed, and no inferential p-value is reported.
5. **Published values are context only.** The paper's ARCENE Table 7 value
   (~33.65 features) is neither a target nor a comparator, and no MATLAB
   runtime parity is claimed.

## Chronology

`configs/stage3b.yaml` is frozen and hashed **before** the first fit or
selection, guarded write-once, with a UTC stamp obtained at runtime. This is an
**in-process, hashed, chronologically ordered freeze** — there is no independent
timestamp attestation, so it is **not** a preregistration. Stage 3A's
contradictory freeze timestamp is preserved unchanged and is never retroactively
repaired.

## Provenance note

Stage 3A's per-run `code_hash` is **not** authoritative: it was recomputed per
task from a source tree being edited mid-sweep, so the 250 records carry 11
distinct values. Stage 3B verifies `protocol_digest` (stable) plus a
selection-code fingerprint instead.
