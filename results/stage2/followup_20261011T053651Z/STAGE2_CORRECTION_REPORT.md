# Stage 2 Correction Report — OSSFS-DD

**Label: CORRECTED_DIAGNOSTIC.** Supersedes the pre-repair Stage 2 smoke; that baseline is preserved under `pre_fix_snapshot.json`.

**Author source:** `doodzhou/OSFS@845a8e9d5625616b43e8cee16e8d22081dbe3f62`

## Verdict

- Discrete-correctness subgate: **DISCRETE_CORRECTNESS_PASS** (10/10 gates)
- Overall Stage 2: **STAGE2_PARTIAL** — governed by the **unchanged** 9/9 gate

> `DISCRETE_CORRECTNESS_PASS` is **not** `STAGE2_READY`. Four paper-`Real` datasets
> still lack verified continuous semantics, so the original 9/9 gate still fails.

## Repairs

| ID | Defect | Confirmed | Primary-mode impact |
|---|---|---|---|
| R1 | `alpha` passed where `_redundancy_pass` needs `beta` | yes | yes — DEXTER removals 25→352 |
| R2 | pool search lacked MATLAB's first-match inner break | yes | yes — ARCENE 12.3s→1.9s |
| R3 | paper_spec warm-up folded the first scores twice | yes | **paper_spec only**; MATLAB unaffected |
| R4 | promotion sorted feature IDs, not arrival positions | yes | yes — changes candidate order |
| R5 | `x or nan` turned a valid alpha of 0.0 into NaN | yes | trace fidelity only |
| R6 | continuous route | unchanged, still fail-closed | none |

Each repair was demonstrated by restoring the committed pre-fix `selector.py` from
git HEAD and re-running the same tests: **9 failed before, all 16 pass after**.

## Corrected five-dataset smoke (seed 11, fold 0, `matlab_compatible_v1`)

| Dataset | Status | n_sel before | n_sel after | Jaccard | direct accepts b/a | redundancy b/a | promotions b/a | secs b/a | RSS GiB |
|---|---|---:|---:|---:|---|---|---|---:|---:|
| ARCENE | COMPLETED | 870 | 863 | 0.991954 | 4/4 | 586/585 | 1452/1444 | 12.261/1.894 | 0.072 |
| DEXTER | COMPLETED | 797 | 470 | 0.589711 | 822/822 | 25/352 | 0/0 | 75.807/49.592 | 0.072 |
| MADELON | COMPLETED | 57 | 57 | 1.0 | 10/10 | 0/0 | 47/47 | 1.605/0.28 | 0.088 |
| GINA | COMPLETED | 116 | 98 | 0.844828 | 55/55 | 7/21 | 68/64 | 4.635/2.127 | 0.114 |
| GISETTE | COMPLETED | 318 | 182 | 0.572327 | 177/177 | 86/187 | 227/189 | 103.034/62.163 | 0.138 |

MADELON is unchanged (Jaccard 1.0): it never triggered a second qualifying pool
pair and never entered a redundancy comparison, so neither R1 nor R2 could apply.
The large DEXTER and GISETTE movements come from R1 (redundancy now uses `2*beta`)
and R2 (promotion no longer over-selects).

## Paper-spec warm-up interpretation

Algorithm 1 gives no decision when only one score exists and `sigma == 0`. The
repaired `paper_spec_v1` folds each score exactly once and then applies the literal
rule, which yields REJECT for the first feature. No epsilon, no lookahead, `alpha < beta`
never relaxed. Versioned as `paper_spec_v1.warmup_bootstrap_r2`; the superseded wording is
retained verbatim in `configs/stage2.yaml`.

## Verification

- Full suite: **135 passed, 0 failed, 0 skipped**
- Stage 1 artifacts: **490 → 490 files, 0 changed**
- Determinism: fresh-process repetitions identical = **True**
- Forbidden datasets read: **0**
- Stage 1 verdict: unchanged `STAGE1_PARTIAL`

## Remaining blocker

`Fisher_Ztest`, called by `OSFS_DD_z2.m`, is absent from the pinned repository
(67 blobs, zero matches). SRBCT, LYMPHOMA, PROSTATE and LEUKEMIA therefore remain
`UNSUPPORTED_UNVERIFIED_SEMANTICS`, and S6/S7/S8 remain `UNRESOLVED`. Resolving this
needs the authors' helper file; I will not contact them without your approval.

MATLAB runtime parity remains `NOT_MEASURED` — `MIToolboxMex` ships only as a Windows
`.mexw64` binary and no MATLAB/Octave exists here.

```
FINAL_VERDICT: STAGE2_PARTIAL | DISCRETE_CORRECTNESS_PASS
```
