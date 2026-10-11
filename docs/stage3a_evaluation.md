# Stage 3A — discrete OSFS-DD evaluation (Workstream A)

Generated 2026-10-11T08:17:33Z · protocol `31daed221a4bc0f3` · code `e8bd699cf609cbdf` · verdict **STAGE3A_READY**

## Scope

Five `EXACT_VERIFIED` discrete datasets x 10 seeds x 5 folds = **250 selections**, each evaluated by three fixed classifiers = **750 classifier outcomes** (raw primary). `DLBCL` and `BREAST` were never read; the four continuous datasets belong to Workstream B and are not read here.

Each seed's **single** `arrival_order` is re-used across that seed's five folds, so the five folds of a seed are **not** five independent feature-order permutations.

## Primary results (raw features, no scaling)

| Dataset | Classifier | Mean acc | SD over seeds | Published | Diff | Mean feats | Published feats | SVM converged |
|---|---|---|---|---|---|---|---|---|
| ARCENE | KNN (k=5) | 0.8090 | 0.0330 | 0.7638 | 0.0452 | 963.6 | 33.6 |  |
| ARCENE | SVM (linear, C=1) | 0.8475 | 0.0270 | 0.7513 | 0.0962 | 963.6 | 33.6 | 48/50 |
| ARCENE | CART (Gini) | 0.7480 | 0.0384 | 0.7012 | 0.0468 | 963.6 | 33.6 |  |
| DEXTER | KNN (k=5) | 0.7230 | 0.0142 | 0.8225 | -0.0995 | 460.2 | 56.7 |  |
| DEXTER | SVM (linear, C=1) | 0.8943 | 0.0075 | 0.8933 | 0.0010 | 460.2 | 56.7 | 50/50 |
| DEXTER | CART (Gini) | 0.8348 | 0.0150 | 0.8358 | -0.0010 | 460.2 | 56.7 |  |
| MADELON | KNN (k=5) | 0.7803 | 0.0104 | 0.8063 | -0.0260 | 56.1 | 15.3 |  |
| MADELON | SVM (linear, C=1) | 0.5134 | 0.0240 | 0.6120 | -0.0986 | 56.1 | 15.3 | 0/50 |
| MADELON | CART (Gini) | 0.7670 | 0.0083 | 0.7799 | -0.0129 | 56.1 | 15.3 |  |
| GINA | KNN (k=5) | 0.8792 | 0.0096 | 0.8573 | 0.0219 | 91.4 | 25.2 |  |
| GINA | SVM (linear, C=1) | 0.4482 | 0.0557 | 0.8286 | -0.3804 | 91.4 | 25.2 | 0/50 |
| GINA | CART (Gini) | 0.8247 | 0.0099 | 0.8178 | 0.0069 | 91.4 | 25.2 |  |
| GISETTE | KNN (k=5) | 0.9596 | 0.0018 | 0.8628 | 0.0968 | 175.6 | 120.8 |  |
| GISETTE | SVM (linear, C=1) | 0.8509 | 0.0216 | 0.8995 | -0.0486 | 175.6 | 120.8 | 0/50 |
| GISETTE | CART (Gini) | 0.9314 | 0.0029 | 0.9287 | 0.0027 | 175.6 | 120.8 |  |

Accuracy is the mean over seeds of that seed's five-fold mean. Published values are from DOI 10.1145/3502737, Tables 3-7, and are **descriptive context only**; no significance test against a published mean was performed. The `SVM converged` column records how many of the 50 raw SVM fits for that dataset reached a converged solution; see the convergence section before comparing SVM rows.

## Selected features

| Dataset | Total | Mean selected | Fraction | Mean selection s | Peak RSS GiB |
|---|---|---|---|---|---|
| ARCENE | 10000 | 963.6 | 0.096 | 2.03 | 0.152 |
| DEXTER | 20000 | 460.2 | 0.023 | 49.48 | 0.160 |
| MADELON | 500 | 56.1 | 0.112 | 0.30 | 0.164 |
| GINA | 970 | 91.4 | 0.094 | 1.99 | 0.197 |
| GISETTE | 5000 | 175.6 | 0.035 | 63.21 | 0.326 |

## A convergence limitation in the primary SVM

The raw-feature linear SVC **fails to converge** on most datasets. liblinear hit the declared iteration cap (`max_iter=2000000`, liblinear's own documented default, made explicit so the run stays inside the declared budget):

| Dataset | Raw SVM fits not converged | Of | Interpretable raw SVM? |
|---|---|---|---|
| ARCENE | 2 | 50 | partially |
| DEXTER | 0 | 50 | yes |
| MADELON | 50 | 50 | **no** |
| GINA | 50 | 50 | **no** |
| GISETTE | 50 | 50 | **no** |

In total **152** raw-primary SVM fits did not converge, versus **0** standardized fits. The cause is feature scale: MADELON's columns differ by ~224x and GINA/GISETTE are similarly badly scaled, which makes liblinear's coordinate descent crawl.

**Consequence for reading the table above:** the raw SVM accuracy for ARCENE, GINA, GISETTE, MADELON is the output of a solver that stopped at its iteration cap, so it is **not a converged optimum** and must not be read as an algorithm result. It is reported because the run was not re-tuned, but the standardized column for those datasets (ARCENE, GINA, GISETTE, MADELON) is the interpretable figure. Only DEXTER has a fully converged raw SVM fit.
MATLAB's `fitcsvm` documents `'Standardize'` with default **false**, so raw features remain the faithful primary path and were not silently replaced. The standardized analysis is reported separately as a labelled sensitivity analysis.

## Standardized sensitivity (labelled, not primary)

| Dataset | Classifier | Mean acc | SD over seeds |
|---|---|---|---|
| ARCENE | KNN (k=5) | 0.8190 | 0.0306 |
| ARCENE | SVM (linear, C=1) | 0.8495 | 0.0201 |
| ARCENE | CART (Gini) | 0.7480 | 0.0384 |
| DEXTER | KNN (k=5) | 0.7550 | 0.0117 |
| DEXTER | SVM (linear, C=1) | 0.8902 | 0.0089 |
| DEXTER | CART (Gini) | 0.8350 | 0.0151 |
| MADELON | KNN (k=5) | 0.6379 | 0.0110 |
| MADELON | SVM (linear, C=1) | 0.6035 | 0.0035 |
| MADELON | CART (Gini) | 0.7675 | 0.0081 |
| GINA | KNN (k=5) | 0.8764 | 0.0078 |
| GINA | SVM (linear, C=1) | 0.8427 | 0.0043 |
| GINA | CART (Gini) | 0.8249 | 0.0099 |
| GISETTE | KNN (k=5) | 0.9605 | 0.0010 |
| GISETTE | SVM (linear, C=1) | 0.9574 | 0.0015 |
| GISETTE | CART (Gini) | 0.9313 | 0.0028 |

## Descriptive spread

Seed-blocked bootstrap over the 10 per-seed means (10,000 resamples) is reported in `results/stage3a/aggregate.json`. Repeated cross-validation over one dataset is not an independent population, so these intervals are **not** generalisation confidence intervals.

## Gates

| Gate | Result |
|---|---|
| exact_allowlist_five | PASS (required=5, observed=5, matches_frozen_allowlist=True, forbidden_reads=0) |
| planned_selector_evaluations | PASS (required=250, observed=250) |
| planned_classifier_outcomes | PASS (required=750, observed_primary=750, observed_standardized_sensitivity=750) |
| all_selections_terminal | PASS () |
| accuracy_bounded_and_denominators_present | PASS (n_checked=750) |
| resource_limits_respected | PASS (memory_limit_gib=32, peak_rss_gib_max=0.326, per_attempt_max_minutes=60, max_single_attempt_minutes=1.1801) |

## What this stage does not establish

- It does not reproduce the authors' continuous route; Stage 2's `UNRESOLVED` verdict for those datasets is unchanged (see the Workstream-B report).

- It makes no runtime claim. The published timings used different hardware and MATLAB, and are shown only as context.

- Published accuracies are point estimates from another implementation; the absolute differences above are descriptive, not tested.

## Figures

- `results/stage3a/figures/accuracy_vs_paper_outcomes_raw_primary.png`
- `results/stage3a/figures/accuracy_vs_paper_outcomes_standardized_sensitivity.png`
- `results/stage3a/figures/feature_counts.png`
- `results/stage3a/figures/per_seed_spread.png`
- `results/stage3a/figures/runtime_context.png`

