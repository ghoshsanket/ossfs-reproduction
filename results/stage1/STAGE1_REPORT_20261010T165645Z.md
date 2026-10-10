# Stage 1 Report — Dataset Acquisition, Provenance and Validation

**Project:** `ossfs-dd-reproduction`  
**Generated (UTC):** 2026-10-10T05:20:50Z  
**Stage:** 1 of 3 — **DATA ONLY**  
**Paper:** Zhou, Zhao, Yan & Wu, *Online Scalable Streaming Feature Selection via Dynamic Decision*, ACM TKDD 16(5) art. 87 (2022), DOI 10.1145/3502737

> No feature selection, no classifier, no accuracy measurement was performed. Stage 1 prepares and audits data only.

## Final verdict

```
STAGE_1_VERDICT: STAGE1_PARTIAL
```

Failed hard gates: exact_verified_provenance

## Preregistered gates

Thresholds come from `configs/stage1.yaml` and were fixed before any download. None was lowered to force a pass.

| Gate | Threshold | Observed | Result | Detail |
|---|---|---|---|---|
| `inventory_attempted` | 11 | 11 | **PASS** | 11/11 registry datasets attempted |
| `exact_dimension_and_label_match` | 11 | 11 | **PASS** | 11/11 match Table 1 (n, p, classes) with aligned, reversible labels; 2 of those have unproven lineage |
| `exact_verified_provenance` | 11 | 9 | **FAIL** | 9/11 EXACT_VERIFIED; 2 shape-matched with lineage recorded as unproven |
| `sample_feature_label_corruption` | 0 | 0 | **PASS** | round-trip, label-alignment and plan-structure checks passed for every dataset |
| `unresolved_nonfinite_or_missing_in_ready_data` | 0 | 0 | **PASS** | 0 non-finite/missing values across ready datasets (0 across all audited datasets); Stage 1 never imputes, so any unresolved value would mark its dataset DATA_QUALITY_BLOCKED instead |
| `unlabeled_challenge_test_examples_used` | 0 | 0 | **PASS** | parsers refuse a 'test' split outright; asserted by test_no_unlabeled_test_data |
| `cv_test_folds_defined` | 550 | 550 | **PASS** | 110 split-plan files covering 550 test folds (11 datasets x 10 seeds x 5 folds = 550) |
| `feature_order_permutations_defined` | 110 | 110 | **PASS** | 110 feature-order permutations (11 datasets x 10 seeds = 110) |
| `stream_leakage_tests_passing` | 1.0 | 22 passed, 0 failed | **PASS** | stream-leakage test failures: 0 of 4 leakage tests |
| `data_or_result_binaries_in_git` | 0 | 0 | **PASS** | git ignore audit from test_git_excludes_data |
| `required_tests_passing` | 1.0 | 22 passed, 0 failed | **PASS** | every required named test ran and passed |

## Dataset status: paper Table 1 vs observed

`features` counts predictor columns only; the class column is never counted as a feature.

| # | Dataset | paper n×p×c | observed n×p×c | source tier | SHA-256 (first 16) | source splits | status |
|---:|---|---|---|---|---|---|---|
| 1 | SRBCT | 63×2308×4 | 63×2308×4 | paper_cited | `d5bb3895e99888bc` | `{"original_labeled": 63}` | **EXACT_VERIFIED** |
| 2 | LYMPHOMA | 62×4026×3 | 62×4026×3 | paper_cited | `b10aa781ff961d9b` | `{"original_labeled": 62}` | **EXACT_VERIFIED** |
| 3 | PROSTATE | 102×6033×2 | 102×6033×2 | paper_cited | `711cef5a04110859` | `{"original_labeled": 102}` | **EXACT_VERIFIED** |
| 4 | LEUKEMIA | 72×7129×2 | 72×7129×2 | paper_cited | `0fd632f7469ffbbe` | `{"original_labeled": 72}` | **EXACT_VERIFIED** |
| 5 | DLBCL | 77×7129×2 | 77×7129×2 | academic_repository | `30bc95073e27e05a` | `{"original_labeled": 77}` | **SHAPE_MATCH_PROVENANCE_UNVERIFIED** |
| 6 | ARCENE | 200×10000×2 | 200×10000×2 | official | `c7a40e39edf23a9b` | `{"train": 100, "valid": 100}` | **EXACT_VERIFIED** |
| 7 | DEXTER | 600×20000×2 | 600×20000×2 | official | `70400df292178d8d` | `{"train": 300, "valid": 300}` | **EXACT_VERIFIED** |
| 8 | BREAST | 97×24481×2 | 97×24481×2 | academic_repository | `9a4971e3bc260ff1` | `{"original_labeled": 97}` | **SHAPE_MATCH_PROVENANCE_UNVERIFIED** |
| 9 | MADELON | 2600×500×2 | 2600×500×2 | official | `eb99a0153ebe7d9f` | `{"train": 2000, "valid": 600}` | **EXACT_VERIFIED** |
| 10 | GINA | 3468×970×2 | 3468×970×2 | official | `d4b95d97b031c4d3` | `{"original_labeled": 3468}` | **EXACT_VERIFIED** |
| 11 | GISETTE | 7000×5000×2 | 7000×5000×2 | official | `430bb631b3e49465` | `{"train": 6000, "valid": 1000}` | **EXACT_VERIFIED** |

## Per-dataset detail, provenance and data quality

### 1. SRBCT — `EXACT_VERIFIED`

- **Expected (Table 1):** 63 samples × 2308 features × 4 classes; paper feature type `Real`
- **Observed:** n=63, p=2308, c=4
- **Upstream origin:** <https://www.cs.binghamton.edu/~lyu/KDD08/data/srbct-std.arff> (tier `paper_cited`)
- **Raw archive:** `dataset/ossfs-dd-reproduction/raw/SRBCT/srbct-std.arff` — SHA-256 `d5bb3895e99888bc52c3e31d747bd2d422fe99d0b0dead771229ac2b84daa7cb`, 1434331 bytes
- **Provenance:** tier `paper_cited`, confidence `high`
- **Citation:** Khan J. et al., Cancer Res 61(16):6469-6476 (2001); NCI-60 small round blue cell tumour panel.
- **Licence:** license_unknown — academic research redistribution, no licence published on the cited page
- **Class counts:** `{"0": 23, "1": 20, "2": 12, "3": 8}`
- **Source split composition:** `{"original_labeled": 63}`
- **Non-finite values:** total=0 (NaN=0, +Inf=0, -Inf=0)
- **Constant columns:** 0; **duplicate column groups:** 0; **duplicate row groups:** 0; **zero-only rows:** 0
- **Value range:** [-3.310311, 7.709031] (all values integral: False); stored as `numpy_dense` with dtype `float64`, density 1.0, zero columns 0
- **Preexisting preprocessing in the source:** Binghamton '-std' release: values arrive standardised by the source collection. Documented as preexisting, NOT applied here.
- **Transformations applied by this pipeline:** none
- **Status reason:** dimensions (63, 2308, 4) match Table 1, labels align and are reversible, no non-finite values, and lineage is substantiated by a paper_cited source
- **Alternatives considered and rejected:**
  - *mixOmics `srbct` (63 samples) and other SRBCT variants with 83 samples* — Not needed: the paper-cited source already matches Table 1 exactly. The 83-sample variant is explicitly out of scope (would add 20 test samples not present in the paper).


### 2. LYMPHOMA — `EXACT_VERIFIED`

- **Expected (Table 1):** 62 samples × 4026 features × 3 classes; paper feature type `Real`
- **Observed:** n=62, p=4026, c=3
- **Upstream origin:** <https://www.cs.binghamton.edu/~lyu/KDD08/data/lymphoma-std.arff> (tier `paper_cited`)
- **Raw archive:** `dataset/ossfs-dd-reproduction/raw/LYMPHOMA/lymphoma-std.arff` — SHA-256 `b10aa781ff961d9b8dbd00b8e0e94f911caffa2c0f3a461373912ef0d76b9e07`, 2445998 bytes
- **Provenance:** tier `paper_cited`, confidence `high`
- **Citation:** Ahn C. et al., Nat Genet 21(3):323-325 (1999) — NCI-60 lymphoma expression data.
- **Licence:** license_unknown — academic research redistribution, no licence published on the cited page
- **Class counts:** `{"0": 42, "1": 9, "2": 11}`
- **Source split composition:** `{"original_labeled": 62}`
- **Non-finite values:** total=0 (NaN=0, +Inf=0, -Inf=0)
- **Constant columns:** 0; **duplicate column groups:** 0; **duplicate row groups:** 0; **zero-only rows:** 0
- **Value range:** [-6.396329, 6.804321] (all values integral: False); stored as `numpy_dense` with dtype `float64`, density 1.0, zero columns 0
- **Preexisting preprocessing in the source:** Binghamton '-std' release: values arrive standardised by the source collection. Documented as preexisting, NOT applied here.
- **Transformations applied by this pipeline:** none
- **Status reason:** dimensions (62, 4026, 3) match Table 1, labels align and are reversible, no non-finite values, and lineage is substantiated by a paper_cited source
- **Alternatives considered and rejected:**
  - *spls R package `lymphoma` (documented 62 x 4026)* — Superseded: identical dimensions, but the paper-cited source ranks higher on lineage.
  - *lymphoma 96 x 4026 (scikit-feature `lymphoma`)* — 96 samples, not 62. Wrong variant.
  - *lymphoma 62 x 4682* — Wrong feature count. Wrong variant.

### 3. PROSTATE — `EXACT_VERIFIED`

- **Expected (Table 1):** 102 samples × 6033 features × 2 classes; paper feature type `Real`
- **Observed:** n=102, p=6033, c=2
- **Upstream origin:** <https://www.cs.binghamton.edu/~lyu/KDD08/data/prostate-std.arff> (tier `paper_cited`)
- **Raw archive:** `dataset/ossfs-dd-reproduction/raw/PROSTATE/prostate-std.arff` — SHA-256 `711cef5a04110859850e95bcd0a033b6da5779190604f598956ece29ee342de6`, 5964489 bytes
- **Provenance:** tier `paper_cited`, confidence `high`
- **Citation:** Singh D. et al., Cancer Res 62(24):7173-7180 (2002) — prostate tumour vs normal expression.
- **Licence:** license_unknown — academic research redistribution, no licence published on the cited page
- **Class counts:** `{"0": 50, "1": 52}`
- **Source split composition:** `{"original_labeled": 102}`
- **Non-finite values:** total=0 (NaN=0, +Inf=0, -Inf=0)
- **Constant columns:** 0; **duplicate column groups:** 0; **duplicate row groups:** 0; **zero-only rows:** 0
- **Value range:** [-7.548211, 9.556945] (all values integral: False); stored as `numpy_dense` with dtype `float64`, density 1.0, zero columns 0
- **Preexisting preprocessing in the source:** Binghamton '-std' release: values arrive standardised by the source collection. Documented as preexisting, NOT applied here.
- **Transformations applied by this pipeline:** none
- **Status reason:** dimensions (102, 6033, 2) match Table 1, labels align and are reversible, no non-finite values, and lineage is substantiated by a paper_cited source
- **Alternatives considered and rejected:**
  - *hastie.su.domains CASI `prostmat.csv` (6033 genes x 102 samples, gene-major)* — Retained as a corroborating lead only. It requires a justified transpose and carries its labels in separate files; the paper-cited source already yields the correct orientation directly.

  - *prostate 102 x 5966 (scikit-feature `Prostate_GE`)* — 5966 features, not 6033. Wrong variant.
  - *prostate 102 x 12600* — Wrong feature count. Wrong variant.

### 4. LEUKEMIA — `EXACT_VERIFIED`

- **Expected (Table 1):** 72 samples × 7129 features × 2 classes; paper feature type `Real`
- **Observed:** n=72, p=7129, c=2
- **Upstream origin:** <https://www.cs.binghamton.edu/~lyu/KDD08/data/leukemia-std.arff> (tier `paper_cited`)
- **Raw archive:** `dataset/ossfs-dd-reproduction/raw/LEUKEMIA/leukemia-std.arff` — SHA-256 `0fd632f7469ffbbe10e6e955f6a29144cd096eb03b71c7f81a959a604356df81`, 5020791 bytes
- **Provenance:** tier `paper_cited`, confidence `high`
- **Citation:** Golub T.R. et al., Science 286(5440):531-537 (1999) — ALL vs AML gene expression classification.
- **Licence:** license_unknown — academic research redistribution, no licence published on the cited page
- **Class counts:** `{"ALL": 47, "AML": 25}`
- **Source split composition:** `{"original_labeled": 72}`
- **Non-finite values:** total=0 (NaN=0, +Inf=0, -Inf=0)
- **Constant columns:** 0; **duplicate column groups:** 0; **duplicate row groups:** 0; **zero-only rows:** 0
- **Value range:** [-8.329711, 8.366156] (all values integral: False); stored as `numpy_dense` with dtype `float64`, density 1.0, zero columns 0
- **Preexisting preprocessing in the source:** Binghamton '-std' release: values arrive standardised by the source collection. Documented as preexisting, NOT applied here.
- **Transformations applied by this pipeline:** none
- **Status reason:** dimensions (72, 7129, 2) match Table 1, labels align and are reversible, no non-finite values, and lineage is substantiated by a paper_cited source
- **Alternatives considered and rejected:**
  - *scikit-feature `ALLAML` (72 x 7129)* — Same dimensions, but a lead rather than the paper's own cited source; the paper-cited source ranks higher on lineage.

  - *scikit-feature `leukemia` (72 x 7070)* — 7070 features. Wrong variant — explicitly warned about in the Stage 1 prompt.
  - *releases with 7128 genes* — Wrong feature count. Wrong variant.

### 5. DLBCL — `SHAPE_MATCH_PROVENANCE_UNVERIFIED`

- **Expected (Table 1):** 77 samples × 7129 features × 2 classes; paper feature type `Integer`
- **Observed:** n=77, p=7129, c=2
- **Upstream origin:** <https://mldata-static.ml.tu-berlin.de/repository/data/download/csv/dlbcl-tumor-from-harvard/index.html> (tier `academic_repository`)
- **Raw archive:** `dataset/ossfs-dd-reproduction/raw/DLBCL/dlbcl-tumor-from-harvard.csv` — SHA-256 `30bc95073e27e05a9aa6e99b46b173e5636c7b47c1bbdf9e6f8f8ee82f0e8e00`, 2276269 bytes
- **Provenance:** tier `academic_repository`, confidence `medium`
- **Citation:** Shipp M.A. et al., Nature Medicine 8(1):68-74 (2002) — DLBCL outcome prediction by gene-expression profiling.
- **Licence:** license_unknown — repository states 'License: unknown (from UCI repository)'
- **Class counts:** `{"DLBCL": 58, "FL": 19}`
- **Source split composition:** `{"original_labeled": 77}`
- **Non-finite values:** total=0 (NaN=0, +Inf=0, -Inf=0)
- **Constant columns:** 0; **duplicate column groups:** 0; **duplicate row groups:** 0; **zero-only rows:** 0
- **Value range:** [-95892.0, 84063.0] (all values integral: True); stored as `numpy_dense` with dtype `float64`, density 1.0, zero columns 6
- **Preexisting preprocessing in the source:** none documented
- **Transformations applied by this pipeline:** none
- **Status reason:** dimensions (77, 7129, 2) and labels match Table 1, but the variant came from an academic_repository record and the paper does not state which release it consumed, so exact scientific equivalence is not proven
- **Alternatives considered and rejected:**
  - *OpenML DLBCL did=45088 (77 instances, 5469 numeric features)* — Sample count matches but the feature count is 5469, not 7129. This is exactly the reprocessed DLBCL variant the Stage 1 prompt says to reject.

  - *TU Berlin `dlbcl-outcome-from-harvard` (58 instances, 7129 attributes)* — 58 instances, not 77 — this is the cured-vs-fatal subset, a different classification problem.
  - *TU Berlin `dlbcl-nih` (240 x 7399)* — Different cohort, different sample and feature counts.
  - *ico2s.org DLBCL (7129 genes x 77 samples, per its index text)* — Dead link — the file is named in the page text but no href exists in the served HTML and the conventional paths return HTTP 404.

### 6. ARCENE — `EXACT_VERIFIED`

- **Expected (Table 1):** 200 samples × 10000 features × 2 classes; paper feature type `Integer`
- **Observed:** n=200, p=10000, c=2
- **Upstream origin:** <https://archive.ics.uci.edu/static/public/167/arcene.zip> (tier `official`)
- **Raw archive:** `dataset/ossfs-dd-reproduction/raw/ARCENE/arcene.zip` — SHA-256 `c7a40e39edf23a9b744eb570592d6e917fd9a1976aaf20ad581e09afd47d5220`, 8648477 bytes
- **Provenance:** tier `official`, confidence `high`
- **Citation:** NIPS 2003 Feature Selection Challenge; Bottou L. et al.
- **Licence:** CC BY 4.0 as published by the UCI Machine Learning Repository
- **Class counts:** `{"-1": 112, "1": 88}`
- **Source split composition:** `{"train": 100, "valid": 100}`
- **Non-finite values:** total=0 (NaN=0, +Inf=0, -Inf=0)
- **Constant columns:** 39; **duplicate column groups:** 46; **duplicate row groups:** 0; **zero-only rows:** 0
- **Value range:** [0.0, 999.0] (all values integral: True); stored as `numpy_dense` with dtype `float64`, density 1.0, zero columns 39
- **Independent source checksum verification** (NIPS `.param` published `Check_sum`):
  - `train`: published `70726744.0`, observed `70726744.0` — **MATCH**; class balance published {'pos': 44, 'neg': 56} vs observed {'pos': 44, 'neg': 56}
  - `valid`: published `71410108.0`, observed `71410108.0` — **MATCH**; class balance published {'pos': 44, 'neg': 56} vs observed {'pos': 44, 'neg': 56}
- **Preexisting preprocessing in the source:** none documented
- **Transformations applied by this pipeline:** none
- **Status reason:** dimensions (200, 10000, 2) match Table 1, labels align and are reversible, no non-finite values, and lineage is substantiated by a official source

### 7. DEXTER — `EXACT_VERIFIED`

- **Expected (Table 1):** 600 samples × 20000 features × 2 classes; paper feature type `Integer`
- **Observed:** n=600, p=20000, c=2
- **Upstream origin:** <https://archive.ics.uci.edu/static/public/168/dexter.zip> (tier `official`)
- **Raw archive:** `dataset/ossfs-dd-reproduction/raw/DEXTER/dexter.zip` — SHA-256 `70400df292178d8d3627411a5631a4cb13fb6b19da4a200b9a206f25255cddcc`, 1323575 bytes
- **Provenance:** tier `official`, confidence `high`
- **Citation:** NIPS 2003 Feature Selection Challenge; Bottou L. et al.
- **Licence:** CC BY 4.0 as published by the UCI Machine Learning Repository
- **Class counts:** `{"-1": 300, "1": 300}`
- **Source split composition:** `{"train": 300, "valid": 300}`
- **Non-finite values:** total=0 (NaN=0, +Inf=0, -Inf=0)
- **Constant columns:** 8965; **duplicate column groups:** 9263; **duplicate row groups:** 0; **zero-only rows:** 0
- **Value range:** [1.0, 943.0] (all values integral: True); stored as `scipy_csr` with dtype `float64`, density 0.0046805833333333335, zero columns 8965
- **Independent source checksum verification** (NIPS `.param` published `Check_sum`):
  - `train`: published `2816528.0`, observed `2816528.0` — **MATCH**; class balance published {'pos': 150, 'neg': 150} vs observed {'pos': 150, 'neg': 150}
  - `valid`: published `2820952.0`, observed `2820952.0` — **MATCH**; class balance published {'pos': 150, 'neg': 150} vs observed {'pos': 150, 'neg': 150}
- **Preexisting preprocessing in the source:** none documented
- **Transformations applied by this pipeline:** none
- **Status reason:** dimensions (600, 20000, 2) match Table 1, labels align and are reversible, no non-finite values, and lineage is substantiated by a official source

### 8. BREAST — `SHAPE_MATCH_PROVENANCE_UNVERIFIED`

- **Expected (Table 1):** 97 samples × 24481 features × 2 classes; paper feature type `Real`
- **Observed:** n=97, p=24481, c=2
- **Upstream origin:** <https://mldata-static.ml.tu-berlin.de/repository/data/download/csv/breast-cancer-kent-ridge-2/index.html> (tier `academic_repository`)
- **Raw archive:** `dataset/ossfs-dd-reproduction/raw/BREAST/breast-cancer-kent-ridge-2.csv` — SHA-256 `9a4971e3bc260ff1f3f68cfbbed3754c04940e4c042fe64faf3bf27737240f64`, 15001176 bytes
- **Provenance:** tier `academic_repository`, confidence `medium`
- **Citation:** Kent Ridge Bioinformatics breast cancer expression data (repository record cites the LibSVM-era collection).
- **Licence:** license_unknown — repository states 'License: unknown (from LibSVMTools repository)'
- **Class counts:** `{"non-relapse": 51, "relapse": 46}`
- **Source split composition:** `{"original_labeled": 97}`
- **Non-finite values:** total=0 (NaN=0, +Inf=0, -Inf=0)
- **Constant columns:** 320; **duplicate column groups:** 340; **duplicate row groups:** 0; **zero-only rows:** 0
- **Value range:** [-2.0, 100.0] (all values integral: False); stored as `numpy_dense` with dtype `float64`, density 1.0, zero columns 29
- **Preexisting preprocessing in the source:** none documented
- **Transformations applied by this pipeline:** none
- **Status reason:** dimensions (97, 24481, 2) and labels match Table 1, but the variant came from an academic_repository record and the paper does not state which release it consumed, so exact scientific equivalence is not proven
- **Alternatives considered and rejected:**
  - *ico2s.org `breast_*` (Naderi et al. 2006, 47293 genes, 128 samples)* — 128 samples and 47293 genes. Different cohort entirely.
  - *seventyGeneData van't Veer / van de Vijver (24481 features)* — 24481 features is a coincidental collision with 117 and 295 samples respectively. Different cohort, different sample count.

  - *TU Berlin `uci-20070111-breast-cancer` / `breast-cancer-ida`* — UCI-style low-dimensional versions, not the 24481-gene microarray matrix.

### 9. MADELON — `EXACT_VERIFIED`

- **Expected (Table 1):** 2600 samples × 500 features × 2 classes; paper feature type `Integer`
- **Observed:** n=2600, p=500, c=2
- **Upstream origin:** <https://archive.ics.uci.edu/static/public/171/madelon.zip> (tier `official`)
- **Raw archive:** `dataset/ossfs-dd-reproduction/raw/MADELON/madelon.zip` — SHA-256 `eb99a0153ebe7d9f4e74551b4b10c54a958b2990b156ce7d76b4e8d71a6f6e82`, 3040503 bytes
- **Provenance:** tier `official`, confidence `high`
- **Citation:** NIPS 2003 Feature Selection Challenge; Guyon I. et al., JMLR 3:1157-1232 (2002).
- **Licence:** CC BY 4.0 as published by the UCI Machine Learning Repository
- **Class counts:** `{"-1": 1300, "1": 1300}`
- **Source split composition:** `{"train": 2000, "valid": 600}`
- **Non-finite values:** total=0 (NaN=0, +Inf=0, -Inf=0)
- **Constant columns:** 0; **duplicate column groups:** 0; **duplicate row groups:** 0; **zero-only rows:** 0
- **Value range:** [0.0, 999.0] (all values integral: True); stored as `numpy_dense` with dtype `float64`, density 1.0, zero columns 0
- **Independent source checksum verification** (NIPS `.param` published `Check_sum`):
  - `train`: published `488083511.0`, observed `488083511.0` — **MATCH**; class balance published {'pos': 1000, 'neg': 1000} vs observed {'pos': 1000, 'neg': 1000}
  - `valid`: published `146395833.0`, observed `146395833.0` — **MATCH**; class balance published {'pos': 300, 'neg': 300} vs observed {'pos': 300, 'neg': 300}
- **Preexisting preprocessing in the source:** none documented
- **Transformations applied by this pipeline:** none
- **Status reason:** dimensions (2600, 500, 2) match Table 1, labels align and are reversible, no non-finite values, and lineage is substantiated by a official source

### 10. GINA — `EXACT_VERIFIED`

- **Expected (Table 1):** 3468 samples × 970 features × 2 classes; paper feature type `Integer`
- **Observed:** n=3468, p=970, c=2
- **Upstream origin:** <https://www.openml.org/data/download/53921/gina_agnostic.arff> (tier `official`)
- **Raw archive:** `dataset/ossfs-dd-reproduction/raw/GINA/gina_agnostic.arff` — SHA-256 `d4b95d97b031c4d331ec2d23d94203f1f8451e1af473bfd95a02d3881761d1b6`, 8544653 bytes
- **Provenance:** tier `official`, confidence `medium`
- **Citation:** ETH Zurich Agnostic Learning Competition, GINA digit-recognition dataset.
- **Licence:** CC BY 4.0 as published by OpenML (dataset did=1038)
- **Class counts:** `{"-1": 1763, "1": 1705}`
- **Source split composition:** `{"original_labeled": 3468}`
- **Non-finite values:** total=0 (NaN=0, +Inf=0, -Inf=0)
- **Constant columns:** 0; **duplicate column groups:** 0; **duplicate row groups:** 0; **zero-only rows:** 0
- **Value range:** [0.0, 255.0] (all values integral: True); stored as `numpy_dense` with dtype `float64`, density 1.0, zero columns 0
- **Preexisting preprocessing in the source:** none documented
- **Transformations applied by this pipeline:** none
- **Status reason:** dimensions (3468, 970, 2) match Table 1, labels align and are reversible, no non-finite values, and lineage is substantiated by a official source
- **Alternatives considered and rejected:**
  - *original ETH Zurich agnostic-track gzip release* — Recorded as the ultimate upstream origin in the ARFF header, but the OpenML-hosted ARFF is the traceable, integrity-checkable copy with a published dataset id; preferred for reproducibility.


### 11. GISETTE — `EXACT_VERIFIED`

- **Expected (Table 1):** 7000 samples × 5000 features × 2 classes; paper feature type `Integer`
- **Observed:** n=7000, p=5000, c=2
- **Upstream origin:** <https://archive.ics.uci.edu/static/public/170/gisette.zip> (tier `official`)
- **Raw archive:** `dataset/ossfs-dd-reproduction/raw/GISETTE/gisette.zip` — SHA-256 `430bb631b3e494659f13763c117f06f5aa8a269b0d8e13ccf5b71d95303f3808`, 21298687 bytes
- **Provenance:** tier `official`, confidence `high`
- **Citation:** NIPS 2003 Feature Selection Challenge; Bottou L. et al.
- **Licence:** CC BY 4.0 as published by the UCI Machine Learning Repository
- **Class counts:** `{"-1": 3500, "1": 3500}`
- **Source split composition:** `{"train": 6000, "valid": 1000}`
- **Non-finite values:** total=0 (NaN=0, +Inf=0, -Inf=0)
- **Constant columns:** 29; **duplicate column groups:** 87; **duplicate row groups:** 0; **zero-only rows:** 0
- **Value range:** [252.0, 999.0] (all values integral: True); stored as `scipy_csr` with dtype `float64`, density 0.12998054285714286, zero columns 29
- **Independent source checksum verification** (NIPS `.param` published `Check_sum`):
  - `train`: published `3164568508.0`, observed `3164568508.0` — **MATCH**; class balance published {'pos': 3000, 'neg': 3000} vs observed {'pos': 3000, 'neg': 3000}
  - `valid`: published `535016668.0`, observed `535016668.0` — **MATCH**; class balance published {'pos': 500, 'neg': 500} vs observed {'pos': 500, 'neg': 500}
- **Preexisting preprocessing in the source:** none documented
- **Transformations applied by this pipeline:** none
- **Status reason:** dimensions (7000, 5000, 2) match Table 1, labels align and are reversible, no non-finite values, and lineage is substantiated by a official source

## Preparation summary

| Artifact tree | Files | Bytes |
|---|---:|---:|
| `dataset/ossfs-dd-reproduction/raw/` | 39 | 264829431 |
| `dataset/ossfs-dd-reproduction/processed/` | 88 | 105848738 |
| `dataset/ossfs-dd-reproduction/splits/` | 110 | 9907922 |
| `dataset/ossfs-dd-reproduction/stream_orders/` | 110 | 24969522 |

- CV test folds defined: **550** (target 550)
- Feature-order permutations defined: **110** (target 110)
- Seeds: `[11, 17, 23, 31, 47, 59, 71, 83, 97, 109]`, `n_splits=5`

## Unresolved items and next actions

| Dataset | Status | Reason | Minimum corrective action |
|---|---|---|---|
| DLBCL | SHAPE_MATCH_PROVENANCE_UNVERIFIED | dimensions (77, 7129, 2) and labels match Table 1, but the variant came from an academic_repository record and the paper does not state which release it consumed, so exact scientific equivalence is not proven | Confirm with the authors, or locate an original Kent Ridge / Shipp et al. release with a published checksum, to substantiate the exact variant. No code change can close this gap. |
| BREAST | SHAPE_MATCH_PROVENANCE_UNVERIFIED | dimensions (97, 24481, 2) and labels match Table 1, but the variant came from an academic_repository record and the paper does not state which release it consumed, so exact scientific equivalence is not proven | Confirm with the authors, or locate an original Kent Ridge / Shipp et al. release with a published checksum, to substantiate the exact variant. No code change can close this gap. |

## Reproduction assumptions (explicit)

- The paper describes 5-fold CV with 10 repetitions of randomized feature arrival orders, but **does not disclose its splits or seeds**. The plans in `dataset/ossfs-dd-reproduction/splits/` and `stream_orders/` use fixed preregistered seeds `[11, 17, 23, 31, 47, 59, 71, 83, 97, 109]`. They are a reproduction assumption, not a recovery of the original experiment.
- Original NIPS-2003 train/valid membership is preserved per row, but is not used as an evaluation holdout: the paper's totals are combined labeled counts.
- No preprocessing of any kind is applied in Stage 1. Any scaling, imputation or discretisation must be fitted on training rows only, separately within each fold, in a later stage.

## Next stage

`DO NOT START STAGE 2` — at least one hard gate fails. Gates must not be weakened to permit Stage 2. See the unresolved-items table above.
