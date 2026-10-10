# Dataset provenance, lineage decisions and uncertainty register

Project: `ossfs-dd-reproduction` — Stage 1 (data only).
Reference: Zhou, Zhao, Yan & Wu, *Online Scalable Streaming Feature Selection via
Dynamic Decision*, ACM TKDD 16(5) art. 87 (2022), DOI [10.1145/3502737](https://doi.org/10.1145/3502737).

Everything below was **observed at Stage 1 runtime on this machine**, not assumed.
Where a statement is an inference rather than an observation, it says so.

---

## 1. Source collection reality check

The paper's Section 4.1.1 footnote cites two collections.

| Cited collection | URL as cited | Observed at Stage 1 runtime |
|---|---|---|
| Binghamton legacy research datasets (Wei Lyu, KDD08) | `http://www.cs.binghamton.edu/~lyu/KDD08/data/` | **LIVE.** Plain HTTP returns a 302 to the `https://` form, which serves an Apache directory index. |
| UCI Machine Learning Repository | `https://archive.ics.uci.edu/` | **LIVE.** |

The Binghamton collection contains exactly seven `.arff` files:

```
colon-std.arff   leukemia-std.arff   lung-std-5000.arff
lungCancer-std.arff   lymphoma-std.arff   prostate-std.arff   srbct-std.arff
```

Four of them correspond one-to-one with Table 1 rows (SRBCT, LYMPHOMA, PROSTATE,
LEUKEMIA). The other three (colon, lung, lungCancer) appear in no Table 1 row and
were **not** downloaded.

So the paper's primary cited collection supplied **4 of the 11** datasets. The
remaining seven came from the second cited collection (UCI, 4 datasets), OpenML
(1), and one academic repository (2).

### 1.1 The `-std` suffix is real and is documented, not hidden

Each Binghamton file is named `*-std.arff` and its ARFF `@relation` field names the
Weka filter chain that produced it. Observed verbatim:

| Dataset | `@relation` as published |
|---|---|
| SRBCT | `SRBCT-weka.filters.unsupervised.attribute.Standardize` |
| PROSTATE | `prost-weka.filters.unsupervised.attribute.Standardize` |
| LYMPHOMA | `leukemia-weka.filters.NormalizationFilter-weka.filters.unsupervised.attribute.Standardize-weka.filters.unsupervised.attribute.Standardize` |
| LEUKEMIA | `leukemia-weka.filters.NormalizationFilter-weka.filters.unsupervised.attribute.Standardize-weka.filters.unsupervised.attribute.Standardize` |

**This is preprocessing that already existed in the source.** Stage 1 did not
standardise, normalise or otherwise touch any value; it parsed what the file
contained. Every manifest records this under
`observed.preexisting_preprocessing_in_source`.

**Open question, recorded rather than resolved:** the paper does not state whether it
consumed the `-std` variants or the unstandardised originals. We took the paper's own
cited URL, which serves the `-std` files. If the authors used unstandardised data, a
Stage 3 accuracy comparison would be measuring a different preprocessing history.
This cannot be settled from public evidence and is listed in §5.

---

## 2. Per-dataset lineage decisions

### 2.1 SRBCT, LYMPHOMA, PROSTATE, LEUKEMIA — `paper_cited`, confidence **high**

Fetched from `https://www.cs.binghamton.edu/~lyu/KDD08/data/<name>-std.arff`, the
collection the paper cites. Observed dimensions matched Table 1 exactly:

| Dataset | rows | ARFF attributes | predictors after removing the class column | Table 1 |
|---|---:|---:|---:|---|
| SRBCT | 63 | 2309 | 2308 | 63 × 2308 ✓ |
| LYMPHOMA | 62 | 4027 | 4026 | 62 × 4026 ✓ |
| PROSTATE | 102 | 6034 | 6033 | 102 × 6033 ✓ |
| LEUKEMIA | 72 | 7130 | 7129 | 72 × 7129 ✓ |

Rejected variants and why:

* **LEUKEMIA 72 × 7070** (scikit-feature `leukemia`). Wrong feature count. Not padded
  or trimmed to 7129.
* **LEUKEMIA 72 × 7128.** Wrong feature count.
* **PROSTATE 102 × 5966** (scikit-feature `Prostate_GE`). Wrong feature count.
* **PROSTATE 102 × 12600.** Wrong feature count.
* **LYMPHOMA 96 × 4026** (scikit-feature `lymphoma`). Wrong sample count.
* **LYMPHOMA 62 × 4682.** Wrong feature count.
* **SRBCT 83 samples** (several releases). Wrong sample count — the paper says 63.
* **PROSTATE 102 × 6033 via `hastie.su.domains/CASI_files/DATA/prostmat.csv`** — a
  6033 × 102 gene-major CSV. Correct dimensions but requires a transpose and splits
  its labels across separate files. Superseded by the paper-cited source, which is
  already sample-major. Kept as a corroborating lead only.

The `ALLAML` and `lymphoma`/`prostate` entries in
`jundongl.github.io/scikit-feature/datasets.html` were checked and their published
dimensions recorded above; none was substituted for the paper-cited source.

### 2.2 ARCENE, DEXTER, MADELON, GISETTE — `official`, confidence **high**

Official UCI archives (`/static/public/{167,168,171,170}/`). These are the second
collection the paper cites.

The Table 1 sample counts are the **sum of the labeled train and validation splits**:

| Dataset | train | valid | combined | Table 1 |
|---|---:|---:|---:|---|
| ARCENE | 100 | 100 | 200 | 200 ✓ |
| DEXTER | 300 | 300 | 600 | 600 ✓ |
| MADELON | 2000 | 600 | 2600 | 2600 ✓ |
| GISETTE | 6000 | 1000 | 7000 | 7000 ✓ |

The unlabeled challenge `*_test.data` files (700 / 2000 / 1800 / 6500 rows) are
**never parsed into a dataset**. `parse_uci_data_labels_pair` raises immediately if a
`test` key is passed, and `test_no_unlabeled_test_data` asserts both that refusal and
that no stored row carries a `test` source-split marker.

#### Source-provided checksum verification

UCI ships the NIPS-2003 `.param` files inside each archive. These declare the data
type, the feature count, and a published value `Check_sum` and positive/negative
counts per split. We used them as an **independent integrity witness**: if the sum of
the values we parsed equals the published checksum, the rows we read are the rows the
challenge authors intended.

| Dataset | split | published `Check_sum` | observed sum | match | published pos/neg | observed pos/neg |
|---|---|---:|---:|:--:|---|---|
| ARCENE | train | 70726744 | 70726744 | yes | 44 / 56 | 44 / 56 |
| ARCENE | valid | 71410108 | 71410108 | yes | 44 / 56 | 44 / 56 |
| DEXTER | train | 2816528 | 2816528 | yes | 150 / 150 | 150 / 150 |
| DEXTER | valid | 2820952 | 2820952 | yes | 150 / 150 | 150 / 150 |
| MADELON | train | 488083511 | 488083511 | yes | 1000 / 1000 | 1000 / 1000 |
| MADELON | valid | 146395833 | 146395833 | yes | 300 / 300 | 300 / 300 |
| GISETTE | train | 3164568508 | 3164568508 | yes | 3000 / 3000 | 3000 / 3000 |
| GISETTE | valid | 535016668 | 535016668 | yes | 500 / 500 | 500 / 500 |

All eight splits match on both the value sum and the class balance. This is the
strongest independent evidence in this stage and it applies to all four UCI datasets.

#### DEXTER sparse layout

`dexter.param` declares `Data type: sparse-integer`, which is the NIPS 2003 format
using **1-based** feature indices. Observed tokens such as `10:105` confirm this. The
parser therefore reads DEXTER as 1-based and materialises **exactly 20000 columns**,
including implicitly all-zero columns. The manifest records
`index_base_convention: 1-based` together with the source `.param` declaration that
justified it.

DEXTER is stored as SciPy CSR rather than a dense array, and no global densification
is performed anywhere in the audit or the stream.

#### An observed UCI packaging quirk

In every one of the four archives, the validation labels file sits at the **archive
root**, not inside the dataset folder:

```
ARCENE/arcene_train.data   ARCENE/arcene_valid.data   arcene_valid.labels
DEXTER/dexter_train.data   DEXTER/dexter_valid.data   dexter_valid.labels
madelon/madelon_train.data madelon/madelon_valid.data madelon_valid.labels
GISETTE/gisette_train.data GISETTE/gisette_valid.data gisette_valid.labels
```

The registry declares both candidate locations for each, so a re-extraction against a
differently-packed archive still resolves. This is recorded, not worked around silently.

### 2.3 GINA — `official` (OpenML), confidence **medium**

Fetched `https://www.openml.org/data/download/53921/gina_agnostic.arff`, the data
file OpenML publishes for dataset id 1038 (`gina_agnostic`), reached from
`https://www.openml.org/d/1038`.

Observed: relation `GINA_Agnostic`, 3468 data rows, **971 attributes of which 970 are
predictors** and the last is the class. The Stage 1 prompt warns against reading 971
attributes as 971 predictors; the parser removes exactly one nominal attribute
(`label`, located as the sole nominal attribute) and stores 970 features, so
3468 × 970 ✓.

Observed class encoding is `{-1, +1}`, consistent with the documented odd/even task.

Confidence is **medium**, not high, for one reason recorded in the ARFF header itself:

```
%Dataset from: http://www.agnostic.inf.ethz.ch/datasets.php
%Modified by TunedIT (converted to ARFF format)
```

The file is an ARFF re-encoding of the original ETH Zurich agnostic-track release,
distributed by OpenML under CC BY 4.0. It is the traceable, integrity-checkable copy
with a published dataset id, which is why it was preferred over the upstream `.gz`,
but it is not the original file byte-for-byte. The original row order is preserved.

### 2.4 DLBCL — `academic_repository`, confidence **medium**

DLBCL is in neither collection the paper cites, so the paper's own source is
unavailable for it.

Selected: `https://mldata-static.ml.tu-berlin.de/repository/data/download/csv/dlbcl-tumor-from-harvard/index.html`
(TU Berlin Machine Learning Repository, dataset record `dlbcl-tumor-from-harvard`).

Observed: 77 data rows × 7130 comma-separated fields = **7129 predictors + 1 class**.
Class counts **DLBCL 58 / FL 19**. Repository index page states, verbatim,
"Data Shape: 7130 attributes, 77 instances".

Provenance chain: the repository describes the record as originating from the
UCI/LibSVM-era Kent Ridge collection and reproduces the description of the two Shipp
et al. (2002) classification problems. The observed class semantics — 58 DLBCL against
19 follicular lymphoma — match the published description of the *first* of those two
problems exactly, which is an internal consistency check the record passes.

Why **not** higher confidence: the paper does not say which DLBCL release it consumed,
so "the paper used this file" cannot be established from public evidence. Status is
therefore `SHAPE_MATCH_PROVENANCE_UNVERIFIED`, not `EXACT_VERIFIED`.

Rejected alternatives:

| Candidate | Observed dimensions | Why rejected |
|---|---|---|
| OpenML `DLBCL` did=45088 | 77 × **5469** | Sample count matches, feature count does not. This is exactly the reprocessed 5469-predictor variant the Stage 1 prompt says to reject. |
| TU Berlin `dlbcl-outcome-from-harvard` | **58** × 7129 | The cured-vs-fatal subset — a different classification problem, and 58 samples. |
| TU Berlin `dlbcl-nih` | 240 × 7399 | Different cohort entirely. |
| `ico2s.org` DLBCL | page text says "7129 genes, 77 samples" | **Dead link.** The phrase appears in the page body but no `href` exists in the served HTML; `dlbcl_10fold_arff.zip`, `dlbcl_loo_arff.zip` and `dlbcl_preprocessed.zip` all return HTTP 404. |

### 2.5 BREAST — `academic_repository`, confidence **medium**

Also absent from both cited collections.

Selected: `https://mldata-static.ml.tu-berlin.de/repository/data/download/csv/breast-cancer-kent-ridge-2/index.html`.

Observed: 97 data rows × 24482 comma-separated fields = **24481 predictors + 1
class**. Class counts **relapse 46 / non-relapse 51**. Repository index page states,
verbatim, "Data Shape: 24482 attributes, 97 instances", which independently matches the
Stage 1 prompt's account of that index.

Rejected alternatives:

| Candidate | Observed dimensions | Why rejected |
|---|---|---|
| ico2s `breast_*` (Naderi et al. 2006) | 128 × 47293 | Different cohort: 128 samples, 47293 genes. |
| `seventyGeneData` van't Veer / van de Vijver | 117 × 24481 and 295 × 24481 | The 24481 feature count is a **coincidental collision** with a completely different study and different sample counts. Rejected on samples. |
| TU Berlin `uci-20070111-breast-cancer`, `breast-cancer-ida` | low-dimensional | UCI-style versions, not the 24481-gene matrix. |

Observed transport detail worth recording: this repository serves the payload itself
at the download path with `Content-Type: text/html`, which is wrong — the body is a
plain CSV. The `arff/` and `matlab/` variants of the same record returned a 58-byte
`An internal server error occurred. Please try again later.` page at Stage 1 runtime.
The downloader rejects such a body as a non-dataset; the CSV variant is the one that
resolved, and the observation is written to the fetch log rather than smoothed over.

---

## 3. Licensing

| Datasets | Licence as stated by the source |
|---|---|
| ARCENE, DEXTER, MADELON, GISETTE | CC BY 4.0 as published by UCI |
| GINA | CC BY 4.0 as published by OpenML (dataset 1038) |
| SRBCT, LYMPHOMA, PROSTATE, LEUKEMIA | `license_unknown` — academic research redistribution, no licence published on the cited page |
| DLBCL | `license_unknown` — repository states "License: unknown (from UCI repository)" |
| BREAST | `license_unknown` — repository states "License: unknown (from LibSVMTools repository)" |

`license_unknown` is recorded as unknown. It is not upgraded to a permissive guess.
Independently of licence, **no data file is committed to git**: everything lives under
the workspace `dataset/` tree, outside the project repository, which
`test_git_excludes_data` enforces against git's own matching rules.

---

## 4. Feature-type disagreement between the paper and its sources

The paper's Table 1 has a "feature type" column. The sources describe their own types
differently. Both are recorded; neither is falsified.

| Dataset | Paper wording | Observed reality |
|---|---|---|
| ARCENE | Integer | ARFF/dataset metadata describes a continuous value space; the values are large non-negative integers. Stored dtype recorded separately. |
| MADELON, GISETTE | Integer | Stored as `float64`; the values happen to be integral. **Not** cast to an integer dtype to match the wording. |
| SRBCT, LYMPHOMA, PROSTATE, LEUKEMIA | Real | Already standardised upstream (§1.1); parsed dtype is `float64`. |
| DEXTER | Integer | Genuinely sparse **integer** values, per the source `.param`. |

`paper_feature_type` is a separate metadata field from the parsed dtype everywhere.
Each manifest carries `observed.value_integrality`, computed on the real values, and
`observed.value_range`.

---

## 5. Uncertainty register

Open items, stated plainly. None of them was resolved by assumption.

| # | Dataset | Uncertainty | Can Stage 1 close it? |
|---|---|---|---|
| U1 | SRBCT, LYMPHOMA, PROSTATE, LEUKEMIA | The paper does not state whether it consumed the `-std` (standardised) variants or unstandardised originals. We used the paper's own cited URL, which serves `-std`. | **No.** Requires author contact or a supplementary appendix. Affects Stage 3 comparability, not Stage 1 validity. |
| U2 | DLBCL | Provenance is an academic repository record consistent with Shipp et al. (2002), not the paper's cited source. The paper does not name its release. | **No.** Needs a published checksum for the original Kent Ridge file, or author confirmation. |
| U3 | BREAST | Same structural gap as U2: Kent Ridge record via an academic repository, paper does not name its release. | **No.** Same remedy. |
| U4 | GINA | The OpenML file is an ARFF re-encoding of the original ETH release, per its own header. | **No**, but the transformation is documented and the distribution channel is authoritative. |
| U5 | All | The paper does not disclose its CV splits or seeds. Ours are a fixed preregistered assumption. | **No.** Explicitly recorded as an assumption in every plan file. |
| U6 | All 11 | No source publishes a SHA-256 for these files. We publish our own hashes so future runs can detect drift, but they are not third-party attestations. | **No.** |
| U7 | SRBCT | The mixOmics `srbct` object and other same-named SRBCT variants exist. We did not compare them element-wise against the Binghamton file, because we do not know which one the paper used, and a coincidental comparison could create false confidence. | Deliberately **not attempted** in Stage 1. |

---

## 6. Everything that was checked and found wrong

Recorded so a later reader does not repeat the work.

* `http://www.cs.binghamton.edu/~lyu/KDD08/data/` (plain HTTP) serves a 302 stub with
  no directory index. The `https://` form works. Not a dead source.
* `csse.szu.edu.cn/staff/zhuzx/datasets.html` — a published mirror listing
  "Breast Cancer 24481 97 2", matching Table 1 exactly. **Unreachable**: the host does
  not respond to HTTP or HTTPS from this machine (`curl` exit with HTTP 000). Not used.
* `www2.stat.duke.edu/~sayan/feature/{ARCENE,DEXTER}/` — alternative mirrors named in
  the Stage 1 prompt. Not needed: the official UCI archive was reachable and ranks
  higher on lineage.
* `biostat.jhsph.edu/~rshepard/labeled_chip_data.html` (the Kent Ridge page) — HTTP 404,
  and the web.archive.org copy of it also 404s.
* `mixomomics.org` SRBCT case-study page and `rdrr.io` `spls`/`BCBCSF` lymphoma
  documentation — usable as corroboration, unnecessary as sources.
* `mldata-static.ml.tu-berlin.de` `breast-cancer-kent-ridge-2` `arff/` and `matlab/`
  variants — server-side error page. The `csv/` variant works.
* OpenML `DLBCL` did=45088 — 5469 features, not 7129. Rejected.
* scikit-feature `leukemia` is 72 × **7070**, not 72 × 7129. Rejected.

---

## 7. Where the evidence lives

| Artefact | Path (relative to `RESEARCH_ROOT`) |
|---|---|
| Table 1 spec, source candidates, lineage notes | `projects/ossfs-dd-reproduction/configs/dataset_registry.yaml` |
| Gates, seeds, retry policy | `projects/ossfs-dd-reproduction/configs/stage1.yaml` |
| Per-dataset provenance manifest | `dataset/ossfs-dd-reproduction/manifests/<KEY>.json` |
| Raw byte inventory with SHA-256 | `dataset/ossfs-dd-reproduction/manifests/<KEY>.raw_inventory.json` |
| Immutable raw files | `dataset/ossfs-dd-reproduction/raw/<KEY>/` |
| Canonical X / y / ids | `dataset/ossfs-dd-reproduction/processed/<KEY>/` |
| Split and order plans | `dataset/ossfs-dd-reproduction/{splits,stream_orders}/<KEY>/` |
| Download / parse / verify logs | `outputs/ossfs-dd-reproduction/stage1/logs/` |
| Committed ledgers and report | `projects/ossfs-dd-reproduction/results/stage1/` |
