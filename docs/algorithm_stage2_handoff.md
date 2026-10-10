# Stage 2 handoff — available data and stream APIs

**This document describes only the interfaces Stage 1 provides. It implements no
part of OSSFS-DD.** No selection policy, no incremental decision logic, no dynamic
decision process, no classifier, no evaluation. Those are Stage 2 and Stage 3 work.

Stage 2 may not begin until the researcher has reviewed
`results/stage1/STAGE1_REPORT.md` and `results/stage1/readiness_summary.json`, and
only if the verdict there is `STAGE1_READY`.

---

## 1. What is on disk

```
${RESEARCH_DATA_ROOT}/ossfs-dd-reproduction/
├── raw/<KEY>/                      immutable downloaded bytes + extracted archives
├── processed/<KEY>/
│   ├── X.npy            OR  X_csr.npz     (n_samples, n_features) predictors only
│   ├── y.npy                                    (n_samples,) label strings
│   ├── sample_ids.npy                            (n_samples,) "<KEY>:<row>"
│   ├── feature_ids.npy                           (p,) int64, always 0..p-1
│   ├── feature_names.json                        (p,) source names or "feature_<i>"
│   ├── source_split.npy                          (n_samples,) "train" | "valid" | ...
│   ├── source_row_index.npy                      (n_samples,) row index within its source file
│   └── schema.json                               dtype, label map, parsing notes, X sha256
├── splits/<KEY>/seed_<S>.json                    preregistered CV plan
├── stream_orders/<KEY>/seed_<S>.json              preregistered arrival permutation
└── manifests/<KEY>.json                           full provenance + data-quality audit
```

`KEY` is one of `SRBCT LYMPHOMA PROSTATE LEUKEMIA DLBCL ARCENE DEXTER BREAST MADELON
GINA GISETTE`.

---

## 2. Loading a dataset

> **Paths stored in `schema.json` and the manifests are relative to
> `RESEARCH_ROOT`, not to the current working directory.** Resolving them
> directly (`np.load(meta["X"]["path"])`) works only if you happen to be sitting
> in `$RESEARCH_ROOT` and silently fails with `FileNotFoundError` anywhere else.
> Always go through `ossfs_dd_data.workspace.resolve`.

```python
import json
import numpy as np
from scipy import sparse
from ossfs_dd_data.workspace import processed_dir, resolve, splits_dir, stream_orders_dir

KEY = "ARCENE"

meta = json.loads((processed_dir(KEY) / "schema.json").read_text(encoding="utf-8"))

xrec = meta["X"]
X = (
    sparse.load_npz(str(resolve(xrec["path"])))
    if xrec["format"] == "scipy_csr"
    else np.load(resolve(xrec["path"]), allow_pickle=False)
)
y = np.load(resolve(meta["y"]["path"]), allow_pickle=False).astype(str)
source_split = np.load(resolve(meta["source_split"]["path"]), allow_pickle=False).astype(str)
sample_ids = np.load(resolve(meta["sample_ids"]["path"]), allow_pickle=False)  # fixed-width, no pickle

assert X.shape == (meta["n_samples"], meta["n_features"])
assert len(y) == len(sample_ids) == X.shape[0]
```

`sample_ids.npy` is a fixed-width Unicode array and **must** be readable with
`allow_pickle=False`. Artifacts written before the Stage 1 follow-up migration
were object-dtype and did require `allow_pickle=True`; those files were migrated
non-lossfully and retained as `sample_ids.npy.v1backup`. If you ever encounter a
`ValueError: Object arrays cannot be loaded when allow_pickle=False`, you are
reading a pre-migration file.

This loader is exercised against **real stored data** (both the dense and the
sparse storage formats) by `tests/test_real_data_loading.py`, so it is not
merely illustrative documentation.

Guarantees a Stage 2 selector may rely on:

* `X.shape[1]` is the number of **predictor columns**. The class column is never
  included. `test_no_target_column_in_features` enforces this.
* `y` is aligned row-for-row with `X`. Labels are the **original source strings**.
  `meta["label_map_original_to_canonical"]` gives a reversible original → canonical
  integer map.
* Column order is stable and is `feature_ids.npy`, i.e. `0..p-1`, mapped to source
  names through `feature_names.json`.
* Values are exactly as published. **Nothing has been imputed, scaled, standardised,
  discretised, clipped, deduplicated or dropped.**
* `X` may be a SciPy CSR matrix. Do not assume `.shape` is the only API you need;
  use `X.tocsr()[:, j]` to read one feature.
* `schema.json["X"]["sha256"]` is the hash of the stored array. Verify it if you cache.

---

## 3. The leakage-safe feature stream

```python
from ossfs_dd_data.stream import FeatureStream

plan = json.loads((splits_dir(KEY) / "seed_11.json").read_text(encoding="utf-8"))
fold = plan["folds"][0]
train_idx = np.asarray(fold["train"], dtype=np.int64)

order_rec = json.loads((stream_orders_dir(KEY) / "seed_11.json").read_text(encoding="utf-8"))
order = np.asarray(order_rec["arrival_order"], dtype=np.int64)

stream = FeatureStream(X, y, train_idx, order)

for original_feature_index, values in stream:
    # values.shape == (len(train_idx),)
    # values is X[train_idx, original_feature_index] — training rows ONLY
    ...
```

Contract, enforced by construction and asserted by
`test_stream_contains_only_training_rows`:

1. Iteration yields `(original_feature_index, column_values)` one feature at a time,
   in `order`.
2. Every yielded column has exactly `len(train_indices)` rows, and equals
   `X[train_indices, feature_index]`.
3. **Held-out rows are unreachable from `__iter__`.** `X` is indexed by
   `train_indices` inside the iterator; there is no code path from the loop to a test
   row.
4. `stream.train_labels` returns labels for training rows only.
5. For sparse `X`, a column is sliced from CSR rather than globally densified.
6. `stream.self_check()` re-verifies 1–3 against real data at runtime.

Deliberate escape hatch: `stream.full_rows_diagnostic(j)` returns a column for **all**
rows including held-out ones. It is a separate, explicitly named method so that
reaching test-row data from a selection loop is a visible decision rather than an
accident. Use it only for debugging and reporting.

The stream contains **no selection policy**. Deciding which features to keep is
Stage 2's job and is deliberately absent.

---

## 4. Split and order plans

```python
plan = json.loads((splits_dir(KEY) / "seed_23.json").read_text())
plan["splitter"]            # "sklearn.model_selection.StratifiedKFold"
plan["random_state"]        # 23
plan["stratified"]          # bool
plan["stratified_skip_reason"]   # non-null only if a class is smaller than n_splits
plan["folds"]               # [{fold, train, test, n_train, n_test}, ...]
```

Guarantees:

* Every fold is a **partition**: `train ∪ test` is exactly `0..n-1`, and
  `train ∩ test = ∅`. Asserted by `test_cv_train_test_disjoint`.
* Indices are sorted within each fold.
* Ten seeds `[11, 17, 23, 31, 47, 59, 71, 83, 97, 109]` × `n_splits=5` = **50 folds
  per dataset, 550 across the eleven**.
* If a class has fewer members than `n_splits`, stratification is mathematically
  impossible. The plan then uses `KFold` and records the reason in
  `stratified_skip_reason`. It does not silently degrade.

```python
order_rec = json.loads((stream_orders_dir(KEY) / "seed_23.json").read_text(encoding="utf-8"))
order_rec["arrival_order"]                 # list of length p, a permutation of 0..p-1
order_rec["generator_api"]                 # exact numpy call that produced it
order_rec["inverse_position_of_feature"]   # str(feature_index) -> arrival position
```

> **The two fields point in opposite directions.** `arrival_order` answers *"which
> feature arrives at step k?"*; `inverse_position_of_feature` answers *"at which
> step does feature j arrive?"*. Conflating them silently produces a wrong
> answer.
>
> ```python
> inverse = {int(k): v for k, v in order_rec["inverse_position_of_feature"].items()}
> assert inverse[j] == order.index(j)   # feature -> position
> assert order[inverse[j]] == j         # and it genuinely inverts
> ```
>
> **Schema history.** Plans written as `stream_order/1.0` stored
> `inverse_position_of_feature` in the *forward* direction, i.e. a verbatim copy
> of `arrival_order`. The Stage 1 follow-up migrated all 110 plans to
> `stream_order/2.0` where the field is the true inverse; originals are retained
> as `*.json.v1backup`. **`arrival_order` was never altered** — only the inverse
> mapping. A `1.0` record's inverse key is unusable and must be re-derived.
>
> Arrival order is preregistered: it is regenerated deterministically from
> `numpy.random.Generator(PCG64(seed)).permutation(p)`, so re-running Stage 1
> reproduces it byte-for-byte.

`test_feature_order_is_permutation` asserts the permutation property and that the
inverse really inverts the forward order, on the real stored plans.

---

## 5. Assumptions Stage 2 must keep carrying

1. **The paper does not disclose its splits or seeds.** These plans are a fixed,
   preregistered reproduction assumption, recorded in every plan file's
   `assumption_notice`. Do not describe them as the paper's original splits.
2. **The original NIPS train/valid partition is metadata, not a holdout.**
   `source_split.npy` records which rows came from the challenge's train and valid
   files. The paper's sample totals are the *combined* labeled counts, so Stage 2/3
   cross-validates over the pooled labeled data. Pooling is not leakage: the split is
   an artefact of how the NIPS files were packaged, not a scientific train/test
   boundary.
3. **Any preprocessing must be fitted on training rows only, inside each fold.**
   Stage 1 performed none. If Stage 2 scales, imputes or discretises, that fit belongs
   on `train_indices` only, recomputed per fold, and must never see `test`.
4. **Preexisting source preprocessing is not Stage 2's to undo.** SRBCT, LYMPHOMA,
   PROSTATE and LEUKEMIA arrive standardised by the Binghamton collection. That is
   recorded in each manifest as preexisting. Silently re-standardising would change
   the data relative to the paper.
5. **Two datasets have unproven lineage** (DLBCL, BREAST — see
   `docs/dataset_provenance.md` §5). They are dimensionally exact and safe to use, but
   any Stage 3 accuracy claim must state that their provenance is `academic_repository`,
   not paper-cited.

---

## 6. Status vocabulary a Stage 2 script can read

```python
from ossfs_dd_data.workspace import manifest_path
import json

verdict = json.loads(manifest_path(KEY).read_text())["stage1_verdict"]
verdict["status"]    # e.g. "EXACT_VERIFIED" | "SHAPE_MATCH_PROVENANCE_UNVERIFIED"
verdict["reason"]    # human-readable justification
```

Only `EXACT_VERIFIED` and `SHAPE_MATCH_PROVENANCE_UNVERIFIED` have usable data and
plans on disk. The remaining statuses mean no plan was written; treat a request to run
one as a hard error rather than improvising a split.
