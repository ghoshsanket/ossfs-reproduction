# OSSFS-DD Reproduction — Stage 1 (Data Only)

Reproduction project for:

> Peng Zhou, Shu Zhao, Yuanting Yan, and Xindong Wu.
> *Online Scalable Streaming Feature Selection via Dynamic Decision*.
> ACM TKDD 16(5), Article 87 (2022). DOI: <https://doi.org/10.1145/3502737>

**This repository currently implements Stage 1 only: dataset acquisition, provenance,
validation, and reproducible feature-stream preparation.**

It does **not** implement OSSFS-DD, does **not** train KNN/SVM/CART, and does **not**
reproduce any classification accuracy. No accuracy number is reported anywhere in this
repository, because none was measured.

---

## 1. What Stage 1 produces

The eleven exact dataset variants of Table 1 of the paper, as immutable raw bytes plus
canonical, audited representations:

```
X                     numeric (n_samples, n_features) — predictor columns only
y                     (n_samples,) observed class labels, with a reversible label map
sample_ids            row identifiers
feature_ids           0..p-1 original feature indices, reversible to source names
splits/<ds>/seed_*.json     preregistered StratifiedKFold train/test index plans
stream_orders/<ds>/*.json   preregistered feature arrival permutations
manifests/<ds>.json         provenance, hashes, parsing decisions, data-quality audit
```

Headline numbers are **never typed by hand**. They are produced by
`scripts/stage1_report.py` into `results/stage1/`, which reads the manifests on disk.

| Artefact | Location |
|---|---|
| Machine-readable status | `results/stage1/dataset_status.csv`, `dataset_status.json` |
| Machine-readable gate summary | `results/stage1/readiness_summary.json` |
| Source registry ledger | `results/stage1/source_registry_ledger.csv` |
| Human-readable report | `results/stage1/STAGE1_REPORT.md` |
| Dataset lineage decisions | `docs/dataset_provenance.md` |
| Stage 2 data API notes | `docs/algorithm_stage2_handoff.md` |

Data itself lives **outside** this repository, under the workspace `dataset/` tree, and
is never committed.

---

## 2. Environment

Target: NVIDIA DGX Spark, Linux **aarch64**, CPython 3.11 managed by the workspace.

```bash
export RESEARCH_PROJECT_NAME=ossfs-dd-reproduction
cd "$RESEARCH_ROOT"
source scripts/research-env.sh          # workspace contract: scripts/research-env.sh
cd "$RESEARCH_PROJECT_ROOT"
uv sync --frozen                        # honours the committed uv.lock
```

AArch64 notes:

* The workspace-managed interpreter is `runtimes/python/cpython-3.11.16-linux-aarch64-gnu/`.
  No system R, no `apt`, no container modification is required or used.
* All ten runtime dependencies ship aarch64 wheels; no source build occurs on `uv sync`.
* `.rda`/`.RData` parsing is implemented but is **not** on the install path: the
  paper-cited source collection supplied every microarray dataset, so no R-format file
  is parsed. `uv sync --extra rdata` adds `pyreadr` if a future fallback source needs it.
* Stage 1 is CPU + disk/network I/O only. It requests **no GPU** and does not read or
  write `CUDA_VISIBLE_DEVICES`.

---

## 3. Running the pipeline

Four independently runnable stages plus the test suite. Each is idempotent: rerunning
on unchanged inputs does not overwrite a successful dataset, change a hash, duplicate
rows, or alter a split permutation.

```bash
export RESEARCH_PROJECT_NAME=ossfs-dd-reproduction
cd "$RESEARCH_ROOT"; source scripts/research-env.sh; cd "$RESEARCH_PROJECT_ROOT"

uv run python scripts/stage1_fetch.py     --all   # immutable raw bytes + download ledger
uv run python scripts/stage1_prepare.py   --all   # canonical X,y + split/order plans
uv run python scripts/stage1_verify.py    --all   # audit + per-dataset status
uv run pytest                                      # contract + unit tests + git audit
uv run python scripts/stage1_report.py              # results/stage1/* — FINAL verdict
```

`pytest` runs **before** `stage1_report.py` on purpose. Three of the eleven gates are
evidenced by the test suite (`stream_leakage_tests_passing`,
`data_or_result_binaries_in_git`, `required_tests_passing`), and a gate whose evidence
is absent is reported as `not measured` — it does **not** pass. `stage1_report.py` is
the final verdict authority because it re-evaluates every gate once the test evidence
exists; `stage1_verify.py` writes the interim data-only verdict and exits non-zero if
a *data* gate fails.

`stage1_report.py` exits non-zero when any hard gate fails, **after** writing all
reports. A partial result is a valid outcome and the exit code reflects it rather than
being suppressed.

Restrict to a subset while debugging:

```bash
uv run python scripts/stage1_fetch.py --dataset ARCENE --dataset DEXTER
uv run python scripts/stage1_verify.py --dataset GINA
```

Logs: `$RESEARCH_OUTPUT_ROOT/ossfs-dd-reproduction/stage1/logs/`.

---

## 4. Project layout

```
src/ossfs_dd_data/
  registry.py     declarative dataset registry -> typed specs
  fetch.py        streaming download, retry, hashing, safe extraction
  parsers.py      ARFF / UCI .data+.labels / libsvm-sparse / CSV adapters
  provenance.py   manifests, hashes, environment snapshot
  validate.py     dimensional + data-quality audit, status assignment
  splits.py       preregistered CV and feature-order plans
  stream.py       leakage-safe one-feature-at-a-time iterator
scripts/          stage1_fetch / prepare / verify / report
configs/
  dataset_registry.yaml   Table 1 expectations + source candidates + lineage notes
  stage1.yaml             fixed seeds, 5-fold protocol, retry policy, gates
tests/            contract + unit tests, including a synthetic test fixture
docs/             provenance register, Stage 2 handoff notes
results/stage1/   committed: small text ledgers, summaries, final report
```

---

## 5. Source restrictions and acquisition policy

* Official / paper-cited URLs are attempted first. The paper's Section 4.1.1 footnote
  cites two collections; both are used, plus one documented academic repository.
* Up to 3 attempts per URL with bounded exponential backoff. 429/5xx/timeouts are
  retried; 404 is not.
* An academic or institutional mirror is used only when the original is unavailable or
  unpublished, and its exact URL, repository, license and equivalence confidence are
  recorded in `configs/dataset_registry.yaml` and `docs/dataset_provenance.md`.
* Downloads stream to a `.partial` file inside the workspace temp area and are renamed
  atomically only after the body completes and the signature is checked. An HTML error
  page, 0-byte body, checksum mismatch, GitHub directory listing, truncated archive or
  login wall is a **failure**, not a dataset.
* Archive extraction rejects path traversal, escaping symlinks/hardlinks and
  unexpected member types.
* No remote install scripts, no unpickling of unknown objects, no executing code from a
  downloaded archive, no access-control bypass.
* `dataset/` is never committed. If a source licence forbade redistribution the files
  would still not be committed; licences are recorded per dataset in the registry.

Matching `(n, p, classes)` is treated as **necessary but not sufficient**. A dataset
whose lineage is plausible but unproven is reported `SHAPE_MATCH_PROVENANCE_UNVERIFIED`,
never `EXACT_VERIFIED`.

---

## 6. Data-quality policy (Stage 1)

Stage 1 audits and reports but **never repairs**. It does not impute, normalize,
standardize, discretize, clip, remove, deduplicate or coerce anything to make a gate
pass. NaN/Inf/placeholder values that cannot be resolved are surfaced as
`DATA_QUALITY_BLOCKED` and escalated to the researcher, because the correct decision is a
train-fold-only preprocessing choice for a later stage.

Any preprocessing already baked into a published source variant is documented as
**preexisting**, not as newly applied by this pipeline.

The paper's `feature_type` wording (Real/Integer) is kept as a separate metadata field
from the dtype actually parsed. Source datasets describe their own types differently —
UCI calls ARCENE continuous — and that disagreement is recorded, never falsified.

---

## 7. Evaluation-plan assumptions (explicit)

The paper describes 5-fold cross-validation with 10 repetitions / randomized feature
arrival orders, with feature selection inside each training fold. It **does not disclose
its splits or seeds**. The plans produced here use fixed preregistered seeds
`[11, 17, 23, 31, 47, 59, 71, 83, 97, 109]` and `StratifiedKFold(n_splits=5)`.

These are therefore **explicit reproduction assumptions**, not claims about the paper's
original random experiments.

The original NIPS-2003 train/validation partition is preserved per row as metadata but is
**not** used as an evaluation holdout: the paper's totals are the *combined* labeled
train+valid counts, and Stage 3 cross-validates over the pooled labeled data.

---

## 8. Known limitations

These are known **before** any result is read, and none of them is a licence to weaken a gate.

1. Provenance strength is uneven across the eleven datasets. Four come from the paper's
   own cited collection; the remainder come from official or academic repositories. The
   registry records, per dataset, exactly which of those applies.
2. Where a repository page states a licence as "unknown", the registry records
   `license_unknown` rather than guessing a permissive licence.
3. Split seeds are an assumption, not a recovery of the paper's splits (§7).
4. Stage 1 produces no accuracy, no runtime and no feature-count result. Those belong to
   Stages 2 and 3.

---

## 9. Status

The current verdict is **not** written here by hand. Read:

* `results/stage1/readiness_summary.json` — the gate results and the stage verdict
* `results/stage1/STAGE1_REPORT.md` — the full human-readable report
* `results/stage1/dataset_status.csv` — per-dataset status

Stage 2 may begin only after the researcher has reviewed those artefacts.


---

# Stage 2 — OSSFS-DD Algorithm (Restricted Nine-Dataset Authorization)

Researcher-authorized: **only the nine Stage-1 `EXACT_VERIFIED` datasets**
(SRBCT, LYMPHOMA, PROSTATE, LEUKEMIA, ARCENE, DEXTER, MADELON, GINA, GISETTE).
**DLBCL and BREAST are excluded** and are refused even if named explicitly.

No classifier is fitted and no accuracy is computed anywhere in Stage 2.
Feature counts and runtimes are engineering diagnostics, not paper results.

## Running it

```bash
export RESEARCH_PROJECT_NAME=ossfs-dd-reproduction
cd "$RESEARCH_ROOT"; source scripts/research-env.sh; cd "$RESEARCH_PROJECT_ROOT"
uv sync --frozen

uv run python scripts/stage2_preflight.py --write   # authorization + env snapshot
uv run python scripts/stage2_selfcheck.py            # mathematical oracles
uv run pytest -q                                     # Stage 1 + Stage 2
uv run python scripts/stage2_run.py --all            # 9 executions, seed 11 fold 0
uv run python scripts/stage2_report.py               # final verdict
```

## Current status

Verdict: **STAGE2_PARTIAL**. Five of nine authorized datasets complete; four
(paper-type `Real`) return `UNSUPPORTED_UNVERIFIED_SEMANTICS`.

Reason: the continuous entry point `OSFS_DD_z2.m` calls `Fisher_Ztest`, which
**does not exist anywhere in the authors' pinned repository**. Its score,
units and conditional test cannot be determined from source, so the selector
raises rather than substituting a proxy statistic. No `1-p`, `|r|`, `p`,
correlation magnitude, discretised MI or `mutual_info_classif` was used.

Read the numbers — never this paragraph — in:

| Artefact | Path |
|---|---|
| Verdict and gates | `results/stage2/stage2_verdict.json` |
| Full report | `results/stage2/STAGE2_REPORT.md` |
| Per-dataset summary | `results/stage2/dataset_smoke_summary.{csv,json}` |
| Score-semantics ledger | `results/stage2/score_semantics_ledger.json` |
| Implementation deltas | `results/stage2/implementation_delta_ledger.json` |
| Determinism proof | `results/stage2/determinism_check.json` |
| Stage-1 immutability proof | `results/stage2/preservation_proof.json` |
| Method spec | `docs/stage2_method_spec.md` |
| Paper vs MATLAB table | `docs/stage2_paper_vs_matlab.md` |
| Source audit | `docs/stage2_source_audit.md` |
| Reproduction guide | `docs/stage2_reproduction_guide.md` |
| Immutable run bundles | `outputs/ossfs-dd-reproduction/stage2/runs/<run_id>/` |
| Authors' MATLAB snapshots | `outputs/ossfs-dd-reproduction/stage2/reference_snapshots/` |

## Two documented modes

`paper_spec_v1` — Algorithm 1 read literally, with the paper's gaps resolved by
preregistered choices (scale-coherent joint information, `>=` promotion,
declared warm-up bootstrap).

`matlab_compatible_v1` — mirrors the released MATLAB, including its warm-up
skip, reject-before-accept ordering, `>2*beta` promotion, tie-breaking, and its
**deliberate mixing of a normalised score with a raw information quantity**.

The two are not numerically interchangeable; results are never compared across
them as if they were.

## Known limitations

1. MATLAB runtime parity is `NOT_MEASURED` — `MIToolboxMex` ships only as a
   Windows binary and no MATLAB/Octave exists on this host. No parity is claimed.
2. Only seed 11 / fold 0 was executed. 9x10x5 is Stage 3 work and needs
   separate authorization.
3. The four `Real` datasets are blocked on unverified continuous semantics, not
   on any defect in the discrete implementation.
4. Split seeds remain an explicit reproduction assumption; the paper does not
   disclose its own.

Stage 3 must not begin until the researcher has reviewed
`results/stage2/STAGE2_REPORT.md` and resolved item 3 above.

## Stage 3A (added after Stage 3)

Stage 3A adds an **evaluation layer only**; it does not change Stage 1 or
Stage 2, and the Stage-2 verdict (`STAGE2_PARTIAL |
DISCRETE_CORRECTNESS_PASS`) stands unchanged, including its `UNRESOLVED`
continuous semantics.

- **Workstream A** — the five discrete `EXACT_VERIFIED` datasets, 10 seeds x
  5 folds = **250 selections** and **750 classifier outcomes** under three
  fixed classifiers, plus a labelled standardized sensitivity analysis.
  Verdict: `STAGE3A_READY`.
- **Workstream B** — a **reconstructed** continuous Fisher-Z estimator
  (`continuous_reconstructed_fisherz_v1`), run train-only on seed 11 / fold 0
  for the four `Real` datasets. The authors' `Fisher_Ztest` helper is absent
  from their repository, so this is research, **not** a reproduction, and it
  does not unlock Stage 2's fail-closed route.

The raw linear SVC does not converge on MADELON, GINA or GISETTE (feature
scale), so those raw SVM fits are flagged rather than reported as results; the
report says so explicitly. Read `docs/stage3a_summary.md` and
`docs/stage3a_evaluation.md` for the numbers, gates and limitations.
