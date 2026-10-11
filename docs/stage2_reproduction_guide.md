# Stage 2 Reproduction Guide

Everything below runs inside the workspace contract and is CPU-only.

## 0. Environment

```bash
export RESEARCH_PROJECT_NAME=ossfs-dd-reproduction
cd "$RESEARCH_ROOT"; source scripts/research-env.sh; cd "$RESEARCH_PROJECT_ROOT"
uv sync --frozen
```

## 1. Command sequence (in order)

```bash
uv run python scripts/stage2_preflight.py --write   # exit 0
uv run python scripts/stage2_selfcheck.py            # exit 0
uv run pytest -q                                     # Stage 1 + Stage 2
uv run python scripts/stage2_run.py --all            # 9 executions
uv run python scripts/stage2_report.py               # final verdict
```

`stage2_run.py` is the only stage that touches dataset arrays, and it touches
only the nine authorized datasets.

## 2. Restricted authorization

`configs/stage2.yaml` carries an explicit allowlist of nine keys. Everything is
checked in `ossfs_dd.runner.authorize` and **fails closed**:

* a key not in the allowlist → refused;
* `DLBCL` / `BREAST` → refused, even when named explicitly on the command line;
* a manifest whose status is not `EXACT_VERIFIED` → refused;
* a Table-1 shape mismatch → refused;
* a processed matrix whose SHA-256 differs from the Stage-1 schema → refused.

```bash
uv run python scripts/stage2_run.py --dataset DLBCL   # exits 2, touches nothing
```

## 3. Determinism

The selector is deterministic: no RNG is used anywhere in `src/ossfs_dd/`.
`--seed` and `--fold` select among Stage-1's persisted plans; they do not
shuffle anything.

To confirm, run the same dataset twice in separate processes and compare:

```bash
uv run python scripts/stage2_run.py --dataset MADELON
uv run python scripts/stage2_run.py --dataset MADELON
```

`results/stage2/determinism_check.json` records the comparison. Timing and RSS
are excluded, as `configs/stage2.yaml` requires.

## 4. Expected result

Five of nine datasets complete (`ARCENE`, `DEXTER`, `MADELON`, `GINA`,
`GISETTE`). Four — `SRBCT`, `LYMPHOMA`, `PROSTATE`, `LEUKEMIA` — report
`UNSUPPORTED_UNVERIFIED_SEMANTICS` because the continuous helper
`Fisher_Ztest` is absent from the authors' repository. This is the expected,
honest outcome and is **not** a defect in the discrete implementation.

## 5. Output locations

| Artefact | Path (relative to `RESEARCH_ROOT`) |
|---|---|
| Run bundles (traces, per-dataset JSON) | `outputs/ossfs-dd-reproduction/stage2/runs/<run_id>/` |
| Logs | `outputs/ossfs-dd-reproduction/stage2/logs/` |
| Authors' MATLAB snapshots | `outputs/ossfs-dd-reproduction/stage2/reference_snapshots/` |
| Reports and ledgers | `projects/ossfs-dd-reproduction/results/stage2/` |

Run bundles stay **outside** Git. Only small JSON/CSV/Markdown summaries are
written into the repository.

## 6. Before Stage 3

Stage 3 needs separate authorization. At minimum it must resolve:

1. whether to implement the continuous route once `Fisher_Ztest` semantics can
   be obtained from the authors, or restrict Stage 3 to the five discrete
   datasets;
2. that the split seeds remain an explicit reproduction assumption;
3. that any preprocessing is fitted on training rows only, per fold.
