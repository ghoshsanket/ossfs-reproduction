# Stage 3D V2 Comparison — Manual Runbook

**The user runs every command. OpenCode only wrote the code.**

## Prerequisites

```bash
cd /home/iemiedc2026/Documents/Sanket/Research
export RESEARCH_PROJECT_NAME=ossfs-dd-reproduction
source scripts/research-env.sh
cd "$RESEARCH_PROJECT_ROOT"
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
RUN_ID="v2cmp_$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$RESEARCH_OUTPUT_ROOT/ossfs-dd-reproduction/stage3d_v2/comparison_${RUN_ID}"
```

## Phase 1: prepare

```bash
python scripts/stage3d_v2_comparison.py prepare \
  --config configs/stage3d_v2_comparison_v1.yaml --run-id "$RUN_ID"
```

## Phase 2: tests and lint

```bash
python -m pytest -o addopts='' -q > "$RUN_DIR/pytest_preflight.log" 2>&1
printf '%s\n' "$?" > "$RUN_DIR/pytest_preflight.exit"
ruff check . > "$RUN_DIR/ruff_preflight.log" 2>&1
printf '%s\n' "$?" > "$RUN_DIR/ruff_preflight.exit"
```

**Do NOT continue to preflight if either exit is nonzero.**

## Phase 3: preflight

```bash
python scripts/stage3d_v2_comparison.py preflight --run-id "$RUN_ID"
```

## Phase 4: smoke

```bash
python scripts/stage3d_v2_comparison.py smoke --run-id "$RUN_ID"
python scripts/stage3d_v2_comparison.py status --run-id "$RUN_ID"
```

## Phase 5: evaluate

```bash
python scripts/stage3d_v2_comparison.py evaluate \
  --run-id "$RUN_ID" --authorize-full-evaluation
```

Resume: `--resume`

## Phase 6: audit and report

```bash
python scripts/stage3d_v2_comparison_audit.py --run-id "$RUN_ID"
python scripts/stage3d_v2_comparison_report.py --run-id "$RUN_ID"
python scripts/stage3d_v2_comparison.py status --run-id "$RUN_ID"
```

## Common blockers

| Blocker | Fix |
|---|---|
| preflight exit 1 | Fix the test/lint failure |
| V1 comparator BLOCKED | Check `$RESEARCH_OUTPUT_ROOT/.../stage3d/20261102T042840Z/raw_records/` |
| evaluate REFUSED | Add `--authorize-full-evaluation` |
| prepare aborts | Choose a new RUN_ID |

## Interpreting outcomes

**`V2_PERFORMANCE_NOT_MEASURED`** — this pipeline presents operational integrity
and descriptive comparisons only.
