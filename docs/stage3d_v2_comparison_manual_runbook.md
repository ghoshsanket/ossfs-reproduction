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

## Phase 5: evaluate (230 selections) then classify (690 outcomes)

```bash
python scripts/stage3d_v2_comparison.py evaluate \
  --run-id "$RUN_ID" --authorize-full-evaluation
python scripts/stage3d_v2_comparison.py classify \
  --run-id "$RUN_ID" --authorize-classification
```

Both refuse to run unless `PREFLIGHT_PASS.json` and `SMOKE_PASS.json` exist and
say `PASS`. `classify` additionally refuses unless all 230 selection records
exist, are unique, and pass integrity checks (key, method_version, finalized,
split/schedule hashes, no duplicate IDs).

Resume an interrupted run of the SAME frozen run ID:

```bash
python scripts/stage3d_v2_comparison.py evaluate \
  --run-id "$RUN_ID" --authorize-full-evaluation --resume
python scripts/stage3d_v2_comparison.py classify \
  --run-id "$RUN_ID" --authorize-classification --resume
```

A corrupt or stale existing record aborts with `CORRUPT_RECORD` /
`STALE_CACHE`; start a NEW `RUN_ID` instead.

## Phase 6: audit and report (independent)

```bash
python scripts/stage3d_v2_comparison_audit.py --run-id "$RUN_ID"; echo "audit_exit=$?"
python scripts/stage3d_v2_comparison_report.py --run-id "$RUN_ID"
python scripts/stage3d_v2_comparison.py status --run-id "$RUN_ID"
```

- The auditor writes all diagnostic artifacts even on failure, but returns exit
  code `0` **only** for `COMPARISON_EVIDENCE_PASS`; any `PARTIAL`/`FAIL` result
  returns a nonzero exit code.
- Do **not** interpret a partial verdict, or a report generated after a partial
  audit, as valid scientific evidence. Check `audit_*/FINAL_VERDICT.json` and
  `audit_*/gate_results.json`.
- E2 now verifies real SHA-256 hashes of every frozen closure file from disk and
  verifies the original config hash against the retained `SOURCE_CONFIG.yaml`
  (never the post-stamped `FROZEN_PROTOCOL.yaml` bytes). A modified protected
  file must block the run; never re-freeze silently.

## HARD STOP CONDITIONS

Do not continue past a failing step:
- pytest or ruff exit nonzero → fix before `preflight`.
- `preflight` exits 1 → see `PREFLIGHT_BLOCKED.json` (exact V1 comparator
  coverage is required; counts alone are insufficient).
- `smoke` fails → see the smoke error.
- `evaluate`/`classify` prints `REFUSED` → a prerequisite is missing.
- `CORRUPT_RECORD` / `STALE_CACHE` → use a new `RUN_ID`.
- `CONTROL_EVIDENCE_BLOCKED` → required V1 comparator raw records/keys missing
  or duplicated.
- audit exits nonzero → inspect `audit_*/FINAL_VERDICT.json`,
  `gate_results.json`, `e2_hash_verification.json`, `paired_join_missing.json`.

## Interpreting outcomes

- **Operational verdict** (`gate_results.json` / auditor exit): 
  `COMPARISON_EVIDENCE_PASS` only when E1–E10 all pass and auditor exit is 0;
  otherwise `COMPARISON_EVIDENCE_PARTIAL` with nonzero exit.
- **Scientific hypotheses** (`independent_recompute.json`): `SUPPORTED` /
  `NOT_SUPPORTED` / `NOT_MEASURED`. A negative result is valid only when backed
  by complete measured evidence.
- **Report**: `V2_PERFORMANCE_MEASURED` only when valid held-out classifier
  records exist and audit evidence is complete; otherwise
  `V2_PERFORMANCE_NOT_MEASURED`. Report is not authoritative over the audit.

## Scientific caveats

- V1's premature batch-pool flush may have affected V1 results; V2 differs only
  in pool lifetime.
- The authors' private Fisher-Z helper is not reproduced.
- Fisher-Z p-values on categorical indicators are Gaussian-reference
  approximations, not exact finite-sample tests.
- Mean CART tolerance ≥ −0.03 is a descriptive criterion, not statistical
  proof of noninferiority.
- Runtime is not hardware-comparable across V1 and V2.
