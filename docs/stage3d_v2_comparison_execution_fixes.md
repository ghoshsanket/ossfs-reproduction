# Stage 3D V2 comparison — execution fixes

All defects below were found by reading the uploaded snapshot. Nothing here was
executed by OpenCode. The owner runs every command.

## Defect A — fatal audit error: JSON parse of a YAML file

**Root cause.** `stage3d_v2_comparison_audit.py` called `json.loads()` on
`FROZEN_PROTOCOL.yaml` before `yaml.safe_load()`. YAML is not JSON, so this
raised `JSONDecodeError` on every run.

**Correction.** The JSON parse is deleted. `load_frozen_protocol()` loads the
file exactly once with `yaml.safe_load()` and raises a *named* error
(`FROZEN_PROTOCOL_MISSING` / `FROZEN_PROTOCOL_UNREADABLE`) instead of a raw
traceback.

## Defect B — CART pairing broke on the first same-dataset record

**Root cause.** The nested loop broke after the first same-dataset/seed/fold
CART record, which is often the wrong arm (e.g. D1 before D2), silently
discarding V1–V2 pairs.

**Correction.** Replaced with `build_paired_cart()`: an indexed, exact,
one-to-one join on `(dataset, seed, fold, mapped_arm)` using the explicit
`ARM_MAP`. Produces `paired_join_ledger.csv` and
`paired_join_missing.json`, and flags duplicate V1 keys.

**Arm mapping (exact, never `.replace('_V2','')`):**

| V2 arm | V1 arm |
|---|---|
| `D2_V2` | `D2` |
| `D2_V2_pilot` | `D2` |
| `C2_V2` | `C2` |
| `C4_V2` | `C4` |

## Defect C — each SVM was fitted twice

**Root cause.** `classify` fitted once for predictions, then fitted a second
`est2` purely to capture convergence warnings.

**Correction.** A single `.fit()` inside `warnings.catch_warnings(record=True)`.
The record now carries `estimator_params`, `n_iter_vector`,
`early_termination_warnings`, `converged`, `subset_ids_sha256`, `dataset`,
`seed`, `fold`, `arm` and `method_version`.

## Defect D — phases not gated

**Root cause.** `evaluate` and `classify` checked only their own authorization
flag; `--resume` did not verify record contents.

**Correction.** Both now require `PREFLIGHT_PASS.json` and `SMOKE_PASS.json`
with `status == PASS`. Resume verifies key, method_version, finalized,
split/schedule hashes and ID uniqueness, and fails with `CORRUPT_RECORD` /
`STALE_CACHE` rather than skipping.

## Defect E — gates were weak

- **E1** now requires exact 230 selection-key equality *and* exact 690
  classifier-key equality with zero duplicates.
- **E2** checks the config hash *and* a non-empty closure map.
- **E3** checks both V1 batch-arm selection keys (≥230) and classifier keys
  (≥690).
- **E4** uses the dataset's true dimensionality (no `10**9` default) and
  checks `n_features_total` matches.
- **E6** now calls the real `marginal_fisherz` over ≥50 distinct cases and
  compares actual `r`, `|Z|` and p with tolerances `1e-8` / `1e-10`.
- **E9** reads the owner's manual pytest/ruff logs and exit files; it launches
  **no** subprocess.

## Defect F — reporting and runbook

- The report now aggregates by `(dataset, V2 arm, classifier)` so C2 and C4
  stay distinct, and adds a convergence table.
- The runbook now includes `classify` between `evaluate` and `audit`, plus
  resume and hard-stop rules.
- The YAML header comment no longer claims `D2_V2_pilot` was removed.

## Negative regression tests added

`tests/test_stage3d_v2_execution_fixes.py` covers: YAML-only parsing, missing /
corrupt frozen protocol, exact arm mapping, wrong-arm CART rejection, pilot→D2
mapping, duplicate V1 keys, single SVM fit, required record fields, missing
preflight/smoke refusal, corrupt-record resume refusal, stale-method resume
refusal, ten-gate presence, conjunction verdict, real Fisher oracle, and
arm-distinct reporting.

## Output schema

- Selection record: `key, dataset, seed, fold, arm, method_version,
  n_selected, selected_feature_ids, n_batches, pool_flushes, finalized,
  selection_seconds, split_train_sha256, schedule_hash`.
- Classifier record: `record_id, classifier, status, dataset, seed, fold, arm,
  method_version, accuracy, balanced_accuracy, n_correct, n_test, n_selected,
  converged, early_termination_warnings, n_iter_vector, estimator_params,
  fit_seconds, subset_ids_sha256, split_train_sha256, schedule_hash`.

## Operator diagnoses

| Message | Meaning |
|---|---|
| `FROZEN_PROTOCOL_MISSING` / `_UNREADABLE` | run `prepare` again / fix the YAML |
| `CONTROL_EVIDENCE_BLOCKED` | V1 raw records absent |
| `CORRUPT_RECORD` | existing record unreadable or duplicate IDs |
| `STALE_CACHE` | record from a different method/config; new `RUN_ID` |
| `REFUSED` | prerequisite pass file missing |

## Scientific caveats

- V1's premature batch-pool flush may have affected V1 results.
- The authors' private Fisher-Z helper is not reproduced.
- Fisher-Z p-values on categorical indicators are Gaussian-reference
  approximations.
- The CART ≥ −0.03 tolerance is descriptive, not a noninferiority proof.
- Runtime is not hardware-comparable across V1 and V2.