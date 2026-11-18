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

## Final audit-contract patch (Stage3D_V2_Final_Audit_and_Test)

### A. Test-fixture repairs
- `tests/test_stage3d_v2_execution_fixes.py::test_resume_rejects_corrupt_record`,
  `test_resume_rejects_wrong_method_version`: now write a minimal **valid**
  frozen protocol (`experiment.arms.D2_V2` covering `A_seed11_fold0__D2_V2`)
  so production `cmd_evaluate()` can build arm mappings before reaching the
  target resume record; tests still assert `CORRUPT_RECORD` / `STALE_CACHE`.
- `tests/test_stage3d_v2_execution_fixes.py::test_audit_has_ten_gates_and_conjunction`:
  supplies a complete valid 690-record evidence fixture and asserts production
  verdicts by mutating mandatory inputs and re-calling
  `audit.evaluate_gates()` (no manual gate-dict rewriting).

### B. Evidence hash verification (E2)
- `scripts/stage3d_v2_comparison_audit.py::verify_closure_hashes`,
  `main`: recomputes SHA-256 for every frozen `closure_hashes.json` entry from
  disk, requires files to exist, requires exact recorded-to-measured equality,
  and verifies the original config hash against the grounded retained
  `SOURCE_CONFIG.yaml` (not post-stamped `FROZEN_PROTOCOL.yaml` bytes).
- `scripts/stage3d_v2_comparison.py::cmd_prepare`: retains `SOURCE_CONFIG.yaml`
  and expands closure to include runner, auditor, report, and protocol files.

### C. Exact comparator coverage (E3)
- `scripts/stage3d_v2_comparison_audit.py::expected_v1_keys_from_v2`,
  `evaluate_gates`, `main`: derives exact required V1 selection/classifier keys
  from each frozen V2 key via `ARM_MAP`, requires every expected key present,
  no duplicates, and reports missing/duplicate keys. Extra V1 records are not
  counted as meeting missing required keys.
- `scripts/stage3d_v2_comparison.py::cmd_prepare`, `cmd_preflight`: control
  evidence marked/blocked using exact required-key coverage (not counts).

### D. Complete paired evidence (E8)
- `scripts/stage3d_v2_comparison_audit.py::build_paired_cart`,
  `arcene_coverage_complete`, `evaluate_gates`, `main`: requires exact expected
  CART pair coverage (230), no unexpected duplicates, finite accuracies for
  measured claims, exact ARCENE within-fold seed/order coverage, and no
  `FIT_ERROR`/incomplete paired accuracy evidence. `FIT_ERROR` or
  `EMPTY_SELECTION` affecting required measured pairs leaves comparison
  incomplete/not measured (no fabricated accuracy).

### E. Process exit semantics
- `scripts/stage3d_v2_comparison_audit.py::main`: returns `0` only for
  `COMPARISON_EVIDENCE_PASS`; returns nonzero for `PARTIAL`/`FAIL` after
  writing all diagnostic artifacts.
- `scripts/stage3d_v2_comparison_report.py`, runbook: report/audit exit is not
  treated as valid evidence when verdict is partial.

### F. Classifier resume integrity
- `scripts/stage3d_v2_comparison.py::cmd_classify`: existing classifier records
  are revalidated against the current verified selection record and frozen
  config (`record_id`, dataset/seed/fold/arm, classifier, status, method
  version, `n_selected`, `subset_ids_sha256`, split/schedule hashes, estimator
  params/result schema). Cross-run/cross-subset stale records are rejected;
  completed validated records are not rerun.

### G. Negative-path mutation tests
- `tests/test_stage3d_v2_audit_final_contract.py`: synthetic fixtures
  demonstrate missing required V1 key despite 230 unrelated keys fails E3;
  changed source file fails E2 hash verification; one missing CART pair fails
  E8; incorrect selection hash prevents classifier resume; partial audit verdict
  produces nonzero process status without executing a full study.

## Defect-to-test traceability

| Defect | Fix (file::function) | Test(s) |
|---|---|---|
| Resume fixtures lacked valid frozen arms | `tests/test_stage3d_v2_execution_fixes.py::_minimal_frozen_protocol_for_a`, `test_resume_rejects_corrupt_record`, `test_resume_rejects_wrong_method_version` | same tests (user-run) |
| Gate-conjunction fixture incomplete / mutated returned gates | `tests/test_stage3d_v2_execution_fixes.py::_complete_gate_evidence`, `test_audit_has_ten_gates_and_conjunction` | same test + `tests/test_stage3d_v2_audit_final_contract.py::test_e3_fails_on_missing_required_v1_key_despite_230_unrelated` |
| E2 structural-only hash checks | `scripts/stage3d_v2_comparison_audit.py::verify_closure_hashes`, `main`; `scripts/stage3d_v2_comparison.py::cmd_prepare` | `tests/test_stage3d_v2_audit_final_contract.py::test_e2_fails_when_source_file_hash_changes` |
| E3 count-only comparator coverage | `scripts/stage3d_v2_comparison_audit.py::expected_v1_keys_from_v2`, `evaluate_gates`, `main`; `scripts/stage3d_v2_comparison.py::cmd_prepare`, `cmd_preflight` | `tests/test_stage3d_v2_audit_final_contract.py::test_e3_fails_on_missing_required_v1_key_despite_230_unrelated` |
| E8 accepted incomplete pairs | `scripts/stage3d_v2_comparison_audit.py::build_paired_cart`, `arcene_coverage_complete`, `evaluate_gates`, `main` | `tests/test_stage3d_v2_audit_final_contract.py::test_e8_fails_on_single_missing_cart_pair` |
| Auditor returned 0 on PARTIAL | `scripts/stage3d_v2_comparison_audit.py::main`; runbook/report interpretation notes | `tests/test_stage3d_v2_audit_final_contract.py::test_partial_verdict_returns_nonzero_exit_without_full_study` |
| Weak classifier resume validation | `scripts/stage3d_v2_comparison.py::cmd_classify` (`validate_existing`, `subset_hash`) | `tests/test_stage3d_v2_audit_final_contract.py::test_incorrect_selection_hash_prevents_classifier_resume` |

## Output schema

- Selection record: `key, dataset, seed, fold, arm, method_version,
  n_features_total, n_selected, selected_feature_ids, n_batches, pool_flushes,
  finalized, selection_seconds, split_train_sha256, schedule_hash`.
- Classifier record (COMPLETED): `record_id, classifier, status, dataset, seed,
  fold, arm, method_version, accuracy, balanced_accuracy, n_correct, n_test,
  n_selected, converged, early_termination_warnings, n_iter_vector,
  estimator_params, fit_seconds, subset_ids_sha256, split_train_sha256,
  schedule_hash`. Terminal non-measured states: `EMPTY_SELECTION` / `FIT_ERROR`
  with explicit status and no fabricated accuracy.

## Operator diagnoses

| Message | Meaning |
|---|---|
| `FROZEN_PROTOCOL_MISSING` / `_UNREADABLE` / `_INVALID` | run `prepare` again / fix the YAML |
| `CONTROL_EVIDENCE_BLOCKED` | required V1 comparator records missing or duplicated |
| `CORRUPT_RECORD` | existing record unreadable or internally inconsistent |
| `STALE_CACHE` | record from a different method/config/subset; new `RUN_ID` |
| `REFUSED` | prerequisite pass file or evidence missing |
| Audit exit nonzero | `COMPARISON_EVIDENCE_PARTIAL/FAIL`; inspect audit diagnostics |

## Scientific caveats

- V1's premature batch-pool flush may have affected V1 results.
- The authors' private Fisher-Z helper is not reproduced.
- Fisher-Z p-values on categorical indicators are Gaussian-reference
  approximations.
- The CART ≥ −0.03 tolerance is descriptive, not a noninferiority proof.
- Runtime is not hardware-comparable across V1 and V2.
- V2 performance remains unmeasured until the user runs the pipeline.