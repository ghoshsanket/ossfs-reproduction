"""H-AUDIT: every mutated fixture must be rejected by the PRODUCTION validators.

These tests build a genuinely valid synthetic evidence bundle, then corrupt it
one dimension at a time and call the same validators the audit entrypoint uses.
A test that merely asserts an absent file is absent does not qualify.
"""

from __future__ import annotations

import copy

import pytest

from ossfs_dd_stage3d.audit_v2_helpers import (
    ValidationFailure,
    derive_verdict,
    expected_classifier_keys,
    expected_selection_keys,
    selection_key,
    validate_classifier_linkage,
    validate_feature_ids,
    validate_finalized,
    validate_oracle,
    validate_schedule_hashes,
    validate_selection_keys,
    validate_v1_untouched,
)


def _valid_bundle() -> dict:
    """A complete, genuinely valid synthetic evidence bundle."""
    keys = sorted(expected_selection_keys())
    records = []
    for k in keys:
        ds, rest = k.split("_seed")
        seed_s, fold_arm = rest.split("_fold")
        fold_s, arm = fold_arm.split("__")
        seed, fold = int(seed_s), int(fold_s)
        records.append({
            "dataset": ds, "seed": int(seed), "fold": int(fold), "arm": arm,
            "n_selected": 2, "selected_feature_ids": [0, 1],
            "batch_size": 10 if arm in ("D2", "C2", "C4") else 1,
            "schedule_hashes": ({"flattened_order_sha256": "a" * 64}
                                if arm in ("D2", "C2", "C4") else None),
        })
    sel_keys = {selection_key(r) for r in records}
    classifiers = [{"record_id": f"{k}__{c}", "classifier": c}
                   for k in sorted(sel_keys) for c in ("knn", "cart", "svm")]
    return {"records": records, "sel_keys": sel_keys, "classifiers": classifiers,
            "oracle": {"oracle_ok": True, "import_ok": True, "import_error": None},
            "manifest": {"method_version": "batch_score_shared_v2", "finalized": True,
                         "pending_pool_end": 0},
            "v1_hashes": {"a.json": "x"}, "v1_hashes_after": {"a.json": "x"}}


def test_valid_bundle_passes_all_validators():
    b = _valid_bundle()
    assert validate_selection_keys(b["records"]) == b["sel_keys"]
    validate_schedule_hashes(b["records"])
    validate_feature_ids(b["records"][0], 10)
    validate_classifier_linkage(b["classifiers"], b["sel_keys"])
    validate_oracle(b["oracle"])
    validate_finalized(b["manifest"])
    validate_v1_untouched(b["v1_hashes"], b["v1_hashes_after"])


# --- mutation 1: missing selection record --------------------------------
def test_mutation_missing_selection_record_fails():
    b = _valid_bundle()
    b["records"] = b["records"][:-1]
    with pytest.raises(ValidationFailure, match="missing keys"):
        validate_selection_keys(b["records"])


# --- mutation 2: count-preserving substitution ---------------------------
def test_mutation_count_preserving_substitution_fails():
    """Swap one key for another so the count stays 460 but the set is wrong."""
    b = _valid_bundle()
    extra = {"dataset": "MADELON", "seed": 11, "fold": 0, "arm": "D1",
             "n_selected": 0, "selected_feature_ids": [], "batch_size": 1,
             "schedule_hashes": None}
    b["records"][-1] = extra
    assert len(b["records"]) == 460, "count must remain constant"
    with pytest.raises(ValidationFailure):
        validate_selection_keys(b["records"])


# --- mutation 3: duplicate classifier / baseline ID ----------------------
def test_mutation_duplicate_classifier_id_fails():
    b = _valid_bundle()
    b["classifiers"][-1] = copy.deepcopy(b["classifiers"][0])
    with pytest.raises(ValidationFailure, match="duplicate classifier"):
        validate_classifier_linkage(b["classifiers"], b["sel_keys"])


# --- mutation 4: erased schedule hash ------------------------------------
def test_mutation_missing_schedule_hash_fails():
    b = _valid_bundle()
    for r in b["records"]:
        if r["arm"] == "D2":
            r["schedule_hashes"] = None
            break
    with pytest.raises(ValidationFailure, match="missing schedule hash"):
        validate_schedule_hashes(b["records"])


# --- mutation 5: forbidden dataset ---------------------------------------
def test_mutation_forbidden_dataset_fails():
    b = _valid_bundle()
    b["records"][0]["dataset"] = "DLBCL"
    with pytest.raises(ValidationFailure):
        validate_selection_keys(b["records"])


# --- mutation 6: mutated selected IDs / count ----------------------------
def test_mutation_out_of_range_feature_id_fails():
    b = _valid_bundle()
    b["records"][0]["selected_feature_ids"] = [0, 10_000_000]
    with pytest.raises(ValidationFailure, match="out of range"):
        validate_feature_ids(b["records"][0], 10)


def test_mutation_n_selected_count_mismatch_fails():
    b = _valid_bundle()
    b["records"][0]["n_selected"] = 99
    with pytest.raises(ValidationFailure, match="n_selected"):
        validate_feature_ids(b["records"][0], 10)


def test_mutation_duplicate_selected_ids_fails():
    b = _valid_bundle()
    b["records"][0]["selected_feature_ids"] = [1, 1]
    b["records"][0]["n_selected"] = 2
    with pytest.raises(ValidationFailure, match="duplicate selected IDs"):
        validate_feature_ids(b["records"][0], 10)


# --- mutation 7: falsified oracle result ---------------------------------
def test_mutation_failed_oracle_fails():
    b = _valid_bundle()
    b["oracle"]["oracle_ok"] = False
    with pytest.raises(ValidationFailure, match="numerical oracle not passed"):
        validate_oracle(b["oracle"])


def test_mutation_broken_import_fails():
    b = _valid_bundle()
    b["oracle"]["import_ok"] = False
    b["oracle"]["import_error"] = "ModuleNotFoundError"
    with pytest.raises(ValidationFailure, match="import check failed"):
        validate_oracle(b["oracle"])


# --- mutation 8: changed finalized / method version ---------------------
def test_mutation_missing_finalization_fails():
    b = _valid_bundle()
    b["manifest"]["finalized"] = False
    with pytest.raises(ValidationFailure, match="finalized is not True"):
        validate_finalized(b["manifest"])


def test_mutation_v1_fallback_disguised_as_v2_fails():
    b = _valid_bundle()
    b["manifest"]["method_version"] = "batch_score_shared_v1"
    with pytest.raises(ValidationFailure, match="is not V2"):
        validate_finalized(b["manifest"])


def test_mutation_pending_pool_after_finalization_fails():
    b = _valid_bundle()
    b["manifest"]["pending_pool_end"] = 3
    with pytest.raises(ValidationFailure, match="pending pool"):
        validate_finalized(b["manifest"])


# --- mutation 9: altered V1 artifact -------------------------------------
def test_mutation_v1_artifact_changed_fails():
    b = _valid_bundle()
    b["v1_hashes_after"] = {"a.json": "y"}
    with pytest.raises(ValidationFailure, match="V1 artifacts changed"):
        validate_v1_untouched(b["v1_hashes"], b["v1_hashes_after"])


# --- mutation 10: auditor trying to write to original preflight ----------
def test_mutation_auditor_write_to_original_preflight_is_detected(tmp_path):
    """A preflight whose digest changes between before/after is corruption."""
    pre = tmp_path / "PREFLIGHT.json"
    pre.write_text('{"utc": "t0"}', encoding="utf-8")
    import hashlib
    before = hashlib.sha256(pre.read_bytes()).hexdigest()
    # an auditor that rewrites it is a violation
    pre.write_text('{"utc": "rewritten"}', encoding="utf-8")
    after = hashlib.sha256(pre.read_bytes()).hexdigest()
    with pytest.raises(ValidationFailure, match="V1 artifacts changed"):
        validate_v1_untouched({"PREFLIGHT.json": before}, {"PREFLIGHT.json": after})


# --- mutation 11: classifier referencing unknown selection ---------------
def test_mutation_broken_classifier_linkage_fails():
    b = _valid_bundle()
    b["classifiers"][0]["record_id"] = "NOPE_seed11_fold0__D1__knn"
    with pytest.raises(ValidationFailure, match="unknown selection"):
        validate_classifier_linkage(b["classifiers"], b["sel_keys"])


# --- verdict derivation ---------------------------------------------------
def test_verdict_is_conjunction_of_gates():
    gates = {f"G{i}": {"outcome": "PASS"} for i in range(1, 11)}
    assert derive_verdict(gates) == "PASS"
    gates["G2"]["outcome"] = "FAIL"
    assert derive_verdict(gates) == "PARTIAL"
    gates["G2"]["outcome"] = "UNKNOWN"
    assert derive_verdict(gates) == "PARTIAL"


def test_expected_key_counts_match_protocol():
    assert len(expected_selection_keys()) == 460
    assert len(expected_classifier_keys(expected_selection_keys())) == 1380
