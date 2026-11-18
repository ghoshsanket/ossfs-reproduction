"""Stage 3D V2 execution fixes: fail-before / pass-after negative regressions.

Written for the human to run. OpenCode executed nothing.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
AUDIT_PATH = PROJECT / "scripts" / "stage3d_v2_comparison_audit.py"
RUNNER_PATH = PROJECT / "scripts" / "stage3d_v2_comparison.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def audit():
    return _load(AUDIT_PATH, "v2_audit_under_test")


@pytest.fixture(scope="module")
def runner():
    return _load(RUNNER_PATH, "v2_runner_under_test")


# ---- A. frozen YAML parsed once, as YAML, with a clear diagnosis

def test_frozen_protocol_loader_is_yaml_only(audit):
    src = inspect.getsource(audit.load_frozen_protocol)
    assert "json.loads" not in src
    assert "safe_load" in src


def test_frozen_protocol_missing_raises_clear_error(audit, tmp_path):
    with pytest.raises(RuntimeError, match="FROZEN_PROTOCOL_MISSING"):
        audit.load_frozen_protocol(tmp_path / "nope.yaml")


def test_frozen_protocol_corrupt_raises_clear_error(audit, tmp_path):
    bad = tmp_path / "FROZEN_PROTOCOL.yaml"
    bad.write_text("a: 1\n  - broken indent\n: bad\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="FROZEN_PROTOCOL_UNREADABLE|FROZEN_PROTOCOL_INVALID"):
        audit.load_frozen_protocol(bad)
    # a plain scalar must also be rejected as invalid protocol
    scalar = tmp_path / "scalar.yaml"
    scalar.write_text("42", encoding="utf-8")
    with pytest.raises(RuntimeError, match="FROZEN_PROTOCOL_INVALID"):
        audit.load_frozen_protocol(scalar)


def test_frozen_protocol_roundtrips(audit, tmp_path):
    good = tmp_path / "FROZEN_PROTOCOL.yaml"
    good.write_text("a: 1\nb:\n  - x\n  - y\n", encoding="utf-8")
    assert audit.load_frozen_protocol(good) == {"a": 1, "b": ["x", "y"]}


# ---- B. CART pairing exact, wrong-arm early break caught

def _v1_cart(dataset, seed, fold, arm, acc=0.8, rid=None):
    return {"record_id": rid or f"{dataset}_seed{seed}_fold{fold}__{arm}__cart",
            "classifier": "cart", "accuracy": acc}


def _v2_cart(dataset, seed, fold, arm, acc=0.82):
    return {"record_id": f"{dataset}_seed{seed}_fold{fold}__{arm}__cart",
            "classifier": "cart", "accuracy": acc}


def test_arm_mapping_is_exact_not_replace(audit):
    assert audit.ARM_MAP == {"D2_V2": "D2", "D2_V2_pilot": "D2", "C2_V2": "C2", "C4_V2": "C4"}
    assert "D2_V2_pilot".replace("_V2", "") != audit.ARM_MAP["D2_V2_pilot"]


def test_cart_pairing_exact_one_to_one(audit):
    v1 = [_v1_cart("ARCENE", 11, 0, "D2"), _v1_cart("MADELON", 11, 0, "D2", acc=0.7)]
    v2 = [_v2_cart("ARCENE", 11, 0, "D2_V2"), _v2_cart("MADELON", 11, 0, "D2_V2_pilot")]
    paired, missing, dupes = audit.build_paired_cart(v2, v1)
    assert len(paired) == 2
    assert missing == [] and dupes == []
    assert all(p["signed_delta_v2_minus_v1"] > 0 for p in paired)


def test_cart_wrong_arm_is_not_used_as_control(audit):
    v1 = [_v1_cart("ARCENE", 11, 0, "D1", acc=0.99)]
    v2 = [_v2_cart("ARCENE", 11, 0, "D2_V2", acc=0.50)]
    paired, missing, _ = audit.build_paired_cart(v2, v1)
    assert paired == []
    assert missing and missing[0][1].startswith("no V1 D2")


def test_cart_pilot_maps_to_d2(audit):
    v1 = [_v1_cart("MADELON", 11, 0, "D2", acc=0.70)]
    v2 = [_v2_cart("MADELON", 11, 0, "D2_V2_pilot", acc=0.75)]
    paired, missing, _ = audit.build_paired_cart(v2, v1)
    assert len(paired) == 1 and paired[0]["v1_arm"] == "D2"
    assert abs(paired[0]["signed_delta_v2_minus_v1"] - 0.05) < 1e-12


def test_cart_duplicate_v1_key_is_flagged(audit):
    # Use two records with the SAME valid composite key to test duplication.
    v1 = [_v1_cart("ARCENE", 11, 0, "D2", acc=0.8),
          _v1_cart("ARCENE", 11, 0, "D2", acc=0.9)]
    v2 = [_v2_cart("ARCENE", 11, 0, "D2_V2")]
    _, _, dupes = audit.build_paired_cart(v2, v1)
    assert dupes, "duplicate V1 keys must be reported, not silently overwritten"


# ---- C. single-fit SVM

def test_runner_fits_svm_once(runner):
    src = inspect.getsource(runner.cmd_classify)
    assert "est2" not in src
    assert src.count("est.fit(") == 1


def test_runner_records_params_and_n_iter(runner):
    src = inspect.getsource(runner.cmd_classify)
    for token in ("estimator_params", "n_iter_vector", "converged",
                  "early_termination_warnings", "subset_ids_sha256"):
        assert token in src


# ---- D. evaluate/classify gated on preflight + smoke

def test_evaluate_requires_authorization(runner):
    args = type("A", (), {"authorize_full_evaluation": False, "run_id": "x", "resume": False})()
    with pytest.raises(SystemExit, match="authorize-full-evaluation"):
        runner.cmd_evaluate(args)


def test_classify_requires_authorization(runner):
    args = type("A", (), {"authorize_classification": False, "run_id": "x", "resume": False})()
    with pytest.raises(SystemExit, match="authorize-classification"):
        runner.cmd_classify(args)


def test_evaluate_refuses_without_preflight_and_smoke(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "OUT_ROOT", tmp_path / "ossfs-dd-reproduction" / "stage3d_v2")
    rd = tmp_path / "ossfs-dd-reproduction" / "stage3d_v2" / "comparison_x"
    rd.mkdir(parents=True)
    args = type("A", (), {"authorize_full_evaluation": True, "run_id": "x", "resume": False})()
    with pytest.raises(SystemExit, match="PREFLIGHT_PASS"):
        runner.cmd_evaluate(args)


def _minimal_frozen_protocol_for_a() -> str:
    """Minimal valid frozen protocol covering A_seed11_fold0__D2_V2 for resume tests."""
    return (
        "config_sha256: abc\n"
        "experiment:\n"
        "  classifiers: [cart]\n"
        "  arms:\n"
        "    D2_V2:\n"
        "      datasets: [A]\n"
        "      seeds: [11]\n"
        "      folds: [0]\n"
        "      metric: discrete_information\n"
    )


def test_classify_refuses_when_selections_missing(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "OUT_ROOT", tmp_path / "ossfs-dd-reproduction" / "stage3d_v2")
    rd = tmp_path / "ossfs-dd-reproduction" / "stage3d_v2" / "comparison_x"
    rd.mkdir(parents=True)
    (rd / "PREFLIGHT_PASS.json").write_text('{"status": "PASS"}', encoding="utf-8")
    (rd / "SMOKE_PASS.json").write_text('{"status": "PASS"}', encoding="utf-8")
    (rd / "comparison_key_manifest.json").write_text(
        json.dumps({"selection_keys": ["A_seed11_fold0__D2_V2"],
                    "classifier_keys": ["A_seed11_fold0__D2_V2__cart"]}), encoding="utf-8")
    (rd / "FROZEN_PROTOCOL.yaml").write_text(_minimal_frozen_protocol_for_a(), encoding="utf-8")
    args = type("A", (), {"authorize_classification": True, "run_id": "x", "resume": False})()
    with pytest.raises(SystemExit, match="selection records missing"):
        runner.cmd_classify(args)


def test_resume_rejects_corrupt_record(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "OUT_ROOT", tmp_path / "ossfs-dd-reproduction" / "stage3d_v2")
    rd = tmp_path / "ossfs-dd-reproduction" / "stage3d_v2" / "comparison_x"
    rd.mkdir(parents=True)
    (rd / "PREFLIGHT_PASS.json").write_text('{"status": "PASS"}', encoding="utf-8")
    (rd / "SMOKE_PASS.json").write_text('{"status": "PASS"}', encoding="utf-8")
    (rd / "comparison_key_manifest.json").write_text(
        json.dumps({"selection_keys": ["A_seed11_fold0__D2_V2"],
                    "classifier_keys": ["A_seed11_fold0__D2_V2__cart"]}), encoding="utf-8")
    (rd / "FROZEN_PROTOCOL.yaml").write_text(_minimal_frozen_protocol_for_a(), encoding="utf-8")
    sel = rd / "selection_records"
    sel.mkdir()
    (sel / "A_seed11_fold0__D2_V2.json").write_text("{not json", encoding="utf-8")
    args = type("A", (), {"authorize_full_evaluation": True, "run_id": "x", "resume": True})()
    with pytest.raises(SystemExit, match="CORRUPT_RECORD"):
        runner.cmd_evaluate(args)


def test_resume_rejects_wrong_method_version(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "OUT_ROOT", tmp_path / "ossfs-dd-reproduction" / "stage3d_v2")
    rd = tmp_path / "ossfs-dd-reproduction" / "stage3d_v2" / "comparison_x"
    rd.mkdir(parents=True)
    (rd / "PREFLIGHT_PASS.json").write_text('{"status": "PASS"}', encoding="utf-8")
    (rd / "SMOKE_PASS.json").write_text('{"status": "PASS"}', encoding="utf-8")
    (rd / "comparison_key_manifest.json").write_text(
        json.dumps({"selection_keys": ["A_seed11_fold0__D2_V2"], "classifier_keys": []}),
        encoding="utf-8")
    (rd / "FROZEN_PROTOCOL.yaml").write_text(_minimal_frozen_protocol_for_a(), encoding="utf-8")
    sel = rd / "selection_records"
    sel.mkdir()
    (sel / "A_seed11_fold0__D2_V2.json").write_text(json.dumps({
        "key": "A_seed11_fold0__D2_V2", "method_version": "batch_score_shared_v1",
        "finalized": True, "split_train_sha256": "x", "schedule_hash": "y",
        "selected_feature_ids": [0]}), encoding="utf-8")
    args = type("A", (), {"authorize_full_evaluation": True, "run_id": "x", "resume": True})()
    with pytest.raises(SystemExit, match="STALE_CACHE"):
        runner.cmd_evaluate(args)


# ---- E/F/G. auditor and report structure

def _complete_gate_evidence(audit):
    """A fully valid synthetic evidence fixture for evaluate_gates (production contract)."""
    sel_keys = {f"DS{i}_seed11_fold0__D2_V2" for i in range(230)}
    clf_keys = {f"{k}__{c}" for k in sel_keys for c in ("knn", "svm", "cart")}
    v1_sel_keys, v1_clf_keys = audit.expected_v1_keys_from_v2(sel_keys)
    v2_clf = []
    for k in sorted(sel_keys):
        for c in ("knn", "svm", "cart"):
            rec = {"record_id": f"{k}__{c}", "status": "COMPLETED", "classifier": c}
            v2_clf.append(rec)
    paired_cart = []
    for k in sorted(sel_keys):
        head, arm = k.rsplit("__", 1)
        ds = head.split("_seed")[0]
        v1_arm = audit.ARM_MAP[arm]
        paired_cart.append({
            "dataset": ds, "seed": 11, "fold": 0,
            "v2_arm": arm, "v1_arm": v1_arm,
            "v2_accuracy": 0.8, "v1_accuracy": 0.75,
            "signed_delta_v2_minus_v1": 0.05,
            "v2_status": "COMPLETED", "v1_status": "COMPLETED",
        })
    return {
        "sel_keys": sel_keys, "expected_sel": sel_keys,
        "clf_keys": clf_keys, "expected_clf": clf_keys,
        "sel_count": 230, "clf_count": 690,
        "v1_batch_sel_keys": v1_sel_keys, "v1_batch_clf_keys": v1_clf_keys,
        "v1_sel_count": len(v1_sel_keys), "v1_clf_count": len(v1_clf_keys),
        "cfg_ok": True, "e2_measured": "ok",
        "map_failures": [], "all_finalized": True,
        "oracle_ok": True, "e6_measured": "ok",
        "v2_clf": v2_clf,
        "v2_j": 0.5, "v1_j": 0.3,
        "paired_cart": paired_cart, "missing_pairs": [], "v1_cart_dupes": [],
        "expected_cart_pairs": 230, "arcene_complete": True,
        "fit_statuses": [r["status"] for r in v2_clf],
        "all_fits_completed": True,
        "pytest_exit": 0, "ruff_exit": 0, "preflight_pass": True,
    }


def test_audit_has_ten_gates_and_conjunction(audit):
    evidence = _complete_gate_evidence(audit)
    gates, verdict = audit.evaluate_gates(evidence)
    assert len(gates) == 10
    assert all(f"E{i}" in gates for i in range(1, 11))
    assert verdict == "COMPARISON_EVIDENCE_PASS"
    # mutate a mandatory input and re-evaluate: production verdict must become PARTIAL
    bad = dict(evidence)
    bad["oracle_ok"] = False
    gates2, verdict2 = audit.evaluate_gates(bad)
    assert gates2["E6"]["outcome"] == "FAIL"
    assert verdict2 == "COMPARISON_EVIDENCE_PARTIAL"


def test_fisher_oracle_is_real(audit):
    src = inspect.getsource(audit.main)
    assert "worst_Z" in src and "worst_p" in src and "marginal_fisherz" in src


def test_report_keeps_arms_distinct():
    rep = _load(PROJECT / "scripts" / "stage3d_v2_comparison_report.py", "rep_mod")
    src = inspect.getsource(rep.main)
    assert "V2 arm" in src and "nonconverged" in src