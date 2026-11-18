"""Narrow mutation/negative tests for Stage 3D V2 final audit contract.

These tests exercise real production logic on isolated synthetic fixtures.
No full study, no external commands.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
AUDIT_PATH = PROJECT / "scripts" / "stage3d_v2_comparison_audit.py"
RUNNER_PATH = PROJECT / "scripts" / "stage3d_v2_comparison.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def audit():
    return _load(AUDIT_PATH, "v2_audit_final_contract")


@pytest.fixture(scope="module")
def runner():
    return _load(RUNNER_PATH, "v2_runner_final_contract")


def _complete_evidence(audit_mod):
    sel_keys = {f"DS{i}_seed11_fold0__D2_V2" for i in range(230)}
    clf_keys = {f"{k}__{c}" for k in sel_keys for c in ("knn", "svm", "cart")}
    v1_sel_keys, v1_clf_keys = audit_mod.expected_v1_keys_from_v2(sel_keys)
    v2_clf = [{"record_id": f"{k}__{c}", "status": "COMPLETED", "classifier": c}
              for k in sorted(sel_keys) for c in ("knn", "svm", "cart")]
    paired_cart = []
    for k in sorted(sel_keys):
        head, arm = k.rsplit("__", 1)
        ds = head.split("_seed")[0]
        paired_cart.append({
            "dataset": ds, "seed": 11, "fold": 0,
            "v2_arm": arm, "v1_arm": audit_mod.ARM_MAP[arm],
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


def test_e3_fails_on_missing_required_v1_key_despite_230_unrelated(audit):
    ev = _complete_evidence(audit)
    # Replace one required V1 key with an unrelated key while keeping count at 230.
    required = sorted(ev["v1_batch_sel_keys"])
    unrelated = "UNRELATED_seed99_fold9__D2"
    v1_sel = set(ev["v1_batch_sel_keys"])
    v1_sel.remove(required[0])
    v1_sel.add(unrelated)
    ev["v1_batch_sel_keys"] = v1_sel
    ev["v1_sel_count"] = len(v1_sel)
    gates, verdict = audit.evaluate_gates(ev)
    assert gates["E3"]["outcome"] == "FAIL"
    assert verdict == "COMPARISON_EVIDENCE_PARTIAL"


def test_e2_fails_when_source_file_hash_changes(audit, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    src = tmp_path / "projsrc" / "src" / "demo.py"
    src.parent.mkdir(parents=True)
    src.write_text("print('ok')\n", encoding="utf-8")
    recorded = audit.sha256_file(src)
    (run_dir / "closure_hashes.json").write_text(
        json.dumps({"src/demo.py": recorded}), encoding="utf-8"
    )
    # retain grounded original config hash
    source_cfg = run_dir / "SOURCE_CONFIG.yaml"
    source_cfg.write_text("a: 1\n", encoding="utf-8")
    frozen = {"config_sha256": audit.sha256_file(source_cfg)}
    ok, measured, details = audit.verify_closure_hashes(run_dir, frozen, tmp_path / "projsrc")
    assert ok is True
    # mutate the protected source file
    src.write_text("print('tampered')\n", encoding="utf-8")
    ok2, measured2, details2 = audit.verify_closure_hashes(run_dir, frozen, tmp_path / "projsrc")
    assert ok2 is False
    assert details2["mismatches"]


def test_e8_fails_on_single_missing_cart_pair(audit):
    ev = _complete_evidence(audit)
    ev["paired_cart"] = ev["paired_cart"][:-1]
    ev["missing_pairs"] = [("DS229_seed11_fold0__D2_V2__cart", "no V1 D2 pair")]
    gates, verdict = audit.evaluate_gates(ev)
    assert gates["E8"]["outcome"] == "FAIL"
    assert verdict == "COMPARISON_EVIDENCE_PARTIAL"


def _runner_tmp(monkeypatch, runner, tmp_path, run_id="x"):
    monkeypatch.setattr(runner, "OUT_ROOT", tmp_path / "ossfs-dd-reproduction" / "stage3d_v2")
    rd = tmp_path / "ossfs-dd-reproduction" / "stage3d_v2" / f"comparison_{run_id}"
    rd.mkdir(parents=True)
    return rd


def _minimal_frozen_yaml():
    return (
        "config_sha256: abc\n"
        "experiment:\n"
        "  classifiers: [knn, svm, cart]\n"
        "  classifier_params:\n"
        "    knn: {n_neighbors: 5, weights: uniform, metric: minkowski, p: 2}\n"
        "    svm: {kernel: linear, C: 1.0, tol: 0.001, max_iter: 2000000, probability: false}\n"
        "    cart: {criterion: gini, splitter: best, random_state: 11}\n"
        "  arms:\n"
        "    D2_V2:\n"
        "      datasets: [A]\n"
        "      seeds: [11]\n"
        "      folds: [0]\n"
        "      metric: discrete_information\n"
    )


def test_incorrect_selection_hash_prevents_classifier_resume(runner, tmp_path, monkeypatch):
    rd = _runner_tmp(monkeypatch, runner, tmp_path)
    (rd / "PREFLIGHT_PASS.json").write_text('{"status":"PASS"}', encoding="utf-8")
    (rd / "SMOKE_PASS.json").write_text('{"status":"PASS"}', encoding="utf-8")
    sel_key = "A_seed11_fold0__D2_V2"
    clf_key = f"{sel_key}__cart"
    (rd / "comparison_key_manifest.json").write_text(
        json.dumps({"selection_keys": [sel_key], "classifier_keys": [clf_key]}),
        encoding="utf-8",
    )
    (rd / "FROZEN_PROTOCOL.yaml").write_text(_minimal_frozen_yaml(), encoding="utf-8")
    sel_dir = rd / "selection_records"
    sel_dir.mkdir()
    sel_rec = {
        "key": sel_key, "dataset": "A", "seed": 11, "fold": 0, "arm": "D2_V2",
        "method_version": "batch_score_shared_v2", "finalized": True,
        "split_train_sha256": "split", "schedule_hash": "sched",
        "selected_feature_ids": [1, 2, 3],
    }
    (sel_dir / f"{sel_key}.json").write_text(json.dumps(sel_rec), encoding="utf-8")
    clf_dir = rd / "classification_records"
    clf_dir.mkdir()
    bad = {
        "record_id": clf_key, "classifier": "cart", "status": "COMPLETED",
        "dataset": "A", "seed": 11, "fold": 0, "arm": "D2_V2",
        "method_version": "batch_score_shared_v2",
        "accuracy": 1.0, "balanced_accuracy": 1.0, "n_correct": 1, "n_test": 1,
        "n_selected": 3, "subset_ids_sha256": "WRONG_HASH",
        "split_train_sha256": "split", "schedule_hash": "sched",
        "estimator_params": {
            "ccp_alpha": 0.0, "class_weight": None, "criterion": "gini",
            "max_depth": None, "max_features": None, "max_leaf_nodes": None,
            "min_impurity_decrease": 0.0, "min_samples_leaf": 1,
            "min_samples_split": 2, "min_weight_fraction_leaf": 0.0,
            "monotonic_cst": None, "random_state": 11, "splitter": "best",
        },
    }
    (clf_dir / f"{clf_key}.json").write_text(json.dumps(bad), encoding="utf-8")
    args = type("A", (), {"authorize_classification": True, "run_id": "x", "resume": False})()
    with pytest.raises(SystemExit, match="subset_ids_sha256 mismatch|STALE_CACHE"):
        runner.cmd_classify(args)


def test_partial_verdict_returns_nonzero_exit_without_full_study(audit, tmp_path, monkeypatch):
    """Invoke auditor main() on a minimal synthetic run that must be PARTIAL."""
    run_id = "synthetic_partial"
    run_root = tmp_path / "out" / "ossfs-dd-reproduction" / "stage3d_v2"
    run_dir = run_root / f"comparison_{run_id}"
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(audit, "OUT_ROOT", run_root)
    monkeypatch.setattr(audit, "PROJECT", tmp_path / "proj")
    (tmp_path / "proj").mkdir()
    monkeypatch.setattr(audit, "V1_ROOT", tmp_path / "v1_missing")

    sel_keys = [f"DS{i}_seed11_fold0__D2_V2" for i in range(230)]
    clf_keys = [f"{k}__{c}" for k in sel_keys for c in ("knn", "svm", "cart")]
    (run_dir / "comparison_key_manifest.json").write_text(
        json.dumps({"selection_keys": sel_keys, "classifier_keys": clf_keys}),
        encoding="utf-8",
    )
    (run_dir / "FROZEN_PROTOCOL.yaml").write_text("config_sha256: x\n", encoding="utf-8")
    (run_dir / "closure_hashes.json").write_text("{}", encoding="utf-8")
    # no selection/classification records, no preflight/smoke/test logs => must be PARTIAL

    rc = audit.main.__wrapped__(type("NS", (), {"run_id": run_id})()) if hasattr(audit.main, "__wrapped__") else None
    # call through argparse-compatible path
    import sys as _sys
    old = _sys.argv
    try:
        _sys.argv = ["audit", "--run-id", run_id]
        rc = audit.main()
    finally:
        _sys.argv = old
    assert rc != 0
    verdict_path = tmp_path / "proj" / "results" / "stage3d_v2" / f"audit_{run_id}" / "FINAL_VERDICT.json"
    assert verdict_path.exists()
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    assert verdict["verdict"] == "COMPARISON_EVIDENCE_PARTIAL"
    assert verdict["v2_performance_status"] == "V2_PERFORMANCE_NOT_MEASURED"
