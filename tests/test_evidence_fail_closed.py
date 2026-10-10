"""Tests that the test-evidence and gate logic fail CLOSED.

The Stage 1 follow-up fixed a defect where ``conftest._git_data_violations()``
returned ``0`` whenever git was unavailable, the project was not a repository,
or a subprocess raised. That converted *missing evidence* into a *passing
gate*. These tests exercise the failure paths directly, because the happy path
proves nothing about them.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import conftest as stage1_conftest
from ossfs_dd_data.gates import evaluate_gates, load_test_results
from ossfs_dd_data.registry import load_stage1_config
from ossfs_dd_data.workspace import results_dir

# --------------------------------------------------------------------------
# git audit must never report 0 when it did not run
# --------------------------------------------------------------------------


def test_git_audit_is_fail_closed(tmp_path, monkeypatch):
    # conftest imports project_root by name, so patch it where it is used.
    monkeypatch.setattr(stage1_conftest, "project_root", lambda: tmp_path)
    out = stage1_conftest._git_data_violations()
    assert out["status"] == "unavailable", out
    assert out["violations"] is None, (
        "a git audit that could not run must report None (not measured), never 0"
    )


def test_git_audit_reports_error_when_git_raises(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise OSError("git is not installed")

    monkeypatch.setattr(stage1_conftest, "project_root", lambda: tmp_path)
    monkeypatch.setattr(subprocess, "run", boom)
    out = stage1_conftest._git_data_violations()
    assert out["status"] == "error", out
    assert out["violations"] is None, (
        "an errored git audit must report None (not measured), never 0"
    )


def test_git_audit_reports_error_when_dry_run_fails(tmp_path, monkeypatch):
    calls = {"n": 0}

    class R:
        returncode = 0
        stdout = "true\n"
        stderr = ""

    def fake_run(cmd, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return R()
        bad = R()
        bad.returncode = 128
        bad.stdout = ""
        bad.stderr = "fatal: not a git repository"
        return bad

    monkeypatch.setattr(stage1_conftest, "project_root", lambda: tmp_path)
    monkeypatch.setattr(subprocess, "run", fake_run)
    out = stage1_conftest._git_data_violations()
    assert out["status"] == "error", out
    assert out["violations"] is None


# --------------------------------------------------------------------------
# gates must not pass on absent evidence
# --------------------------------------------------------------------------


def _write_results(payload: dict):
    results_dir().mkdir(parents=True, exist_ok=True)
    (results_dir() / "test_results.json").write_text(json.dumps(payload), encoding="utf-8")


def _evaluate():
    cfg = load_stage1_config()
    return evaluate_gates(cfg, 11, 5, list(cfg["evaluation_plan"]["seeds"]))


def test_gate_fails_when_git_violations_are_unknown():
    _write_results({
        "passed": 54, "failed": 0, "skipped": 0,
        "stream_leakage_tests": ["a"], "stream_leakage_failures": [],
        "missing_required_tests": [], "skipped_required_tests": [],
        "config_error": None,
        "git_audit_status": "error", "git_data_violations": None,
        "git_audit_detail": "git is not installed",
    })
    try:
        g = _evaluate()
        gate = next(x for x in g["gates"] if x["name"] == "data_or_result_binaries_in_git")
        assert gate["pass"] is False, "gate passed on absent git evidence"
        assert gate["observed"] == "not measured", gate
        assert "not measured" in gate["detail"].lower()
    finally:
        (results_dir() / "test_results.json").unlink(missing_ok=True)


def test_gate_fails_when_a_required_test_is_skipped():
    _write_results({
        "passed": 40, "failed": 0, "skipped": 1,
        "stream_leakage_tests": ["a"], "stream_leakage_failures": [],
        "missing_required_tests": [], "skipped_required_tests": ["test_dataset_roundtrip"],
        "config_error": None,
        "git_audit_status": "ok", "git_data_violations": 0, "git_audit_detail": "clean",
    })
    try:
        g = _evaluate()
        gate = next(x for x in g["gates"] if x["name"] == "required_tests_passing")
        assert gate["pass"] is False, "a skipped required test was treated as a pass"
        assert "test_dataset_roundtrip" in gate["detail"]
    finally:
        (results_dir() / "test_results.json").unlink(missing_ok=True)


def test_gate_fails_when_config_could_not_be_loaded():
    _write_results({
        "passed": 54, "failed": 0, "skipped": 0,
        "stream_leakage_tests": ["a"], "stream_leakage_failures": [],
        "missing_required_tests": [], "skipped_required_tests": [],
        "config_error": "FileNotFoundError: configs/stage1.yaml",
        "git_audit_status": "ok", "git_data_violations": 0, "git_audit_detail": "clean",
    })
    try:
        g = _evaluate()
        gate = next(x for x in g["gates"] if x["name"] == "required_tests_passing")
        assert gate["pass"] is False, "an unloaded gate config was treated as a pass"
        assert "config_error" in gate["detail"]
    finally:
        (results_dir() / "test_results.json").unlink(missing_ok=True)


def test_gate_fails_when_results_file_is_absent():
    p = results_dir() / "test_results.json"
    backup = None
    if p.exists():
        backup = p.read_text(encoding="utf-8")
        p.unlink()
    try:
        assert load_test_results() is None
        g = _evaluate()
        for name in ("stream_leakage_tests_passing", "data_or_result_binaries_in_git",
                     "required_tests_passing"):
            gate = next(x for x in g["gates"] if x["name"] == name)
            assert gate["pass"] is False, f"{name} passed with no evidence file"
            assert gate["observed"] == "not measured"
    finally:
        if backup is not None:
            p.write_text(backup, encoding="utf-8")


def test_preregistered_gates_were_not_lowered():
    """The follow-up must not have relaxed any threshold to make things pass."""
    cfg = load_stage1_config()
    g = cfg["gates"]
    assert g["exact_verified_provenance"]["required"] == 11
    assert g["exact_dimension_and_label_match"]["required"] == 11
    assert g["inventory_attempted"]["required"] == 11
    assert g["unresolved_corrupt_rows_or_columns"]["max_allowed"] == 0
    assert g["unlabeled_challenge_test_examples_used"]["max_allowed"] == 0
    assert g["cv_test_folds_defined"]["required"] == 550
    assert g["feature_order_permutations_defined"]["required"] == 110


def test_git_helpers_are_unused_legacy_absent():
    """The v1 fail-open helper must be gone, not merely bypassed."""
    src = Path(__file__).resolve().parents[1] / "tests" / "conftest.py"
    text = src.read_text(encoding="utf-8")
    assert "_GIT_CACHE" not in text, "the v1 git-audit cache is still present"
    assert text.count("def _git_data_violations") == 1, "duplicate git audit helper"
