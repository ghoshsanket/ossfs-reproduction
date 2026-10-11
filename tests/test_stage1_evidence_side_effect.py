"""Regression tests for the Stage 1 evidence-writing side effect (M6).

`tests/conftest.py` used to write ``results/stage1/test_results.json`` on every
pytest session, so any unrelated suite rewrote the Stage 1 gate artifact with
its own totals. These tests pin the opt-in behaviour and the preserved schema.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
STAGE1_RESULTS = PROJECT / "results" / "stage1" / "test_results.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def test_flag_is_registered_and_defaults_off():
    sys.path.insert(0, str(PROJECT / "tests"))
    import conftest

    captured: dict[str, object] = {}

    class Parser:
        def addoption(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

    conftest.pytest_addoption(Parser())
    assert conftest.WRITE_EVIDENCE_FLAG in captured["args"]
    assert captured["kwargs"]["action"] == "store_true"
    assert captured["kwargs"]["default"] is False


def test_evidence_not_requested_by_default():
    sys.path.insert(0, str(PROJECT / "tests"))
    import conftest

    class Config:
        def getoption(self, name, *a, **k):
            assert name == conftest.WRITE_EVIDENCE_FLAG
            return False

    assert conftest.stage1_evidence_requested(Config()) is False

    class ConfigOn(Config):
        def getoption(self, name, *a, **k):
            return True

    assert conftest.stage1_evidence_requested(ConfigOn()) is True


def test_sessionfinish_is_a_no_op_without_the_flag():
    """The hook must return before doing any work when the flag is absent."""
    sys.path.insert(0, str(PROJECT / "tests"))
    import conftest

    class Config:
        def getoption(self, name, *a, **k):
            return False

    class Session:
        config = Config()

    before = _sha256(STAGE1_RESULTS)
    conftest.pytest_sessionfinish(Session(), 0)
    assert _sha256(STAGE1_RESULTS) == before


@pytest.mark.slow
def test_ordinary_pytest_run_leaves_stage1_artifact_byte_identical(tmp_path):
    """A real pytest session without the flag must not touch the Stage 1 file."""
    before = _sha256(STAGE1_RESULTS)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "--collect-only", "-k", "zzz_no_such_test_name"],
        cwd=PROJECT, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode in (0, 5), proc.stdout[-2000:] + proc.stderr[-2000:]
    assert _sha256(STAGE1_RESULTS) == before, (
        "an ordinary pytest run rewrote the Stage 1 evidence file"
    )


def test_stage1_evidence_schema_is_preserved():
    """The opt-in path must still emit the schema gates.py validates."""
    payload = json.loads(STAGE1_RESULTS.read_text(encoding="utf-8"))
    for field in (
        "schema", "exitstatus", "passed", "failed", "skipped", "total",
        "required_named_tests", "missing_required_tests", "skipped_required_tests",
        "observed_test_names", "stream_leakage_tests", "stream_leakage_failures",
        "git_audit_status", "git_data_violations", "config_error", "evidence_complete",
    ):
        assert field in payload, f"Stage 1 evidence schema lost field {field!r}"
    assert payload["schema"] == "ossfs-dd-reproduction/stage1/testresults/2.0"
    assert payload["passed"] == 135, "committed Stage 1 evidence must remain the 135-test run"
    assert payload["failed"] == 0
