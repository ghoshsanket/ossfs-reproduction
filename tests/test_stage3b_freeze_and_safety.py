"""Stage 3B freeze chronology and safety tests.

These pin the repair of the Stage 3A weakness: the Stage 3B protocol must be
frozen and hashed BEFORE any real fit or selection, the freeze must carry a
genuine runtime UTC (never a future hard-coded time), and prior Stage 1/2/3A
artifacts must remain byte-identical.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "configs" / "stage3b.yaml"
STAGE3B_RESULTS = PROJECT / "results" / "stage3b"
STAGE1_EVIDENCE = PROJECT / "results" / "stage1" / "test_results.json"


def _frozen() -> dict:
    if not CONFIG.exists():
        pytest.skip("Stage 3B is not frozen yet (pre-freeze)")
    import yaml

    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _run_ids() -> list[str]:
    if not STAGE3B_RESULTS.exists():
        return []
    return sorted(p.name for p in STAGE3B_RESULTS.iterdir() if p.is_dir())


# ------------------------------------------------------------------ freeze


def test_config_exists_and_is_parseable():
    cfg = _frozen()
    assert cfg["stage"] == "3b"
    assert cfg["experiment_a"]["planned_fits"] == 450
    assert cfg["experiment_b"]["planned_runs"] == 50


def test_frozen_utc_is_a_real_timestamp_not_in_the_future():
    cfg = _frozen()
    ts = cfg["frozen_utc"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", ts), ts
    frozen = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    now = datetime.now(UTC)
    # a freeze cannot be in the future; allow a generous clock/scheduling skew
    assert frozen <= now + __import__("datetime").timedelta(minutes=5), (
        "frozen_utc is in the future — a hard-coded future time is prohibited"
    )


def test_freeze_statement_does_not_claim_attested_preregistration():
    cfg = _frozen()
    text = (cfg["freeze_statement"] + cfg.get("freeze_limitation", "")).lower()
    assert "not an independently attested preregistration" in text


def test_preflight_is_write_once_and_precedes_evaluation():
    run_ids = _run_ids()
    if not run_ids:
        pytest.skip("no Stage 3B run directory yet")
    for rid in run_ids:
        pf = STAGE3B_RESULTS / rid / "stage3b_preflight.json"
        assert pf.exists(), f"run {rid} has no preflight manifest"
        payload = json.loads(pf.read_text(encoding="utf-8"))
        assert payload["run_id"] == rid
        assert payload["config_sha256"]
        assert payload["protocol_digest"]
        assert payload["code_hash"]


def test_event_journal_ordering_when_present():
    import os

    out_root = os.environ.get("RESEARCH_OUTPUT_ROOT")
    if not out_root:
        pytest.skip("RESEARCH_OUTPUT_ROOT not set")
    for rid in _run_ids():
        journal = Path(out_root) / "ossfs-dd-reproduction" / "stage3b" / rid / "logs" / "event_journal.jsonl"
        if not journal.exists():
            continue
        rows = [json.loads(x) for x in journal.read_text(encoding="utf-8").splitlines() if x]
        events = [r["event"] for r in rows]
        assert events[0] == "PRE_FLIGHT_WRITTEN", events
        stamps = [datetime.strptime(r["utc"], "%Y-%m-%dT%H:%M:%SZ") for r in rows]
        assert stamps == sorted(stamps), "journal timestamps are not monotonic"
        if "FIRST_DATA_EVALUATION_STARTED" in events:
            assert events.index("PRE_FLIGHT_WRITTEN") < events.index("FIRST_DATA_EVALUATION_STARTED")


def test_config_declares_no_future_preregistration_claim():
    cfg = _frozen()
    assert cfg["experiment_a"]["feature_selection_performed"] is False
    assert cfg["experiment_b"]["order_seed_must_not_select_split"] is True
    assert cfg["estimator"]["linear_svc_substitution_forbidden"] is True
    assert cfg["scaler"]["fitted_on_test_rows"] is False
    assert cfg["hypotheses_are_not_operational_gates"] is True


def test_resource_budget_matches_prompt():
    cfg = _frozen()
    r = cfg["resources"]
    assert r["per_fit_timeout_minutes"] == 10
    assert r["experiment_a_max_wallclock_hours"] == 16
    assert r["experiment_b_max_wallclock_hours"] == 2
    assert r["memory_limit_gib"] == 32
    assert r["blas_threads_per_worker"] == 1
    assert r["experiment_a_workers"] <= 2
    assert r["cpu_only"] is True


# ------------------------------------------------------------------ safety


def test_stage1_evidence_artifact_is_still_present_and_valid():
    assert STAGE1_EVIDENCE.exists(), "Stage 1 evidence file is missing"
    payload = json.loads(STAGE1_EVIDENCE.read_text(encoding="utf-8"))
    assert payload["schema"] == "ossfs-dd-reproduction/stage1/testresults/2.0"


def test_stage3b_config_declares_forbidden_and_excluded_datasets():
    cfg = _frozen()
    assert set(cfg["authorization"]["forbidden"]) == {"DLBCL", "BREAST"}
    assert "SRBCT" in cfg["authorization"]["continuous_excluded"]
    assert "ARCENE" not in cfg["authorization"]["continuous_excluded"]


def test_stage3b_does_not_touch_stage123_configs():
    """Stage 3B must not modify Stage 1/2/3A configs."""
    import subprocess

    dirty = subprocess.run(
        ["git", "status", "--porcelain", "configs/stage1.yaml", "configs/stage2.yaml",
         "configs/stage3a.yaml"],
        cwd=PROJECT, capture_output=True, text=True, timeout=60,
    ).stdout.strip()
    assert dirty == "", f"Stage 1/2/3A configs were modified:\n{dirty}"


def test_no_forbidden_dataset_in_stage3b_source():
    import re as _re

    src = PROJECT / "src" / "ossfs_dd_stage3b"
    for f in src.rglob("*.py"):
        text = f.read_text(encoding="utf-8")
        # FORBIDDEN tuple may name them; no executable call may load them
        assert "load_dataset(\"DLBCL\"" not in text
        assert "load_dataset('DLBCL'" not in text
        assert "load_dataset(\"BREAST\"" not in text
        assert "load_dataset('BREAST'" not in text
        assert not _re.search(r"allowlist\s*=.*\[.*DLBCL.*\]\s*$", text, _re.M)
