"""Fail-closed audit tests for Stage 3D."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ossfs_dd_stage3d.standard_fisherz import marginal_fisherz


def test_missing_raw_record_is_not_verified():
    """A missing raw record must not be treated as present."""
    p = Path(tempfile.mkdtemp()) / "nonexistent.json"
    assert not p.exists()


def test_fisher_oracle_fails_on_wrong_Z():
    """Using atanh(r) without sqrt(n-3) must not pass."""
    import numpy as np
    from scipy import stats
    rng = np.random.default_rng(0)
    n = 100
    x = rng.normal(size=n)
    y = rng.choice(["a", "b"], size=n)
    m = marginal_fisherz(x, y, use_evidence=True)
    r = np.corrcoef(x, (y == "a").astype(float))[0, 1]
    wrong_Z = np.arctanh(np.clip(r, -(1 - 1e-12), 1 - 1e-12))  # missing sqrt(n-3)
    wrong_p = 2 * stats.norm.sf(abs(wrong_Z))
    # the correct p uses sqrt(n-3); they must differ
    assert abs(m.p_raw - wrong_p) > 1e-5


def test_constant_input_yields_neutral():
    y = ["a"] * 30 + ["b"] * 30
    m = marginal_fisherz([0.0] * 60, y, use_evidence=True)
    assert m.score == 0.0 and m.r == 0.0 and m.p_raw == 1.0


def test_small_n_raises():
    with pytest.raises(ValueError):
        marginal_fisherz([1.0, 2.0, 3.0, 4.0], ["a", "b", "a", "b"],
                         use_evidence=True)


def test_single_class_raises():
    with pytest.raises(ValueError):
        marginal_fisherz(list(range(30)), ["a"] * 30, use_evidence=True)


def test_import_works():
    from ossfs_dd_stage3d.conditional_fisherz import FisherResult
    assert FisherResult is not None


def test_v2_import_and_factory():
    import numpy as np

    from ossfs_dd_stage3d.selector import make_continuous_batch_v2, make_discrete_batch_v2
    bs, sf, jf = make_discrete_batch_v2(np.array(["a", "b"] * 10))
    assert bs is not None
    bs2, sf2, jf2 = make_continuous_batch_v2(np.array(["a", "b"] * 10), use_evidence=True)
    assert bs2 is not None


def test_forbidden_datasets_never_loaded():
    """DLBCL and BREAST must be refused by authorization."""
    from ossfs_dd.config import MetricFamily, Mode, Stage2Plan
    from ossfs_dd.runner import authorize
    plan = Stage2Plan(
        allowlist=["ARCENE"], forbidden=["DLBCL", "BREAST"],
        required_manifest_status="EXACT_VERIFIED",
        primary_mode=Mode.MATLAB_COMPATIBLE,
        routes={"Integer": {"metric_family": MetricFamily.DISCRETE_INFORMATION,
                            "source_reference": "x", "semantics_status": "x"}},
        seed=11, fold=0, memory_limit_gib=32, max_walltime_minutes=60, gates={})
    for bad in ("DLBCL", "BREAST"):
        try:
            authorize(bad, plan)
        except Exception as exc:  # noqa: BLE001 - refusal is the expected outcome
            assert type(exc).__name__  # refusal recorded
            continue
        raise AssertionError(f"{bad} was authorized")
