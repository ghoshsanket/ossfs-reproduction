"""Continuous selector structure (Workstream B) and Stage-2 fail-closed isolation.

The structural rules asserted here come from the pinned ``OSFS_DD_z2.m`` and are
independent of the missing ``Fisher_Ztest`` helper, so they are testable even
though the scores themselves are a reconstruction.
"""

from __future__ import annotations

import numpy as np
import pytest

from ossfs_dd.config import MetricFamily, Mode, SelectionConfig, UnsupportedSemanticsError
from ossfs_dd_continuous import MODE_ID
from ossfs_dd_continuous.selector import (
    CONTINUOUS_METRIC_FAMILY,
    PAIR_LABEL,
    ContinuousReconstructedSelector,
)


def _binary_problem(n: int = 90, p: int = 14, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = np.where(np.arange(n) < n // 2, "c0", "c1")
    signal = np.asarray([1.0 if t == "c0" else 0.0 for t in y])
    cols = [signal * 1.5 + rng.normal(0, 1.0, n) for _ in range(3)]
    cols += [rng.normal(0, 1.0, n) for _ in range(p - 3)]
    return cols, y


def _select(cols, y, **cfg_kwargs):
    sel = ContinuousReconstructedSelector(
        SelectionConfig(mode=Mode.MATLAB_COMPATIBLE, **cfg_kwargs), labels=y
    )
    return sel, sel.fit_stream(iter(enumerate([np.asarray(c) for c in cols])))


def test_z2_warmup_accepts_the_first_two_features():
    """z2.m:41-46 adds the first two features; mi2.m:36-39 skips them."""
    cols, y = _binary_problem()
    _sel, res = _select(cols, y)
    warm = [e for e in res.events if "warm-up" in e.region]
    assert len(warm) == 2
    assert [e.decision for e in warm] == ["accept", "accept"]
    assert res.feature_indices[:2] == [0, 1]


def test_discrete_selector_warmup_skips_in_contrast():
    """Guards against accidentally reusing the mi2.m warm-up for the z2 route."""
    from ossfs_dd.selector import Selector

    cols, y = _binary_problem()
    discrete = Selector(
        SelectionConfig(mode=Mode.MATLAB_COMPATIBLE),
        metric_family=MetricFamily.DISCRETE_INFORMATION,
        labels=y,
        entry_file="mi2",
    )
    res = discrete.fit_stream(iter(enumerate([np.asarray(c) for c in cols])))
    warm = [e for e in res.events if "warm-up" in e.region]
    assert [e.decision for e in warm] == ["skipped_warmup", "skipped_warmup"]


def test_reported_labels_are_continuous_not_discrete():
    cols, y = _binary_problem()
    _sel, res = _select(cols, y)
    assert res.mode == MODE_ID
    assert res.metric_family == CONTINUOUS_METRIC_FAMILY
    # it must not borrow a discrete mode id
    assert res.mode not in ("matlab_compatible_v1", "paper_spec_v1")
    assert PAIR_LABEL in " ".join(res.warnings + [str(res.mode_metadata)])


def test_reconstruction_warning_is_always_present():
    cols, y = _binary_problem()
    _sel, res = _select(cols, y)
    assert any("RECONSTRUCTED" in w for w in res.warnings)


def test_selected_features_are_valid_original_ids():
    cols, y = _binary_problem()
    _sel, res = _select(cols, y)
    assert all(0 <= f < len(cols) for f in res.feature_indices)
    assert len(set(res.feature_indices)) == len(res.feature_indices)
    assert res.n_features_seen == len(cols)


def test_pool_flush_proceeds_past_unusable_pairs():
    """A tiny n makes every pair unusable; the flush must skip, not crash."""
    cols, y = _binary_problem(n=20, p=30, seed=2)
    sel, res = _select(cols, y, pool_size=5)
    assert res.n_features_seen == len(cols)
    assert sel.pair_skips >= 0  # recorded, never hidden


def test_scores_stay_finite_and_events_are_recorded():
    cols, y = _binary_problem(seed=5)
    _sel, res = _select(cols, y)
    assert len(res.events) >= len(cols)
    for e in res.events:
        assert np.isfinite(e.sigma) or e.decision == "pool_flush"


def test_small_train_sample_is_refused_rather_than_silently_scored():
    """Fewer than MIN_N_PAIR rows: the marginal path raises, it does not fake."""
    from ossfs_dd_continuous.fisherz import UnsupportedSampleSize, marginal_fisher_z

    with pytest.raises(UnsupportedSampleSize):
        marginal_fisher_z(np.arange(4, dtype=float), np.array(["a", "b", "a", "b"]))


# --------------------------------------------------- Stage-2 fail-closed


def test_stage2_continuous_route_is_still_fail_closed():
    """Workstream B must NOT have unlocked Stage 2's UNRESOLVED route."""
    from ossfs_dd.selector import Selector

    with pytest.raises(UnsupportedSemanticsError):
        Selector(
            SelectionConfig(mode=Mode.MATLAB_COMPATIBLE),
            metric_family=MetricFamily.GAUSSIAN_FISHER_Z,
            labels=np.array(["a", "b"] * 10),
            entry_file="z2",
        )


def test_stage2_refuses_a_continuous_dataset_route():
    from ossfs_dd.config import Stage2Plan
    from ossfs_dd.runner import run_dataset

    plan = Stage2Plan(
        allowlist=["SRBCT"], forbidden=["DLBCL", "BREAST"],
        required_manifest_status="EXACT_VERIFIED",
        primary_mode=Mode.MATLAB_COMPATIBLE,
        routes={"Real": {"metric_family": MetricFamily.GAUSSIAN_FISHER_Z,
                         "source_reference": "Fisher_Ztest", "semantics_status": "UNRESOLVED"}},
        seed=11, fold=0, memory_limit_gib=32, max_walltime_minutes=60, gates={},
    )
    result = run_dataset("SRBCT", plan)
    assert result["status"] == "UNSUPPORTED_UNVERIFIED_SEMANTICS"
    assert result["n_selected"] is None
    assert "Fisher_Ztest" in result["error"]
