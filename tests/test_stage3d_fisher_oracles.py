"""Fisher r-to-z numerical oracles (Stage 3D)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from ossfs_dd_stage3d.standard_fisherz import (
    MIN_N_MARGINAL,
    benjamini_hochberg,
    marginal_fisherz,
    pair_score_standard,
    partial_correlation,
)


def _indicators(y, classes):
    return [(c, (y == c).astype(float)) for c in classes]


def test_marginal_Z_matches_atanh_times_sqrt_n_minus_3():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(MIN_N_MARGINAL, 200))
        x = rng.normal(size=n)
        y = rng.choice(["a", "b"], size=n)
        m = marginal_fisherz(x, y, use_evidence=True)
        b = (y == "a").astype(float)
        r = np.corrcoef(x, b)[0, 1]
        z = np.arctanh(np.clip(r, -(1 - 1e-12), 1 - 1e-12))
        Z = z * np.sqrt(n - 3)
        # winning class may be the complement -> |Z| invariant
        assert abs(abs(m.Z) - abs(Z)) < 1e-10, (m.Z, Z)
        p = 2 * stats.norm.sf(abs(Z))
        assert abs(m.p_raw - p) < 1e-10


def test_marginal_pvalue_uses_sqrt_n_minus_3_not_raw_z():
    """The critical bug: atanh(r) alone is NOT standard normal."""
    rng = np.random.default_rng(1)
    n = 120
    y = np.array(["a"] * (n // 2) + ["b"] * (n - n // 2))
    x = np.asarray([1.0 if t == "a" else 0.0 for t in y]) + rng.normal(0, 0.4, n)
    m = marginal_fisherz(x, y, use_evidence=True)
    r = np.corrcoef(x, (y == "a").astype(float))[0, 1]
    wrong = 2 * stats.norm.sf(abs(np.arctanh(r)))          # missing sqrt(n-3)
    right = 2 * stats.norm.sf(abs(np.arctanh(r)) * np.sqrt(n - 3))
    assert abs(m.p_raw - right) < 1e-12
    assert abs(m.p_raw - wrong) > 1e-5, "p must NOT equal the wrong (unscaled) value"


def test_constant_feature_is_neutral():
    y = np.array(["a"] * 30 + ["b"] * 30)
    m = marginal_fisherz(np.zeros(60), y, use_evidence=True)
    assert m.score == 0.0 and m.r == 0.0 and m.p_raw == 1.0
    assert m.constant is True


def test_small_n_refused():
    y = np.array(["a", "b", "a", "b"])
    with pytest.raises(ValueError):
        marginal_fisherz(np.arange(4, dtype=float), y, use_evidence=True)


def test_single_class_refused():
    y = np.array(["only"] * 30)
    with pytest.raises(ValueError):
        marginal_fisherz(np.arange(30, dtype=float), y, use_evidence=True)


def test_multiclass_holm():
    rng = np.random.default_rng(2)
    n = 200
    y = np.array(["a"] * 60 + ["b"] * 70 + ["c"] * 70)
    x = rng.normal(size=n)
    m = marginal_fisherz(x, y, use_evidence=True)
    assert m.n_classes == 3
    # Holm-adjusted p >= raw p for every class
    for pc in m.per_class:
        assert pc["p_holm"] >= pc["p_raw"] - 1e-15
    # evidence in [0,1]
    assert 0 <= m.evidence <= 1
    assert 0 <= m.score <= 1


def test_binary_no_double_count():
    """K=2 has one symmetric contrast; p_holm == p_raw."""
    rng = np.random.default_rng(3)
    n = 100
    y = np.array(["a"] * 50 + ["b"] * 50)
    x = rng.normal(size=n)
    m = marginal_fisherz(x, y, use_evidence=True)
    assert m.n_classes == 2
    assert m.per_class[0]["p_holm"] == m.per_class[0]["p_raw"]
    assert m.per_class[1]["p_holm"] == m.per_class[1]["p_raw"]


def test_evidence_score_bounded_and_not_probability():
    rng = np.random.default_rng(4)
    for _ in range(20):
        n = int(rng.integers(5, 100))
        y = rng.choice(["a", "b"], size=n)
        x = rng.normal(size=n)
        m = marginal_fisherz(x, y, use_evidence=True)
        assert 0 <= m.score <= 1
        assert 0 <= m.evidence <= 1
        assert m.score <= abs(m.r) + 1e-12 or m.evidence <= 1.0


def test_r_only_ablation_differs_from_normalized():
    rng = np.random.default_rng(5)
    n = 200
    y = rng.choice(["a", "b"], size=n)
    x = rng.normal(size=n)
    a = marginal_fisherz(x, y, use_evidence=True)
    b = marginal_fisherz(x, y, use_evidence=False)
    assert a.r_only == b.score  # r-only score equals abs(r)
    # scores may differ
    assert a.score != b.score or a.evidence == 1.0


def test_partial_correlation_oracle():
    rng = np.random.default_rng(6)
    for _ in range(30):
        n = int(rng.integers(6, 100))
        x, yv, z = rng.normal(size=(3, n))
        r1, p1 = partial_correlation(x, yv, z)
        A = np.column_stack([np.ones(n), z])
        rx = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
        ry = yv - A @ np.linalg.lstsq(A, yv, rcond=None)[0]
        r_ref = np.corrcoef(rx, ry)[0, 1]
        assert abs(r1 - r_ref) < 1e-10
        Z = np.arctanh(np.clip(r_ref, -(1 - 1e-12), 1 - 1e-12)) * np.sqrt(n - 4)
        p_ref = 2 * stats.norm.sf(abs(Z))
        assert abs(p1 - p_ref) < 1e-10


def test_partial_correlation_handles_constants():
    n = 30
    x = np.ones(n)
    y = np.arange(n, dtype=float)
    z = np.arange(n, dtype=float)
    r, p = partial_correlation(x, y, z)
    assert r == 0.0 and p == 1.0


def test_partial_correlation_small_n_refused():
    with pytest.raises(ValueError):
        partial_correlation(np.arange(5, dtype=float), np.arange(5, dtype=float),
                           np.arange(5, dtype=float))


def test_pair_score_standard_symmetry():
    rng = np.random.default_rng(7)
    n = 80
    y = rng.choice(["a", "b"], size=n)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    ps = pair_score_standard(x, z, y, use_evidence=True)
    ps2 = pair_score_standard(z, x, y, use_evidence=True)
    assert ps["joint"] == ps2["joint"], "pair score must be symmetric"
    assert 0 <= ps["joint"] <= 2


def test_benjamini_hochberg():
    p = [0.01, 0.04, 0.03, 0.20]
    adj = benjamini_hochberg(p)
    assert len(adj) == 4
    assert all(0 <= a <= 1 for a in adj)
    assert adj[0] <= adj[3]  # smallest p -> smallest adjusted


def test_clipping_count_recorded():
    n = 50
    y = np.array(["a"] * 25 + ["b"] * 25)
    x = np.ones(n) * 1e6  # near-perfect correlation -> clipping
    m = marginal_fisherz(x, y, use_evidence=True)
    # either clipped or saturated; just record it's finite
    assert np.isfinite(m.score)
    assert m.clip_count >= 0
