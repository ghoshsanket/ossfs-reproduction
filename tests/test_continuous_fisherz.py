"""The reconstructed Fisher-Z estimator (Workstream B).

These tests verify the *declared* reconstruction matches its own written
specification and a trusted oracle. They never claim equivalence to the
authors' absent ``Fisher_Ztest``; that claim is impossible to test.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from ossfs_dd_continuous import ALLOWLIST, FORBIDDEN, MODE_ID, SEMANTICS_STATUS
from ossfs_dd_continuous.authorization import ContinuousAuthError, authorize_continuous
from ossfs_dd_continuous.fisherz import (
    MIN_N_MARGINAL,
    MIN_N_PAIR,
    UnsupportedSampleSize,
    _bonferroni_q,
    marginal_fisher_z,
    pair_heuristic_fisher_z,
)


def _labels(n: int, k: int = 2) -> np.ndarray:
    return np.array([f"c{i % k}" for i in range(n)])


def test_mode_is_declared_reconstructed_not_source_verified():
    assert MODE_ID == "continuous_reconstructed_fisherz_v1"
    assert SEMANTICS_STATUS == "RECONSTRUCTED_NOT_SOURCE_VERIFIED"


def test_marginal_score_matches_declared_formula():
    rng = np.random.default_rng(0)
    n = 150
    y = _labels(n)
    x = np.asarray([1.0 if t == "c0" else 0.0 for t in y]) + rng.normal(0, 0.5, n)
    got = marginal_fisher_z(x, y)

    # independent recomputation of the declared formula
    q = stats.norm.ppf(1 - 0.05 / (2 * 2))
    best = 0.0
    for c in ("c0", "c1"):
        b = (y == c).astype(float)
        r = np.corrcoef(x, b)[0, 1]
        z = np.arctanh(r)
        se = 1 / np.sqrt(n - 3)
        best = max(best, np.tanh(max(0.0, abs(z) - q * se)))
    assert got.score == pytest.approx(best, rel=1e-12)


def test_marginal_pvalue_matches_scipy_fisher_statistic():
    """The two-sided p must use z*sqrt(n-3), not the raw z."""
    rng = np.random.default_rng(1)
    n = 120
    y = _labels(n)
    x = np.asarray([1.0 if t == "c0" else 0.0 for t in y]) + rng.normal(0, 0.4, n)
    got = marginal_fisher_z(x, y)
    r = np.corrcoef(x, (y == "c0").astype(float))[0, 1]
    ref = 2 * stats.norm.sf(abs(np.arctanh(r)) * np.sqrt(n - 3))
    assert got.p_two_sided == pytest.approx(ref, rel=1e-10)
    assert got.p_two_sided < 1e-20  # a strongly informative feature


def test_constant_feature_is_neutral_zero():
    y = _labels(60)
    assert marginal_fisher_z(np.zeros(60), y).score == 0.0
    assert marginal_fisher_z(np.ones(60), y).score == 0.0


def test_pure_noise_scores_near_zero():
    rng = np.random.default_rng(2)
    y = _labels(400)
    noise = marginal_fisher_z(rng.normal(0, 1, 400), y)
    assert noise.score < 0.05


def test_strong_signal_scores_high():
    n = 300
    y = _labels(n)
    x = np.asarray([1.0 if t == "c0" else 0.0 for t in y])
    assert marginal_fisher_z(x, y).score > 0.5


def test_multiclass_bonferroni_is_harder_than_binary():
    rng = np.random.default_rng(4)
    n = 400
    y3 = _labels(n, 3)
    x = np.asarray([1.0 if t == "c0" else 0.0 for t in y3]) + rng.normal(0, 1.0, n)
    y2 = np.asarray(["b" if t == "c0" else "a" for t in y3])
    assert marginal_fisher_z(x, y3).q > marginal_fisher_z(x, y2).q
    assert marginal_fisher_z(x, y3).score <= marginal_fisher_z(x, y2).score


def test_scores_always_in_unit_interval():
    rng = np.random.default_rng(5)
    for _ in range(20):
        n = int(rng.integers(MIN_N_MARGINAL, 200))
        y = _labels(n, 2)
        x = rng.normal(0, 1, n)
        assert 0.0 <= marginal_fisher_z(x, y).score <= 1.0


def test_small_sample_is_refused_not_silently_zeroed():
    y = _labels(MIN_N_MARGINAL - 1)
    with pytest.raises(UnsupportedSampleSize):
        marginal_fisher_z(np.arange(MIN_N_MARGINAL - 1, dtype=float), y)


def test_single_class_is_refused():
    y = np.array(["only"] * 30)
    with pytest.raises(ValueError):
        marginal_fisher_z(np.arange(30, dtype=float), y)


def test_bonferroni_q_requires_two_classes():
    with pytest.raises(ValueError):
        _bonferroni_q(1)


def test_rng_is_seed_reproducible():
    a = np.random.default_rng(9).normal(size=50)
    b = np.random.default_rng(9).normal(size=50)
    assert np.array_equal(a, b)


def test_pair_score_is_bounded_and_heuristic():
    rng = np.random.default_rng(6)
    n = 120
    y = _labels(n)
    a = np.asarray([1.0 if t == "c0" else 0.0 for t in y]) + rng.normal(0, 0.3, n)
    z = rng.normal(0, 1, n)
    got = pair_heuristic_fisher_z(a, z, y)
    assert 0.0 <= got.score <= 1.0
    assert got.status == "heuristic_joint_v1"
    assert np.isfinite(got.condition_number)
    # informative a should dominate a noise partner
    assert got.r_multiple > 0.5


def test_pair_score_detects_a_real_joint_relationship():
    """When both partners carry signal, the pair correlation rises."""
    rng = np.random.default_rng(7)
    n = 150
    y = _labels(n)
    signal = np.asarray([1.0 if t == "c0" else 0.0 for t in y])
    a = signal + rng.normal(0, 0.4, n)
    z = signal + rng.normal(0, 0.4, n)
    both = pair_heuristic_fisher_z(a, z, y)
    one = pair_heuristic_fisher_z(a, rng.normal(0, 1, n), y)
    assert both.r_multiple > one.r_multiple


def test_pair_small_sample_is_refused():
    y = _labels(MIN_N_PAIR - 1)
    x = np.arange(MIN_N_PAIR - 1, dtype=float)
    with pytest.raises(UnsupportedSampleSize):
        pair_heuristic_fisher_z(x, x, y)


def test_pair_handles_rank_deficient_columns():
    """Identical predictors are rank deficient; the result stays finite."""
    rng = np.random.default_rng(8)
    n = 60
    y = _labels(n)
    a = rng.normal(0, 1, n)
    got = pair_heuristic_fisher_z(a, a.copy(), y)
    assert np.isfinite(got.score) and 0.0 <= got.score <= 1.0


# ----------------------------------------------------------- authorization


@pytest.mark.parametrize("dataset", FORBIDDEN)
def test_continuous_workstream_refuses_forbidden(dataset):
    with pytest.raises(ContinuousAuthError):
        authorize_continuous(dataset, seed=11, fold=0)


@pytest.mark.parametrize("dataset", ["ARCENE", "MADELON", "GINA"])
def test_continuous_workstream_refuses_discrete_datasets(dataset):
    with pytest.raises(ContinuousAuthError):
        authorize_continuous(dataset, seed=11, fold=0)


@pytest.mark.parametrize("dataset", ALLOWLIST)
def test_continuous_allowlist_is_authorized(dataset):
    auth = authorize_continuous(dataset, seed=11, fold=0)
    assert auth.manifest_status == "EXACT_VERIFIED"
    assert auth.paper_feature_type == "Real"
