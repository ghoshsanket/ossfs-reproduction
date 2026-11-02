"""V2 lifecycle tests: cross-batch promotion, capacity, terminal flush, idempotence."""

from __future__ import annotations

import numpy as np
import pytest

from ossfs_dd.config import Mode, SelectionConfig
from ossfs_dd_stage3d.batch_selector_v2 import BatchSelectorV2


def _make(labels, scores, pool_size=100, k1=1.0, k2=2.0):
    it = iter(scores)
    def sf(col, lab): return next(it)
    def jf(a, b, lab, ma): return ma + 1.5, "x"
    bs = BatchSelectorV2(
        SelectionConfig(mode=Mode.MATLAB_COMPATIBLE, k1=k1, k2=k2, pool_size=pool_size),
        metric_family="x", score_fn=sf, joint_fn=jf,
        is_redundant_fn=lambda j, beta: (False, ""),
        promotes_pair_fn=lambda j, beta: (j > 2 * beta, ""),
        labels=labels, pool_size=pool_size, k1=k1, k2=k2)
    return bs


def test_v2_pool_persists_across_batches():
    labels = np.array(["a", "b"] * 50)
    bs = _make(labels, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0] + [0.05] * 90)
    bs.consume_batch(list(range(10)), [np.zeros(100) for _ in range(10)])
    assert len(bs.pool) == 2, f"expected 2 deferred in pool after batch1, got {len(bs.pool)}"
    assert bs.state.pool_flushes == 0
    bs.consume_batch(list(range(10, 100)), [np.zeros(100) for _ in range(90)])
    bs.finalize_stream()
    assert len(bs.pool) == 0
    assert bs.state.pool_flushes == 1


def test_v2_cross_batch_promotion_actually_occurs():
    """A deferred feature from batch A must be promoted via interaction with batch B."""
    labels = np.array(["a", "b"] * 50)
    # wide thresholds so 0.5 lands in the delay band; joint_fn makes joint > 2*beta
    scores = [0.85, 0.5] + [0.5] * 99
    bs = _make(labels, scores, pool_size=100, k1=-10.0, k2=10.0)
    bs.consume_batch([0, 1], [np.zeros(100) for _ in range(2)])
    deferred_after_b1 = len(bs.pool)
    bs.consume_batch(list(range(2, 101)), [np.zeros(100) for _ in range(99)])
    bs.finalize_stream()
    assert deferred_after_b1 > 0, "batch1 must defer at least one feature"
    assert bs.state.promoted_from_pool > 0, "cross-batch promotion must occur"
    assert bs.state.pool_flushes >= 1


def test_v2_capacity_flush_at_100():
    labels = np.array(["a", "b"] * 50)
    # 101 features with alternating scores -> all land in the delay band
    scores = [0.0 if i % 2 == 0 else 1.0 for i in range(101)]
    bs = _make(labels, scores, pool_size=100, k1=-10.0, k2=10.0)  # everything delays
    bs.consume_batch(list(range(101)), [np.zeros(100) for _ in range(101)])
    assert bs.state.pool_flushes >= 1, "capacity flush must fire at pool_size=100"
    assert bs.state.pending_pool_at_end <= 100


def test_v2_terminal_flush_exactly_once():
    labels = np.array(["a", "b"] * 50)
    scores = [0.5] * 5
    bs = _make(labels, scores, pool_size=100, k1=0.0, k2=10.0)
    bs.consume_batch(list(range(5)), [np.zeros(100) for _ in range(5)])
    bs.finalize_stream()
    f1 = bs.state.pool_flushes
    bs.finalize_stream()  # idempotent
    assert bs.state.pool_flushes == f1, "finalize_stream must be idempotent"
    assert len(bs.pool) == 0


def test_v2_consume_after_finalize_raises():
    labels = np.array(["a", "b"] * 50)
    scores = [0.5] * 5
    bs = _make(labels, scores, pool_size=100, k1=0.0, k2=10.0)
    bs.consume_batch(list(range(5)), [np.zeros(100) for _ in range(5)])
    bs.finalize_stream()
    with pytest.raises(RuntimeError, match="finalized|closed"):
        bs.consume_batch([99], [np.zeros(100)])


def test_v2_no_duplicate_final_ids():
    labels = np.array(["a", "b"] * 50)
    scores = [0.5] * 50
    bs = _make(labels, scores, pool_size=100, k1=0.0, k2=10.0)
    bs.consume_batch(list(range(50)), [np.zeros(100) for _ in range(50)])
    bs.finalize_stream()
    assert len(set(bs.candidates.feature_ids())) == len(bs.candidates.feature_ids())


def test_v2_moments_updated_once_per_score():
    labels = np.array(["a", "b"] * 50)
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    bs = _make(labels, scores, pool_size=100, k1=0.0, k2=10.0)
    bs.consume_batch(list(range(5)), [np.zeros(100) for _ in range(5)])
    # n_scores seen == 5
    assert len(bs.scores) == 5
    assert bs.state.arrival_count == 5


def test_v2_thresholds_frozen_within_batch():
    """alpha/beta computed once per batch, not per feature."""
    labels = np.array(["a", "b"] * 50)
    scores = [0.1, 0.9, 0.5, 0.3, 0.7]
    bs = _make(labels, scores, pool_size=100, k1=1.0, k2=2.0)
    bs.consume_batch(list(range(5)), [np.zeros(100) for _ in range(5)])
    # all events in this batch share the same alpha/beta
    alphas = {e["alpha"] for e in bs.state.events if "alpha" in e}
    betas = {e["beta"] for e in bs.state.events if "beta" in e}
    assert len(alphas) == 1, f"alpha varied within batch: {alphas}"
    assert len(betas) == 1, f"beta varied within batch: {betas}"


def test_v2_deterministic_rerun():
    labels = np.array(["a", "b"] * 50)
    scores = [0.3, 0.7, 0.1, 0.9, 0.5, 0.2, 0.8, 0.4, 0.6, 0.0]
    def run():
        bs = _make(labels, list(scores), pool_size=100, k1=1.0, k2=2.0)
        bs.consume_batch(list(range(10)), [np.zeros(100) for _ in range(10)])
        bs.finalize_stream()
        return list(bs.candidates.feature_ids()), bs.state.pool_flushes
    a = run()
    b = run()
    assert a == b, f"non-deterministic: {a} vs {b}"


def test_v2_boundary_batch_sizes():
    for p in (2000, 2001, 10000, 10001):
        from ossfs_dd_stage3d.scheduler import batch_size_for_p
        b = batch_size_for_p(p)
        assert b in (10, 25, 50)
        assert (p + b - 1) // b >= 1
