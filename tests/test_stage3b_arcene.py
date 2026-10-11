"""Experiment B ARCENE crossed-design tests — SYNTHETIC inputs only."""

from __future__ import annotations

import numpy as np
import pytest

from ossfs_dd_stage3b.analysis import b_h1_outcome, b_h2_outcome, jaccard
from ossfs_dd_stage3b.arcene_experiment import (
    run_arcene_selection,
    verify_event_positions,
    verify_selected_ids,
)
from ossfs_dd_stage3b.protocol import ArceneRunID
from ossfs_dd_stage3b.trace import (
    checkpoint_rows,
    event_row,
    recompute_thresholds,
)


def _tiny_stream(n=60, p=6, seed=0):
    rng = np.random.default_rng(seed)
    y = np.where(np.arange(n) < n // 2, "a", "b")
    sig = (y == "a").astype(float)
    cols = [sig * 2 + rng.normal(0, 0.5, n) for _ in range(2)]
    cols += [rng.normal(0, 1, n) for _ in range(p - 2)]
    return np.column_stack(cols), y


def test_crossed_design_changes_order_but_holds_train_rows_fixed():
    X, y = _tiny_stream()
    train = np.arange(40)
    orders = [np.arange(X.shape[1]), np.arange(X.shape[1])[::-1]]
    results = []
    for perm in orders:
        results.append(run_arcene_selection(ArceneRunID(0, 11), X, y, train,
                                            np.asarray(perm),
                                            k1=1.0, k2=2.0, pool_size=5))
    # identical training rows used for both
    assert all(r["n_train"] == 40 for r in results)
    assert all(r["test_rows_read"] == 0 for r in results)
    assert all(r["classifier_fitted"] is False for r in results)


def test_order_seed_does_not_select_the_split():
    """Two different order seeds on the SAME fold must use the same split."""
    X, y = _tiny_stream()
    train = np.arange(40)
    a = run_arcene_selection(ArceneRunID(2, 11), X, y, train,
                             np.arange(X.shape[1]), k1=1.0, k2=2.0, pool_size=5)
    b = run_arcene_selection(ArceneRunID(2, 109), X, y, train,
                             np.arange(X.shape[1])[::-1], k1=1.0, k2=2.0, pool_size=5)
    assert a["fold"] == b["fold"] == 2
    assert a["n_train"] == b["n_train"]
    assert a["order_seed"] == 11 and b["order_seed"] == 109


def test_selected_ids_unique_and_bounded():
    X, y = _tiny_stream()
    train = np.arange(40)
    rec = run_arcene_selection(ArceneRunID(0, 17), X, y, train,
                               np.arange(X.shape[1]), k1=1.0, k2=2.0, pool_size=5)
    check = verify_selected_ids(rec, X.shape[1])
    assert check["ids_unique"] and check["all_in_bounds"]
    assert check["ids_and_positions_parallel"] and check["ok"]


def test_selection_reads_no_test_rows():
    X, y = _tiny_stream()
    train = np.arange(30)
    rec = run_arcene_selection(ArceneRunID(0, 11), X, y, train,
                               np.arange(X.shape[1]), k1=1.0, k2=2.0, pool_size=5)
    assert rec["test_rows_read"] == 0
    assert rec["n_train"] == 30


def test_event_positions_match_the_arrival_permutation():
    from ossfs_dd.config import MetricFamily, Mode, SelectionConfig
    from ossfs_dd.selector import Selector

    X, y = _tiny_stream()
    train = np.arange(40)
    order = np.arange(X.shape[1])[::-1]
    from ossfs_dd_data.stream import FeatureStream

    stream = FeatureStream(X, y, train, order)
    sel = Selector(SelectionConfig(mode=Mode.MATLAB_COMPATIBLE, k1=1.0, k2=2.0, pool_size=5),
                   metric_family=MetricFamily.DISCRETE_INFORMATION,
                   labels=stream.train_labels, entry_file="mi2")
    res = sel.fit_stream(iter(stream))
    check = verify_event_positions(res.events, order)
    assert check["ok"] and check["n_mismatches"] == 0


def test_recompute_thresholds_matches_ddof1_convention():
    s = np.array([0.2, 0.4, 0.9, 0.1, 0.7])
    got = recompute_thresholds(s, 1.0, 2.0)
    assert got["mu"] == pytest.approx(s.mean())
    assert got["sigma"] == pytest.approx(s.std(ddof=1))
    assert got["alpha"] == pytest.approx(s.mean() + s.std(ddof=1))
    assert got["beta"] == pytest.approx(s.mean() + 2 * s.std(ddof=1))


def test_recompute_handles_degenerate_and_empty():
    assert recompute_thresholds(np.array([]), 1.0, 2.0)["mu"] is None
    single = recompute_thresholds(np.array([0.5]), 1.0, 2.0)
    assert single["sigma"] == 0.0


def test_event_row_is_json_safe_and_scalar_only():
    import json

    from ossfs_dd.config import MetricFamily, Mode, SelectionConfig
    from ossfs_dd.selector import Selector
    from ossfs_dd_data.stream import FeatureStream

    X, y = _tiny_stream()
    order = np.arange(X.shape[1])
    stream = FeatureStream(X, y, np.arange(40), order)
    sel = Selector(SelectionConfig(mode=Mode.MATLAB_COMPATIBLE, pool_size=5),
                   metric_family=MetricFamily.DISCRETE_INFORMATION,
                   labels=stream.train_labels, entry_file="mi2")
    res = sel.fit_stream(iter(stream))
    rows = [event_row(e) for e in res.events]
    # every row must serialize without NaN/Infinity literals
    blob = json.dumps(rows, allow_nan=False)
    assert "NaN" not in blob and "Infinity" not in blob
    for row in rows:
        assert isinstance(row["decision"], str)
        for v in row.values():
            assert isinstance(v, (str, int, float, bool, type(None), list))


def test_checkpoint_rows_verify_thresholds():
    from ossfs_dd.config import MetricFamily, Mode, SelectionConfig
    from ossfs_dd.selector import Selector
    from ossfs_dd_data.stream import FeatureStream

    X, y = _tiny_stream(n=140, p=120)
    order = np.arange(X.shape[1])
    stream = FeatureStream(X, y, np.arange(100), order)
    sel = Selector(SelectionConfig(mode=Mode.MATLAB_COMPATIBLE, pool_size=5),
                   metric_family=MetricFamily.DISCRETE_INFORMATION,
                   labels=stream.train_labels, entry_file="mi2")
    res = sel.fit_stream(iter(stream))
    rows = checkpoint_rows(res.events, list(sel.scores), 1.0, 2.0)
    assert rows, "expected at least one checkpoint on the pinned 100-grid"
    assert rows[0]["arrival_position"] == 100
    for row in rows:
        if row["alpha_matches_recompute"] is not None:
            assert row["alpha_matches_recompute"] is True
        if row["beta_matches_recompute"] is not None:
            assert row["beta_matches_recompute"] is True


# ------------------------------------------------------------- hypotheses


def test_jaccard_empty_conventions():
    assert jaccard(set(), set()) == 1.0
    assert jaccard(set(), {1}) == 0.0
    assert jaccard({1, 2}, {1, 2}) == 1.0
    assert jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3)


def _arcene_records(fold_counts):
    recs = []
    for fold, counts in fold_counts.items():
        for i, n in enumerate(counts):
            recs.append({
                "run_id": f"ARCENE_split11_fold{fold}_order{10 + i}",
                "dataset": "ARCENE", "split_seed": 11, "fold": fold,
                "order_seed": 10 + i, "n_selected": n,
                "counters": {"redundancy_removals": n // 2, "promoted_from_pool": n // 3,
                             "direct_accepts": n, "peak_candidates": n,
                             "pool_flushes": 2, "immediate_rejects": 10},
            })
    return recs


def test_b_h1_supported_when_three_folds_exceed_threshold():
    # folds 0-2 span >=500; folds 3-4 stay narrow, so the criterion is met
    # exactly at the preregistered minimum of three folds.
    counts = {
        0: [34, 1561] + [200] * 8,
        1: [100] + [1200] + [300] * 8,
        2: [40] + [900] + [250] * 8,
        3: [100 + i for i in range(10)],
        4: [100 + i for i in range(10)],
    }
    out = b_h1_outcome(_arcene_records(counts))
    assert out["overall"] == "SUPPORTED"
    assert out["n_folds_meeting_threshold"] >= 3


def test_b_h1_refuted_when_ranges_are_small():
    counts = {f: [100 + i for i in range(10)] for f in range(5)}
    out = b_h1_outcome(_arcene_records(counts))
    assert out["overall"] == "REFUTED"
    assert out["n_folds_meeting_threshold"] < 3


def test_b_h1_inconclusive_when_runs_missing():
    counts = {0: [34, 1561] + [200] * 8, 1: [100 + i for i in range(10)],
              2: [100 + i for i in range(10)], 3: [100 + i for i in range(10)],
              4: [100 + i for i in range(10)]}
    recs = _arcene_records(counts)
    recs = [r for r in recs if not (r["fold"] == 1 and r["order_seed"] == 19)]
    out = b_h1_outcome(recs)
    assert out["overall"] == "INCONCLUSIVE_MISSING_RUNS"


def test_b_h2_never_claims_a_mechanism():
    counts = {f: [100 + i * 7 for i in range(10)] for f in range(5)}
    out = b_h2_outcome(_arcene_records(counts))
    assert out["status"] in ("ASSOCIATION_OBSERVED_NOT_ISOLATED", "MECHANISM_UNRESOLVED")
    assert "does not establish" in out["caution"]
