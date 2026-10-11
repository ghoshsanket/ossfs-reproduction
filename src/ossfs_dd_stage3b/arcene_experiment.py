"""Experiment B: ARCENE feature-arrival-order sensitivity, split held fixed.

The crossed design is the whole point: the CV partition is **always** Stage 1
``seed_11.json`` fold ``f``, and only the arrival permutation varies across the
ten order seeds. ``order_seed`` selects a permutation and must never silently
select the matching CV split.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ossfs_dd.config import MetricFamily, Mode, SelectionConfig
from ossfs_dd.selector import Selector
from ossfs_dd_data.stream import FeatureStream

from .protocol import B_SPLIT_SEED, ArceneRunID

ANCHOR_ORDER_SEED = 11


def run_arcene_selection(run_id: ArceneRunID, X: Any, y: np.ndarray,
                         train_idx: np.ndarray, order: np.ndarray,
                         *, k1: float, k2: float, pool_size: int) -> dict[str, Any]:
    """One crossed (fold, order) selection. Train rows only; never a test row."""
    stream = FeatureStream(X, y, train_idx, order)
    labels = stream.train_labels

    sel_cfg = SelectionConfig(mode=Mode.MATLAB_COMPATIBLE, k1=k1, k2=k2,
                              pool_size=pool_size)
    t0 = time.perf_counter()
    selector = Selector(
        sel_cfg,
        metric_family=MetricFamily.DISCRETE_INFORMATION,
        labels=labels,
        entry_file="mi2",
    )
    result = selector.fit_stream(iter(stream))
    seconds = time.perf_counter() - t0

    counts = result.state.__dict__
    return {
        "schema": "ossfs-dd-reproduction/stage3b/arcene_run/1.0",
        "run_id": run_id.run_id,
        "dataset": "ARCENE",
        "split_seed": B_SPLIT_SEED,
        "fold": run_id.fold,
        "order_seed": run_id.order_seed,
        "is_anchor": run_id.order_seed == ANCHOR_ORDER_SEED,
        "mode": "matlab_compatible_v1",
        "entry_file": "mi2",
        "k1": k1, "k2": k2, "pool_size": pool_size,
        "n_train": int(train_idx.size),
        "n_features_total": int(X.shape[1]),
        "n_features_seen": int(result.n_features_seen),
        "n_selected": int(len(result.feature_indices)),
        "selected_feature_ids": list(result.feature_indices),
        "selected_arrival_positions": list(result.arrival_positions),
        "score_min": result.score_min,
        "score_max": result.score_max,
        "counters": {
            "direct_accepts": counts["direct_accepts"],
            "immediate_rejects": counts["immediate_rejects"],
            "deferred": counts["deferred"],
            "redundancy_removals": counts["redundancy_removals"],
            "promoted_from_pool": counts["promoted_from_pool"],
            "pool_flushes": counts["pool_flushes"],
            "joint_evaluations": counts["joint_evaluations"],
            "peak_candidates": counts["peak_candidates"],
            "peak_pool": counts["peak_pool"],
            "pending_pool_at_end": counts["pending_pool_at_end"],
        },
        "duplicate_final_ids": result.state.duplicate_final_ids,
        "warnings": list(result.warnings),
        "seconds": round(seconds, 4),
        "test_rows_read": 0,
        "classifier_fitted": False,
    }


def verify_event_positions(events: list[Any], order: np.ndarray) -> dict[str, Any]:
    """Check every arrival event against the selected arrival permutation."""
    bad: list[dict[str, Any]] = []
    arrivals = [e for e in events if e.decision != "pool_flush"]
    for i, ev in enumerate(arrivals):
        if i >= len(order):
            bad.append({"index": i, "reason": "more arrival events than order length"})
            break
        if ev.feature_id != int(order[i]):
            bad.append({"index": i, "expected": int(order[i]), "observed": ev.feature_id,
                        "reason": "feature_id does not match arrival permutation"})
        if ev.arrival_position != i:
            bad.append({"index": i, "expected_pos": i, "observed_pos": ev.arrival_position,
                        "reason": "arrival_position mismatch"})
        if len(bad) >= 5:
            break
    return {"n_arrivals_checked": len(arrivals), "n_mismatches": len(bad),
            "mismatches": bad[:5], "ok": not bad}


def verify_selected_ids(result_record: dict[str, Any], n_features: int) -> dict[str, Any]:
    ids = result_record["selected_feature_ids"]
    pos = result_record["selected_arrival_positions"]
    return {
        "n_selected": len(ids),
        "ids_unique": len(set(ids)) == len(ids),
        "positions_unique": len(set(pos)) == len(pos),
        "all_in_bounds": all(isinstance(i, int) and 0 <= i < n_features for i in ids),
        "ids_and_positions_parallel": len(ids) == len(pos),
        "ok": (len(set(ids)) == len(ids) and len(ids) == len(pos)
               and all(isinstance(i, int) and 0 <= i < n_features for i in ids)),
    }
