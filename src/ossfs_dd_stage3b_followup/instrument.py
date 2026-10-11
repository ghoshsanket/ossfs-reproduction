"""Chronological, scalar-only instrumentation of the Stage 2 selector.

Produces, for one ARCENE (fold, order) cell:

* ``events``      — one record per arrival plus every pool flush, replayable
                   without any held-out label;
* ``checkpoints`` — prefix snapshots at arrivals 100, 200, ... plus a terminal
                   summary taken **after** the final pool flush;
* an independent **operation ledger** that is built from candidate-list
  insertions and applied removals rather than from ``state`` counters, so the
  two can be reconciled against each other.

No feature vectors, patient rows, label arrays or matrices are emitted; every
value is a scalar. Nonfinite quantities become ``None`` with a reason, never a
``NaN``/``Infinity`` JSON token.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ossfs_dd.config import MetricFamily, Mode, SelectionConfig
from ossfs_dd.selector import Selector
from ossfs_dd_data.stream import FeatureStream
from ossfs_dd_stage3b.trace import checkpoint_rows, event_row

CHECKPOINTS = tuple(range(100, 10_001, 100))
TOLERANCE = 1e-10

ACCEPT = "accept"
REJECT = "reject"
DELAY = "delay"
WARMUP = "skipped_warmup"
POOL_FLUSH = "pool_flush"


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


class OperationLedger:
    """Candidate-list changes observed from events, independent of ``state``.

    Counts *list length*, not set size, so duplicates remain visible.
    """

    def __init__(self) -> None:
        self.candidate_ids: list[int] = []
        self.direct_inserts = 0
        self.pool_inserts = 0
        self.applied_removals: list[int] = []
        self.pool_flushes = 0
        self.promoted_ids: list[int] = []
        self.duplicate_insertions: list[int] = []

    def _insert(self, fid: int, *, from_pool: bool) -> None:
        if fid in self.candidate_ids:
            self.duplicate_insertions.append(fid)
        self.candidate_ids.append(fid)
        if from_pool:
            self.pool_inserts += 1
        else:
            self.direct_inserts += 1

    def apply_event(self, ev: Any) -> None:
        if ev.decision == POOL_FLUSH:
            self.pool_flushes += 1
            details = getattr(ev, "details", None) or {}
            for fid in details.get("promoted", []) or []:
                self._insert(int(fid), from_pool=True)
                self.promoted_ids.append(int(fid))
            return
        if ev.decision == WARMUP:
            # mi2.m warm-up SKIPS the first two arrivals (`continue` before any
            # insertion), so they must NOT be counted as candidate insertions.
            # This is discrete/mi2 semantics; the continuous z2 route accepts
            # them instead, which is why the two routes are not interchangeable.
            return
        if ev.decision == ACCEPT:
            self._insert(int(ev.feature_id), from_pool=False)
            details = getattr(ev, "details", None) or {}
            for fid in details.get("removed", []) or []:
                fid = int(fid)
                if fid in self.candidate_ids:
                    self.candidate_ids.remove(fid)
                self.applied_removals.append(fid)
            return
        # reject / delay add nothing to the candidate list

    def reconcile(self, selector_state: Any) -> dict[str, Any]:
        """Compare the event ledger against the selector's own counters."""
        list_len = len(self.candidate_ids)
        direct = self.direct_inserts
        pool = self.pool_inserts
        removed = len(self.applied_removals)
        expected = direct + pool - removed
        return {
            "ledger_candidate_list_length": list_len,
            "ledger_direct_inserts": direct,
            "ledger_pool_inserts": pool,
            "ledger_applied_removals": removed,
            "ledger_expected_length": expected,
            "ledger_identity_holds": bool(expected == list_len),
            "state_final_candidates": int(selector_state.final_candidates),
            "state_direct_accepts": int(selector_state.direct_accepts),
            "state_promoted_from_pool": int(selector_state.promoted_from_pool),
            "state_redundancy_removals": int(selector_state.redundancy_removals),
            "state_matches_ledger": bool(
                int(selector_state.final_candidates) == list_len
                and int(selector_state.direct_accepts) == direct
                and int(selector_state.promoted_from_pool) == pool
                and int(selector_state.redundancy_removals) == removed
            ),
            "duplicate_insertions": self.duplicate_insertions,
            "n_duplicate_insertions": len(self.duplicate_insertions),
        }


def run_cell(*, X: Any, y: np.ndarray, train_idx: np.ndarray, order: np.ndarray,
             fold: int, order_seed: int, split_seed: int,
             pool_size: int = 100) -> dict[str, Any]:
    """Run one crossed ARCENE cell and return scalar-only instrumentation."""
    stream = FeatureStream(X, y, train_idx, order)
    selector = Selector(
        SelectionConfig(mode=Mode.MATLAB_COMPATIBLE, k1=1.0, k2=2.0, pool_size=pool_size),
        metric_family=MetricFamily.DISCRETE_INFORMATION,
        labels=stream.train_labels,
        entry_file="mi2",
    )
    result = selector.fit_stream(iter(stream))

    ledger = OperationLedger()
    for ev in result.events:
        ledger.apply_event(ev)

    events = [event_row(ev) for ev in result.events]
    arrivals = [e for e in events if e["kind"] == "arrival"]
    flushes = [e for e in events if e["kind"] == "pool_flush"]

    # attach candidate/pool occupancy to each arrival, replayed chronologically
    cand_len = 0
    pool_occ = 0
    for row in events:
        if row["kind"] == "pool_flush":
            pool_occ = int(row.get("pool_size") or 0)
            row["candidate_count"] = cand_len
            continue
        if row["decision"] in (WARMUP, ACCEPT):
            cand_len += 1
        row["candidate_count"] = cand_len
        if row["decision"] in (DELAY,):
            pool_occ += 1
        row["pool_occupancy"] = pool_occ

    checkpoints = checkpoint_rows(result.events, list(selector.scores), 1.0, 2.0)
    terminal = next((c for c in checkpoints if c.get("is_terminal")), None)
    arrival_checkpoints = [c for c in checkpoints if not c.get("is_terminal")]

    return {
        "run_id": f"ARCENE_split{split_seed}_fold{fold}_order{order_seed}",
        "split_seed": split_seed,
        "fold": fold,
        "order_seed": order_seed,
        "n_train": int(train_idx.size),
        "n_features_total": int(X.shape[1]),
        "n_features_seen": int(result.n_features_seen),
        "n_selected": int(len(result.feature_indices)),
        "selected_feature_ids": list(result.feature_indices),
        "selected_arrival_positions": list(result.arrival_positions),
        "counters": {
            "direct_accepts": int(result.state.direct_accepts),
            "immediate_rejects": int(result.state.immediate_rejects),
            "deferred": int(result.state.deferred),
            "redundancy_removals": int(result.state.redundancy_removals),
            "promoted_from_pool": int(result.state.promoted_from_pool),
            "pool_flushes": int(result.state.pool_flushes),
            "joint_evaluations": int(result.state.joint_evaluations),
            "peak_candidates": int(result.state.peak_candidates),
            "peak_pool": int(result.state.peak_pool),
            "final_candidates": int(result.state.final_candidates),
            "duplicate_final_ids": list(result.state.duplicate_final_ids),
        },
        "ledger": ledger.reconcile(result.state),
        "trace_summary": {
            "n_arrival_events": len(arrivals),
            "n_pool_flush_events": len(flushes),
            "n_checkpoints": len(arrival_checkpoints),
            "has_terminal_checkpoint": terminal is not None,
            "terminal_candidate_count": (
                terminal["cumulative"].get("accept") if terminal else None
            ),
        },
        "test_rows_read": 0,
        "classifier_fitted": False,
    }


def threshold_oracle(checkpoints: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise the 1e-10 threshold-recomputation agreement."""
    checked = 0
    mismatch = 0
    for cp in checkpoints:
        for key in ("alpha_matches_recompute", "beta_matches_recompute"):
            if cp.get(key) is None:
                continue
            checked += 1
            if cp[key] is False:
                mismatch += 1
    return {"n_checked": checked, "n_mismatch": mismatch,
            "tolerance": TOLERANCE, "ok": mismatch == 0}


def divergence_bin(pair_a: dict, pair_b: dict, *, threshold: int = 100) -> dict[str, Any]:
    """First predeclared 100-arrival bin where the candidate-count gap > threshold."""
    a = {c["arrival_position"]: c for c in pair_a}
    b = {c["arrival_position"]: c for c in pair_b}
    for pos in sorted(set(a) & set(b)):
        gap = abs(a[pos]["cumulative"]["accept"] - b[pos]["cumulative"]["accept"])
        if gap > threshold:
            return {"bin": pos, "gap": gap, "outcome": "CROSSING"}
    return {"bin": None, "gap": None, "outcome": "NO_CROSSING"}
