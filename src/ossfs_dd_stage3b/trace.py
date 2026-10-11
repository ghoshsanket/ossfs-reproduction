"""Scalar-only per-arrival traces and independently recomputed thresholds.

Selector decisions are never rewritten. The trace records what
``SelectionResult.events`` already emitted (arrival position, feature id,
relevance score, alpha, beta, sigma, decision, region) plus pool-flush details,
and adds checkpoint summaries derived from the observed score prefix.

Nonfinite values become ``null`` with a reason, never a nonstandard JSON NaN.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

CHECKPOINTS = tuple(range(100, 10_001, 100))
TOLERANCE = 1e-10


def _num(v: Any) -> float | None:
    """JSON-safe scalar: nonfinite becomes None."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def event_row(ev: Any) -> dict[str, Any]:
    """One arrival (or pool-flush) event, scalar fields only."""
    row: dict[str, Any] = {
        "kind": "pool_flush" if ev.decision == "pool_flush" else "arrival",
        "arrival_position": ev.arrival_position,
        "feature_id": ev.feature_id,
        "score": _num(ev.score),
        "decision": ev.decision,
        "region": ev.region,
        "alpha": _num(ev.alpha),
        "beta": _num(ev.beta),
        "sigma": _num(ev.sigma),
    }
    details = getattr(ev, "details", None) or {}
    if isinstance(details, dict):
        for key in ("removed", "promoted", "promoted_arrival_positions",
                    "unique_pairs", "pool_size", "pool_size_now", "candidates_now"):
            if key in details:
                val = details[key]
                row[key] = [int(v) for v in val] if isinstance(val, list) else (
                    _num(val) if isinstance(val, (int, float)) else val
                )
    if row["alpha"] is None or row["beta"] is None:
        row["threshold_state"] = "WARMUP_OR_DEFINITE_NAN"
    return row


def recompute_thresholds(scores: np.ndarray, k1: float, k2: float) -> dict[str, Any]:
    """Independently recompute mu/sigma and alpha/beta from a score prefix.

    Uses the selector's sample-SD convention (ddof=1). Returns nulls when the
    prefix is too short or degenerate, and never forces alpha < beta.
    """
    finite = np.asarray([s for s in scores if s is not None and math.isfinite(float(s))],
                        dtype=float)
    n = finite.size
    if n == 0:
        return {"mu": None, "sigma": None, "alpha": None, "beta": None,
                "n": 0, "reason": "no finite scores"}
    mu = float(finite.mean())
    sigma = float(finite.std(ddof=1)) if n > 1 else 0.0
    return {
        "mu": mu, "sigma": sigma,
        "alpha": mu + k1 * sigma, "beta": mu + k2 * sigma,
        "n": n, "reason": None,
    }


def checkpoint_rows(events: list[Any], all_scores: list[float], k1: float,
                    k2: float) -> list[dict[str, Any]]:
    """Chronological prefix checkpoints at 100, 200, ..., plus a terminal row.

    Diagnostic-only correction (Stage 3B follow-up D2).

    The previous implementation accumulated **every** event into ``cumulative``
    before entering the checkpoint loop, so the checkpoint at arrival 100
    reported decisions that had not happened yet -- in practice the full run
    totals. Events are now consumed in chronological order and the counters are
    snapshotted at each checkpoint's own prefix.

    Semantics fixed here:

    * a ``pool_flush`` event is **not** an arrival; it carries the sentinel
      position ``-1`` and is attributed to the arrival that triggered it
      (a flush appears in the stream immediately after its triggering arrival);
    * the post-stream **final flush** happens after the last arrival, so it is
      excluded from every arrival checkpoint and reported only in the terminal
      summary;
    * ``removed`` counts the applied-removal ledger reported by accept events
      (see the D1 repair in ``ossfs_dd.selector._redundancy_pass``).
    """
    out: list[dict[str, Any]] = []
    by_position: dict[int, Any] = {
        e.arrival_position: e for e in events if e.decision != "pool_flush"
    }
    cumulative = {"accept": 0, "reject": 0, "delay": 0, "pool_flush": 0,
                  "promoted": 0, "removed": 0}

    def apply(ev: Any) -> None:
        if ev.decision == "pool_flush":
            cumulative["pool_flush"] += 1
            details = getattr(ev, "details", None) or {}
            cumulative["promoted"] += len(details.get("promoted", []) or [])
        elif ev.decision == "accept":
            cumulative["accept"] += 1
            details = getattr(ev, "details", None) or {}
            cumulative["removed"] += len(details.get("removed", []) or [])
        elif ev.decision == "reject":
            cumulative["reject"] += 1
        elif ev.decision == "delay":
            cumulative["delay"] += 1

    def snapshot(pos: int, terminal: bool = False) -> dict[str, Any]:
        prefix = all_scores[:pos]
        recomputed = recompute_thresholds(np.asarray(prefix), k1, k2)
        observed = by_position.get(pos - 1)
        obs_alpha = _num(observed.alpha) if observed is not None else None
        obs_beta = _num(observed.beta) if observed is not None else None
        agree_alpha = (recomputed["alpha"] is not None and obs_alpha is not None
                       and abs(recomputed["alpha"] - obs_alpha) <= TOLERANCE)
        agree_beta = (recomputed["beta"] is not None and obs_beta is not None
                      and abs(recomputed["beta"] - obs_beta) <= TOLERANCE)
        finite = [float(x) for x in prefix if x is not None and math.isfinite(float(x))]
        row = {
            "arrival_position": pos,
            "n_arrivals": len(prefix),
            "n_scores": len(prefix),
            "is_terminal": terminal,
            "score_mean": float(np.mean(finite)) if finite else None,
            "score_sd_ddof1": float(np.std(finite, ddof=1)) if len(finite) > 1 else None,
            "recomputed": recomputed,
            "observed_alpha": obs_alpha,
            "observed_beta": obs_beta,
            "alpha_matches_recompute": agree_alpha if obs_alpha is not None else None,
            "beta_matches_recompute": agree_beta if obs_beta is not None else None,
            "cumulative": dict(cumulative),
        }
        out.append(row)
        return row

    checkpoints = set(CHECKPOINTS)
    seen_arrivals = 0
    for ev in events:
        if ev.decision == "pool_flush":
            # Attributed to the arrival just processed. A flush at the very end
            # of the stream (after every arrival) is deferred to the terminal row.
            if seen_arrivals >= len(all_scores):
                apply(ev)
                continue
            apply(ev)
            continue
        seen_arrivals += 1
        apply(ev)
        if seen_arrivals in checkpoints and seen_arrivals <= len(all_scores):
            snapshot(seen_arrivals)

    # Any checkpoint position beyond the final arrival is not emitted.
    terminal_pos = len(all_scores)
    if terminal_pos and terminal_pos not in {r["arrival_position"] for r in out}:
        snapshot(terminal_pos, terminal=False)
    snapshot(terminal_pos, terminal=True)
    return out
