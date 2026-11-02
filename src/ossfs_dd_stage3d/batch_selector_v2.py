"""Versioned v2 batch selector: deferred pool persists across batches.

v1 (``batch_selector.BatchSelector``) unconditionally flushes the deferred pool
at the end of **every** batch. That is a deviation from the declared
``pool_size=100`` / end-of-stream semantics: a feature deferred in batch 1 can
never pair with a feature deferred in batch 2.

v2 fixes this with a separate lifecycle:

  * ``consume_batch(ids, cols)`` scores and decides for one batch. The pool is
    flushed **only** when it reaches ``pool_size`` (capacity), using the frozen
    beta of the batch in which capacity was reached.
  * ``finalize_stream()`` flushes any remaining pool **exactly once** at the
    true end of the stream, using the last batch's beta.

v1 is left untouched and its existing outputs are not overwritten. v2 is a
separately identifiable method (`batch_score_shared_v2`).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ossfs_dd.pools import Candidate as _Cand

from .batch_selector import BatchResult, BatchSelector


class BatchSelectorV2(BatchSelector):
    """Deferred pool persists across batch boundaries."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._finalized = False
        self._v2 = True

    def consume_batch(self, feature_ids: list[int], columns: list[Any]) -> None:
        """Score + decide for one batch. NO terminal flush here."""
        # Reuse the parent's scoring/decision logic but skip the trailing
        # stream_end flush. We do that by calling the parent's consume_batch
        # and then... the parent always flushes. So instead reimplement the
        # decision tail without the trailing flush.
        if len(feature_ids) != len(columns):
            raise ValueError("feature_ids and columns must have equal length")
        if not feature_ids:
            return
        self.state.n_batches += 1
        batch = sorted(zip(feature_ids, columns, strict=True), key=lambda z: int(z[0]))
        ids_sorted = [int(fid) for fid, _ in batch]
        cols_sorted = [np.asarray(c) for _, c in batch]

        batch_scores: list[tuple[int, Any, float]] = []
        for fid, col in zip(ids_sorted, cols_sorted, strict=True):
            try:
                sc = float(self.score_fn(col, self.labels))
            except Exception as exc:  # noqa: BLE001
                self.warnings.append(f"score failed for feature {fid}: {exc}")
                sc = float("nan")
            if math.isnan(sc):
                self.state.score_clamped_nonfinite += 1
            self.scores.append(sc)
            batch_scores.append((fid, col, sc))

        for _fid, _col, sc in batch_scores:
            mean, std = self.moments.update(sc)
            self.moments.mean = mean

        alpha = self.moments.mean + self.k1 * self.moments.std
        beta = self.moments.mean + self.k2 * self.moments.std
        self.last_alpha, self.last_beta = alpha, beta
        if alpha > 1.0:
            self.state.alpha_exceeds_one += 1
        if beta > 1.0:
            self.state.beta_exceeds_one += 1
        if self.moments.degenerate:
            self.state.degenerate_sigma_events += 1

        order = sorted(range(len(batch_scores)),
                       key=lambda i: (-_safe_score(batch_scores[i][2]), batch_scores[i][0]))
        for i in order:
            fid, col, sc = batch_scores[i]
            self._arrival_counter += 1
            pos = self._arrival_counter
            self.state.arrival_count += 1

            if not math.isfinite(sc):
                self.state.immediate_rejects += 1
                self.state.events.append({"arrival_position": pos, "feature_id": fid,
                                          "score": sc, "decision": "reject",
                                          "region": "non-finite score",
                                          "alpha": alpha, "beta": beta,
                                          "sigma": self.moments.std})
                continue
            if sc <= alpha:
                self.state.immediate_rejects += 1
                self.state.events.append({"arrival_position": pos, "feature_id": fid,
                                          "score": sc, "decision": "reject",
                                          "region": "score <= alpha",
                                          "alpha": alpha, "beta": beta,
                                          "sigma": self.moments.std})
                continue
            if sc >= beta:
                self.state.direct_accepts += 1
                self.candidates.add(_Cand(fid, pos, sc, col))
                self.state.peak_candidates = max(self.state.peak_candidates, len(self.candidates))
                removed = self._redundancy_pass(self.candidates.items[-1], beta)
                self.state.events.append({"arrival_position": pos, "feature_id": fid,
                                          "score": sc, "decision": "accept",
                                          "region": "score >= beta",
                                          "alpha": alpha, "beta": beta,
                                          "sigma": self.moments.std,
                                          "removed": removed})
                continue
            self.state.deferred += 1
            self.pool.add(_Cand(fid, pos, sc, col))
            self.state.peak_pool = max(self.state.peak_pool, len(self.pool))
            self.state.events.append({"arrival_position": pos, "feature_id": fid,
                                      "score": sc, "decision": "delay",
                                      "region": "alpha < score < beta",
                                      "alpha": alpha, "beta": beta,
                                      "sigma": self.moments.std,
                                      "pool_size_now": len(self.pool)})
            if len(self.pool) >= self.pool_size:
                self._flush_pool("capacity", beta)

        self.state.final_candidates = len(self.candidates)
        self.state.pending_pool_at_end = len(self.pool)
        self.state.duplicate_final_ids = self.candidates.duplicates()
        # NO terminal flush here: the pool persists across batches.

    def finalize_stream(self) -> None:
        """Flush any remaining pool exactly once at the true end of the stream."""
        if self._finalized:
            return
        self._finalized = True
        if len(self.pool):
            beta = self.last_beta if self.last_beta is not None else 0.0
            self._flush_pool("stream_end", beta)
        self.state.final_candidates = len(self.candidates)
        self.state.pending_pool_at_end = len(self.pool)
        self.state.duplicate_final_ids = self.candidates.duplicates()

    def run(self, feature_ids: list[int], columns: list[Any]) -> BatchResult:
        raise RuntimeError("v2 requires consume_batch per batch then finalize_stream(); "
                           "use the runner helper")


def _safe_score(s: float) -> float:
    if not __import__("math").isfinite(s):
        return float("-inf")
    return s

