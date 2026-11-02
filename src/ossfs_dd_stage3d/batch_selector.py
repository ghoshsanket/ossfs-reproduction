"""Atomic-batch streaming selector (Stage 3D variant `batch_score_shared_v1`).

For each arrived batch (size <= B):

1. Score every feature in the batch (canonical original-ID order) and update
   the running moments **once per score** (ddof=1).
2. Compute **one** batch-level alpha = mu + 1*sigma, beta = mu + 2*sigma
   after all batch scores are observed. Freeze them for the batch.
3. Sort the batch's features by (descending score, ascending feature ID) and
   apply three-way decisions (score<=alpha reject, score>=beta accept, else
   delay) using the frozen batch beta for redundancy/pool work.
4. Redundancy and delayed-pool analysis use the supplied pair operator and the
   frozen beta; bounded buffer capacity 100 with a flush at capacity and at
   stream completion. Applied removals only.

All accounting (redundancy_removals) counts applied removals only.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ossfs_dd.config import SelectionConfig
from ossfs_dd.moments import RunningMoments
from ossfs_dd.pools import Candidate, CandidateSet, UndeterminedPool


@dataclass
class BatchState:
    mode: str
    metric_family: str
    n_batches: int = 0
    arrival_count: int = 0
    immediate_rejects: int = 0
    direct_accepts: int = 0
    deferred: int = 0
    redundancy_removals: int = 0
    promoted_from_pool: int = 0
    pool_flushes: int = 0
    joint_evaluations: int = 0
    score_clamped_nonfinite: int = 0
    alpha_exceeds_one: int = 0
    beta_exceeds_one: int = 0
    degenerate_sigma_events: int = 0
    peak_candidates: int = 0
    peak_pool: int = 0
    final_candidates: int = 0
    pending_pool_at_end: int = 0
    duplicate_final_ids: list[int] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class BatchResult:
    mode: str
    metric_family: str
    feature_indices: list[int]
    arrival_positions: list[int]
    n_features_seen: int
    n_batches: int
    state: BatchState
    score_min: float | None
    score_max: float | None
    warnings: list[str] = field(default_factory=list)


class BatchSelector:
    def __init__(self, config: SelectionConfig, *, metric_family: str,
                 score_fn: Callable[[Any, Any], float],
                 joint_fn: Callable[[Any, Any, Any, float], tuple[float, str]],
                 is_redundant_fn: Callable[[float, float], tuple[bool, str]],
                 promotes_pair_fn: Callable[[float, float], tuple[bool, str]],
                 labels: Any, pool_size: int = 100,
                 k1: float = 1.0, k2: float = 2.0) -> None:
        self.cfg = config
        self.metric_family = metric_family
        self.labels = np.asarray(labels).astype(str)
        self.score_fn = score_fn
        self.joint_fn = joint_fn
        self.is_redundant_fn = is_redundant_fn
        self.promotes_pair_fn = promotes_pair_fn
        self.k1 = k1
        self.k2 = k2
        self.pool_size = pool_size
        self.moments = RunningMoments()
        self.candidates = CandidateSet()
        self.pool = UndeterminedPool(pool_size=pool_size)
        self.state = BatchState(mode=config.mode.value, metric_family=metric_family)
        self.warnings: list[str] = []
        self.scores: list[float] = []
        self._arrival_counter = 0
        self.last_alpha: float | None = None
        self.last_beta: float | None = None

    def _redundancy_pass(self, new: Candidate, beta: float) -> list[int]:
        if len(self.candidates) <= 1:
            return []
        existing = [c for c in self.candidates if c.feature_id != new.feature_id]
        to_remove: list[int] = []
        drop_new = False
        for c in existing:
            self.state.joint_evaluations += 1
            joint, _ = self.joint_fn(c.vector, new.vector, self.labels, c.score)
            redundant, _ = self.is_redundant_fn(joint, beta)
            if not redundant:
                continue
            if new.score < c.score:
                drop_new = True
                break
            to_remove.append(c.feature_id)
        applied: list[int] = []
        if drop_new:
            if self.candidates.remove(new.feature_id):
                applied.append(new.feature_id)
        else:
            for fid in to_remove:
                if self.candidates.remove(fid):
                    applied.append(fid)
        self.state.redundancy_removals += len(applied)
        return applied

    def _flush_pool(self, trigger: str, beta: float) -> None:
        pool = list(self.pool)
        n = len(pool)
        promoted_positions: list[int] = []
        for i in range(n):
            for j in range(i + 1, n):
                a, b = pool[i], pool[j]
                self.state.joint_evaluations += 1
                joint, _ = self.joint_fn(a.vector, b.vector, self.labels, a.score)
                ok, _ = self.promotes_pair_fn(joint, beta)
                if ok:
                    promoted_positions.extend([a.arrival_position, b.arrival_position])
                    break
        unique = sorted(set(promoted_positions))
        added = 0
        for pos in unique:
            for c in pool:
                if c.arrival_position == pos:
                    self.candidates.add(c)
                    added += 1
                    break
        self.state.promoted_from_pool += added
        self.state.peak_candidates = max(self.state.peak_candidates, len(self.candidates))
        self.pool.clear()
        self.state.pool_flushes += 1
        self.state.events.append({
            "kind": "pool_flush", "trigger": trigger, "pool_size": n,
            "promoted_positions": unique, "added": added,
        })

    def consume_batch(self, feature_ids: list[int], columns: list[Any]) -> None:
        if len(feature_ids) != len(columns):
            raise ValueError("feature_ids and columns must have equal length")
        if not feature_ids:
            return
        self.state.n_batches += 1
        batch = sorted(zip(feature_ids, columns, strict=True), key=lambda z: int(z[0]))
        ids_sorted = [int(fid) for fid, _ in batch]
        cols_sorted = [np.asarray(c) for _, c in batch]

        # 1. score every feature once, in canonical original-ID order
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

        # running moments: update once per score, in canonical order (ddof=1)
        for _fid, _col, sc in batch_scores:
            mean, std = self.moments.update(sc)
            self.moments.mean = mean

        # 2. one batch-level alpha/beta
        alpha = self.moments.mean + self.k1 * self.moments.std
        beta = self.moments.mean + self.k2 * self.moments.std
        self.last_alpha, self.last_beta = alpha, beta
        if alpha > 1.0:
            self.state.alpha_exceeds_one += 1
        if beta > 1.0:
            self.state.beta_exceeds_one += 1
        if self.moments.degenerate:
            self.state.degenerate_sigma_events += 1

        # 3. three-way decisions in (descending score, ascending feature id) order
        order = sorted(range(len(batch_scores)),
                       key=lambda i: (-_safe_score(batch_scores[i][2]), batch_scores[i][0]))
        for i in order:
            fid, col, sc = batch_scores[i]
            self._arrival_counter += 1
            pos = self._arrival_counter
            self.state.arrival_count += 1

            if not math.isfinite(sc):
                self.state.immediate_rejects += 1
                self.state.events.append({
                    "arrival_position": pos, "feature_id": fid, "score": sc,
                    "decision": "reject", "region": "non-finite score",
                    "alpha": alpha, "beta": beta, "sigma": self.moments.std,
                })
                continue
            if sc <= alpha:
                self.state.immediate_rejects += 1
                self.state.events.append({
                    "arrival_position": pos, "feature_id": fid, "score": sc,
                    "decision": "reject", "region": "score <= alpha",
                    "alpha": alpha, "beta": beta, "sigma": self.moments.std,
                })
                continue
            if sc >= beta:
                self.state.direct_accepts += 1
                self.candidates.add(Candidate(fid, pos, sc, col))
                self.state.peak_candidates = max(self.state.peak_candidates, len(self.candidates))
                removed = self._redundancy_pass(self.candidates.items[-1], beta)
                self.state.events.append({
                    "arrival_position": pos, "feature_id": fid, "score": sc,
                    "decision": "accept", "region": "score >= beta",
                    "alpha": alpha, "beta": beta, "sigma": self.moments.std,
                    "removed": removed,
                })
                continue
            self.state.deferred += 1
            self.pool.add(Candidate(fid, pos, sc, col))
            self.state.peak_pool = max(self.state.peak_pool, len(self.pool))
            self.state.events.append({
                "arrival_position": pos, "feature_id": fid, "score": sc,
                "decision": "delay", "region": "alpha < score < beta",
                "alpha": alpha, "beta": beta, "sigma": self.moments.std,
                "pool_size_now": len(self.pool),
            })
            if len(self.pool) >= self.pool_size:
                self._flush_pool("capacity", beta)

        if len(self.pool):
            self._flush_pool("stream_end", beta)

        self.state.final_candidates = len(self.candidates)
        self.state.pending_pool_at_end = len(self.pool)
        self.state.duplicate_final_ids = self.candidates.duplicates()

    def run(self, feature_ids: list[int], columns: list[Any]) -> BatchResult:
        self.consume_batch(feature_ids, columns)
        finite = [s for s in self.scores if math.isfinite(s)]
        return BatchResult(
            mode=self.cfg.mode.value,
            metric_family=self.metric_family,
            feature_indices=self.candidates.feature_ids(),
            arrival_positions=[c.arrival_position for c in self.candidates],
            n_features_seen=self.state.arrival_count,
            n_batches=self.state.n_batches,
            state=self.state,
            score_min=min(finite) if finite else None,
            score_max=max(finite) if finite else None,
            warnings=list(self.warnings),
        )


def _safe_score(s: float) -> float:
    if not math.isfinite(s):
        return float("-inf")
    return s


# CandidateSet.items is private-ish; expose via a helper property.
if not hasattr(CandidateSet, "items"):
    pass


# Patch CandidateSet to expose items as a list attribute (it already is).

