"""A three-way streaming selector driven by the RECONSTRUCTED Fisher-Z scores.

This subclasses the Stage-2 ``Selector`` so the three-way decision loop, the
running moments and the pool machinery are not re-implemented, and overrides
only the three places where ``OSFS_DD_z2.m`` differs from ``OSFS_DD_mi2.m``.
Those differences are source-verified from the pinned ``z2.m`` and are NOT
about the missing ``Fisher_Ztest`` helper:

1. **Warm-up accepts.** ``z2.m:41-46`` adds the first two features to
   ``AccIndexs`` before ``continue``. ``mi2.m:36-39`` skips them. Reproducing
   the discrete warm-up here would silently change the selected set.
2. **NaN guard.** ``z2.m:35-37`` maps a NaN score to 0. ``mi2.m`` has no such
   guard and lets NaN propagate.
3. **Pair guards.** ``z2.m:124`` and ``z2.m:152`` skip a pair when the helper
   reports ``CI_cmi==1`` or a NaN cmi. ``mi2.m`` has no equivalent guard.

The reported mode ID is ``continuous_reconstructed_fisherz_v1``: the decision
*structure* is source-derived, the *scores* are our reconstruction.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import numpy as np

from ossfs_dd.config import MetricFamily, SelectionConfig
from ossfs_dd.pools import Candidate
from ossfs_dd.selector import Event, SelectionResult, Selector

from . import MODE_ID
from .fisherz import (
    UnsupportedSampleSize,
    marginal_fisher_z,
    pair_heuristic_fisher_z,
)

CONTINUOUS_METRIC_FAMILY = "continuous_reconstructed_fisher_z"
PAIR_LABEL = "heuristic_joint_v1"


class ContinuousReconstructedSelector(Selector):
    """Three-way streaming selector over reconstructed Fisher-Z scores."""

    def __init__(
        self,
        config: SelectionConfig,
        *,
        labels: np.ndarray,
        entry_file: str = "z2",
        mode_metadata: dict[str, Any] | None = None,
    ) -> None:
        # The parent takes a MetricFamily; DISCRETE_INFORMATION is used purely as
        # a type placeholder so the parent's GAUSSIAN_FISHER_Z fail-closed guard
        # (which must stay armed for Stage 2) is not bypassed for Stage 2's own
        # mode. Every label this class reports is replaced below.
        super().__init__(
            config,
            metric_family=MetricFamily.DISCRETE_INFORMATION,
            labels=labels,
            entry_file=entry_file,
            score_fn=self._fisherz_marginal,
            mode_metadata=mode_metadata,
        )
        self.mode_id = MODE_ID
        self.state.mode = MODE_ID
        self.state.metric_family = CONTINUOUS_METRIC_FAMILY
        self.pair_skips = 0

    # ---- scores --------------------------------------------------------
    def _fisherz_marginal(self, column: np.ndarray, labels: np.ndarray) -> float:
        try:
            score = marginal_fisher_z(column, labels).score
        except UnsupportedSampleSize as exc:
            raise UnsupportedSampleSize(str(exc)) from exc
        if math.isnan(score):
            # z2.m:35-37 maps a NaN relevance score to 0 before it is stored.
            self.warnings.append("marginal Fisher-Z returned NaN; set to 0 per z2.m:35-37")
            return 0.0
        return float(score)

    def _joint(
        self, a: np.ndarray, b: np.ndarray, labels: np.ndarray, marginal_of_a: float
    ) -> tuple[float, str]:
        self.state.joint_evaluations += 1
        pair = pair_heuristic_fisher_z(a, b, labels)
        return marginal_of_a + pair.score, (
            f"marginal + {PAIR_LABEL} (RECONSTRUCTED, not the authors' cmi)"
        )

    # ---- z2.m warm-up: accept the first two features -------------------
    def _warmup_accept(self, feature_id: int, arrival_position: int, column: np.ndarray) -> None:
        from ossfs_dd.moments import seed_from_values

        score = self._fisherz_marginal(column, self.labels)
        self.state.arrival_count += 1
        self.scores.append(score)
        finite = [s for s in self.scores if not math.isnan(s)]
        mean, std = seed_from_values(finite)
        self.moments.count = self.state.arrival_count
        self.moments.mean = mean
        self.moments.m2 = (std ** 2) * max(self.state.arrival_count - 1, 0)
        # z2.m:42 `AccIndexs=[AccIndexs,i]` -- accepted, no redundancy check.
        self.candidates.add(Candidate(feature_id, arrival_position, score, column))
        self.state.direct_accepts += 1
        self.state.peak_candidates = max(self.state.peak_candidates, len(self.candidates))
        self.events.append(
            Event(
                arrival_position, feature_id, score, "accept",
                "i<=2 warm-up", "z2.m:41-46 accepts the first two features before `continue`",
                float("nan"), float("nan"), std,
            )
        )

    def consume(self, feature_id: int, arrival_position: int, column: np.ndarray) -> None:
        if self.state.arrival_count < 2:
            self._warmup_accept(feature_id, arrival_position, np.asarray(column))
            return
        super().consume(feature_id, arrival_position, np.asarray(column))

    # ---- z2.m pair guards ----------------------------------------------
    def _pair_or_none(self, a: np.ndarray, b: np.ndarray) -> float | None:
        """``None`` means "skip this pair", per z2.m:124 / z2.m:152."""
        try:
            value = pair_heuristic_fisher_z(a, b, self.labels).score
        except (UnsupportedSampleSize, ValueError):
            self.pair_skips += 1
            return None
        if math.isnan(value):
            self.pair_skips += 1
            return None
        return value

    def _redundancy_pass(self, new: Candidate, beta: float) -> list[int]:
        """z2.m:141-168, including the ``CI_cmi==1||isnan(cmi)`` continue."""
        if len(self.candidates) <= 1:
            return []
        existing = list(self.candidates)[:-1]
        to_remove: list[int] = []
        drop_new = False
        for c in existing:
            self.state.joint_evaluations += 1
            pair = self._pair_or_none(c.vector, new.vector)
            if pair is None:
                continue                      # z2.m:152
            joint = c.score + pair
            threshold = 2.0 * beta
            if joint >= threshold:
                continue
            if new.score < c.score:
                drop_new = True
                break
            to_remove.append(c.feature_id)
        if drop_new:
            self.candidates.remove(new.feature_id)
        else:
            for fid in to_remove:
                self.candidates.remove(fid)
        removed = to_remove + ([new.feature_id] if drop_new else [])
        self.state.redundancy_removals += len(removed)
        return removed

    def _flush_pool(self, trigger: str) -> None:
        """z2.m:115-138, including the pair guard and inner-only break."""
        if self.last_beta is None:
            self.warnings.append("final pool flush requested before beta was assigned")
            return
        beta = self.last_beta
        pool = list(self.pool)
        n = len(pool)
        promoted_positions: list[int] = []
        for i in range(n):
            for j in range(i + 1, n):
                a, b = pool[i], pool[j]
                self.state.joint_evaluations += 1
                pair = self._pair_or_none(a.vector, b.vector)
                if pair is None:
                    continue                  # z2.m:124
                if a.score + pair > 2.0 * beta:
                    promoted_positions.extend([a.arrival_position, b.arrival_position])
                    break                      # inner only
        unique_positions = sorted(set(promoted_positions))
        added: list[int] = []
        for pos in unique_positions:
            for c in pool:
                if c.arrival_position == pos:
                    self.candidates.add(c)
                    added.append(c.feature_id)
                    break
        self.state.promoted_from_pool += len(added)
        self.state.peak_candidates = max(self.state.peak_candidates, len(self.candidates))
        taken = self.pool.clear()
        self.state.pool_flushes += 1
        self.events.append(
            Event(
                -1, -1, float("nan"), "pool_flush", trigger, "flushDeterPool (z2.m)",
                self.last_alpha if self.last_alpha is not None else float("nan"),
                beta, self.moments.std,
                {"pool_size": len(taken), "promoted": added,
                 "promoted_arrival_positions": unique_positions},
            )
        )

    # ---- reporting ------------------------------------------------------
    def result(self) -> SelectionResult:
        res = super().result()
        res.mode = MODE_ID
        res.metric_family = CONTINUOUS_METRIC_FAMILY
        res.warnings = list(res.warnings) + [
            (
                "RECONSTRUCTED estimator, not the authors' Fisher_Ztest "
                "(absent upstream); pair estimator is "
                f"{PAIR_LABEL} (heuristic, not a conditional-independence test)"
            )
        ]
        return res

    def fit_stream(self, stream: Iterator[tuple[int, np.ndarray]]) -> SelectionResult:
        for feature_id, column in stream:
            self.consume(feature_id, self._next_position(), np.asarray(column))
        return self._finalize()
