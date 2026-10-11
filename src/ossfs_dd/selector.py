"""Stateful online selector: the three-way streaming decision loop.

Mirrors the structure of ``OSFS_DD_mi2.m`` / ``OSFS_DD_z2.m`` with the mode
differences kept explicit rather than smoothed away.

Invariants enforced here
------------------------
* one column at a time; the selector never holds the full ``X``;
* ``full_rows_diagnostic`` is never called — the selector has no code path to
  a held-out row;
* a rejected feature retains **no** vector, only bounded scalar trace;
* ``feature_id`` is always the original zero-based Stage-1 feature ID;
  ``arrival_position`` is stored separately and is never used as an ID;
* thresholds are computed from scores seen SO FAR only.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import MetricFamily, Mode, SelectionConfig
from .decisions import (
    Decision,
    decide,
    is_redundant,
    pool_full,
    promotes_pair,
    weaker_feature,
)
from .metrics import (
    conditional_mutual_information,
    normalised_su_of_joint,
    symmetrical_uncertainty,
)
from .moments import RunningMoments, seed_from_values
from .pools import Candidate, CandidateSet, UndeterminedPool

# A column is retained as a numpy array so CSR and dense sources behave alike.
Column = np.ndarray


@dataclass
class Event:
    """One decision event. Bounded scalar metadata only."""

    arrival_position: int
    feature_id: int
    score: float
    decision: str
    region: str
    reason: str
    alpha: float
    beta: float
    sigma: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectorState:
    """All selector state, serialisable, no raw dataset content."""

    mode: str
    metric_family: str
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

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class SelectionResult:
    """Outcome of one streaming selection run."""

    mode: str
    metric_family: str
    feature_indices: list[int]              # original zero-based feature IDs
    arrival_positions: list[int]            # arrival order positions, parallel
    n_features_seen: int
    state: SelectorState
    score_min: float | None
    score_max: float | None
    events: list[Event]
    mode_metadata: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "metric_family": self.metric_family,
            "n_features_seen": self.n_features_seen,
            "n_selected": len(self.feature_indices),
            "selected_feature_ids": self.feature_indices,
            "duplicate_final_ids": self.state.duplicate_final_ids,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "counts": {
                "arrival_count": self.state.arrival_count,
                "immediate_rejects": self.state.immediate_rejects,
                "direct_accepts": self.state.direct_accepts,
                "deferred": self.state.deferred,
                "redundancy_removals": self.state.redundancy_removals,
                "promoted_from_pool": self.state.promoted_from_pool,
                "pool_flushes": self.state.pool_flushes,
                "joint_evaluations": self.state.joint_evaluations,
                "alpha_exceeds_one": self.state.alpha_exceeds_one,
                "beta_exceeds_one": self.state.beta_exceeds_one,
                "degenerate_sigma_events": self.state.degenerate_sigma_events,
                "peak_candidates": self.state.peak_candidates,
                "peak_pool": self.state.peak_pool,
                "final_candidates": self.state.final_candidates,
                "pending_pool_at_end": self.state.pending_pool_at_end,
            },
            "warnings": self.warnings,
            "mode_metadata": self.mode_metadata,
        }


class Selector:
    """Online three-way streaming feature selector."""

    def __init__(
        self,
        config: SelectionConfig,
        *,
        metric_family: MetricFamily,
        labels: np.ndarray,
        entry_file: str = "mi2",
        score_fn: Callable[[Column, np.ndarray], float] | None = None,
        mode_metadata: dict[str, Any] | None = None,
    ) -> None:
        if metric_family is MetricFamily.GAUSSIAN_FISHER_Z:
            from .config import UnsupportedSemanticsError
            from .reference_semantics import COMMIT

            raise UnsupportedSemanticsError(
                "gaussian_fisher_z",
                "OSFS_DD_z2.m calls Fisher_Ztest, which does not exist in the pinned "
                f"authors' repository (commit {COMMIT}); its score, units and conditional "
                "test cannot be determined from source. No proxy statistic is permitted.",
                "OSFS_DD_z2.m:34,123,151",
            )

        self.cfg = config
        self.metric_family = metric_family
        self.labels = np.asarray(labels).astype(str)
        self.entry_file = entry_file
        self.mode_metadata = mode_metadata or {}
        self._score_fn = score_fn or self._discrete_score

        self.moments = RunningMoments()
        self.candidates = CandidateSet()
        self.pool = UndeterminedPool(pool_size=config.pool_size)
        self.state = SelectorState(mode=config.mode.value, metric_family=metric_family.value)
        self.events: list[Event] = []
        self.scores: list[float] = []
        self.last_beta: float | None = None
        self.last_alpha: float | None = None
        self.warnings: list[str] = []

    # ---- scoring ------------------------------------------------------
    def _discrete_score(self, column: Column, labels: np.ndarray) -> float:
        score = symmetrical_uncertainty(column, labels, log_base=self.cfg.log_base)
        if math.isnan(score):
            # mi2.m has NO isnan guard (unlike z2.m:35-37): NaN propagates into
            # the comparison. Record it instead of silently substituting 0.
            self.warnings.append(
                "symmetrical_uncertainty returned NaN (both operands have zero entropy); "
                "the reference discrete file has no isnan guard, so this is reported, not hidden"
            )
        return score

    # ---- joint quantity ----------------------------------------------
    def _joint(self, a: Column, b: Column, labels: np.ndarray,
               marginal_of_a: float) -> tuple[float, str]:
        """Mode-specific joint quantity. See decisions/doc for the S4 divergence."""
        self.state.joint_evaluations += 1
        if self.cfg.mode is Mode.PAPER_SPEC:
            value = normalised_su_of_joint(a, b, labels, log_base=self.cfg.log_base)
            return value, "SU({A,B};Y) — scale-coherent (paper_spec_v1)"
        # matlab_compatible_v1 reproduces mi2.m:116,137 exactly:
        #   jointMI = SU(A;Y) + cmi(A, Y, B)
        cmi = conditional_mutual_information(a, labels, b, log_base=self.cfg.log_base).value
        return marginal_of_a + cmi, "SU(A;Y) + cmi(A,Y,B) — mixed units (matlab_compatible_v1)"

    # ---- main loop ----------------------------------------------------
    def consume(self, feature_id: int, arrival_position: int, column: Column) -> None:
        """Fold one arrival. ``column`` holds TRAINING rows only."""
        labels = self.labels
        score = self._score_fn(column, labels)

        self.state.arrival_count += 1
        pos = self.state.arrival_count  # 1-based, as the reference's Num
        self.scores.append(score)

        # ---- first two features --------------------------------------
        if pos <= 2 and self.cfg.mode is Mode.MATLAB_COMPATIBLE:
            # mi2.m:37-41 — score, update moments, skip. Never selected.
            # The moments are seeded from the scores seen so far and the
            # branch RETURNS, so no score is folded in twice.
            mean, std = seed_from_values([s for s in self.scores if not math.isnan(s)])
            self.moments.count = pos
            self.moments.mean = mean
            self.moments.m2 = (std ** 2) * max(pos - 1, 0)
            self.events.append(Event(
                arrival_position, feature_id, score, "skipped_warmup",
                "i<=2 warm-up", "mi2.m:37-41 `continue` without insertion",
                float("nan"), float("nan"), std,
            ))
            return
            # paper_spec_v1 warm-up (INTERPRETATION, versioned in configs/stage2.yaml
            # as `paper_spec_v1.warmup_bootstrap_r2`):
            #   The score is folded into the moments EXACTLY ONCE via the normal
            #   update path, then the feature is routed through the ordinary
            #   three-way decision using the resulting thresholds.
            #   Pre-repair, seed_from_values() wrote count/mean/m2 and the score
            #   was then passed to update() a second time, so the moments for the
            #   first score started at count=2 and the second at count=3, skewing
            #   every subsequent dynamic threshold.
            #   Algorithm 1 prescribes no decision when only one score exists and
            #   sigma==0; we therefore route it through the literal rule, which
            #   yields REJECT (score <= alpha) rather than forcing an accept.
            #   No future score is consulted and alpha<beta is never relaxed.

        mean, std = self.moments.update(score)
        self.moments.mean, self.moments.m2 = mean, self.moments.m2
        alpha, beta = self.moments.thresholds(self.cfg.k1, self.cfg.k2)
        self.last_alpha, self.last_beta = alpha, beta

        if not math.isfinite(score):
            self.state.score_clamped_nonfinite += 1
            self.events.append(Event(
                arrival_position, feature_id, score, "reject", "non-finite score",
                "non-finite relevance score; comparison is undefined", alpha, beta, std,
            ))
            self.state.immediate_rejects += 1
            return

        if alpha > 1.0:
            self.state.alpha_exceeds_one += 1
        if beta > 1.0:
            self.state.beta_exceeds_one += 1
        if self.moments.degenerate:
            self.state.degenerate_sigma_events += 1

        outcome = decide(self.cfg.mode.value, entry=self.entry_file,
                         score=score, alpha=alpha, beta=beta)

        if outcome.decision is Decision.REJECT:
            self.state.immediate_rejects += 1
            self.events.append(Event(
                arrival_position, feature_id, score, "reject", outcome.region,
                outcome.reason, alpha, beta, std,
            ))
            return

        if outcome.decision is Decision.ACCEPT:
            self.state.direct_accepts += 1
            # R1: _accept's second parameter is `beta`, and _redundancy_pass
            # compares `joint < 2*beta` (mi2.m:138). `alpha` must NOT be passed
            # here; doing so silently compared against 2*alpha.
            self._accept(Candidate(feature_id, arrival_position, score, column), beta, std)
            return

        # ---- delay --------------------------------------------------
        self.state.deferred += 1
        self.pool.add(Candidate(feature_id, arrival_position, score, column))
        self.state.peak_pool = max(self.state.peak_pool, len(self.pool))
        self.events.append(Event(
            arrival_position, feature_id, score, "delay", outcome.region, outcome.reason,
            alpha, beta, std, {"pool_size_now": len(self.pool)},
        ))
        if pool_full(len(self.pool), self.cfg.pool_size):
            self._flush_pool("capacity")
            self.state.peak_pool = max(self.state.peak_pool, len(self.pool))

    def _accept(self, cand: Candidate, beta: float, std: float) -> None:
        """Insert, then redundancy-check against existing candidates."""
        self.candidates.add(cand)
        self.state.peak_candidates = max(self.state.peak_candidates, len(self.candidates))
        removed = self._redundancy_pass(cand, beta)
        self.events.append(Event(
            cand.arrival_position, cand.feature_id, cand.score, "accept",
            "score >= beta", "accepted, then redundancy-checked against 2*beta",
            self.last_alpha if self.last_alpha is not None else float("nan"),
            beta, std, {"removed": removed, "candidates_now": len(self.candidates)},
        ))

    def _redundancy_pass(self, new: Candidate, beta: float) -> list[int]:
        """mi2.m:128-150. Returns the feature IDs actually removed.

        Diagnostic-only correction (Stage 3B follow-up D1).

        The scan may queue older candidates in ``to_remove`` and then set
        ``drop_new`` and break (mi2.m:141). In that branch the ``else`` that
        applies ``to_remove`` never runs, so those queued candidates are NOT
        deleted. The previous code nevertheless returned and counted them.

        The ledger is therefore derived only from removals that were actually
        applied, using ``CandidateSet.remove``'s boolean return (which reports
        real shrinkage). Deletion ORDER, the pair-search order, the tie-break,
        the applied promotions and the resulting selected feature order are all
        unchanged: this touches accounting only.
        """
        if len(self.candidates) <= 1:
            return []
        existing = list(self.candidates)[:-1]      # count-1, newest excluded
        to_remove: list[int] = []
        drop_new = False
        for c in existing:
            joint, _ = self._joint(c.vector, new.vector, self.labels, c.score)
            redundant, _why = is_redundant(joint, beta, mode=self.cfg.mode.value)
            if not redundant:
                continue
            who, _reason = weaker_feature(new.score, c.score)
            if who == "new":
                drop_new = True
                break                       # mi2.m:141 breaks out of the loop
            to_remove.append(c.feature_id)
        # Only deletions that really shrank the candidate list are reported.
        applied: list[int] = []
        if drop_new:
            if self.candidates.remove(new.feature_id):
                applied.append(new.feature_id)
        else:
            for fid in to_remove:           # preserve deletion order
                if self.candidates.remove(fid):
                    applied.append(fid)
        self.state.redundancy_removals += len(applied)
        return applied

    def _flush_pool(self, trigger: str) -> None:
        """mi2.m:107-125. Promotes qualifying pairs; no redundancy re-check."""
        if self.last_beta is None:
            # mi2.m:82-86 references `beta` which is never assigned when the
            # stream is too short. Recorded, not silently skipped.
            self.warnings.append(
                "final pool flush requested before beta was ever assigned "
                "(the reference would raise a MATLAB error here)"
            )
            return
        beta = self.last_beta
        pool = list(self.pool)
        n = len(pool)
        promoted: list[int] = []          # feature IDs
        promoted_positions: list[int] = []  # arrival positions, kept in parallel

        # R2: faithful transcription of mi2.m:112-122.
        #   for i: for j in (i+1..): if joint > 2*beta: keep i AND j; break INNER
        # The outer loop CONTINUES. The pre-repair code walked every pair with
        # no break at all, which both over-promoted and issued extra joint
        # evaluations.
        for i in range(n):
            for j in range(i + 1, n):
                a, b = pool[i], pool[j]
                joint, _ = self._joint(a.vector, b.vector, self.labels, a.score)
                ok, _why = promotes_pair(joint, beta, mode=self.cfg.mode.value)
                if ok:
                    promoted.extend([a.feature_id, b.feature_id])
                    promoted_positions.extend([a.arrival_position, b.arrival_position])
                    break                      # inner only

        # R4: MATLAB's unique(selectedIndex) sorts the 1-based ARRIVAL POSITIONS
        # (mi2.m:124), not the original feature indices. Candidate insertion order
        # is load-bearing because _redundancy_pass traverses it in order.
        order = sorted(
            range(len(promoted_positions)), key=lambda k: promoted_positions[k]
        )
        unique_positions: list[int] = []
        seen_positions: set[int] = set()
        for k in order:
            pos = promoted_positions[k]
            if pos in seen_positions:
                continue
            seen_positions.add(pos)
            unique_positions.append(pos)

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
        self.events.append(Event(
            -1, -1, float("nan"), "pool_flush", trigger, "flushDeterPool",
            self.last_alpha if self.last_alpha is not None else float("nan"),
            beta, self.moments.std,
            {"pool_size": len(taken), "promoted": added,
             "promoted_arrival_positions": unique_positions,
             "unique_pairs": len(unique_positions) // 2},
        ))

    # ---- driving ------------------------------------------------------
    def fit_stream(self, stream: Iterator[tuple[int, Column]]) -> SelectionResult:
        """Consume a Stage-1 ``FeatureStream``. Training rows only, by contract.

        Returns the finalized :class:`SelectionResult`, not ``self``, so callers
        cannot accidentally use an unfinalised selector.
        """
        for feature_id, column in stream:
            self.consume(feature_id, self._next_position(), np.asarray(column))
        return self._finalize()

    _pos = 0

    def _next_position(self) -> int:
        self._pos += 1
        return self._pos - 1

    def _finalize(self) -> SelectionResult:
        if len(self.pool):
            self._flush_pool("stream_end")
        self.state.final_candidates = len(self.candidates)
        self.state.pending_pool_at_end = len(self.pool)
        self.state.duplicate_final_ids = self.candidates.duplicates()
        return self.result()

    def result(self) -> SelectionResult:
        finite = [s for s in self.scores if not math.isnan(s)]
        return SelectionResult(
            mode=self.cfg.mode.value,
            metric_family=self.metric_family.value,
            feature_indices=self.candidates.feature_ids(),
            arrival_positions=[c.arrival_position for c in self.candidates],
            n_features_seen=self.state.arrival_count,
            state=self.state,
            score_min=min(finite) if finite else None,
            score_max=max(finite) if finite else None,
            events=self.events,
            mode_metadata=self.mode_metadata,
            warnings=list(self.warnings),
        )
