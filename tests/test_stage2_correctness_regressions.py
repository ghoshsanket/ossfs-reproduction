"""Stage 2 correctness regressions — R1 through R5.

Each test is written to FAIL on the pre-repair code and PASS after the fix.
`results/stage2/followup_<stamp>/regression_fail_before_pass_after.json` records
the observed before/after values, captured by running this file against the
unmodified selector before patching.

Every expected outcome is derived from an INDEPENDENT oracle — a hand-written
nested-loop reimplementation of `OSFS_DD_mi2.m:107-125`, or NumPy reductions —
never from the code under test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ossfs_dd.config import MetricFamily, Mode, SelectionConfig
from ossfs_dd.decisions import is_redundant
from ossfs_dd.moments import RunningMoments
from ossfs_dd.pools import Candidate
from ossfs_dd.selector import Selector

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def make_selector(labels, *, mode=Mode.MATLAB_COMPATIBLE, score_fn=None, pool_size=100):
    cfg = SelectionConfig(mode=mode, pool_size=pool_size)
    return Selector(
        cfg,
        metric_family=MetricFamily.DISCRETE_INFORMATION,
        labels=labels,
        score_fn=score_fn,
    )


def force_thresholds(sel: Selector, alpha: float, beta: float, mean=0.15, std=0.05) -> None:
    """Pin the dynamic thresholds so a test can place a joint value between them.

    This stubs only the threshold arithmetic. Decision routing, `_accept`,
    `_redundancy_pass` and the pool logic all remain the real code.
    """
    sel.moments.thresholds = lambda k1, k2: (alpha, beta)  # type: ignore[method-assign]
    original_update = sel.moments.update

    def update(value):
        original_update(value)
        return mean, std

    sel.moments.update = update  # type: ignore[method-assign]


LABELS = np.array(["a", "b"] * 20, dtype=object)


def const_column(n=40, value=1):
    return np.full(n, value, dtype=np.int64)


def identity_vector(index: int) -> np.ndarray:
    """A column that encodes its own pool index.

    ``_joint`` receives VECTORS (a.vector, b.vector), not Candidates, so a stub
    must recover identity from the payload. An earlier version of this file
    assumed Candidates and raised AttributeError — which is an error in the test,
    not evidence of the defect.
    """
    return np.full(4, float(index), dtype=np.int64)


# ==========================================================================
# R1 — wrong threshold forwarded to the redundancy check
# ==========================================================================


def test_R1_redundancy_uses_beta_not_alpha():
    """Integration path: consume -> ACCEPT -> _accept -> _redundancy_pass.

    alpha=0.10, beta=0.20, joint=0.30.
    Reference says redundant:  0.30 < 2*0.20 = 0.40  -> TRUE
    Pre-repair bug checks:     0.30 < 2*0.10 = 0.20  -> FALSE
    """
    sel = make_selector(LABELS, score_fn=lambda col, y: 0.9)
    force_thresholds(sel, alpha=0.10, beta=0.20)

    # joint value is supplied by stubbing only the metric, not the logic
    sel._joint = lambda a, b, labels, marginal: (0.30, "stub")  # type: ignore[method-assign]

    sel.consume(0, 0, const_column())
    sel.consume(1, 1, const_column())
    sel.consume(2, 2, const_column())   # first ACCEPT -> becomes a candidate
    sel.consume(3, 3, const_column())   # second ACCEPT -> redundancy check runs

    accepts = [e for e in sel.events if e.decision == "accept"]
    assert accepts, "expected at least two acceptance events"
    last = accepts[-1]
    assert last.details["removed"], (
        "feature 3 should have been removed as redundant (0.30 < 2*beta=0.40); "
        f"removed={last.details['removed']}. The redundancy check received "
        f"{last.beta} where it must receive beta=0.20."
    )
    assert sel.candidates.feature_ids() == [3]


def test_R1_forwarded_threshold_is_beta():
    """Direct proof of the parameter plumbing, independent of any score values."""
    sel = make_selector(LABELS, score_fn=lambda col, y: 0.9)
    force_thresholds(sel, alpha=0.10, beta=0.20)
    captured: dict[str, float] = {}
    original = sel._redundancy_pass

    def spy(new, beta):
        captured["beta"] = beta
        return original(new, beta)

    sel._redundancy_pass = spy  # type: ignore[method-assign]
    sel._joint = lambda a, b, labels, marginal: (0.30, "stub")  # type: ignore[method-assign]

    sel.consume(0, 0, const_column())
    sel.consume(1, 1, const_column())
    sel.consume(2, 2, const_column())
    sel.consume(3, 3, const_column())

    assert captured.get("beta") == pytest.approx(0.20), (
        f"_redundancy_pass received {captured.get('beta')}; it must receive beta (0.20), "
        "not alpha (0.10)"
    )


def test_R1_reference_predicate_agrees_with_expectation():
    """The standalone predicate itself is correct — the defect is the call site."""
    assert is_redundant(0.30, 0.20, mode="matlab_compatible_v1")[0] is True
    assert is_redundant(0.30, 0.10, mode="matlab_compatible_v1")[0] is False


# ==========================================================================
# R2 — MATLAB's first-match inner break is missing
# ==========================================================================


def _oracle_flush_deter_pool(pool_ids, qualifies):
    """Independent transcription of OSFS_DD_mi2.m:112-124.

    for i: for j in (i+1..): if qualifies(i,j): keep i,j; break INNER
    Returns (evaluated_pairs, qualifying_pairs, promoted_arrival_positions).
    """
    evaluated, qualifying, promoted = [], [], []
    n = len(pool_ids)
    for i in range(n):
        for j in range(i + 1, n):
            evaluated.append((i, j))
            if qualifies(i, j):
                qualifying.append((i, j))
                promoted.extend([i, j])       # arrival positions
                break
    # mi2.m:124 unique() sorts ARRIVAL POSITIONS, not original feature IDs
    return evaluated, qualifying, sorted(set(promoted))


def test_R2_pool_flush_breaks_after_first_match_per_outer_iteration():
    sel = make_selector(LABELS, score_fn=lambda col, y: 0.5)
    force_thresholds(sel, alpha=0.0, beta=0.1)
    sel.last_beta, sel.last_alpha = 0.1, 0.0

    pool_ids = [10, 20, 30, 40]
    for idx, pid in enumerate(pool_ids):
        sel.pool.add(Candidate(pid, idx, 0.5, identity_vector(idx)))

    # (0,1) qualifies and (0,2) would also qualify: MATLAB must stop after (0,1)
    def qualifies(i, j):
        return (i, j) in {(0, 1), (0, 2), (1, 2)}
    evaluated: list[tuple[int, int]] = []

    def stub(a, b, labels, marginal):
        i, j = int(a[0]), int(b[0])
        evaluated.append((i, j))
        return (1.0 if qualifies(i, j) else 0.0, "stub")

    sel._joint = stub  # type: ignore[method-assign]

    exp_evaluated, exp_qualifying, exp_positions = _oracle_flush_deter_pool(pool_ids, qualifies)
    sel._flush_pool("test")

    assert evaluated == exp_evaluated, (
        f"evaluated pair sequence {evaluated} != oracle {exp_evaluated}; "
        "the inner loop must break after the first qualifying pair for each outer i"
    )
    got_positions = sorted(c.arrival_position for c in sel.candidates)
    assert got_positions == exp_positions, (
        f"promoted arrival positions {got_positions} != oracle {exp_positions}"
    )
    assert (0, 2) not in evaluated, (
        "pair (0,2) qualifies but is unreachable: (0,1) matched first"
    )
    assert sel.state.pool_flushes == 1


def test_R2_outer_loop_continues_after_inner_break():
    """A later outer i must still be processed after an earlier i matched."""
    sel = make_selector(LABELS, score_fn=lambda col, y: 0.5)
    force_thresholds(sel, alpha=0.0, beta=0.1)
    sel.last_beta, sel.last_alpha = 0.1, 0.0
    pool_ids = [10, 20, 30, 40]
    for i, pid in enumerate(pool_ids):
        sel.pool.add(Candidate(pid, i, 0.5, identity_vector(i)))
    # only (0,1) and (2,3) qualify; each outer i matches on its FIRST j
    def qualifies(i, j):
        return (i, j) in {(0, 1), (2, 3)}
    evaluated: list[tuple[int, int]] = []

    def stub(a, b, labels, marginal):
        i, j = int(a[0]), int(b[0])
        evaluated.append((i, j))
        return (1.0 if qualifies(i, j) else 0.0, "stub")

    sel._joint = stub  # type: ignore[method-assign]
    exp_evaluated, _, exp_positions = _oracle_flush_deter_pool(pool_ids, qualifies)
    sel._flush_pool("test")
    assert evaluated == exp_evaluated
    assert sorted(c.arrival_position for c in sel.candidates) == exp_positions == [0, 1, 2, 3]
    assert sorted(sel.candidates.feature_ids()) == [10, 20, 30, 40]
    assert sel.state.pool_flushes == 1


# ==========================================================================
# R3 — paper_spec warm-up double counts the first scores
# ==========================================================================


def _replay_moments(sel: Selector):
    """Replay the recorded score history through a clean RunningMoments."""
    rm = RunningMoments()
    out = []
    for s in sel.scores:
        rm.update(s)
        out.append((rm.count, rm.mean, rm.std))
    return out


@pytest.mark.parametrize("mode", [Mode.PAPER_SPEC, Mode.MATLAB_COMPATIBLE])
def test_R3_moment_count_matches_scores_seen(mode):
    """The running count must equal the number of scores observed, at EVERY step.

    Asserted inside the consume loop; checking only after the loop would compare
    against a fully-populated counter and prove nothing.
    """
    fixed = [0.10, 0.40, 0.15, 0.75, 0.25]
    sel = make_selector(LABELS, mode=mode)
    seq = iter(fixed)
    sel._score_fn = lambda col, y, _s=seq: next(_s)  # type: ignore[method-assign]
    for i in range(len(fixed)):
        sel.consume(i, i, const_column())
        assert sel.moments.count == i + 1, (
            f"{mode.value}: after {i + 1} arrivals count={sel.moments.count}, "
            f"expected {i + 1}"
        )


@pytest.mark.parametrize("mode", [Mode.PAPER_SPEC, Mode.MATLAB_COMPATIBLE])
def test_R3_moments_match_numpy_prefix(mode):
    """At every arrival: mean == np.mean(prefix), std == np.std(prefix, ddof=1)."""
    fixed = [0.10, 0.40, 0.15, 0.75, 0.25]
    sel = make_selector(LABELS, mode=mode)
    seq = iter(fixed)
    sel._score_fn = lambda col, y, _s=seq: next(_s)  # type: ignore[method-assign]
    for i in range(len(fixed)):
        sel.consume(i, i, const_column())
        prefix = np.asarray(fixed[: i + 1], dtype=np.float64)
        assert sel.moments.count == i + 1
        assert abs(sel.moments.mean - prefix.mean()) <= 1e-12, (
            f"{mode.value} at prefix {i + 1}: mean {sel.moments.mean} vs {prefix.mean()}"
        )
        if i + 1 >= 2:
            assert abs(sel.moments.std - prefix.std(ddof=1)) <= 1e-10, (
                f"{mode.value} at prefix {i + 1}: std {sel.moments.std} vs {prefix.std(ddof=1)}"
            )


def test_R3_matlab_mode_warmup_skips_are_unaffected():
    """The MATLAB route returns inside the warm-up branch and must not double count."""
    fixed = [0.10, 0.40, 0.15]
    sel = make_selector(LABELS, mode=Mode.MATLAB_COMPATIBLE)
    seq = iter(fixed)
    sel._score_fn = lambda col, y, _s=seq: next(_s)  # type: ignore[method-assign]
    for i in range(3):
        sel.consume(i, i, const_column())
    assert sel.moments.count == 3
    assert abs(sel.moments.mean - np.mean(fixed)) <= 1e-12


# ==========================================================================
# R4 — promotion order must follow arrival position, not original feature ID
# ==========================================================================


def test_R4_promotion_sorted_by_arrival_position_not_feature_id():
    """Scrambled original IDs with monotonic arrival positions.

    MATLAB `unique()` sorts the 1-based ARRIVAL POSITIONS. Sorting original
    feature IDs instead produces a different candidate insertion order, which
    changes every subsequent redundancy traversal.
    """
    # arrival positions 0,1,2 carry original IDs 900, 5, 100 (deliberately scrambled)
    ids_by_position = [(900, 0), (5, 1), (100, 2)]
    sel = make_selector(LABELS, score_fn=lambda col, y: 0.5)
    force_thresholds(sel, alpha=0.0, beta=0.1)
    sel.last_beta, sel.last_alpha = 0.1, 0.0
    for fid, pos in ids_by_position:
        sel.pool.add(Candidate(fid, pos, 0.5, identity_vector(pos)))
    sel._joint = lambda a, b, labels, marginal: (1.0, "stub")  # type: ignore[method-assign]

    sel._flush_pool("test")

    promoted_ids = sel.candidates.feature_ids()
    promoted_positions = [c.arrival_position for c in sel.candidates]

    assert promoted_positions == sorted(promoted_positions), (
        f"promoted arrival positions {promoted_positions} are not ascending; MATLAB's "
        "unique() sorts arrival positions"
    )
    assert promoted_ids == [900, 5, 100], (
        f"promoted original IDs {promoted_ids} must follow arrival-position order "
        "[900, 5, 100], not feature-ID order [5, 100, 900]"
    )


def test_R4_arrival_position_to_id_mapping_is_preserved():
    sel = make_selector(LABELS, score_fn=lambda col, y: 0.5)
    force_thresholds(sel, alpha=0.0, beta=0.1)
    sel.last_beta, sel.last_alpha = 0.1, 0.0
    pairs = [(900, 0), (5, 1), (100, 2)]
    for fid, pos in pairs:
        sel.pool.add(Candidate(fid, pos, 0.5, identity_vector(pos)))
    sel._joint = lambda a, b, labels, marginal: (1.0, "stub")  # type: ignore[method-assign]
    sel._flush_pool("test")
    got = {c.arrival_position: c.feature_id for c in sel.candidates}
    assert got == {0: 900, 1: 5, 2: 100}


# ==========================================================================
# R5 — `x or nan` turns a legitimate alpha of 0.0 into NaN
# ==========================================================================


def test_R5_zero_alpha_is_recorded_as_zero_not_nan():
    sel = make_selector(LABELS, score_fn=lambda col, y: 0.9)
    force_thresholds(sel, alpha=0.0, beta=0.2)
    sel._joint = lambda a, b, labels, marginal: (1.0, "stub")  # type: ignore[method-assign]
    for i in range(4):
        sel.consume(i, i, const_column())
    accepts = [e for e in sel.events if e.decision == "accept"]
    assert accepts, "expected acceptance events"
    assert not math.isnan(accepts[0].alpha), (
        "alpha was recorded as NaN; the legitimate value is 0.0 "
        "(`self.last_alpha or float('nan')` is a truthiness bug)"
    )
    assert accepts[0].alpha == pytest.approx(0.0)


def test_R5_pool_flush_event_records_zero_alpha():
    sel = make_selector(LABELS, score_fn=lambda col, y: 0.5)
    force_thresholds(sel, alpha=0.0, beta=0.1)
    sel.last_beta, sel.last_alpha = 0.1, 0.0
    for i in range(4):
        sel.pool.add(Candidate(10 + i, i, 0.5, identity_vector(i)))
    sel._joint = lambda a, b, labels, marginal: (1.0, "stub")  # type: ignore[method-assign]
    sel._flush_pool("test")
    ev = [e for e in sel.events if e.decision == "pool_flush"][-1]
    assert not math.isnan(ev.alpha), "pool flush event lost a legitimate alpha of 0.0"


# ==========================================================================
# R6 — continuous route must remain fail-closed
# ==========================================================================


def test_R6_continuous_route_still_refuses():
    from ossfs_dd.config import UnsupportedSemanticsError

    with pytest.raises(UnsupportedSemanticsError):
        make_selector(LABELS).__class__(
            SelectionConfig(), metric_family=MetricFamily.GAUSSIAN_FISHER_Z, labels=LABELS
        )


def test_R6_unresolved_semantics_stay_unresolved():
    from ossfs_dd.reference_semantics import REFERENCE_SEMANTICS

    for sid in ("S6", "S7", "S8"):
        assert REFERENCE_SEMANTICS[sid]["status"] == "UNRESOLVED"
