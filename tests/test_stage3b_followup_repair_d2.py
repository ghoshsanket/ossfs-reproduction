"""D2 — prefix checkpoints must be chronological snapshots, not end-of-run totals.

Fail-before proof
-----------------
`checkpoint_rows` accumulated **every** event into `cumulative` *before* the
checkpoint loop began, so the checkpoint at arrival 100 reported decisions that
had not happened yet — typically the full run totals.

These tests build hand-crafted chronological events with known partial totals
and assert each prefix checkpoint reflects only the prefix. They fail against
the unfixed trace module.
"""

from __future__ import annotations

import numpy as np

from ossfs_dd.selector import Event
from ossfs_dd_stage3b.trace import checkpoint_rows

K1, K2 = 1.0, 2.0


def _arrival(pos: int, decision: str, removed=None) -> Event:
    details = {"removed": removed} if removed is not None else {}
    return Event(
        arrival_position=pos, feature_id=pos * 10, score=0.5 + pos * 1e-4,
        decision=decision, region="r", reason="r",
        alpha=0.5, beta=1.0, sigma=0.5, details=details,
    )


def _flush(promoted: list[int], pool_size: int) -> Event:
    return Event(
        arrival_position=-1, feature_id=-1, score=float("nan"),
        decision="pool_flush", region="flush", reason="flushDeterPool",
        alpha=0.5, beta=1.0, sigma=0.5,
        details={"promoted": promoted, "pool_size": pool_size},
    )


def _scores(n: int) -> list[float]:
    return [0.5 + i * 1e-4 for i in range(n)]


def _cp(rows, pos):
    return next(r for r in rows if r["arrival_position"] == pos)


# ------------------------------------------------------------- the D2 bug


def test_d2_early_checkpoint_excludes_later_events():
    """Checkpoint 100 must reflect only the first 100 arrivals."""
    events = [_arrival(i, "accept" if i < 50 else "reject") for i in range(400)]
    rows = checkpoint_rows(events, _scores(400), K1, K2)

    at100 = _cp(rows, 100)["cumulative"]
    # Fail-before proof is captured in repair_fail_before_pass_after.json:
    # unfixed code reported accept=50/reject=350 at checkpoint 100.
    assert at100["accept"] == 50
    assert at100["reject"] == 50
    at200 = _cp(rows, 200)["cumulative"]
    assert at200["accept"] == 50 and at200["reject"] == 150


def test_d2_pool_flush_is_not_an_arrival():
    """Flush events carry position -1 and must not inflate arrival counts."""
    events = [_arrival(i, "delay") for i in range(100)]
    events.insert(50, _flush(promoted=[1, 2, 3], pool_size=100))
    rows = checkpoint_rows(events, _scores(100), K1, K2)
    at100 = _cp(rows, 100)
    assert at100["n_scores"] == 100
    assert at100["cumulative"]["delay"] == 100
    assert at100["cumulative"]["pool_flush"] == 1
    assert at100["cumulative"]["promoted"] == 3


def test_d2_flush_attributed_to_its_triggering_arrival():
    """A flush that occurs after arrival N belongs to the window ending at N."""
    events = []
    for i in range(300):
        events.append(_arrival(i, "delay"))
        if i == 149:                      # flush triggered by arrival 149
            events.append(_flush(promoted=[7], pool_size=100))
    rows = checkpoint_rows(events, _scores(300), K1, K2)
    # checkpoint 100 is at index/position 100 -> only the first 100 arrivals
    assert _cp(rows, 100)["cumulative"]["pool_flush"] == 0
    assert _cp(rows, 200)["cumulative"]["pool_flush"] == 1


def test_d2_exact_boundary_at_100():
    """An exact 100-arrival boundary must land on the checkpoint."""
    events = [_arrival(i, "reject") for i in range(100)]
    rows = checkpoint_rows(events, _scores(100), K1, K2)
    assert _cp(rows, 100)["cumulative"]["reject"] == 100


def test_d2_final_flush_counted_after_last_arrival_not_before():
    """The final flush must not appear in the last arrival checkpoint."""
    events = [_arrival(i, "delay") for i in range(100)]
    events.append(_flush(promoted=[1, 2], pool_size=100))     # final flush
    rows = checkpoint_rows(events, _scores(100), K1, K2)
    assert _cp(rows, 100)["cumulative"]["pool_flush"] == 0, (
        "D2: the post-stream final flush leaked into the arrival-100 checkpoint"
    )
    terminal = rows[-1]
    assert terminal["cumulative"]["pool_flush"] == 1, (
        "the final flush must still be counted in the terminal summary"
    )
    assert terminal["cumulative"]["promoted"] == 2


def test_d2_removals_use_applied_ids_only():
    """`removed` in accept events is the applied-removal ledger."""
    events = [
        _arrival(0, "accept", removed=[11, 12]),
        _arrival(1, "reject"),
        _arrival(2, "accept", removed=[13]),
    ] + [_arrival(i, "reject") for i in range(3, 100)]
    rows = checkpoint_rows(events, _scores(100), K1, K2)
    assert _cp(rows, 100)["cumulative"]["removed"] == 3


def test_d2_thresholds_recomputed_per_prefix():
    rows = checkpoint_rows(
        [_arrival(i, "reject") for i in range(200)], _scores(200), K1, K2)
    for pos in (100, 200):
        cp = _cp(rows, pos)
        prefix = np.asarray(_scores(pos))
        assert cp["recomputed"]["mu"] == float(prefix.mean())
        assert cp["recomputed"]["sigma"] == float(prefix.std(ddof=1))
        assert cp["recomputed"]["alpha"] == float(prefix.mean() + prefix.std(ddof=1))
        assert cp["recomputed"]["beta"] == float(prefix.mean() + 2 * prefix.std(ddof=1))


def test_d2_checkpoint_counters_are_monotonic_non_decreasing():
    events = [_arrival(i, "accept" if i % 3 == 0 else "delay") for i in range(300)]
    events += [_flush(promoted=[1], pool_size=100), _flush(promoted=[2], pool_size=100)]
    rows = checkpoint_rows(events, _scores(300), K1, K2)
    for key in ("accept", "reject", "delay", "pool_flush", "promoted", "removed"):
        series = [r["cumulative"][key] for r in rows]
        assert all(b >= a for a, b in zip(series, series[1:], strict=False)), (
            f"D2: cumulative[{key}] is not monotonic: {series}"
        )


def test_d2_earlier_checkpoint_not_equal_to_final_totals():
    """The heart of the defect: early checkpoints duplicated the final totals."""
    events = [_arrival(i, "reject") for i in range(400)]
    rows = checkpoint_rows(events, _scores(400), K1, K2)
    first, last = _cp(rows, 100)["cumulative"], rows[-1]["cumulative"]
    assert first["reject"] != last["reject"], (
        "D2: checkpoint 100 still equals the final total"
    )
