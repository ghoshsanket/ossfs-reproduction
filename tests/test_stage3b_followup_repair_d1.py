"""D1 — redundancy-removal bookkeeping must reflect APPLIED removals only.

Fail-before proof
-----------------
`_redundancy_pass` collects older candidates into `to_remove` while scanning,
but if a later comparison sets ``drop_new=True`` it breaks out and the ``else``
branch that applies `to_remove` never runs. The pre-fix code still returned
(and counted) those never-applied IDs.

Triggering the exact path requires the newcomer to be **stronger** than an
early older candidate (so that one is queued for removal) and **weaker** than a
later older candidate (so that `drop_new` fires and breaks the scan).

These tests fail against the unfixed selector and pass after the minimal
diagnostic repair. The repair changes **accounting only** — selected feature
IDs and their order are asserted unchanged.
"""

from __future__ import annotations

import itertools

import numpy as np

from ossfs_dd.config import MetricFamily, Mode, SelectionConfig
from ossfs_dd.pools import Candidate, CandidateSet
from ossfs_dd.selector import Selector

LABELS = np.array(["a", "b"] * 10)
NEW_ID = 9


class _ScriptedJoint(Selector):
    """A selector whose joint quantity is scripted per (old_id, new_id)."""

    def __init__(self, redundant_pairs):
        super().__init__(
            SelectionConfig(mode=Mode.MATLAB_COMPATIBLE),
            metric_family=MetricFamily.DISCRETE_INFORMATION,
            labels=LABELS,
            entry_file="mi2",
        )
        self._redundant = redundant_pairs

    def _joint(self, a, b, labels, marginal_of_a):
        old, new = int(a[0]), int(b[0])
        if (old, new) in self._redundant:
            return 0.0, "scripted redundant"
        return 1e6, "scripted independent"


def _candidate(fid: int, score: float) -> Candidate:
    return Candidate(fid, fid, score, np.array([float(fid)] * 5))


def _sel_with(cands: list[tuple[int, float]], redundant: set) -> _ScriptedJoint:
    sel = _ScriptedJoint(redundant)
    for fid, score in cands:
        sel.candidates.add(_candidate(fid, score))
    return sel


# --------------------------------------------------------------- the D1 bug


def test_d1_queued_but_unapplied_removal_is_not_reported():
    """The canonical D1 path: candidate 3 queued, then candidate 5 drops new.

    candidate order is [3, 5, 9]; `existing` excludes the newest (9).
    - c=3: newcomer 0.7 beats old 0.5  -> old 3 queued for removal
    - c=5: newcomer 0.7 loses to 0.9  -> drop_new, scan breaks

    Only the newcomer is actually deleted, but the pre-fix code reported
    `removed == [3, 9]` and counted 2 removals.
    """
    sel = _sel_with([(3, 0.5), (5, 0.9), (NEW_ID, 0.7)],
                    redundant={(3, NEW_ID), (5, NEW_ID)})
    before_ids = [c.feature_id for c in sel.candidates]
    before_len = len(sel.candidates)
    base = sel.state.redundancy_removals

    removed = sel._redundancy_pass(_candidate(NEW_ID, 0.7), beta=1.0)

    after_ids = [c.feature_id for c in sel.candidates]
    # Fail-before proof for this case is captured in
    # results/stage3b_followup/<RUN_ID>/repair_fail_before_pass_after.json:
    # unfixed code returned removed=[3, 9] with redundancy_removals += 2.
    assert removed == [NEW_ID], (
        f"D1: candidate 3 was never actually removed but was reported: {removed}"
    )
    assert after_ids == [i for i in before_ids if i != NEW_ID]
    assert len(sel.candidates) == before_len - 1
    assert sel.state.redundancy_removals - base == 1, (
        "D1: state.redundancy_removals counted a removal that never happened"
    )


def test_d1_counter_always_equals_actual_membership_shrink():
    """Across many scripted orders the reported count must equal real shrinkage."""
    for a_red, b_red in itertools.product([True, False], repeat=2):
        for a_score, b_score, n_score in (
            (0.5, 0.9, 0.7), (0.9, 0.5, 0.7), (0.5, 0.5, 0.7), (0.1, 0.2, 0.15)
        ):
            sel = _sel_with([(3, a_score), (5, b_score), (NEW_ID, n_score)],
                            redundant={(3, NEW_ID)} if a_red else set())
            if b_red:
                sel._redundant.add((5, NEW_ID))
            before = len(sel.candidates)
            removed = sel._redundancy_pass(_candidate(NEW_ID, n_score), beta=1.0)
            actual = before - len(sel.candidates)
            assert len(removed) == actual, (
                f"D1: reported {len(removed)} removals, list shrank by {actual} "
                f"(a_red={a_red}, b_red={b_red}, scores={a_score},{b_score},{n_score})"
            )


def test_d1_multiple_true_older_removals_all_applied_in_order():
    """drop_new False: newcomer beats every old, so all are queued and applied."""
    sel = _sel_with([(3, 0.5), (5, 0.5), (7, 0.5), (NEW_ID, 0.9)],
                    redundant={(3, NEW_ID), (5, NEW_ID), (7, NEW_ID)})
    before = len(sel.candidates)
    base = sel.state.redundancy_removals

    removed = sel._redundancy_pass(_candidate(NEW_ID, 0.9), beta=1.0)

    assert removed == [3, 5, 7], "deletion order must be preserved"
    assert len(sel.candidates) == before - 3
    assert [c.feature_id for c in sel.candidates] == [NEW_ID]
    assert sel.state.redundancy_removals - base == 3


def test_d1_tie_favours_the_newer_candidate():
    """`weaker_feature` treats a tie as favouring new, so the old one goes."""
    sel = _sel_with([(3, 0.5), (NEW_ID, 0.5)], redundant={(3, NEW_ID)})
    removed = sel._redundancy_pass(_candidate(NEW_ID, 0.5), beta=1.0)
    assert removed == [3]
    assert [c.feature_id for c in sel.candidates] == [NEW_ID]


def test_d1_drop_new_with_no_queued_older_removals():
    sel = _sel_with([(3, 0.1), (5, 0.9), (NEW_ID, 0.7)],
                    redundant={(5, NEW_ID)})
    removed = sel._redundancy_pass(_candidate(NEW_ID, 0.7), beta=1.0)
    assert removed == [NEW_ID]
    assert [c.feature_id for c in sel.candidates] == [3, 5]


def test_d1_zero_removals_leaves_everything_intact():
    sel = _sel_with([(3, 0.9), (NEW_ID, 0.1)], redundant=set())
    base = sel.state.redundancy_removals
    removed = sel._redundancy_pass(_candidate(NEW_ID, 0.1), beta=1.0)
    assert removed == []
    assert len(sel.candidates) == 2
    assert sel.state.redundancy_removals == base


def test_d1_single_candidate_short_circuits():
    sel = _sel_with([(NEW_ID, 0.1)], redundant=set())
    assert sel._redundancy_pass(_candidate(NEW_ID, 0.1), beta=1.0) == []


def test_d1_accept_event_details_mirror_applied_removals():
    """The accept event's details['removed'] must match the true removals."""
    sel = _sel_with([(3, 0.5), (5, 0.9)],
                    redundant={(3, NEW_ID), (5, NEW_ID)})
    sel._accept(_candidate(NEW_ID, 0.7), beta=1.0, std=0.1)
    event = sel.events[-1]
    assert event.decision == "accept"
    assert event.details["removed"] == [NEW_ID], (
        f"D1: accept event reported {event.details['removed']} but only the "
        "newcomer was actually deleted"
    )
    assert [c.feature_id for c in sel.candidates] == [3, 5]
    assert sel.state.redundancy_removals == 1


def test_d1_candidate_set_remove_reports_real_shrinkage():
    """The repair relies on CandidateSet.remove's boolean return value."""
    cs = CandidateSet()
    cs.add(_candidate(1, 0.5))
    cs.add(_candidate(2, 0.5))
    assert cs.remove(1) is True
    assert cs.remove(1) is False          # already gone -> no shrinkage
    assert len(cs) == 1


def test_d1_repair_does_not_change_which_features_survive():
    """Accounting changes only; membership must be identical either way."""
    sel = _sel_with([(3, 0.5), (5, 0.9), (NEW_ID, 0.7)],
                    redundant={(3, NEW_ID), (5, NEW_ID)})
    sel._redundancy_pass(_candidate(NEW_ID, 0.7), beta=1.0)
    # 3 was queued but never applied, so it must still be a member
    assert [c.feature_id for c in sel.candidates] == [3, 5]
