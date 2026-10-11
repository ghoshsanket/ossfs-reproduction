"""Three-way decision regions and their mode-specific boundary rules.

The reference MATLAB is not self-consistent between its two entry files, so the
comparison operators are mode-specific and are NOT shared:

``matlab_compatible_v1`` / discrete (``OSFS_DD_mi2.m``)
    reject is tested FIRST with ``<=``, then accept with ``>=``
    (lines 52, 56). When ``sigma == 0`` the score equals the mean, so BOTH
    conditions hold and **reject wins**.

``matlab_compatible_v1`` / continuous (``OSFS_DD_z2.m``)
    accept is tested FIRST with ``>=``, then reject with ``<=``
    (lines 58, 72). Under the same degeneracy, **accept wins**.

That divergence is observable and is reproduced rather than smoothed over.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class Decision(enum.Enum):
    REJECT = "reject"
    ACCEPT = "accept"
    DELAY = "delay"


@dataclass(frozen=True)
class DecisionOutcome:
    decision: Decision
    region: str
    reason: str


def decide_matlab_discrete(score: float, alpha: float, beta: float) -> DecisionOutcome:
    """``mi2.m:52-69`` — reject first, then accept."""
    if score <= alpha:
        return DecisionOutcome(
            Decision.REJECT, "score <= alpha", "mi2.m:52 `if (mi_F<=alpha)` evaluated first"
        )
    if score >= beta:
        return DecisionOutcome(
            Decision.ACCEPT, "score >= beta", "mi2.m:56 `elseif (mi_F>=beta)`"
        )
    return DecisionOutcome(
        Decision.DELAY, "alpha < score < beta", "mi2.m:69 `else` -> undetermined pool"
    )


def decide_matlab_continuous(score: float, alpha: float, beta: float) -> DecisionOutcome:
    """``z2.m:58-74`` — accept first, then reject.

    Only reachable if continuous semantics are ever verified; see
    ``docs/stage2_source_audit.md`` §2 for why they currently are not.
    """
    if score >= beta:
        return DecisionOutcome(
            Decision.ACCEPT, "score >= beta", "z2.m:58 `if(fz_F>=beta)` evaluated first"
        )
    if score <= alpha:
        return DecisionOutcome(
            Decision.REJECT, "score <= alpha", "z2.m:72 `elseif(fz_F<=alpha)`"
        )
    return DecisionOutcome(
        Decision.DELAY, "alpha < score < beta", "z2.m:75 `else` -> undetermined pool"
    )


def decide_paper_spec(score: float, alpha: float, beta: float) -> DecisionOutcome:
    """Literal three-way specification: ``<=alpha`` discard, ``>=beta`` select.

    Identical in outcome to the discrete reference; named separately because
    ``paper_spec_v1`` also routes the first two features through this path
    rather than skipping them.
    """
    if score <= alpha:
        return DecisionOutcome(
            Decision.REJECT, "score <= alpha", "paper_spec: literal discard region"
        )
    if score >= beta:
        return DecisionOutcome(
            Decision.ACCEPT, "score >= beta", "paper_spec: literal select region"
        )
    return DecisionOutcome(
        Decision.DELAY, "alpha < score < beta", "paper_spec: literal delay region"
    )


def decide(mode: str, *, entry: str, score: float, alpha: float, beta: float) -> DecisionOutcome:
    """Dispatch by mode and by which author entry file is being mirrored."""
    if mode == "paper_spec_v1":
        return decide_paper_spec(score, alpha, beta)
    if entry == "z2":
        return decide_matlab_continuous(score, alpha, beta)
    return decide_matlab_discrete(score, alpha, beta)


# --------------------------------------------------------------------------
# pair-level comparators
# --------------------------------------------------------------------------


def is_redundant(
    joint: float, beta: float, *, mode: str
) -> tuple[bool, str]:
    """Is a candidate pair jointly redundant?

    Both modes use a strict ``< 2*beta`` (``mi2.m:138``). The difference is
    only in how ``joint`` was computed, not in the comparison.
    """
    threshold = 2.0 * beta
    redundant = joint < threshold
    return redundant, (
        f"joint={joint:.6g} {'<' if redundant else '>='} 2*beta={threshold:.6g}"
    )


def promotes_pair(
    joint: float, beta: float, *, mode: str
) -> tuple[bool, str]:
    """Does an uncertain pair qualify for promotion?

    * ``matlab_compatible_v1``: strict ``> 2*beta`` (``mi2.m:117``).
    * ``paper_spec_v1``: ``>= 2*beta`` — declared interpretation, since the
      paper writes the rule with ``>=`` while the code uses ``>``.
    """
    threshold = 2.0 * beta
    if mode == "paper_spec_v1":
        promote = joint >= threshold
        op = ">="
    else:
        promote = joint > threshold
        op = ">"
    return promote, f"joint={joint:.6g} {op} 2*beta={threshold:.6g}"


def pool_full(size: int, pool_size: int) -> bool:
    """``>=`` pool size, per ``mi2.m:73``."""
    return size >= pool_size


def weaker_feature(new_score: float, existing_score: float) -> tuple[str, str]:
    """Which member of a redundant pair is removed.

    ``mi2.m:139-144``: if ``newMI < indMI`` the NEW feature is dropped and the
    loop breaks; otherwise the EXISTING feature is dropped and the loop
    continues. A tie therefore falls to the ``else`` branch and removes the
    EXISTING feature — ties favour the newer feature.
    """
    if new_score < existing_score:
        return "new", "newMI < indMI -> discard the new feature"
    return "existing", "newMI >= indMI -> discard the existing feature (ties favour the new one)"
