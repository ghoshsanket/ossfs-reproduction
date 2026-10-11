"""The reconstructed Fisher-Z statistics, implemented exactly as configured.

Everything here is a DECLARED RECONSTRUCTION. ``Fisher_Ztest`` is absent from
the authors' repository, so none of these formulas is claimed to reproduce it.
They are a transparent, testable stand-in that lets the continuous route be
exercised and reasoned about.

Formulas (see ``configs/continuous_rnd_v1.yaml`` for the frozen text):

marginal
    For each training class ``c``:  ``b_c = 1[y == c]``, ``r_c = pearson(x, b_c)``
    (neutral 0 if ``x`` or ``b_c`` is constant), require ``n >= 5``,
    ``z_c = atanh(clip(r_c))``, ``SE = 1/sqrt(n-3)``,
    ``q = Phi^-1(1 - 0.05/(2K))``, ``score_c = tanh(max(0, |z_c| - q*SE))``,
    ``score(x, y) = max_c score_c``.

pair (heuristic)
    Fit ``b_c ~ 1 + x + z`` on training rows only, take the in-sample multiple
    correlation ``R_c``, and ``score_pair_c = tanh(max(0, atanh(R_c) - q/sqrt(n-4)))``.
    Require ``n >= 6``. This is NOT a Fisher conditional-independence statistic;
    the label ``heuristic_joint_v1`` is carried everywhere it is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import norm

ALPHA = 0.05
MIN_N_MARGINAL = 5
MIN_N_PAIR = 6
_R_CLIP = 1.0 - 1e-12


class UnsupportedSampleSize(ValueError):
    """Raised when ``n`` is too small for the declared estimator."""


@dataclass(frozen=True)
class MarginalScore:
    score: float
    r: float
    z: float
    p_two_sided: float
    n: int
    n_classes: int
    q: float
    constant: bool = False
    per_class: list[dict[str, Any]] = field(default_factory=list)


def _bonferroni_q(n_classes: int, alpha: float = ALPHA) -> float:
    if n_classes < 2:
        raise ValueError(f"q needs >= 2 classes, got {n_classes}")
    return float(norm.ppf(1.0 - alpha / (2.0 * n_classes)))


def marginal_fisher_z(
    x: np.ndarray, y: np.ndarray, *, alpha: float = ALPHA
) -> MarginalScore:
    """Reconstructed marginal score for one feature column."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y).astype(str).ravel()
    if x.shape != y.shape:
        raise ValueError(f"x/y length mismatch: {x.shape} vs {y.shape}")
    n = int(x.size)
    if n < MIN_N_MARGINAL:
        raise UnsupportedSampleSize(f"marginal needs n>={MIN_N_MARGINAL}, got {n}")
    classes = np.unique(y)
    if classes.size < 2:
        raise ValueError(f"need >= 2 classes, got {classes.size}")
    q = _bonferroni_q(int(classes.size), alpha)

    per_class: list[dict[str, Any]] = []
    best = 0.0
    best_r = 0.0
    best_z = 0.0
    best_p = 1.0
    constant = bool(np.ptp(x) == 0.0)

    for c in classes:
        b = (y == c).astype(np.float64)
        if np.ptp(x) == 0.0 or np.ptp(b) == 0.0:
            r = 0.0                      # declared neutral 0
            z_val, p_val, s = 0.0, 1.0, 0.0
        else:
            r = float(np.corrcoef(x, b)[0, 1])
            r = float(np.clip(r, -_R_CLIP, _R_CLIP))
            z_val = float(np.arctanh(r))
            se = 1.0 / np.sqrt(n - 3.0)
            # The Fisher-Z test statistic is z * sqrt(n-3); the p-value must use
            # that statistic, not the raw z (which is not standard normal).
            p_val = float(2.0 * norm.sf(abs(z_val / se)))
            s = float(np.tanh(max(0.0, abs(z_val) - q * se)))
        per_class.append({"class": str(c), "r": r, "z": z_val, "p_two_sided": p_val, "score": s})
        if s > best:
            best, best_r, best_z, best_p = s, r, z_val, p_val

    return MarginalScore(
        score=float(best), r=best_r, z=best_z, p_two_sided=best_p,
        n=n, n_classes=int(classes.size), q=q, constant=constant, per_class=per_class,
    )


@dataclass(frozen=True)
class PairScore:
    score: float
    r_multiple: float
    condition_number: float
    n: int
    q: float
    status: str = "heuristic_joint_v1"


def _fit_multiple_correlation(x: np.ndarray, z: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """In-sample multiple correlation of ``b`` on ``(1, x, z)``; pinv if singular."""
    A = np.column_stack([np.ones_like(x), x, z])
    if np.ptp(b) == 0.0:
        return 0.0, 1.0
    try:
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        cond = float(np.linalg.cond(A))
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(A) @ b
        cond = float("inf")
    resid = b - A @ coef
    ss_tot = float(((b - b.mean()) ** 2).sum())
    ss_res = float((resid ** 2).sum())
    r2 = 0.0 if ss_tot == 0.0 else max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
    return float(np.sqrt(r2)), cond


def pair_heuristic_fisher_z(
    a: np.ndarray, z: np.ndarray, y: np.ndarray, *, alpha: float = ALPHA
) -> PairScore:
    """Reconstructed PAIR score. Heuristic, not a conditional-independence test."""
    a = np.asarray(a, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64).ravel()
    y = np.asarray(y).astype(str).ravel()
    n = int(a.size)
    if n < MIN_N_PAIR:
        raise UnsupportedSampleSize(f"pair needs n>={MIN_N_PAIR}, got {n}")
    classes = np.unique(y)
    if classes.size < 2:
        raise ValueError(f"need >= 2 classes, got {classes.size}")
    q = _bonferroni_q(int(classes.size), alpha)

    best, best_r, best_cond = 0.0, 0.0, 1.0
    for c in classes:
        b = (y == c).astype(np.float64)
        r_m, cond = _fit_multiple_correlation(a, z, b)
        r_c = float(np.clip(r_m, 0.0, _R_CLIP))
        s = float(np.tanh(max(0.0, float(np.arctanh(r_c)) - q / np.sqrt(n - 4.0))))
        if s > best:
            best, best_r, best_cond = s, r_c, cond
    return PairScore(
        score=best, r_multiple=best_r, condition_number=best_cond, n=n, q=q
    )
