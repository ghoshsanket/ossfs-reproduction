"""Information-theoretic primitives with MIToolbox-compatible semantics.

Every quantity here is computed in **bits**, confirmed from the authors'
vendored C source: `mi/Entropy.c:50` accumulates `-= p*log(p)` and line 54 does
`entropy /= log(2.0)`; `mi/MutualInformation.c:63` does the same.

Implementation notes
--------------------
* States are the **original unique values**. No binning, quantisation or
  discretisation of continuous values ever happens here.
* No n-dimensional joint state is materialised. `H(X,Z)` is built by pairing
  the two 1-D state vectors into tuples, which is `O(n)` memory, not
  `O(prod(|X_i|))`.
* No feature-by-feature Gram matrix is formed.
* Floating-point noise must not produce a negative mutual information: raw MI
  is clamped at 0.0 when it lands below zero within tolerance, and the clamp
  is counted so callers can see it happened.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

NEGATIVE_MI_TOLERANCE = 1e-12


@dataclass(frozen=True)
class InformationResult:
    value: float
    negative_clamped: bool = False
    n_states: int = 0


def _codes(values: np.ndarray) -> np.ndarray:
    """Dense integer codes for the original unique states.

    Accepts 1-D ``(n,)`` and 2-D ``(k, n)``. A 2-D input is treated as *k*
    separate variables and encoded as tuples of per-column codes, so
    ``stack([X, Z])`` becomes one joint categorical variable. This keeps the
    memory cost O(k*n) instead of materialising a product state space.
    """
    v = np.asarray(values)
    if v.ndim == 1:
        _, inv = np.unique(v, return_inverse=True)
        return inv.astype(np.int64, copy=False).ravel()
    if v.ndim == 2:
        n_rows = v.shape[1]
        if n_rows == 0:
            return np.zeros(0, dtype=np.int64)
        # Mixed-radix product encoding: each column's codes are appended as a
        # new (more significant) digit. Adding offsets instead would collide —
        # (x=0,y=1) and (x=1,y=0) would share a code — and silently merge joint
        # states, which is exactly what the brute-force CMI oracle caught.
        combined = np.zeros(n_rows, dtype=np.int64)
        for row in v:
            _, inv = np.unique(row, return_inverse=True)
            n_states = int(inv.max()) + 1 if inv.size else 1
            combined = combined * n_states + inv.astype(np.int64, copy=False)
        return combined
    raise ValueError(f"expected a 1-D or 2-D array, got shape {v.shape}")


def _counts_from_codes(codes: np.ndarray, n_states: int) -> np.ndarray:
    return np.bincount(codes, minlength=n_states).astype(np.float64)


def entropy(x: np.ndarray, *, log_base: float = 2.0) -> InformationResult:
    """H(X) = -sum p log p, in units of ``log_base``."""
    codes = _codes(x)
    n = codes.size
    if n == 0:
        return InformationResult(0.0, n_states=0)
    counts = _counts_from_codes(codes, int(codes.max()) + 1)
    probs = counts[counts > 0] / n
    value = float(-np.sum(probs * np.log(probs)) / math.log(log_base))
    return InformationResult(value, n_states=int(counts.size))


def joint_entropy(x: np.ndarray, y: np.ndarray, *, log_base: float = 2.0) -> InformationResult:
    """H(X,Y) via paired state tuples — O(n) memory, no n-dim table."""
    cx, cy = _codes(x), _codes(y)
    if cx.size != cy.size:
        raise ValueError(f"length mismatch: {cx.size} vs {cy.size}")
    n = cx.size
    if n == 0:
        return InformationResult(0.0)
    nx = int(cx.max()) + 1
    ny = int(cy.max()) + 1
    flat = cx * ny + cy
    counts = np.bincount(flat, minlength=nx * ny).astype(np.float64)
    probs = counts[counts > 0] / n
    value = float(-np.sum(probs * np.log(probs)) / math.log(log_base))
    return InformationResult(value, n_states=int((counts > 0).sum()))


def mutual_information(x: np.ndarray, y: np.ndarray, *, log_base: float = 2.0) -> InformationResult:
    """I(X;Y) = H(X) + H(Y) - H(X,Y)."""
    hx = entropy(x, log_base=log_base).value
    hy = entropy(y, log_base=log_base).value
    hxy = joint_entropy(x, y, log_base=log_base).value
    raw = hx + hy - hxy
    negative = raw < 0.0
    if negative and raw > -NEGATIVE_MI_TOLERANCE:
        raw = 0.0
    return InformationResult(float(raw), negative_clamped=bool(negative))


def conditional_mutual_information(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, *, log_base: float = 2.0
) -> InformationResult:
    """I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(Z) - H(X,Y,Z).

    Equivalently I(X;Y|Z) = I(X;Y) - I(X;Z) + I(Y;Z); the entropy form is used
    because it matches the conditional-entropy primitive MIToolbox exposes.
    """
    hxz = joint_entropy(x, z, log_base=log_base).value
    hyz = joint_entropy(y, z, log_base=log_base).value
    hz = entropy(z, log_base=log_base).value
    hxyz = joint_entropy(np.stack([x, y]), z, log_base=log_base).value  # (2, n)
    raw = hxz + hyz - hz - hxyz
    negative = raw < 0.0
    if negative and raw > -NEGATIVE_MI_TOLERANCE:
        raw = 0.0
    return InformationResult(float(raw), negative_clamped=bool(negative))


def symmetrical_uncertainty(x: np.ndarray, y: np.ndarray, *, log_base: float = 2.0) -> float:
    """SU = 2*I(X;Y) / (H(X)+H(Y)), in [0,1].

    This is exactly ``SU.m``: lines 3-7 compute ``(2*iXY)/(hX+hY)``.

    Degenerate case: if both entropies are 0 the MATLAB expression is ``0/0``
    and yields NaN. ``matlab_compatible_v1`` reproduces that; ``paper_spec_v1``
    passes ``neutral_zero_entropy=True`` to get the documented neutral 0.0.
    Returning an invented "perfect association" for a constant feature would be
    wrong, so neither mode ever returns 1.0 here.
    """
    hx = entropy(x, log_base=log_base).value
    hy = entropy(y, log_base=log_base).value
    denom = hx + hy
    if denom == 0.0:
        return float("nan")
    mi = mutual_information(x, y, log_base=log_base).value
    return float(2.0 * mi / denom)


def normalised_su_of_joint(
    x: np.ndarray, z: np.ndarray, y: np.ndarray, *, log_base: float = 2.0
) -> float:
    """SU({X,Z};Y) — the scale-coherent joint definition used by paper_spec_v1.

    Treating ``(X,Z)`` as one joint categorical variable keeps every term on
    the same normalised [0,1] scale as the marginal score, so comparing it
    against ``2*beta`` is meaningful. This is deliberately NOT what the authors'
    code computes.
    """
    joint = np.stack([x, z])
    hxz = entropy(joint, log_base=log_base).value
    hy = entropy(y, log_base=log_base).value
    denom = hxz + hy
    if denom == 0.0:
        return float("nan")
    mi = mutual_information(joint, y, log_base=log_base).value
    return float(2.0 * mi / denom)


# --------------------------------------------------------------------------
# Continuous / Fisher-Z machinery — deliberately NOT invented.
# --------------------------------------------------------------------------


def fisher_z_blocked(reason: str, source: str) -> None:
    """Always raises. See ``docs/stage2_source_audit.md`` §2."""
    from .config import UnsupportedSemanticsError

    raise UnsupportedSemanticsError("gaussian_fisher_z", reason, source)
