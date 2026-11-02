"""Standard Gaussian Fisher r-to-z relevance for continuous features (Stage 3D).

StandardFisherZNormalizedP: full r -> z -> p -> Holm -> normalized p-evidence.
StandardFisherZEffectOnly: effect-size only (abs(r)), identical otherwise.
Both are DECLARED, independently specified methods. They are NOT the authors'
missing Fisher_Ztest helper, nor the Stage 2 reconstructed heuristic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import norm

EPS_P = 1e-12
L_EVIDENCE = 12.0
MIN_N_MARGINAL = 5
_R_CLIP = 1.0 - 1e-12


@dataclass(frozen=True)
class FisherResult:
    score: float
    r_only: float
    r: float
    z: float
    Z: float
    p_raw: float
    p_holm: float
    evidence: float
    n: int
    n_classes: int
    winning_class: str
    clip_count: int
    constant: bool
    per_class: list[dict[str, Any]] = field(default_factory=list)


def _holm(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p_values[idx])
        running = max(running, val)
        adj[idx] = running
    return adj


def marginal_fisherz(x: np.ndarray, y: np.ndarray, *, use_evidence: bool) -> FisherResult:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y).astype(str).ravel()
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch {x.shape} vs {y.shape}")
    n = int(x.size)
    if n < MIN_N_MARGINAL:
        raise ValueError(f"n={n} < {MIN_N_MARGINAL}")
    classes = np.unique(y)
    if classes.size < 2:
        raise ValueError(f"need >=2 classes, got {classes.size}")
    if not np.all(np.isfinite(x)):
        raise ValueError("non-finite x")

    x_const = bool(np.ptp(x) == 0.0)

    per_class = []
    clip_count = 0
    for c in classes:
        b = (y == c).astype(float)
        if x_const or np.ptp(b) == 0.0:
            r = 0.0
            z = 0.0
            Z = 0.0
            p_raw = 1.0
            const = True
        else:
            r = float(np.corrcoef(x, b)[0, 1])
            if not np.isfinite(r):
                r = 0.0
                const = True
                z = 0.0
                Z = 0.0
                p_raw = 1.0
            else:
                const = False
                if abs(r) > _R_CLIP:
                    clip_count += 1
                    r = float(np.clip(r, -_R_CLIP, _R_CLIP))
                z = float(np.arctanh(r))
                Z = z * math.sqrt(n - 3.0)
                p_raw = float(2.0 * norm.sf(abs(Z)))
        per_class.append({"class": str(c), "r": r, "z": z, "Z": Z,
                          "p_raw": p_raw, "constant": const})

    p_raws = [pc["p_raw"] for pc in per_class]
    p_holms = list(p_raws) if classes.size == 2 else _holm(p_raws)
    for pc, ph in zip(per_class, p_holms, strict=True):
        pc["p_holm"] = ph
        pc["evidence"] = float(np.clip(-np.log10(max(ph, EPS_P)) / L_EVIDENCE, 0.0, 1.0))
        if use_evidence:
            pc["score"] = abs(pc["r"]) * pc["evidence"]
        else:
            pc["score"] = abs(pc["r"])

    # winner: max score, tie-break by ascending class label
    best_idx = 0
    for i in range(1, len(per_class)):
        if (per_class[i]["score"] > per_class[best_idx]["score"]
                or (per_class[i]["score"] == per_class[best_idx]["score"]
                    and per_class[i]["class"] < per_class[best_idx]["class"])):
            best_idx = i
    w = per_class[best_idx]
    r_only = max(abs(pc["r"]) for pc in per_class)
    return FisherResult(
        score=float(w["score"]), r_only=float(r_only), r=float(w["r"]),
        z=float(w["z"]), Z=float(w["Z"]), p_raw=float(w["p_raw"]),
        p_holm=float(w["p_holm"]), evidence=float(w["evidence"]),
        n=n, n_classes=int(classes.size), winning_class=w["class"],
        clip_count=clip_count, constant=x_const, per_class=per_class)


def partial_correlation(x, y, z) -> tuple[float, float]:
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    z = np.asarray(z, float).ravel()
    n = x.size
    if n < 6:
        raise ValueError(f"n={n} < 6")
    if not (x.shape == y.shape == z.shape):
        raise ValueError("shape mismatch")
    if np.ptp(x) == 0 or np.ptp(y) == 0 or np.ptp(z) == 0:
        return 0.0, 1.0

    def resid(v, w):
        A = np.column_stack([np.ones_like(w), w])
        beta = np.linalg.lstsq(A, v, rcond=None)[0]
        return v - A @ beta

    rx = resid(x, z)
    ry = resid(y, z)
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return 0.0, 1.0
    r = float(np.corrcoef(rx, ry)[0, 1])
    if not np.isfinite(r):
        return 0.0, 1.0
    r = float(np.clip(r, -_R_CLIP, _R_CLIP))
    Z = float(np.arctanh(r)) * math.sqrt(n - 4.0)
    p = float(2.0 * norm.sf(abs(Z)))
    return r, p


def pair_score_standard(x, z, y, *, use_evidence: bool) -> dict:
    mx = marginal_fisherz(x, y, use_evidence=use_evidence)
    mz = marginal_fisherz(z, y, use_evidence=use_evidence)

    def cond_score(cond_var, target_var):
        best = 0.0
        best_r = 0.0
        best_p = 1.0
        classes = np.unique(y)
        for c in classes:
            b = (y == c).astype(float)
            if np.ptp(b) == 0:
                continue
            r_p, p_p = partial_correlation(target_var, b, cond_var)
            if np.ptp(target_var) == 0 or np.ptp(cond_var) == 0:
                continue
            if use_evidence:
                E = float(np.clip(-np.log10(max(p_p, EPS_P)) / L_EVIDENCE, 0.0, 1.0))
                sc = abs(r_p) * E
            else:
                sc = abs(r_p)
            if sc > best:
                best, best_r, best_p = sc, r_p, p_p
        return best, best_r, best_p

    cs_zx, r_zx, p_zx = cond_score(x, z)
    cs_xz, r_xz, p_xz = cond_score(z, x)
    joint_a = mx.score + cs_zx
    joint_b = mz.score + cs_xz
    joint = float(max(joint_a, joint_b))
    return {
        "joint": joint,
        "marginal_x": mx.score, "marginal_z": mz.score,
        "conditional_z_given_x": cs_zx, "conditional_x_given_z": cs_xz,
        "r_partial_z_given_x": r_zx, "r_partial_x_given_z": r_xz,
        "p_partial_z_given_x": p_zx, "p_partial_x_given_z": p_xz,
        "symmetry": "max of the two directed conditionals",
        "label": "partial_correlation_heuristic",
    }


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adj = [0.0] * m
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        val = min(prev, p_values[idx] * m / (rank + 1))
        adj[idx] = min(1.0, val)
        prev = adj[idx]
    return adj
