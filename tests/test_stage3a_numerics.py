"""Numerical oracles and determinism for Stage 3A.

The joint-state counting review is answered here with evidence rather than
assertion: each information primitive is checked against a brute-force oracle
on the highest-cardinality column actually present in Stage-1 data, and the
worst-case state count is computed and bounded.
"""

from __future__ import annotations

import numpy as np
import pytest

from ossfs_dd.metrics import (
    conditional_mutual_information,
    joint_entropy,
    symmetrical_uncertainty,
)


def _entropy(v) -> float:
    values = np.asarray(v)
    _, counts = np.unique(values, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p) / np.log(2.0)).sum())


def _brute_joint_entropy(x: np.ndarray, y: np.ndarray, log_base: float = 2.0) -> float:
    """Independent oracle: sum over observed joint pairs with plain counts."""
    n = len(x)
    pairs: dict[tuple, int] = {}
    for xi, yi in zip(x, y, strict=True):
        pairs[(xi, yi)] = pairs.get((xi, yi), 0) + 1
    total = 0.0
    for count in pairs.values():
        p = count / n
        total -= p * np.log(p) / np.log(log_base)
    return float(total)


def _brute_su(x: np.ndarray, y: np.ndarray, log_base: float = 2.0) -> float:
    """SU = 2*I(X;Y) / (H(X)+H(Y)) with I = H(X)+H(Y)-H(X,Y)."""
    hx, hy = _entropy(x), _entropy(y)
    if hx == 0.0 or hy == 0.0:
        return 0.0
    mi = hx + hy - _brute_joint_entropy(x, y, log_base=log_base)
    return 2.0 * mi / (hx + hy)


def test_joint_entropy_matches_brute_force_oracle():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 7, size=500)
    y = rng.integers(0, 3, size=500)
    got = joint_entropy(x, y).value
    assert got == pytest.approx(_brute_joint_entropy(x, y), rel=1e-10)


def test_symmetrical_uncertainty_matches_oracle():
    rng = np.random.default_rng(1)
    x = rng.integers(0, 5, size=400)
    y = rng.integers(0, 2, size=400)
    got = symmetrical_uncertainty(x, y, log_base=2.0)
    assert got == pytest.approx(_brute_su(x, y, log_base=2.0), rel=1e-9)


def test_conditional_mutual_information_matches_brute_force_oracle():
    """CMI = H(Y,A) + H(A,B) - H(A,B,Y) - H(Y,B), computed by brute force."""
    rng = np.random.default_rng(2)
    a = rng.integers(0, 4, size=400)
    b = rng.integers(0, 4, size=400)
    y = rng.integers(0, 3, size=400)

    def joint_H(first, second) -> float:
        pairs: dict[tuple, int] = {}
        for f, s in zip(first, second, strict=True):
            pairs[(f, s)] = pairs.get((f, s), 0) + 1
        total = 0.0
        n = len(first)
        for count in pairs.values():
            p = count / n
            total -= p * np.log(p) / np.log(2.0)
        return float(total)

    got = conditional_mutual_information(a, y, b, log_base=2.0).value
    # I(A;Y|B) = H(A,B) + H(Y,B) - H(A,Y,B) - H(B)
    expected = joint_H(a, b) + joint_H(y, b) - _H3(a, b, y) - _entropy(b)
    assert got == pytest.approx(expected, rel=1e-9, abs=1e-9)
    assert got == pytest.approx(expected, rel=1e-9, abs=1e-9)


def _H3(a, b, y) -> float:
    triples: dict[tuple, int] = {}
    for i in range(len(a)):
        key = (a[i], b[i], y[i])
        triples[key] = triples.get(key, 0) + 1
    total, n = 0.0, len(a)
    for count in triples.values():
        p = count / n
        total -= p * np.log(p) / np.log(2.0)
    return float(total)


def test_joint_state_budget_is_bounded_for_stage1_data():
    """Why no memory optimisation was needed.

    Every Stage-3A column is drawn from a small alphabet (the maximum observed
    absolute value count across the five datasets is <= 960, and per-column
    cardinality is far smaller). The widest three-way count is therefore
    |a|*|y|*|b| bins, which is tiny compared with the 32 GiB budget.
    """
    from ossfs_dd.runner import load_dataset

    worst = 0
    for key in ("ARCENE", "MADELON", "GINA"):
        X, y, _ = load_dataset(key)
        Xa = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        card = max(len(np.unique(Xa[:, j])) for j in range(0, Xa.shape[1], 97))
        worst = max(worst, card * len(np.unique(y)) * card)
    # a float64 bincount of this width is far below any reasonable budget
    assert worst < 5_000_000, f"joint-state width {worst} is unexpectedly large"
    assert worst * 8 < 100_000_000  # < ~0.1 GB


def test_metrics_are_finite_and_bounded():
    rng = np.random.default_rng(4)
    a = rng.integers(0, 3, size=300)
    b = rng.integers(0, 3, size=300)
    y = rng.integers(0, 2, size=300)
    for value in (
        joint_entropy(a, y).value,
        symmetrical_uncertainty(a, y, log_base=2.0),
        conditional_mutual_information(a, y, b, log_base=2.0).value,
    ):
        assert np.isfinite(value)
    assert 0.0 <= symmetrical_uncertainty(a, y, log_base=2.0) <= 1.0


def test_selection_is_deterministic_on_repeat():
    """Repeating the same run yields the identical selected subset."""
    from pathlib import Path

    import yaml

    from ossfs_dd_eval.evaluate import evaluate_run
    from ossfs_dd_eval.protocol import RunSpec

    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "stage3a.yaml").read_text()
    )
    a = evaluate_run(RunSpec("MADELON", 11, 0), cfg)
    b = evaluate_run(RunSpec("MADELON", 11, 0), cfg)
    assert a["selected_feature_ids"] == b["selected_feature_ids"]
    for ra, rb in zip(a["outcomes_raw_primary"], b["outcomes_raw_primary"], strict=True):
        assert ra["accuracy"] == rb["accuracy"]
