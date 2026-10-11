"""Experiment B: ARCENE decision trajectory + downstream classifier accuracy.

For each of the 50 crossed (fold, order) cells the repaired selector is
re-run, its ordered selected IDs are compared against the saved Stage 3B run,
and three frozen classifiers are trained on that subset using the **same fixed
fold**. Plus all-feature baselines for contextual comparison.

Leakage contract: every subset comes from training rows only. Held-out labels
are touched only to score frozen predictions; nothing is retuned.
"""

from __future__ import annotations

import hashlib
import json
import resource
import time
import warnings
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

CART_SEED = 11          # frozen across all ten orders within a fold
SVM_MAX_ITER = 2_000_000


def _sha(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def extract_columns(X: Any, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    if sparse.issparse(X):
        return np.column_stack([np.asarray(X[rows, c].todense()).ravel() for c in cols])
    return np.column_stack([np.asarray(X[rows, c]).ravel() for c in cols])


def build_estimator(name: str):
    """The three frozen classifiers. Parameters are never tuned."""
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=5, weights="uniform",
                                    metric="minkowski", p=2)
    if name == "cart":
        return DecisionTreeClassifier(criterion="gini", splitter="best",
                                      random_state=CART_SEED)
    if name == "svm":
        return SVC(kernel="linear", C=1.0, tol=0.001, max_iter=SVM_MAX_ITER)
    raise ValueError(f"unknown classifier {name!r}")


def evaluate_classifier(name: str, Xtr: np.ndarray, ytr: np.ndarray,
                        Xte: np.ndarray, yte: np.ndarray) -> dict[str, Any]:
    """Fit one frozen classifier and score it on the held-out rows."""
    est = build_estimator(name)
    t0 = time.perf_counter()
    early: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            est.fit(Xtr, ytr)
            early = [str(w.message) for w in caught
                     if "Convergence" in str(w.message)
                     or "terminated early" in str(w.message)]
        yhat = est.predict(Xte)
    except Exception as exc:  # noqa: BLE001
        return {"classifier": name, "status": "FIT_ERROR", "accuracy": None,
                "balanced_accuracy": None, "converged": None,
                "error": f"{type(exc).__name__}: {exc}",
                "seconds": round(time.perf_counter() - t0, 4)}
    seconds = time.perf_counter() - t0

    n_iter = getattr(est, "n_iter_", None)
    n_iter_vec = [int(v) for v in np.ravel(n_iter)] if n_iter is not None else None
    hit_cap = bool(n_iter_vec) and max(n_iter_vec) >= SVM_MAX_ITER
    converged = (not hit_cap) and not early
    n_test = int(yte.shape[0])
    n_correct = int(np.sum(yhat == yte))
    return {
        "classifier": name,
        "status": "COMPLETED",
        "n_train": int(Xtr.shape[0]),
        "n_test": n_test,
        "n_correct": n_correct,
        "accuracy": (n_correct / n_test) if n_test else None,
        "balanced_accuracy": (float(balanced_accuracy_score(yte, yhat))
                              if n_test else None),
        "converged": bool(converged),
        "n_iter_vector": n_iter_vec,
        "n_iter_max": (max(n_iter_vec) if n_iter_vec else None),
        "early_termination_warnings": early,
        "seconds": round(seconds, 4),
        "peak_rss_gib": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2), 4),
        "estimator_params": {
            k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
            for k, v in sorted(est.get_params(deep=False).items())
        },
        "error": None,
    }


def subset_feature_hash(cols: list[int]) -> str:
    return _sha([int(c) for c in cols])
