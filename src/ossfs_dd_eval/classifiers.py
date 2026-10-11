"""The three fixed classifiers, and the empty-selection contract.

The paper does not publish the exact sklearn configuration, so the frozen
Stage-3A config declares one and it is not tuned. Primary results use raw stored
feature values with no scaling; a standardized variant exists as a separately
labelled sensitivity analysis, never a silent default.

A raw-scale linear SVC does not reach convergence on MADELON (a 224x
per-column scale ratio). Rather than silently re-tune or silently standardize,
the solver's own documented iteration cap is made explicit and every SVM
outcome carries a ``converged`` flag so a non-converged fit is visible instead
of being reported as a converged accuracy.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

EMPTY_STATUS = "NOT_EVALUABLE_EMPTY_SELECTION"

# liblinear's documented iteration cap; surfaced so the run is bounded.
SVM_MAX_ITER = 2_000_000


@dataclass(frozen=True)
class ClassifierSpec:
    name: str
    factory: Any
    seed_dependent: bool

    def build(self, seed: int) -> Any:
        """Un-tuned estimator. ``seed`` only reaches the tree for determinism."""
        if self.name == "cart":
            return self.factory(random_state=seed)
        return self.factory()


# Frozen, un-tuned, and identical for every dataset/seed/fold.
CLASSIFIERS: dict[str, ClassifierSpec] = {
    "knn": ClassifierSpec(
        "knn",
        lambda: KNeighborsClassifier(
            n_neighbors=5, weights="uniform", metric="minkowski", p=2
        ),
        seed_dependent=False,
    ),
    # SVC's `probability` default is already False; passing it explicitly only
    # emits a sklearn FutureWarning without changing behaviour.
    "svm": ClassifierSpec(
        "svm",
        lambda: SVC(kernel="linear", C=1.0, max_iter=SVM_MAX_ITER),
        seed_dependent=False,
    ),
    "cart": ClassifierSpec(
        "cart",
        lambda **kw: DecisionTreeClassifier(criterion="gini", splitter="best", **kw),
        seed_dependent=True,
    ),
}


def prepare(X_train: np.ndarray, X_test: np.ndarray, *, standardize: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return the matrices to classify on.

    ``standardize=False`` (the primary path) passes raw values through. The
    optional variant fits the scaler on TRAINING ROWS ONLY.
    """
    if not standardize:
        return X_train, X_test
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    train = scaler.fit_transform(X_train)  # fitted on train rows only
    return train, scaler.transform(X_test)


def fit_predict(
    name: str,
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one estimator and return its predictions plus convergence metadata.

    ``ConvergenceWarning`` is captured rather than allowed to scroll past: an
    unconverged SVM is a real limitation of the primary configuration and is
    reported as such.
    """
    est = CLASSIFIERS[name].build(seed)
    meta: dict[str, Any] = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        est.fit(X_train, y_train)
        for w in caught:
            # sklearn signals early solver termination as "Solver terminated early
            # (max_iter=...)" for liblinear and as a ConvergenceWarning elsewhere.
            # Matching only "Convergence" would mislabel a capped liblinear run
            # as converged, which is exactly the failure this flag exists to catch.
            msg = str(w.message)
            if "Convergence" in msg or "terminated early" in msg or "max_iter" in msg:
                meta["converged"] = False
                meta["warning"] = msg.split(".")[0]
    if name == "svm":
        it = getattr(est, "n_iter_", None)
        if it is not None:
            n_iter = int(np.ravel(it)[0])
            meta["n_iter"] = n_iter
            # An iteration count that lands exactly on the cap is non-convergence
            # even when the solver emits no warning.
            if n_iter >= SVM_MAX_ITER:
                meta["converged"] = False
                meta.setdefault("warning", f"n_iter reached the max_iter cap {SVM_MAX_ITER}")
    if "converged" not in meta:
        meta["converged"] = True
    return est.predict(X_test), meta
