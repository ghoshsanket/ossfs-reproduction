"""Experiment A: one standardized SVC fit on a fixed Stage 3A subset.

Standardization is frozen on for every cell and fitted on the selected
TRAINING rows only. Subsets are read verbatim from the Stage 3A raw records;
nothing is reselected, sorted or extended.

Status semantics are kept strictly separate:

  CONVERGED    solver finished without hitting the cap and without warning
  NONCONVERGED predictions exist but the cap was hit / early termination warned;
               its accuracy is an **incomplete-fit diagnostic** and is stored in
               its own field so it can never be pooled with converged accuracy
  TIMEOUT      worker killed at the hard wall-clock limit; accuracy is None
  FIT_ERROR    the fit raised; accuracy is None
  BLOCKED_INPUT a required Stage 3A input was missing or inconsistent
"""

from __future__ import annotations

import resource
import time
import warnings
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

A_SEEDS = (11, 17, 23)
A_FOLDS = (0, 1, 2, 3, 4)
A_C_VALUES = (0.1, 1.0, 10.0)
A_TOL_VALUES = (0.001, 0.0001)
A_MAX_ITER = 2_000_000
A_DATASETS = ("MADELON", "GINA", "GISETTE")

CONVERGED, NONCONVERGED = "CONVERGED", "NONCONVERGED"
TIMEOUT, FIT_ERROR, BLOCKED_INPUT = "TIMEOUT", "FIT_ERROR", "BLOCKED_INPUT"


def fit_ids() -> list[str]:
    """The 270 planned fit identifiers, deterministic order."""
    return [
        f"{ds}_seed{seed}_fold{fold}__C{c}__tol{tol}"
        for ds in A_DATASETS
        for seed in A_SEEDS
        for fold in A_FOLDS
        for c in A_C_VALUES
        for tol in A_TOL_VALUES
    ]


def extract_columns(X: Any, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    if sparse.issparse(X):
        return np.column_stack([np.asarray(X[rows, c].todense()).ravel() for c in cols])
    return np.column_stack([np.asarray(X[rows, c]).ravel() for c in cols])


def _sha(obj: Any) -> str:
    import hashlib
    import json

    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def run_one_fit(*, fit_id: str, dataset: str, seed: int, fold: int,
                subset_record: dict[str, Any], X: Any, y: np.ndarray,
                train_idx: np.ndarray, test_idx: np.ndarray,
                C: float, tol: float) -> dict[str, Any]:
    """Fit one (subset, C, tol) cell on standardized selected features."""
    record: dict[str, Any] = {
        "schema": "ossfs-dd-reproduction/stage3b_followup/svm_fit/1.0",
        "fit_id": fit_id,
        "dataset": dataset, "split_seed": seed, "fold": fold,
        "C": C, "tol": tol, "max_iter": A_MAX_ITER,
        "scale_variant": "standardized",
        "kernel": "linear",
        "n_selected": len(subset_record["selected_feature_ids"]),
        "subset_ids_sha256": _sha(subset_record["selected_feature_ids"]),
        "train_index_sha256": _sha([int(v) for v in train_idx]),
        "test_index_sha256": _sha([int(v) for v in test_idx]),
        "stage3a_run_id": subset_record["run_id"],
    }

    try:
        cols = np.asarray(subset_record["selected_feature_ids"], dtype=np.int64)
        if len(set(cols.tolist())) != cols.size:
            return {**record, "status": BLOCKED_INPUT, "accuracy": None,
                    "error": "duplicate feature ids in the Stage 3A subset"}
        Xtr = extract_columns(X, train_idx, cols)
        Xte = extract_columns(X, test_idx, cols)
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)          # SELECTED TRAINING ROWS ONLY
        Xte = scaler.transform(Xte)
    except Exception as exc:  # noqa: BLE001
        return {**record, "status": BLOCKED_INPUT, "accuracy": None,
                "error": f"{type(exc).__name__}: {exc}"}

    ytr, yte = y[train_idx], y[test_idx]
    record["n_train"] = int(Xtr.shape[0])
    record["n_test"] = int(Xte.shape[0])
    record["class_counts_train"] = {str(k): int(v)
                                    for k, v in zip(*np.unique(ytr, return_counts=True), strict=True)}
    record["class_counts_test"] = {str(k): int(v)
                                   for k, v in zip(*np.unique(yte, return_counts=True), strict=True)}
    record["scaler_mean_sha256"] = _sha([round(float(v), 12) for v in scaler.mean_])
    record["scaler_scale_sha256"] = _sha([round(float(v), 12) for v in scaler.scale_])
    record["n_nonfinite_train"] = int((~np.isfinite(Xtr)).sum())
    record["n_nonfinite_test"] = int((~np.isfinite(Xte)).sum())

    est = SVC(kernel="linear", C=float(C), tol=float(tol), max_iter=A_MAX_ITER)
    record["estimator"] = {
        "class_module": type(est).__module__,
        "class_name": type(est).__name__,
        "backend": "libsvm",
        "params": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                   for k, v in sorted(est.get_params(deep=False).items())},
    }

    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            est.fit(Xtr, ytr)
            early = [str(w.message) for w in caught
                     if "Convergence" in str(w.message)
                     or "terminated early" in str(w.message)
                     or "max_iter" in str(w.message)]
        yhat = est.predict(Xte)
    except Exception as exc:  # noqa: BLE001
        return {**record, "status": FIT_ERROR, "accuracy": None, "converged": None,
                "seconds": round(time.perf_counter() - t0, 4),
                "error": f"{type(exc).__name__}: {exc}"}
    seconds = time.perf_counter() - t0

    n_iter_vec = getattr(est, "n_iter_", None)
    n_iter_vec = [int(v) for v in np.ravel(n_iter_vec)] if n_iter_vec is not None else None
    fit_status = getattr(est, "fit_status_", None)
    if hasattr(fit_status, "item"):
        fit_status = fit_status.item()
    hit_cap = bool(n_iter_vec) and max(n_iter_vec) >= A_MAX_ITER
    converged = (not hit_cap) and not early

    n_test = int(yte.shape[0])
    n_correct = int(np.sum(yhat == yte))
    record.update(
        status=CONVERGED if converged else NONCONVERGED,
        converged=converged,
        hit_iteration_cap=hit_cap,
        early_termination_warnings=early,
        n_iter_vector=n_iter_vec,
        n_iter_max=(max(n_iter_vec) if n_iter_vec else None),
        fit_status=fit_status,
        n_correct=n_correct,
        accuracy=(n_correct / n_test) if n_test else None,
        accuracy_is_incomplete_fit_diagnostic=not converged,
        seconds=round(seconds, 4),
        peak_rss_gib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2), 4),
        error=None,
    )
    return record
