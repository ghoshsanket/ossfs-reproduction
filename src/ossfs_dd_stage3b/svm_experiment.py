"""Experiment A: SVM conditioning on the already-selected Stage 3A subsets.

No feature selection happens here. Each run reads a Stage 3A record's
``selected_feature_ids`` verbatim and refits ``SVC(kernel='linear')`` under a
prespecified (scale_variant, max_iter) cell.

Status semantics (never silently collapsed):
  CONVERGED     -- fit completed and the solver did not hit its cap
  NONCONVERGED  -- predictions exist but the solver stopped at the cap or
                   warned about early termination. Accuracy is still recorded,
                   but ONLY as an incomplete-fit diagnostic, kept in a separate
                   column from converged accuracies everywhere.
  TIMEOUT       -- the worker was killed at the wall-clock limit; accuracy null
  FIT_ERROR     -- the fit raised; accuracy null
  BLOCKED_INPUT -- a required Stage 3A input was missing or inconsistent
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

from .protocol import FitID

STATUS_CONVERGED = "CONVERGED"
STATUS_NONCONVERGED = "NONCONVERGED"
STATUS_TIMEOUT = "TIMEOUT"
STATUS_FIT_ERROR = "FIT_ERROR"
STATUS_BLOCKED_INPUT = "BLOCKED_INPUT"


def extract_columns(X: Any, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Dense column block for the given rows, CSR- and ndarray-safe."""
    if sparse.issparse(X):
        return np.column_stack([
            np.asarray(X[rows, c].todense()).ravel() for c in cols
        ])
    return np.column_stack([np.asarray(X[rows, c]).ravel() for c in cols])


def prepare_inputs(X: Any, train_idx: np.ndarray, test_idx: np.ndarray,
                   cols: list[int], scale_variant: str) -> dict[str, Any]:
    """Hold the subset fixed; vary only the training-row-only scaling."""
    col_arr = np.asarray(cols, dtype=np.int64)
    Xtr = extract_columns(X, train_idx, col_arr)
    Xte = extract_columns(X, test_idx, col_arr)
    meta: dict[str, Any] = {
        "scale_variant": scale_variant,
        "n_features_used": int(col_arr.size),
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "scaler_fitted_on": "selected_training_rows_only" if scale_variant == "standardized" else None,
    }
    if scale_variant == "raw":
        meta["scaler_mean_sha256"] = None
        meta["scaler_scale_sha256"] = None
    elif scale_variant == "standardized":
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)          # TRAIN ROWS ONLY
        Xte = scaler.transform(Xte)              # test uses the train-fitted scaler
        from .provenance import sha256_json
        meta["scaler_mean_sha256"] = sha256_json([round(float(v), 12) for v in scaler.mean_])
        meta["scaler_scale_sha256"] = sha256_json([round(float(v), 12) for v in scaler.scale_])
    else:
        raise ValueError(f"unknown scale_variant {scale_variant!r}")
    meta["n_nonfinite_train"] = int((~np.isfinite(Xtr)).sum())
    meta["n_nonfinite_test"] = int((~np.isfinite(Xte)).sum())
    return {"X_train": Xtr, "X_test": Xte, "meta": meta}


def run_one_fit(fit_id: FitID, X: Any, y: np.ndarray, train_idx: np.ndarray,
                test_idx: np.ndarray, cols: list[int],
                subset_record: dict[str, Any]) -> dict[str, Any]:
    """Fit one (subset, scaling, cap) cell and classify its terminal status."""
    from .provenance import subset_fingerprint

    record: dict[str, Any] = {
        "schema": "ossfs-dd-reproduction/stage3b/svm_fit/1.0",
        "fit_id": fit_id.fit_id,
        "dataset": fit_id.subset.dataset,
        "split_seed": fit_id.subset.seed,
        "fold": fit_id.subset.fold,
        "scale_variant": fit_id.scale_variant,
        "max_iter": fit_id.max_iter,
        "n_selected": len(cols),
        "subset": subset_fingerprint(fit_id.subset.dataset, cols),
        "stage3a_run_id": fit_id.subset.stage3a_run_id,
        "n_train_expected": int(subset_record["n_train"]),
        "n_test_expected": int(subset_record["n_test"]),
    }

    try:
        prep = prepare_inputs(X, train_idx, test_idx, cols, fit_id.scale_variant)
    except Exception as exc:  # noqa: BLE001 - recorded, never hidden
        record.update(status=STATUS_FIT_ERROR, accuracy=None, error=f"prepare: {exc!r}")
        return record

    Xtr, Xte = prep["X_train"], prep["X_test"]
    ytr, yte = y[train_idx], y[test_idx]
    record["input"] = prep["meta"]
    record["n_train"] = int(Xtr.shape[0])
    record["n_test"] = int(Xte.shape[0])

    est = SVC(kernel="linear", C=1.0, max_iter=int(fit_id.max_iter))
    record["estimator"] = {
        "class_module": type(est).__module__,
        "class_name": type(est).__name__,
        "params": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                   for k, v in sorted(est.get_params(deep=False).items())},
    }

    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            est.fit(Xtr, ytr)
            early = [str(w.message) for w in caught
                     if "Convergence" in str(w.message) or "terminated early" in str(w.message)
                     or "max_iter" in str(w.message)]
        predict_start = time.perf_counter()
        yhat = est.predict(Xte)
        predict_seconds = time.perf_counter() - predict_start
    except Exception as exc:  # noqa: BLE001
        record.update(status=STATUS_FIT_ERROR, accuracy=None,
                      seconds=round(time.perf_counter() - t0, 4),
                      error=f"{type(exc).__name__}: {exc}")
        return record
    seconds = time.perf_counter() - t0

    n_iter = getattr(est, "n_iter_", None)
    n_iter_vec = None
    if n_iter is not None:
        n_iter_vec = [int(v) for v in np.ravel(n_iter)]
    fit_status = getattr(est, "fit_status_", None)
    if hasattr(fit_status, "item"):
        fit_status = fit_status.item()

    hit_cap = bool(n_iter_vec) and max(n_iter_vec) >= int(fit_id.max_iter)
    converged = (not hit_cap) and not early
    n_test = int(yte.shape[0])
    n_correct = int(np.sum(yhat == yte))

    record.update(
        status=STATUS_CONVERGED if converged else STATUS_NONCONVERGED,
        converged=converged,
        hit_iteration_cap=hit_cap,
        early_termination_warnings=early,
        n_iter_vector=n_iter_vec,
        n_iter_max=(max(n_iter_vec) if n_iter_vec else None),
        fit_status=fit_status,
        n_test=n_test,
        n_correct=n_correct,
        # Recorded even for NONCONVERGED, but tagged so it can never be pooled
        # with converged accuracy by accident.
        accuracy=(n_correct / n_test) if n_test else None,
        accuracy_is_incomplete_fit_diagnostic=not converged,
        seconds=round(seconds, 4),
        predict_seconds=round(predict_seconds, 4),
        peak_rss_gib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2), 4),
        error=None,
    )
    return record
