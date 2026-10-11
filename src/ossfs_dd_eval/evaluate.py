"""Run one Stage-3A selection and its three held-out classifier outcomes.

The ordering below is the whole leakage contract:

1. authorize;
2. load the immutable Stage-1 matrix, split and the seed's ``arrival_order``;
3. build a Stage-1 ``FeatureStream`` over TRAINING ROWS ONLY and select;
4. only then read the test rows, and only for the selected columns.

The selector never receives a held-out row. A test row is first touched in
step 4, after the subset is final.

Each selection is evaluated under two preprocessing variants from the *same*
selected subset: the raw primary path and a clearly labelled standardized
sensitivity analysis. Selection is the expensive step and is not repeated.
"""

from __future__ import annotations

import json
import resource
import time
from typing import Any

import numpy as np
from scipy import sparse

from ossfs_dd.config import SelectionConfig
from ossfs_dd.selector import Selector
from ossfs_dd_data.stream import FeatureStream
from ossfs_dd_data.workspace import splits_dir, stream_orders_dir

from .auth import authorize_run
from .classifiers import CLASSIFIERS, fit_predict
from .metrics import empty_selection_outcome, score_outcome
from .protocol import RunSpec


def _select(spec: RunSpec, config: dict[str, Any]) -> dict[str, Any]:
    """Selection only. Touches training rows; never a test row."""
    auth, plan = authorize_run(spec.dataset, config, seed=spec.seed, fold=spec.fold)

    from ossfs_dd.runner import load_dataset

    X, y, _schema = load_dataset(spec.dataset)
    split = json.loads(
        (splits_dir(spec.dataset) / f"seed_{spec.seed}.json").read_text(encoding="utf-8")
    )
    train_idx = np.asarray(split["folds"][spec.fold]["train"], dtype=np.int64)
    rec = json.loads(
        (stream_orders_dir(spec.dataset) / f"seed_{spec.seed}.json").read_text(encoding="utf-8")
    )
    if rec.get("schema") != "ossfs-dd-reproduction/stage1/stream_order/2.0":
        raise ValueError(f"{spec.dataset}: unexpected stream-order schema {rec.get('schema')!r}")
    order = np.asarray(rec["arrival_order"], dtype=np.int64)

    stream = FeatureStream(X, y, train_idx, order)
    sel_cfg = SelectionConfig(mode=plan.primary_mode)

    t0 = time.perf_counter()
    selector = Selector(
        sel_cfg,
        metric_family=auth.metric_family,
        labels=stream.train_labels,
        entry_file="mi2",
    )
    selection = selector.fit_stream(iter(stream))
    seconds = time.perf_counter() - t0
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)

    return {
        "X": X,
        "y": y,
        "train_idx": train_idx,
        "feature_indices": list(selection.feature_indices),
        "seconds": seconds,
        "peak_rss_gib": peak_rss,
        "selection_summary": selection.summary(),
        "n_features": auth.n_features,
        "n_train": int(train_idx.size),
        "n_test": int(X.shape[0] - train_idx.size),
    }


def _column(matrix: Any, idx: np.ndarray, column: int) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix[idx, column].todense()).ravel()
    return np.asarray(matrix[idx, column]).ravel()


def _evaluate_variant(
    spec: RunSpec,
    config: dict[str, Any],
    X: Any,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    cols: list[int],
    *,
    standardize: bool,
) -> list[dict[str, Any]]:
    zero_division = int(config["metrics"]["zero_division"])
    outcomes: list[dict[str, Any]] = []
    if not cols:
        for name in CLASSIFIERS:
            outcomes.append(empty_selection_outcome(name, 0))
        return outcomes
    sub = np.asarray(cols, dtype=np.int64)
    Xtr = np.column_stack([_column(X, train_idx, c) for c in sub])
    Xte = np.column_stack([_column(X, test_idx, c) for c in sub])
    ytr, yte = y[train_idx], y[test_idx]
    for name in CLASSIFIERS:
        Xtr_v, Xte_v = _prepare(Xtr, Xte, standardize)
        yhat, meta = fit_predict(name, spec.seed, Xtr_v, ytr, Xte_v)
        outcomes.append(
            score_outcome(
                name, yte, yhat, zero_division=zero_division, fit_meta=meta, standardized=standardize
            )
        )
    return outcomes


def _prepare(Xtr: np.ndarray, Xte: np.ndarray, standardize: bool) -> tuple[np.ndarray, np.ndarray]:
    from .classifiers import prepare

    return prepare(Xtr, Xte, standardize=standardize)


def evaluate_run(spec: RunSpec, config: dict[str, Any]) -> dict[str, Any]:
    """One selection plus three classifiers, under raw and standardized inputs."""
    payload = _select(spec, config)
    X, y, train_idx = payload["X"], payload["y"], payload["train_idx"]
    cols = payload["feature_indices"]
    test_idx = np.setdiff1d(np.arange(X.shape[0]), train_idx, assume_unique=False)

    raw = _evaluate_variant(
        spec, config, X, y, train_idx, test_idx, cols, standardize=False
    )
    std = _evaluate_variant(
        spec, config, X, y, train_idx, test_idx, cols, standardize=True
    )

    return {
        "run_id": spec.run_id,
        **spec.as_dict(),
        "mode": "matlab_compatible_v1",
        "status": "COMPLETED",
        "n_features_total": payload["n_features"],
        "n_train": payload["n_train"],
        "n_test": payload["n_test"],
        "n_selected": len(cols),
        "selected_fraction": (len(cols) / payload["n_features"]) if payload["n_features"] else None,
        "selected_feature_ids": cols,
        "selection_seconds": round(payload["seconds"], 3),
        "peak_rss_gib": round(payload["peak_rss_gib"], 3),
        "selection_summary": payload["selection_summary"],
        "outcomes_raw_primary": raw,
        "outcomes_standardized_sensitivity": std,
    }
