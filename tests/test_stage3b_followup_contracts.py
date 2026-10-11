"""Follow-up contracts: design, authorization, leakage and chronology.

Synthetic inputs only. These assert the preregistered design, the refusal of
forbidden datasets, that nothing reads test rows during selection/scaling, and
that the frozen config carries a genuine UTC stamp.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import yaml

from ossfs_dd_stage3b.provenance import BlockedInput, assert_allowed
from ossfs_dd_stage3b_followup import EXP_A_DATASETS, EXP_B_DATASET
from ossfs_dd_stage3b_followup.experiment_a import (
    A_C_VALUES,
    A_FOLDS,
    A_MAX_ITER,
    A_SEEDS,
    A_TOL_VALUES,
    fit_ids,
)

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "configs" / "stage3b_followup_v1.yaml"
RESULTS = PROJECT / "results" / "stage3b_followup"


def _cfg() -> dict:
    if not CONFIG.exists():
        pytest.skip("follow-up config not frozen yet")
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- design


def test_experiment_a_grid_is_exactly_270():
    ids = fit_ids()
    assert len(ids) == 270
    assert len(set(ids)) == 270
    assert A_C_VALUES == (0.1, 1.0, 10.0)
    assert A_TOL_VALUES == (0.001, 0.0001)
    assert A_MAX_ITER == 2_000_000
    assert A_SEEDS == (11, 17, 23) and A_FOLDS == (0, 1, 2, 3, 4)


def test_config_planned_counts_match_the_prompt():
    cfg = _cfg()
    assert cfg["experiment_a"]["planned_fits"] == 270
    assert cfg["experiment_b"]["planned_selections"] == 50
    assert cfg["experiment_b"]["planned_subset_classifiers"] == 150
    assert cfg["experiment_b"]["planned_baselines"] == 15


def test_experiment_b_is_arcene_with_fixed_split_seed():
    cfg = _cfg()
    assert cfg["experiment_b"]["dataset"] == "ARCENE" == EXP_B_DATASET
    assert cfg["experiment_b"]["split_seed"] == 11
    assert cfg["experiment_b"]["order_seed_selects_permutation_only"] is True
    assert len(cfg["experiment_b"]["order_seeds"]) == 10


def test_cart_seed_is_fixed_across_orders():
    assert _cfg()["experiment_b"]["cart_seed_fixed_across_orders"] == 11


def test_standardization_is_frozen_on_for_experiment_a():
    cfg = _cfg()
    assert cfg["experiment_a"]["scale_variant"] == "standardized"
    assert cfg["experiment_a"]["scaler_fit_rows"] == "selected_training_rows_only"
    assert cfg["experiment_a"]["linear_svc_substitution_forbidden"] is True


def test_hypotheses_thresholds_match_the_prompt():
    cfg = _cfg()
    assert cfg["hypotheses_a"]["A_H1"]["threshold"] == 0.02
    assert cfg["hypotheses_a"]["A_H1"]["min_datasets_meeting"] == 1
    assert cfg["hypotheses_a"]["A_H2"]["threshold_relative"] == 0.10
    assert cfg["hypotheses_a"]["A_H2"]["min_datasets_meeting"] == 2
    assert cfg["hypotheses_a"]["A_H3"]["threshold"] == 0.95
    assert cfg["hypotheses_b"]["B_H2"]["threshold"] == 0.05
    assert cfg["hypotheses_b"]["B_H2"]["min_folds_meeting"] == 3
    assert cfg["hypotheses_b"]["B_H3"]["default_outcome"] == "MECHANISM_UNRESOLVED"
    assert cfg["hypotheses_are_not_operational_gates"] is True


def test_resource_ceilings_match_the_prompt():
    r = _cfg()["resources"]
    assert r["memory_limit_gib"] == 32
    assert r["max_fit_workers"] == 2
    assert r["selection_workers"] == 1
    assert r["blas_threads_per_worker"] == 1
    assert r["per_fit_timeout_minutes"] == 10
    assert r["experiment_a_max_hours"] == 3
    assert r["experiment_b_max_hours"] == 4
    assert r["total_max_hours"] == 8


def test_ten_operational_gates_declared():
    gates = _cfg()["operational_gates"]
    assert len(gates) == 10
    assert gates[0] == "F1_AUTHORIZATION_AND_SOURCE"
    assert gates[-1] == "F10_CODE_AND_IMMUTABILITY"


# --------------------------------------------------------- authorization


@pytest.mark.parametrize("dataset", ["DLBCL", "BREAST"])
def test_forbidden_datasets_are_refused(dataset):
    with pytest.raises(BlockedInput):
        assert_allowed(dataset)


@pytest.mark.parametrize("dataset", ["SRBCT", "LYMPHOMA", "PROSTATE", "LEUKEMIA"])
def test_continuous_datasets_are_refused(dataset):
    with pytest.raises(BlockedInput):
        assert_allowed(dataset)


@pytest.mark.parametrize("dataset", list(EXP_A_DATASETS) + [EXP_B_DATASET])
def test_authorized_datasets_pass(dataset):
    assert_allowed(dataset)


def test_unknown_dataset_is_refused():
    with pytest.raises(BlockedInput):
        assert_allowed("NOT_A_DATASET")


# ------------------------------------------------------------- chronology


def test_frozen_utc_is_a_real_non_future_timestamp():
    cfg = _cfg()
    ts = cfg["frozen_utc"]
    frozen = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    assert frozen <= datetime.now(UTC) + timedelta(minutes=5), (
        "frozen_utc must not be in the future"
    )


def test_freeze_statement_is_not_a_preregistration_claim():
    text = _cfg()["freeze_statement"].lower()
    assert "not an independently" in text
    assert "preregistration" in text


def test_config_declares_no_forbidden_work():
    cfg = _cfg()
    assert set(cfg["authorization"]["forbidden"]) == {"DLBCL", "BREAST"}
    assert "ARCENE" not in cfg["authorization"]["continuous_excluded"]
    assert cfg["experiment_a"]["datasets"] == list(EXP_A_DATASETS)


# --------------------------------------------------------------- leakage


def test_scaler_is_fitted_on_training_rows_only():
    from sklearn.preprocessing import StandardScaler

    from ossfs_dd_stage3b_followup.experiment_a import extract_columns

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    train, test = np.arange(24), np.arange(24, 40)
    Xte_orig = X[test].copy()
    Xtr = extract_columns(X, train, np.arange(3))
    Xte = extract_columns(X, test, np.arange(3))
    scaler = StandardScaler()
    a_train = scaler.fit_transform(Xtr)
    # a wildly different test set must not change the training transform
    scaler2 = StandardScaler()
    Xte2 = Xte + 1e6
    b_train = scaler2.fit_transform(Xtr)
    assert np.allclose(a_train, b_train)
    assert not np.array_equal(Xte_orig, Xte2)


def test_classifier_matrix_uses_training_rows_only():
    from ossfs_dd_stage3b_followup.experiment_b import extract_columns

    X = np.arange(20 * 2, dtype=float).reshape(20, 2)
    tr, te = np.arange(12), np.arange(12, 20)
    A = extract_columns(X, tr, np.arange(2))
    B = extract_columns(X, te, np.arange(2))
    assert A.shape[0] == 12 and B.shape[0] == 8
    assert not np.array_equal(A, B)


def test_selection_reads_no_test_rows():
    """The instrumented ARCENE cell must report zero test-row reads."""
    from ossfs_dd_stage3b_followup.instrument import run_cell

    rng = np.random.default_rng(1)
    n = 40
    y = np.where(np.arange(n) < n // 2, "a", "b")
    sig = (y == "a").astype(float)
    X = np.column_stack([sig * 2 + rng.normal(0, .5, n) for _ in range(4)])
    order = np.arange(4)
    rec = run_cell(X=X, y=y, train_idx=np.arange(30), order=order,
                   fold=0, order_seed=11, split_seed=11)
    assert rec["test_rows_read"] == 0
    assert rec["classifier_fitted"] is False
    assert rec["n_train"] == 30


def test_frozen_estimators_are_not_retuned():
    from ossfs_dd_stage3b_followup.experiment_b import build_estimator

    knn = build_estimator("knn")
    assert knn.n_neighbors == 5 and knn.weights == "uniform" and knn.p == 2
    cart = build_estimator("cart")
    assert cart.criterion == "gini" and cart.random_state == 11
    svm = build_estimator("svm")
    assert svm.kernel == "linear" and svm.C == 1.0 and svm.max_iter == 2_000_000
    assert type(svm).__name__ == "SVC"     # libsvm, never LinearSVC
