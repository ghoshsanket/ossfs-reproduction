"""Stage-3A protocol, authorization and leakage contracts.

These are the tests that would have caught a wrong-dataset or test-row leak
before any result was recorded, so they are written to fail loudly rather than
to accommodate the implementation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from ossfs_dd_eval.auth import Stage3AError, authorize_run
from ossfs_dd_eval.classifiers import CLASSIFIERS, prepare
from ossfs_dd_eval.metrics import empty_selection_outcome, score_outcome
from ossfs_dd_eval.protocol import (
    CONTINUOUS_DATASETS,
    PLANNED_OUTCOMES,
    PLANNED_SELECTIONS,
    STAGE3A_DATASETS,
    RunSpec,
    all_runs,
    assert_protocol,
    protocol_digest,
)

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load((PROJECT / "configs" / "stage3a.yaml").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- protocol


def test_planned_sizes_are_derived_and_correct():
    assert PLANNED_SELECTIONS == 250
    assert PLANNED_OUTCOMES == 750
    assert PLANNED_SELECTIONS * len(CLASSIFIERS) == PLANNED_OUTCOMES


def test_run_ids_are_unique_and_complete():
    runs = all_runs()
    assert len(runs) == PLANNED_SELECTIONS
    assert len({r.run_id for r in runs}) == PLANNED_SELECTIONS


def test_one_order_per_dataset_seed_not_per_fold():
    """The five folds of a seed share one feature order; they are not five
    independent permutations."""
    orders = {r.order_key for r in all_runs()}
    assert len(orders) == len(STAGE3A_DATASETS) * 10
    folds_for_one_seed = [r for r in all_runs() if r.dataset == "MADELON" and r.seed == 11]
    assert len({r.order_key for r in folds_for_one_seed}) == 1
    assert len({r.fold for r in folds_for_one_seed}) == 5


def test_frozen_config_agrees_with_protocol(config):
    assert_protocol(config)  # raises on any drift


def test_protocol_digest_changes_with_selection_semantics(config):
    other = dict(config)
    other["selection"] = {**config["selection"], "pool_size": 99}
    assert protocol_digest(config) != protocol_digest(other)


# ------------------------------------------------------------ authorization


@pytest.mark.parametrize("dataset", ["DLBCL", "BREAST"])
def test_forbidden_datasets_are_refused(config, dataset):
    with pytest.raises((Stage3AError, Exception)):
        authorize_run(dataset, config, seed=11, fold=0)


@pytest.mark.parametrize("dataset", CONTINUOUS_DATASETS)
def test_continuous_datasets_are_refused_by_workstream_a(config, dataset):
    with pytest.raises(Stage3AError):
        authorize_run(dataset, config, seed=11, fold=0)


def test_unknown_dataset_is_refused(config):
    with pytest.raises(Stage3AError):
        authorize_run("NOT_A_DATASET", config, seed=11, fold=0)


# --------------------------------------------------------- empty selection


def test_empty_selection_is_not_evaluable_and_not_zero():
    for name in CLASSIFIERS:
        outcome = empty_selection_outcome(name, 0)
        assert outcome["accuracy"] is None
        assert outcome["status"] == "NOT_EVALUABLE_EMPTY_SELECTION"


def test_accuracy_is_bounded_and_counted():
    y = np.array(["a", "a", "b", "b"])
    correct = score_outcome("knn", y, y, zero_division=0)
    assert correct["accuracy"] == 1.0
    assert correct["n_correct"] == 4 and correct["n_test"] == 4
    wrong = score_outcome("knn", y, np.array(["b", "b", "a", "a"]), zero_division=0)
    assert wrong["accuracy"] == 0.0
    assert 0.0 <= wrong["accuracy"] <= 1.0


# ------------------------------------------------------------- classifiers


def test_classifier_parameters_are_frozen():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    knn = CLASSIFIERS["knn"].build(11)
    assert isinstance(knn, KNeighborsClassifier)
    assert knn.n_neighbors == 5 and knn.weights == "uniform"
    assert knn.metric == "minkowski" and knn.p == 2
    svm = CLASSIFIERS["svm"].build(11)
    assert isinstance(svm, SVC) and svm.kernel == "linear" and svm.C == 1.0
    cart = CLASSIFIERS["cart"].build(11)
    assert isinstance(cart, DecisionTreeClassifier)
    assert cart.criterion == "gini" and cart.splitter == "best"


def test_cart_is_seed_deterministic():
    a = CLASSIFIERS["cart"].build(11).fit(np.eye(6), np.array(list("aabbcc")))
    b = CLASSIFIERS["cart"].build(11).fit(np.eye(6), np.array(list("aabbcc")))
    assert a.tree_.node_count == b.tree_.node_count


def test_standardization_fits_on_training_rows_only():
    """A scaler fitted on train rows must ignore the test-row scale."""
    xtr = np.array([[0.0], [1.0], [2.0], [3.0]])
    xte = np.array([[1000.0]])
    tr, te = prepare(xtr, xte, standardize=True)
    assert tr.shape == (4, 1) and te.shape == (1, 1)
    # the enormous test value stays enormous; it was not folded into the scaler
    assert te[0, 0] > 100.0
    assert abs(float(np.mean(tr))) < 1e-12


def test_raw_preparation_is_identity():
    xtr, xte = np.array([[1.0]]), np.array([[2.0]])
    a, b = prepare(xtr, xte, standardize=False)
    assert a[0, 0] == 1.0 and b[0, 0] == 2.0


def test_svm_convergence_cap_is_declared():
    from ossfs_dd_eval.classifiers import SVM_MAX_ITER

    assert CLASSIFIERS["svm"].build(11).max_iter == SVM_MAX_ITER


# ----------------------------------------------------------------- leakage


def test_stream_yields_training_rows_only():
    """Every column the selector can see equals X[train_idx, feature]."""
    import json

    from scipy import sparse

    from ossfs_dd.runner import load_dataset
    from ossfs_dd_data.stream import FeatureStream
    from ossfs_dd_data.workspace import splits_dir, stream_orders_dir

    key, seed, fold = "MADELON", 11, 0
    X, y, _ = load_dataset(key)
    split = json.loads((splits_dir(key) / f"seed_{seed}.json").read_text(encoding="utf-8"))
    train_idx = np.asarray(split["folds"][fold]["train"], dtype=np.int64)
    order = np.asarray(
        json.loads((stream_orders_dir(key) / f"seed_{seed}.json").read_text(encoding="utf-8"))[
            "arrival_order"
        ],
        dtype=np.int64,
    )
    stream = FeatureStream(X, y, train_idx, order)
    test_idx = np.setdiff1d(np.arange(X.shape[0]), train_idx)
    dense = X.toarray() if sparse.issparse(X) else np.asarray(X)

    checked = 0
    for fid, column in stream:
        assert np.array_equal(np.asarray(column), dense[train_idx, fid]), (
            f"stream column {fid} does not match the training rows"
        )
        # and it must not be the test-row column
        assert not np.array_equal(np.asarray(column), dense[test_idx, fid])
        checked += 1
        if checked >= 40:
            break
    assert checked > 0


def test_classifier_inputs_come_only_from_selected_columns():
    """The evaluated matrices are exactly (train, selected) and (test, selected)."""
    from ossfs_dd_eval import evaluate as ev

    captured: dict[str, list] = {}
    original = ev.fit_predict

    def spy(name, seed, X_train, y_train, X_test):
        # fit_predict is called once per classifier per variant: the RAW primary
        # first, then the standardized sensitivity. Keep both, assert on the raw.
        captured.setdefault(name, []).append((X_train, X_test))
        return original(name, seed, X_train, y_train, X_test)

    ev.fit_predict = spy  # type: ignore[assignment]  # noqa: SLF001
    try:
        record = ev.evaluate_run(RunSpec("MADELON", 11, 0), _load_cfg())
    finally:
        ev.fit_predict = original  # type: ignore[assignment]  # noqa: SLF001

    cols = np.asarray(record["selected_feature_ids"], dtype=np.int64)
    import json

    from scipy import sparse

    from ossfs_dd.runner import load_dataset
    from ossfs_dd_data.workspace import splits_dir

    X, _, _ = load_dataset("MADELON")
    dense = X.toarray() if sparse.issparse(X) else np.asarray(X)
    split = json.loads((splits_dir("MADELON") / "seed_11.json").read_text(encoding="utf-8"))
    train_idx = np.asarray(split["folds"][0]["train"], dtype=np.int64)
    test_idx = np.setdiff1d(np.arange(X.shape[0]), train_idx)

    assert len(captured["knn"]) == 2  # raw then standardized
    knn_train, knn_test = captured["knn"][0]  # RAW primary
    assert knn_train.shape == (len(train_idx), len(cols))
    assert knn_test.shape == (len(test_idx), len(cols))
    # the first captured feature must be exactly the first selected column
    assert np.array_equal(knn_train[:, 0], dense[train_idx, cols[0]])
    assert np.array_equal(knn_test[:, 0], dense[test_idx, cols[0]])


def _load_cfg() -> dict:
    return yaml.safe_load((PROJECT / "configs" / "stage3a.yaml").read_text(encoding="utf-8"))
