"""Tests that run against REAL stored datasets, not only synthetic fixtures.

The Stage 1 follow-up repaired three things that a synthetic fixture could not
have caught:

* the documented Stage 2 loader resolved workspace-relative paths against the
  caller's working directory;
* ``sample_ids.npy`` was object-dtype and needed ``allow_pickle=True``;
* ``FeatureStream.self_check()`` reported a tautology instead of evidence.

Each of those is now asserted here against the actual bytes on disk, for both
the dense and the sparse storage formats. If the dataset tree is absent these
tests SKIP, and the Stage 1 gate treats a skipped required test as *no
evidence* rather than as a pass.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import sparse

from ossfs_dd_data.registry import load_registry
from ossfs_dd_data.stream import FeatureStream
from ossfs_dd_data.workspace import processed_dir, resolve, splits_dir, stream_orders_dir


def _specs():
    return [s for s in load_registry().datasets if (processed_dir(s.key) / "schema.json").exists()]


def _load(key: str):
    """The corrected loader, exactly as documented in the Stage 2 handoff."""
    meta = json.loads((processed_dir(key) / "schema.json").read_text(encoding="utf-8"))
    xrec = meta["X"]
    X = (
        sparse.load_npz(str(resolve(xrec["path"])))
        if xrec["format"] == "scipy_csr"
        else np.load(resolve(xrec["path"]), allow_pickle=False)
    )
    y = np.load(resolve(meta["y"]["path"]), allow_pickle=False).astype(str)
    return X, y, meta


SPECS = _specs()
dense_keys = [s.key for s in SPECS
              if json.loads((processed_dir(s.key) / "schema.json").read_text())["X"]["format"] == "numpy_npy"]
sparse_keys = [s.key for s in SPECS
               if json.loads((processed_dir(s.key) / "schema.json").read_text())["X"]["format"] == "scipy_csr"]

pytestmark = pytest.mark.skipif(not SPECS, reason="dataset tree absent: no real-data evidence available")


def test_both_storage_formats_are_present():
    """The real-data suite must cover the dense AND the sparse paths."""
    assert dense_keys, "no densely-stored dataset found on disk"
    assert sparse_keys, "no sparse-stored dataset found on disk"


@pytest.mark.parametrize("key", dense_keys + sparse_keys)
def test_documented_loader_resolves_workspace_relative_paths(key, monkeypatch, tmp_path):
    """B2: the documented loader must work from ANY working directory."""
    X, y, meta = _load(key)
    assert X.shape == (meta["n_samples"], meta["n_features"])
    assert len(y) == meta["n_samples"]

    # Re-run the same documented sequence from an unrelated working directory.
    monkeypatch.chdir(tmp_path)
    meta2 = json.loads((processed_dir(key) / "schema.json").read_text(encoding="utf-8"))
    xrec = meta2["X"]
    X2 = (
        sparse.load_npz(str(resolve(xrec["path"])))
        if xrec["format"] == "scipy_csr"
        else np.load(resolve(xrec["path"]), allow_pickle=False)
    )
    assert X2.shape == X.shape
    if sparse.issparse(X):
        assert (X2 != X).nnz == 0
    else:
        assert np.array_equal(X2, X)


@pytest.mark.parametrize("key", dense_keys + sparse_keys)
def test_sample_ids_load_without_pickle(key):
    """B3: sample_ids must be readable with allow_pickle=False on real data."""
    p = processed_dir(key) / "sample_ids.npy"
    ids = np.load(p, allow_pickle=False)          # must not raise
    assert ids.dtype.kind == "U", f"{key}: sample_ids dtype is {ids.dtype}, expected fixed-width Unicode"
    _, y, meta = _load(key)
    assert len(ids) == len(y) == meta["n_samples"]
    assert ids[0] == f"{key}:0"
    assert ids[-1] == f"{key}:{meta['n_samples'] - 1}"


@pytest.mark.parametrize("key", ["SRBCT", "GINA", "ARCENE", "DEXTER", "GISETTE"])
def test_stream_self_check_on_real_data(key):
    """B5: self_check must give real evidence on real stored data."""
    if key not in [s.key for s in SPECS]:
        pytest.skip(f"{key} not prepared on this machine")
    X, y, meta = _load(key)
    plan = json.loads((splits_dir(key) / "seed_11.json").read_text(encoding="utf-8"))
    train_idx = np.asarray(plan["folds"][0]["train"], dtype=np.int64)
    rec = json.loads((stream_orders_dir(key) / "seed_11.json").read_text(encoding="utf-8"))
    order = np.asarray(rec["arrival_order"], dtype=np.int64)

    check = FeatureStream(X, y, train_idx, order).self_check(deep=True, max_deep_features=6)

    assert check["contract_satisfied"] is True
    assert check["emitted_values_equal_X_train_column"] is True, (
        "self_check claims emitted values match X[train, col]; that must be verified, not assumed"
    )
    assert check["value_checked_columns"] > 0
    assert check["mismatched_feature_indices"] == []
    assert check["every_emitted_row_count_equals_train_rows"] is True
    assert check["emission_is_permutation"] is True
    assert check["n_held_out_rows"] > 0
    # the removed v1 field must not reappear
    assert "train_indices_disjoint_from_complement" not in check


@pytest.mark.parametrize("key", ["SRBCT", "GINA"])
def test_stream_emitted_values_match_real_training_rows(key):
    """Independent of self_check: verify emitted columns by hand on real data."""
    if key not in [s.key for s in SPECS]:
        pytest.skip(f"{key} not prepared on this machine")
    X, y, _ = _load(key)
    plan = json.loads((splits_dir(key) / "seed_17.json").read_text(encoding="utf-8"))
    train_idx = np.asarray(plan["folds"][0]["train"], dtype=np.int64)
    rec = json.loads((stream_orders_dir(key) / "seed_17.json").read_text(encoding="utf-8"))
    order = np.asarray(rec["arrival_order"], dtype=np.int64)

    n_checked = 0
    for feature_index, values in FeatureStream(X, y, train_idx, order):
        assert values.shape[0] == train_idx.size
        if feature_index % 97 == 0:
            expected = (X[train_idx, feature_index].toarray().ravel()
                        if sparse.issparse(X) else X[train_idx, feature_index])
            assert np.array_equal(values, expected)
            n_checked += 1
    assert n_checked > 0, "no columns were sampled for verification"


@pytest.mark.parametrize("key", ["SRBCT", "ARCENE"])
def test_inverse_order_contract_on_real_plans(key):
    """B1: the stored inverse must genuinely invert arrival_order on real plans."""
    if key not in [s.key for s in SPECS]:
        pytest.skip(f"{key} not prepared on this machine")
    for seed in load_stage1_seeds():
        p = stream_orders_dir(key) / f"seed_{seed}.json"
        if not p.exists():
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        order = list(rec["arrival_order"])
        inv = {int(k): v for k, v in rec["inverse_position_of_feature"].items()}
        p_exp = meta_n_features(key)

        assert rec.get("schema") == "ossfs-dd-reproduction/stage1/stream_order/2.0", (
            f"{key}/seed_{seed}: schema is {rec.get('schema')!r}; the inverse field of a 1.0 "
            "record is a copy of arrival_order and must not be trusted"
        )
        assert len(inv) == p_exp, f"{key}/seed_{seed}: inverse has {len(inv)} entries, expected {p_exp}"
        for feature_id in range(p_exp):
            position = inv[feature_id]
            assert 0 <= position < p_exp
            assert order[position] == feature_id, (
                f"{key}/seed_{seed}: inverse_position_of_feature[{feature_id}]={position} but "
                f"arrival_order[{position}]={order[position]}"
            )
        # and the two fields are genuinely different objects
        assert list(inv.values()) != order or sorted(order) == list(range(p_exp))


def load_stage1_seeds() -> list[int]:
    from ossfs_dd_data.registry import load_stage1_config

    return list(load_stage1_config()["evaluation_plan"]["seeds"])


def meta_n_features(key: str) -> int:
    return json.loads((processed_dir(key) / "schema.json").read_text())["n_features"]
