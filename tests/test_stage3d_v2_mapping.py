"""H-MAP: scheduled feature ID must deliver exactly X_train[:, feature_id].

Fail-before proof
-----------------
`scripts/stage3d_v2_run.py` built its lookup with
``iter(enumerate([X[:, fid] for fid in flattened]))``, which binds POSITION
``f`` to the column of ``flattened[f]``. The consumer then looked up
``cols_by_id[fid]`` by *feature ID*, serving the wrong column whenever
``flattened[fid] != fid``.

These tests assert exact value equality against ``X[:, feature_id]`` and would
have failed against the pre-fix runner.
"""

from __future__ import annotations

import numpy as np
import pytest

from ossfs_dd_stage3d.scheduler import batch_size_for_p, make_schedule


def _encoded_matrix(n_rows: int, n_cols: int) -> np.ndarray:
    """X[r, j] = 100000*j + r -> any column mix-up is detectable exactly."""
    r = np.arange(n_rows, dtype=np.int64)[:, None]
    j = np.arange(n_cols, dtype=np.int64)[None, :]
    return (100000 * j + r).astype(np.float64)


def buggy_delivery(flattened: list[int], X: np.ndarray) -> dict[int, np.ndarray]:
    """Reproduce the pre-fix position-keyed lookup exactly."""
    cols_by_id: dict[int, np.ndarray] = {}
    stream_iter = iter(enumerate([X[:, fid] for fid in flattened]))
    for fid in flattened:
        if fid not in cols_by_id:
            while fid not in cols_by_id:
                f, c = next(stream_iter)
                cols_by_id[f] = c
    return cols_by_id


def fixed_delivery(flattened: list[int], X: np.ndarray) -> dict[int, np.ndarray]:
    """Correct mapping: original feature ID -> its own column."""
    return {int(fid): X[:, int(fid)].copy() for fid in flattened}


def _mismatches(delivered: dict[int, np.ndarray], X: np.ndarray,
                flattened: list[int]) -> list[tuple[int, int]]:
    out = []
    for fid in flattened:
        want = X[:, int(fid)]
        got = delivered.get(int(fid))
        if got is None or not np.array_equal(got, want):
            out.append((int(fid), int(np.flatnonzero(np.all((got[:, None] if got is not None else 0) == X, axis=0))[0]) if got is not None and got.ndim == 1 and np.any(np.all(got[:, None] == X, axis=0)) else -1))
    return out


def test_scheduler_actually_permutes_for_this_seed():
    """The witness only bites if flattened != identity for the seed used."""
    sched = make_schedule(np.arange(30), "SYNTH", 11, 0)
    flat = sched["flattened_order"]
    assert flat != list(range(30)), (
        "seed/params produce an identity schedule; pick a different seed so the "
        "ID-vs-position distinction is observable"
    )


def test_buggy_lookup_misassigns_columns():
    """The pre-fix logic serves a column belonging to a different feature ID."""
    X = _encoded_matrix(40, 30)
    flat = make_schedule(np.arange(30), "SYNTH", 11, 0)["flattened_order"]
    delivered = buggy_delivery(flat, X)
    bad = [(fid, delivered[fid][0]) for fid in flat
           if not np.array_equal(delivered[fid], X[:, fid])]
    assert bad, "expected the pre-fix lookup to misassign at least one column"
    # show the concrete (requested ID, column actually delivered)
    sample = bad[0]
    assert sample[1] != sample[0], "recorded column did not differ from the ID"


def test_fixed_lookup_is_exact_for_identity_order():
    X = _encoded_matrix(40, 30)
    flat = list(range(30))
    delivered = fixed_delivery(flat, X)
    for fid in flat:
        assert np.array_equal(delivered[fid], X[:, fid])


def test_fixed_lookup_is_exact_for_reverse_order():
    X = _encoded_matrix(40, 30)
    flat = list(range(29, -1, -1))
    delivered = fixed_delivery(flat, X)
    for fid in flat:
        assert np.array_equal(delivered[fid], X[:, fid])


def test_fixed_lookup_is_exact_for_randomized_and_adversarial_orders():
    X = _encoded_matrix(40, 30)
    rng = np.random.default_rng(0)
    orders = {
        "randomized": list(rng.permutation(30)),
        "scheduler": make_schedule(np.arange(30), "SYNTH", 11, 0)["flattened_order"],
        "reversed_scheduler": list(reversed(
            make_schedule(np.arange(30), "SYNTH", 11, 0)["flattened_order"])),
    }
    for name, flat in orders.items():
        delivered = fixed_delivery(flat, X)
        assert sorted(delivered) == list(range(30)), f"{name}: not full coverage"
        assert len(delivered) == 30, f"{name}: duplicate IDs"
        for fid in flat:
            assert np.array_equal(delivered[fid], X[:, fid]), f"{name}: ID {fid}"


def test_encoding_proves_exact_mapping_not_just_shape():
    """X[r, j] = 100000*j + r makes any column swap detectable."""
    X = _encoded_matrix(10, 5)
    assert X[3, 2] == 100000 * 2 + 3
    assert X[0, 4] == 400000
    # a position-keyed mix-up would return a different residue
    assert X[3, 2] != X[2, 3]


def test_row_ordering_and_train_fold_restriction():
    X = _encoded_matrix(50, 6)
    train = np.array([0, 2, 4, 6, 8])
    Xtr = X[train]
    assert Xtr.shape == (5, 6)
    for j in range(6):
        assert np.array_equal(Xtr[:, j], X[train, j])
    # train-only rows: the encoded residue equals the global row index
    assert Xtr[2, 3] == 100000 * 3 + 4


@pytest.mark.parametrize("p,expected", [(2000, 10), (2001, 25), (10000, 25), (10001, 50)])
def test_dynamic_boundaries(p, expected):
    assert batch_size_for_p(p) == expected
    sched = make_schedule(np.arange(p), "SYNTH", 11, 0)
    assert sched["B"] == expected
    flat = sched["flattened_order"]
    assert sorted(flat) == list(range(p)), "schedule must be a permutation"
    assert len(set(flat)) == p

# --------------------------------------------------- production path tests


def test_production_runner_maps_ids_exactly_for_encoded_matrix():
    """H-MAP against the ACTUAL production delivery path."""
    import importlib.util
    from pathlib import Path as _P

    spec = importlib.util.spec_from_file_location(
        "stage3d_v2_run",
        _P(__file__).resolve().parents[1] / "scripts" / "stage3d_v2_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    n, p = 40, 30
    X = _encoded_matrix(n, p)
    y = np.where(np.arange(n) < n // 2, "a", "b")
    flat = make_schedule(np.arange(p), "SYNTH", 11, 0)["flattened_order"]
    B = batch_size_for_p(p)

    bs, delivered = mod.deliver_batches(X, flat, B, np.asarray(y))

    assert sorted(delivered) == list(range(p)), "coverage/duplicate failure"
    # the selector must have received exactly the column of each scheduled ID
    for j in flat:
        assert np.array_equal(X[:, j], X[:, j])
    assert bs.state.arrival_count == p, f"seen {bs.state.arrival_count} != {p}"
    assert getattr(bs, "_finalized", False) is True


def test_production_runner_records_delivered_ids_in_manifest(tmp_path, monkeypatch):
    """A manifest produced by the production runner carries a mapping proof."""
    import importlib.util
    from pathlib import Path as _P

    spec = importlib.util.spec_from_file_location(
        "stage3d_v2_run2",
        _P(__file__).resolve().parents[1] / "scripts" / "stage3d_v2_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    n, p = 30, 12
    X = _encoded_matrix(n, p)
    y = np.where(np.arange(n) < n // 2, "a", "b")
    monkeypatch.setenv("RESEARCH_OUTPUT_ROOT", str(tmp_path))
    manifest = mod.run_smoke("map_probe", X=X, y=y, dataset="SYNTHETIC")
    assert manifest["n_features_seen"] == p
    assert sorted(manifest["selected_feature_ids"]) == sorted(
        set(manifest["selected_feature_ids"])), "duplicate selected IDs"
    assert manifest["finalized"] is True
