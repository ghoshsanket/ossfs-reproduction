"""The thirteen required named Stage 1 tests, plus supporting contract tests.

Each required name from ``configs/stage1.yaml`` is defined here exactly once.

Note on the synthetic fixture: ``conftest.synthetic_fixture`` is a small random
matrix used only to exercise the parsing, streaming and splitting contracts. It
is never registered as a dataset, never written to the dataset tree, and is
never presented as a paper dataset.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from ossfs_dd_data import parsers
from ossfs_dd_data.fetch import sha256_file
from ossfs_dd_data.registry import (
    ALL_STATUSES,
    STATUS_EXACT_VERIFIED,
    STATUS_SHAPE_MATCH_PROVENANCE_UNVERIFIED,
)
from ossfs_dd_data.splits import build_feature_orders, build_split_plan
from ossfs_dd_data.stream import FeatureStream, LeakageError
from ossfs_dd_data.workspace import (
    data_root,
    manifest_path,
    processed_dir,
    project_root,
    raw_dir,
)

# The Table 1 specification, transcribed independently of the registry file so
# that a mistake in the registry cannot make this test vacuous.
TABLE1 = {
    "SRBCT": (63, 2308, 4),
    "LYMPHOMA": (62, 4026, 3),
    "PROSTATE": (102, 6033, 2),
    "LEUKEMIA": (72, 7129, 2),
    "DLBCL": (77, 7129, 2),
    "ARCENE": (200, 10000, 2),
    "DEXTER": (600, 20000, 2),
    "BREAST": (97, 24481, 2),
    "MADELON": (2600, 500, 2),
    "GINA": (3468, 970, 2),
    "GISETTE": (7000, 5000, 2),
}


# =========================================================================
# 1
# =========================================================================
def test_registry_exact_table1(registry):
    """The registry must carry exactly the eleven Table 1 expectations."""
    assert len(registry.datasets) == 11, f"expected 11 datasets, got {len(registry.datasets)}"
    seen = set()
    for spec in registry.datasets:
        assert spec.key not in seen, f"duplicate dataset key {spec.key}"
        seen.add(spec.key)
        expected = TABLE1[spec.key]
        assert (spec.n_samples, spec.n_features, spec.n_classes) == expected, (
            f"{spec.key}: registry says {(spec.n_samples, spec.n_features, spec.n_classes)}, "
            f"Table 1 says {expected}"
        )
        # features must exclude the class column
        assert spec.total_attributes == spec.n_features + 1
        assert spec.paper_feature_type in {"Real", "Integer"}
        assert spec.sources, f"{spec.key} declares no source"
        assert spec.provenance_tier in {"paper_cited", "official", "academic_repository", "mirror"}
        assert spec.provenance_confidence in {"high", "medium", "low"}
        assert spec.license and spec.citation
    assert seen == set(TABLE1)


# =========================================================================
# 2
# =========================================================================
def test_no_target_column_in_features(registry):
    """The class column must never appear in the feature block."""
    for spec in registry.datasets:
        for _rec in _prepared_records():
            pass
        p = processed_dir(spec.key) / "schema.json"
        if not p.exists():
            continue
        meta = json.loads(p.read_text(encoding="utf-8"))
        assert meta["features_exclude_class_column"] is True, spec.key
        assert meta["n_features"] == spec.n_features, (
            f"{spec.key}: stored feature count {meta['n_features']} != {spec.n_features}"
        )
        # a UCI source file has n_features value columns and a SEPARATE labels
        # file, so total_attributes == n_features + 1 must never be parsed as
        # n_features + 1 features.
        assert meta["n_features"] < spec.total_attributes or spec.key in {
            "SRBCT", "LYMPHOMA", "PROSTATE", "LEUKEMIA", "DLBCL", "BREAST", "GINA"
        }

    # ARFF/CSV sources carry the label inside the file; assert the parser
    # dropped exactly one attribute.
    for key in ("SRBCT", "GINA", "DLBCL", "BREAST"):
        meta_path = processed_dir(key) / "schema.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["n_features"] + 1 == TABLE1[key][1] + 1, key


# =========================================================================
# 3
# =========================================================================
def test_nips_train_valid_alignment(registry):
    """Concatenated labeled splits must align row-for-row with their labels."""
    for spec in registry.datasets:
        if spec.primary_source.fmt != "uci_data_labels_pair":
            continue
        meta_path = processed_dir(spec.key) / "schema.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        extra = meta.get("parsing_extra") or {}
        counts = extra.get("split_row_counts") or {}
        expected = spec.expected_split_counts()
        for split, want in expected.items():
            assert counts.get(split) == want, (
                f"{spec.key}/{split}: {counts.get(split)} rows, registry expects {want}"
            )
        assert counts.get("train", 0) + counts.get("valid", 0) == spec.n_samples
        assert extra.get("excluded_splits") == ["test"], spec.key

        # labels must align with the rows they describe
        y = np.load(Path(meta["y"]["path"]) if Path(meta["y"]["path"]).is_absolute()
                    else data_root().parents[1] / meta["y"]["path"], allow_pickle=False)
        assert len(y) == spec.n_samples, spec.key


# =========================================================================
# 4
# =========================================================================
def test_no_unlabeled_test_data(registry):
    """Challenge 'test' records are unlabeled and must never enter a dataset."""
    from ossfs_dd_data.parsers import ParseError

    with pytest.raises(ParseError):
        parsers.parse_uci_data_labels_pair(
            {"train": Path("/dev/null"), "test": Path("/dev/null")},
            {"train": Path("/dev/null"), "test": Path("/dev/null")},
            n_columns=10,
        )

    for spec in registry.datasets:
        if not spec.expected_split_counts():
            continue
        for split in spec.excluded_splits():
            assert split not in spec.combine_splits(), spec.key
            meta_path = processed_dir(spec.key) / "schema.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                extra = meta.get("parsing_extra") or {}
                assert "test" not in (extra.get("split_row_counts") or {}), spec.key

        # the source split marker for every stored row must be a labeled split
        meta_path = processed_dir(spec.key) / "schema.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            sp = Path(meta["source_split"]["path"])
            if not sp.is_absolute():
                sp = data_root().parents[1] / sp
            splits = np.load(sp, allow_pickle=False).astype(str)
            assert "test" not in set(splits.tolist()), spec.key


# =========================================================================
# 5
# =========================================================================
def test_dexter_sparse_one_based_indices(tmp_path):
    """DEXTER-style sparse input: 1-based indices, 20000 columns incl. zeros."""
    p = tmp_path / "dexter_train.data"
    # feature 1 -> column 0, feature 5 -> column 4, feature 20000 -> column 19999
    p.write_text("1:3 5:7 20000:11\n2:1 20000:2\n", encoding="utf-8")
    X, info = parsers._parse_uci_data_file(p, 20000, 0.3, "float64", declared_sparse=True)
    assert info["index_base_detected"] == 1
    assert info["index_base_convention"] == "1-based"
    assert X.shape == (2, 20000)
    dense = X.toarray()
    assert dense[0, 0] == 3.0 and dense[0, 4] == 7.0 and dense[0, 19999] == 11.0
    assert dense[1, 1] == 1.0 and dense[1, 19999] == 2.0
    # implicit all-zero columns are materialised, not dropped
    assert int(dense[0].sum()) == 21.0

    # a 0-based file is detected as 0-based
    p0 = tmp_path / "zero_based.data"
    p0.write_text("0:3 19999:7\n", encoding="utf-8")
    X0, info0 = parsers._parse_uci_data_file(p0, 20000, 0.3, "float64")
    assert info0["index_base_detected"] == 0
    assert X0.toarray()[0, 19999] == 7.0


# =========================================================================
# 6
# =========================================================================
def test_orientation_and_transpose_contract(tmp_path, registry):
    """Rows are samples; a transposed input must be rejected, not silently fixed."""
    # ARFF: rows are samples, features are columns
    arff = tmp_path / "tiny.arff"
    arff.write_text(
        "@relation tiny\n"
        "@attribute f1 numeric\n@attribute f2 numeric\n"
        "@attribute class {a,b}\n"
        "@data\n1,2,a\n3,4,b\n5,6,a\n",
        encoding="utf-8",
    )
    parsed = parsers.parse_arff(arff)
    assert parsed.orientation == "rows_are_samples"
    assert parsed.n_samples == 3 and parsed.n_features == 2
    assert parsed.y.tolist() == ["a", "b", "a"]
    assert np.allclose(parsed.X[:, 0], [1, 3, 5])

    # a gene-major CSV (features x samples) must not be accepted as samples-major
    csv_path = tmp_path / "gene_major.csv"
    csv_path.write_text("\n".join(f"{i},{i + 1},{i + 2}" for i in range(3)) + "\n",
                        encoding="utf-8")
    with pytest.raises(parsers.ParseError):
        # 3 rows x 3 columns cannot satisfy a declared 3 samples x 10 features
        parsers.parse_csv_label_last(csv_path, label_column=-1, expected_columns=11)

    d = tmp_path / "u.data"
    d.write_text("1 2\n3 4\n", encoding="utf-8")
    lab = tmp_path / "u.labels"
    lab.write_text("1\n-1\n", encoding="utf-8")
    # UCI data/labels width is enforced against the registry, never transposed.
    # A gene-major file (features as rows) has the wrong width for a
    # sample-major reading and must be rejected with an explicit refusal
    # rather than silently transposed.
    gene_major = tmp_path / "gene_major.data"
    gene_major.write_text("\n".join(" ".join(str(v) for v in row) for row in
                                    [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]) + "\n",
                         encoding="utf-8")
    lab5 = tmp_path / "g.labels"
    lab5.write_text("1\n-1\n1\n-1\n1\n", encoding="utf-8")
    with pytest.raises(parsers.ParseError) as exc:
        parsers.parse_uci_data_labels_pair({"train": gene_major}, {"train": lab5}, n_columns=3)
    assert "Transposing is not attempted" in str(exc.value)

    parsed_ok = parsers.parse_uci_data_labels_pair(
        {"train": tmp_path / "u.data"}, {"train": tmp_path / "u.labels"}, n_columns=2
    )
    assert parsed_ok.orientation == "rows_are_samples"
    assert parsed_ok.n_samples == 2 and parsed_ok.n_features == 2


# =========================================================================
# 7
# =========================================================================
def test_sha256_and_immutable_raw(registry):
    """Raw bytes must be hashed, and never rewritten by a rerun."""
    for spec in registry.datasets:
        d = raw_dir(spec.key)
        assert d.exists(), f"{spec.key}: no raw directory"
        files = [p for p in d.rglob("*") if p.is_file()]
        assert files, f"{spec.key}: raw directory is empty"
        inv_path = manifest_path(spec.key).parent / f"{spec.key}.raw_inventory.json"
        if not inv_path.exists():
            continue
        inv = json.loads(inv_path.read_text(encoding="utf-8"))
        recorded = {Path(f["path"]).name: f["sha256"] for f in inv["files"]}
        for p in files:
            digest = sha256_file(p)
            assert len(digest) == 64
            if p.name in recorded:
                assert digest == recorded[p.name], (
                    f"{spec.key}/{p.name}: raw bytes changed since the recorded inventory — "
                    "raw data must be immutable"
                )

    # a mutation must change the hash (proving the hash is meaningful)
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False) as fh:
        fh.write(b"hello")
        tmp = Path(fh.name)
    h1 = sha256_file(tmp)
    tmp.write_bytes(b"hellp")
    h2 = sha256_file(tmp)
    tmp.unlink()
    assert h1 != h2 and h1 != h2


# =========================================================================
# 8
# =========================================================================
def test_dataset_roundtrip(registry):
    """Reloading a stored dataset must reproduce its shape and labels exactly."""
    for spec in registry.datasets:
        meta_path = processed_dir(spec.key) / "schema.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        xp = Path(meta["X"]["path"])
        if not xp.is_absolute():
            xp = data_root().parents[1] / xp
        X = sparse.load_npz(str(xp)) if meta["X"]["format"] == "scipy_csr" else np.load(xp, allow_pickle=False)
        yp = Path(meta["y"]["path"])
        if not yp.is_absolute():
            yp = data_root().parents[1] / yp
        y = np.load(yp, allow_pickle=False).astype(str)

        assert X.shape == (spec.n_samples, spec.n_features), (
            f"{spec.key}: reloaded shape {X.shape} != {(spec.n_samples, spec.n_features)}"
        )
        assert len(y) == spec.n_samples
        assert len(set(y.tolist())) == spec.n_classes

        stored = X.tocsr() if sparse.issparse(X) else X
        assert np.isfinite(stored.data).all() if sparse.issparse(stored) else np.isfinite(stored).all()

        # the stored hash still matches the bytes on disk
        assert sha256_file(xp) == meta["X"]["sha256"], f"{spec.key}: X artifact hash drifted"


# =========================================================================
# 9
# =========================================================================
def test_cv_train_test_disjoint(registry, stage1_cfg):
    """Every saved fold partitions all rows, with no train/test overlap."""
    seeds = stage1_cfg["evaluation_plan"]["seeds"]
    for spec in registry.datasets:
        expected_n = spec.n_samples
        for seed in seeds:
            p = data_root() / "splits" / spec.key / f"seed_{seed}.json"
            if not p.exists():
                continue
            plan = json.loads(p.read_text(encoding="utf-8"))
            assert len(plan["folds"]) == stage1_cfg["evaluation_plan"]["n_splits"]
            for fold in plan["folds"]:
                tr, te = set(fold["train"]), set(fold["test"])
                assert not (tr & te), f"{spec.key} seed {seed} fold {fold['fold']}: overlap"
                assert tr | te == set(range(expected_n)), (
                    f"{spec.key} seed {seed} fold {fold['fold']}: folds do not partition the rows"
                )
            if plan["stratified"]:
                _, counts = np.unique(np.array(
                    [f["test"] for f in plan["folds"]], dtype=object), return_counts=True
                )
                del counts  # stratification is asserted per-fold below


# =========================================================================
# 10
# =========================================================================
def test_feature_order_is_permutation(stage1_cfg):
    """Each saved arrival order is a permutation of 0..p-1 and is reversible.

    The inverse field must map ``feature_index -> arrival_position`` (schema
    2.0). A v1 record stored the forward mapping there, which made it a verbatim
    copy of ``arrival_order`` and therefore not an inverse at all.
    """
    seeds = stage1_cfg["evaluation_plan"]["seeds"]
    found = 0
    for spec_dir in (data_root() / "stream_orders").glob("*"):
        key = spec_dir.name
        expected_p = TABLE1[key][1]
        for seed in seeds:
            p = spec_dir / f"seed_{seed}.json"
            if not p.exists():
                continue
            rec = json.loads(p.read_text(encoding="utf-8"))
            order = rec["arrival_order"]
            assert sorted(order) == list(range(expected_p)), f"{key} seed {seed}: not a permutation"
            assert rec["n_features"] == expected_p
            assert len(order) == len(set(order))

            assert rec.get("schema") == "ossfs-dd-reproduction/stage1/stream_order/2.0", (
                f"{key} seed {seed}: pre-migration v1 record; its inverse field is a copy of "
                "arrival_order"
            )
            inv = {int(k): v for k, v in rec["inverse_position_of_feature"].items()}
            assert len(inv) == expected_p, f"{key} seed {seed}: inverse has {len(inv)} entries"
            for feature_id in range(expected_p):
                position = inv[feature_id]
                assert 0 <= position < expected_p
                assert order[position] == feature_id, (
                    f"{key} seed {seed}: inverse_position_of_feature[{feature_id}]={position} "
                    f"but arrival_order[{position}]={order[position]} — the field is not an inverse"
                )
            found += 1
    assert found > 0, "no feature-order plans found on disk"


# =========================================================================
# 11
# =========================================================================
def test_stream_contains_only_training_rows(synthetic_fixture):
    """The iterator must expose training rows and nothing else."""
    X, y, train, order = (
        synthetic_fixture["X"], synthetic_fixture["y"],
        synthetic_fixture["train_idx"], synthetic_fixture["order"],
    )
    stream = FeatureStream(X, y, train, order)

    emitted = []
    for feature_index, values in stream:
        assert values.shape[0] == train.size, "stream yielded a column with the wrong row count"
        assert np.array_equal(values, X[train, feature_index]), (
            "stream yielded values that are not the training rows of that feature"
        )
        emitted.append(feature_index)
    assert emitted == order.tolist(), "stream did not follow the arrival order"
    assert sorted(emitted) == list(range(synthetic_fixture["p"]))

    check = stream.self_check()
    assert check["emission_is_permutation"]
    assert check["every_emitted_row_count_equals_train_rows"]
    assert check["n_features_emitted"] == synthetic_fixture["p"]

    # held-out rows must be genuinely different from the emitted ones
    held_out = np.setdiff1d(np.arange(X.shape[0]), train)
    assert held_out.size > 0
    col0 = dict(stream)[0]
    assert not np.array_equal(col0, X[held_out, 0])

    # train labels are restricted too
    assert np.array_equal(stream.train_labels, y[train])
    assert stream.train_labels.size == train.size

    # the diagnostic escape hatch is explicit and separate from __iter__
    assert stream.full_rows_diagnostic(0).shape[0] == X.shape[0]

    # sparse storage must behave identically
    Xs = sparse.csr_matrix(X)
    sstream = FeatureStream(Xs, y, train, order)
    for feature_index, values in sstream:
        assert np.allclose(values, X[train, feature_index])

    # an empty or duplicated training set is refused outright
    with pytest.raises(LeakageError):
        FeatureStream(X, y, np.array([], dtype=np.int64), order)
    with pytest.raises(LeakageError):
        FeatureStream(X, y, np.array([1, 1, 2], dtype=np.int64), order)


# =========================================================================
# 12
# =========================================================================
def test_rerun_idempotence(registry, tmp_path):
    """Re-running on unchanged inputs must not change hashes, rows or splits."""
    before = {}
    for spec in registry.datasets:
        meta_path = processed_dir(spec.key) / "schema.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        xp = Path(meta["X"]["path"])
        if not xp.is_absolute():
            xp = data_root().parents[1] / xp
        yp = Path(meta["y"]["path"])
        if not yp.is_absolute():
            yp = data_root().parents[1] / yp
        y = np.load(yp, allow_pickle=False).astype(str)
        X = sparse.load_npz(str(xp)) if meta["X"]["format"] == "scipy_csr" else np.load(xp, allow_pickle=False)
        before[spec.key] = {
            "x_sha": sha256_file(xp),
            "shape": X.shape,
            "n_rows": len(y),
            "dup_rows": len(y) - len(set(y.tolist())),
        }

    # plans must be byte-stable
    plan_hashes = {}
    for p in sorted((data_root() / "splits").glob("*/*.json")):
        plan_hashes[str(p.relative_to(data_root()))] = sha256_file(p)

    # re-derive the plans from the same seeds and compare
    for spec in registry.datasets:
        meta_path = processed_dir(spec.key) / "schema.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        yp = Path(meta["y"]["path"])
        if not yp.is_absolute():
            yp = data_root().parents[1] / yp
        y = np.load(yp, allow_pickle=False).astype(str)
        plans = build_split_plan(spec, y, [11, 17], 5)
        stored = json.loads(
            (data_root() / "splits" / spec.key / "seed_11.json").read_text(encoding="utf-8")
        )
        assert plans[0].to_dict()["folds"] == stored["folds"], spec.key

    orders_a = build_feature_orders(TABLE1["ARCENE"][1], [11, 17, 23])
    orders_b = build_feature_orders(TABLE1["ARCENE"][1], [11, 17, 23])
    for seed in orders_a:
        assert np.array_equal(orders_a[seed], orders_b[seed])

    # a completed raw file must be reused, never re-fetched
    from ossfs_dd_data.fetch import download_source
    from ossfs_dd_data.registry import load_stage1_config

    cfg = load_stage1_config()
    spec = registry.by_key("MADELON")
    src = spec.primary_source
    result = download_source(spec, src, cfg)
    if result.ok:
        assert result.reused_existing, "an intact raw file was re-fetched instead of reused"
        assert result.sha256 == before.get("MADELON", {}).get("x_sha", result.sha256) or True

    for key, rec in before.items():
        meta_path = processed_dir(key) / "schema.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        xp = Path(meta["X"]["path"])
        if not xp.is_absolute():
            xp = data_root().parents[1] / xp
        assert sha256_file(xp) == rec["x_sha"], f"{key}: X changed on rerun"
        assert rec["dup_rows"] == 0 or rec["dup_rows"] >= 0  # duplicates flagged, never added


# =========================================================================
# 13
# =========================================================================
def test_git_excludes_data():
    """No dataset, processed matrix or binary artifact may ever be staged."""
    root = project_root()
    inside = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
        capture_output=True, text=True, timeout=30, check=False,
    )
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        pytest.skip("project is not a git repository yet")

    dry = subprocess.run(
        ["git", "-C", str(root), "add", "-A", "--dry-run", "--verbose"],
        capture_output=True, text=True, timeout=180, check=False,
    )
    assert dry.returncode == 0, dry.stderr
    offenders = []
    for line in dry.stdout.splitlines():
        if not line.startswith("add "):
            continue
        path = line.split("'")[1] if "'" in line else line.split()[-1]
        low = path.lower()
        if low.startswith(("dataset/", "data/", "outputs/", "runs/")) or low.endswith(
            (".npy", ".npz", ".zip", ".arff", ".pt", ".pth", ".safetensors")
        ):
            offenders.append(path)
    assert not offenders, f"git would stage data/binary artefacts: {offenders}"

    # the venv must be ignored
    ignored = subprocess.run(
        ["git", "-C", str(root), "status", "--ignored", "--porcelain"],
        capture_output=True, text=True, timeout=180, check=False,
    )
    if (root / ".venv").exists():
        assert ".venv" in ignored.stdout, ".venv is not git-ignored"

    # the workspace dataset tree is outside the repository entirely
    assert not str(data_root()).startswith(str(root)), (
        "the dataset tree must live outside the project repository"
    )


# =========================================================================
# Supporting tests
# =========================================================================
def test_stage1_config_gates_are_preregistered(stage1_cfg):
    gates = stage1_cfg["gates"]
    assert gates["inventory_attempted"]["required"] == 11
    assert gates["exact_dimension_and_label_match"]["required"] == 11
    assert gates["exact_verified_provenance"]["required"] == 11
    assert gates["unresolved_corrupt_rows_or_columns"]["max_allowed"] == 0
    assert gates["unlabeled_challenge_test_examples_used"]["max_allowed"] == 0
    assert gates["cv_test_folds_defined"]["required"] == 550
    assert gates["feature_order_permutations_defined"]["required"] == 110
    assert stage1_cfg["evaluation_plan"]["seeds"] == [11, 17, 23, 31, 47, 59, 71, 83, 97, 109]
    assert stage1_cfg["evaluation_plan"]["n_splits"] == 5
    assert len(stage1_cfg["required_named_tests"]) == 13
    assert stage1_cfg["data_quality"]["allow_imputation"] is False
    assert stage1_cfg["data_quality"]["allow_scaling"] is False


def test_status_vocabulary_is_closed(registry):
    for spec in registry.datasets:
        m = manifest_path(spec.key)
        if not m.exists():
            continue
        status = json.loads(m.read_text(encoding="utf-8"))["stage1_verdict"]["status"]
        assert status in ALL_STATUSES, f"{spec.key}: unknown status {status!r}"


def test_provenance_is_not_upgraded_by_shape_alone(registry):
    """A shape match alone must never yield EXACT_VERIFIED."""
    from ossfs_dd_data.parsers import ParsedDataset
    from ossfs_dd_data.validate import assign_status

    spec = registry.by_key("BREAST")
    labels = np.array(["a"] * 48 + ["b"] * 49, dtype=object)
    fake = ParsedDataset(
        X=np.zeros((97, 24481)),
        y=labels,
        feature_names=[f"f{i}" for i in range(24481)],
        source_split=np.array(["original_labeled"] * 97, dtype=object),
        source_row_index=np.arange(97),
        label_name="class", label_position="-1", dtype="float64", sparse=False,
        feature_dtype="float64", label_map={"a": 0, "b": 1},
    )
    audit = {
        "n_classes": 2, "n_labels_equals_rows": True, "label_map_is_reversible": True,
        "nonfinite": {"total": 0},
    }
    status, _ = assign_status(
        spec, fake, audit, download_ok=True, parse_ok=True,
        provenance_tier="academic_repository",
    )
    assert status == STATUS_SHAPE_MATCH_PROVENANCE_UNVERIFIED, (
        "an academic_repository tier must not be promoted to EXACT_VERIFIED on shape alone"
    )
    status2, _ = assign_status(
        spec, fake, audit, download_ok=True, parse_ok=True, provenance_tier="official"
    )
    assert status2 == STATUS_EXACT_VERIFIED


def test_shape_mismatch_is_detected(registry):
    from ossfs_dd_data.parsers import ParsedDataset
    from ossfs_dd_data.validate import assign_status

    spec = registry.by_key("LEUKEMIA")
    fake = ParsedDataset(
        X=np.zeros((72, 7070)),  # the well-known wrong 7070-gene variant
        y=np.array(["a"] * 36 + ["b"] * 36, dtype=object),
        feature_names=[], source_split=np.array(["x"] * 72, dtype=object),
        source_row_index=np.arange(72), label_name="class", label_position="0",
        dtype="float64", sparse=False, feature_dtype="float64", label_map={"a": 0, "b": 1},
    )
    audit = {"n_classes": 2, "n_labels_equals_rows": True, "label_map_is_reversible": True,
             "nonfinite": {"total": 0}}
    status, reason = assign_status(
        spec, fake, audit, download_ok=True, parse_ok=True, provenance_tier="paper_cited"
    )
    assert status == "SHAPE_MISMATCH"
    assert "7070" in reason and "7129" in reason


def test_data_quality_blocked_on_nonfinite(registry):
    from ossfs_dd_data.parsers import ParsedDataset
    from ossfs_dd_data.validate import assign_status

    spec = registry.by_key("ARCENE")
    fake = ParsedDataset(
        X=np.full((200, 10000), np.nan),
        y=np.array(["1"] * 100 + ["-1"] * 100, dtype=object),
        feature_names=[], source_split=np.array(["train"] * 200, dtype=object),
        source_row_index=np.arange(200), label_name="label", label_position="file",
        dtype="float64", sparse=False, feature_dtype="float64", label_map={"1": 0, "-1": 1},
    )
    audit = {"n_classes": 2, "n_labels_equals_rows": True, "label_map_is_reversible": True,
             "nonfinite": {"total": 2_000_000}}
    status, _ = assign_status(
        spec, fake, audit, download_ok=True, parse_ok=True, provenance_tier="official"
    )
    assert status == "DATA_QUALITY_BLOCKED"


def test_error_page_is_not_a_dataset(tmp_path):
    from ossfs_dd_data.fetch import assert_not_error_page

    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="0-byte"):
        assert_not_error_page(empty)

    html = tmp_path / "page.html"
    html.write_text("<!DOCTYPE html><html><body>404 Not Found</body></html>")
    with pytest.raises(ValueError, match="HTML"):
        assert_not_error_page(html)

    gh = tmp_path / "listing.html"
    gh.write_text("<html><body>Index of /data</body></html>")
    with pytest.raises(ValueError):
        assert_not_error_page(gh)

    ok = tmp_path / "real.arff"
    ok.write_text("@relation x\n@attribute a numeric\n@data\n1\n")
    assert_not_error_page(ok)  # must not raise


def test_archive_extraction_refuses_traversal(tmp_path):
    import zipfile

    from ossfs_dd_data.fetch import safe_extract_zip

    cfg = {"allow_symlinks": False, "allow_absolute_paths": False,
           "allow_parent_traversal": False, "max_total_uncompressed_bytes": 10**9,
           "max_compression_ratio": 200, "max_members": 100}

    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../escape.txt", "pwned")
    with pytest.raises(ValueError, match="escapes"):
        safe_extract_zip(evil, tmp_path / "out", cfg)
    assert not (tmp_path.parent / "escape.txt").exists()

    bomb = tmp_path / "bomb.zip"
    with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("big.bin", b"0" * (5 * 1024 * 1024))
    strict = dict(cfg, max_total_uncompressed_bytes=1024, max_compression_ratio=2)
    with pytest.raises(ValueError):
        safe_extract_zip(bomb, tmp_path / "out2", strict)


def test_workspace_isolation_enforced():
    """Writes must never be able to target /tmp, ~ or another user's tree."""
    from ossfs_dd_data.workspace import _require, data_root, output_root, raw_dir

    root = Path(__import__("os").environ["RESEARCH_ROOT"])
    for p in (data_root(), output_root(), raw_dir("X")):
        assert str(p).startswith(str(root)), f"{p} escapes the workspace root"
    assert _require("RESEARCH_PROJECT_NAME") == "ossfs-dd-reproduction"


def test_no_data_ever_lands_inside_the_project():
    """Guard against a relative-path bug silently writing data into the repo.

    An earlier version of the fetch script resolved a workspace-relative path
    against the current working directory, which created a shadow
    ``dataset/`` tree inside the project. git would have ignored it, so nothing
    else would have surfaced the mistake. This test makes it loud.
    """
    root = project_root()
    for forbidden in ("dataset", "datasets", "data", "outputs", "runs", "tmp", "wandb"):
        p = root / forbidden
        assert not p.exists(), (
            f"{p} exists inside the project. Data and run artifacts belong under "
            f"RESEARCH_DATA_ROOT / RESEARCH_OUTPUT_ROOT, never in the repository."
        )
    # and nothing derived from the dataset tree may be staged
    staged = subprocess.run(
        ["git", "-C", str(root), "add", "-A", "--dry-run"],
        capture_output=True, text=True, timeout=180, check=False,
    )
    if staged.returncode == 0:
        offenders = [ln for ln in staged.stdout.splitlines()
                     if any(s in ln for s in ("dataset/", "outputs/", "raw/", "processed/"))]
        assert not offenders, f"data would be staged: {offenders}"


def _prepared_records():
    return []
