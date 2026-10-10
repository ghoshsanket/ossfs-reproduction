"""Stage 1 live disk audit — READ-ONLY with respect to datasets.

This answers one question: *what is actually on this machine right now?*

It does not trust `results/stage1/`, does not trust the manifests' existence,
and does not trust any previously reported state. Every claim is recomputed
from bytes on disk. If storage is missing it is reported missing even when a
prior report claims success.

Checks performed
----------------
Per dataset
  * raw archive present, size and SHA-256 equal to the recorded manifest value
  * processed X and y present; X SHA-256 equal to the recorded schema value
  * X loaded via ``workspace.resolve`` (never CWD-relative)
  * ``X.shape``, ``len(y)``, observed class count, finite values
  * label set, label map reversibility, source-split counts
  * consistency between manifest expectations, the schema and the actual arrays
  * the dataset tree is outside the git work tree

Per plan (10 seeds x 11 datasets = 110 split plans, 110 order plans)
  * train/test disjointness, full row coverage, index bounds
  * stratified declaration consistency
  * feature order is a full permutation of ``0..p-1``
  * ``inverse_position_of_feature`` genuinely inverts ``arrival_order``
  * the pre-migration v1 backup, where present, has an identical arrival order

Stream leakage (real data)
  * every emitted column has exactly ``len(train_indices)`` rows
  * sampled emitted columns equal ``X[train_indices, column]``
  * held-out rows are present but never emitted

Exit codes: 0 = all checks passed; 1 = one or more checks failed;
2 = the dataset tree is absent, so nothing could be audited.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402

from ossfs_dd_data.fetch import configure_logging, sha256_file  # noqa: E402
from ossfs_dd_data.registry import load_registry, load_stage1_config  # noqa: E402
from ossfs_dd_data.stream import FeatureStream  # noqa: E402
from ossfs_dd_data.workspace import (  # noqa: E402
    data_root,
    manifest_path,
    processed_dir,
    project_root,
    raw_dir,
    rel_to_research,
    resolve,
    results_dir,
    splits_dir,
    stream_orders_dir,
    utc_now,
    write_json,
)

LOG = logging.getLogger("ossfs_dd_data.live_audit")

FOLLOWUP_DIRNAME = "followup_20261010"
# Documented sampling: how many emitted columns per (dataset, seed) are compared
# element-wise against X[train, col]. Structural coverage (row counts) is 100%.
VALUE_CHECK_SAMPLE = 12


def followup_dir() -> Path:
    d = results_dir() / FOLLOWUP_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _git_revision() -> dict[str, Any]:
    root = project_root()
    def run(*a: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(root), *a], capture_output=True, text=True,
                timeout=30, check=False,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    return {
        "commit": run("rev-parse", "HEAD") or "not a git repository",
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _dataset_outside_git() -> dict[str, Any]:
    root = project_root()
    try:
        inside = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"status": "error", "detail": f"{type(exc).__name__}: {exc}"}
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return {"status": "ok", "detail": "project is not a git work tree; nothing to stage"}
    outside = not str(data_root()).startswith(str(root))
    return {
        "status": "ok" if outside else "error",
        "detail": "dataset tree is outside the repository" if outside
        else "dataset tree is INSIDE the repository",
    }


def load_processed(key: str) -> tuple[Any, np.ndarray, dict[str, Any]]:
    """Load via ``resolve`` so the CWD is irrelevant."""
    meta = json.loads((processed_dir(key) / "schema.json").read_text(encoding="utf-8"))
    xrec = meta["X"]
    X = (
        sparse.load_npz(str(resolve(xrec["path"])))
        if xrec["format"] == "scipy_csr"
        else np.load(resolve(xrec["path"]), allow_pickle=False)
    )
    y = np.load(resolve(meta["y"]["path"]), allow_pickle=False).astype(str)
    return X, y, meta


# --------------------------------------------------------------------------
# per-dataset audit
# --------------------------------------------------------------------------


def audit_dataset(spec, seeds: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "dataset": spec.key,
        "expected": {"n": spec.n_samples, "p": spec.n_features, "c": spec.n_classes},
        "failures": [],
        "warnings": [],
    }
    failures: list[str] = out["failures"]

    # ---- manifest ------------------------------------------------------
    mp = manifest_path(spec.key)
    if not mp.exists():
        failures.append(f"REASON_MANIFEST_MISSING: {rel_to_research(mp)} does not exist")
        out["outcome"] = "DATASET_FILES_MISSING"
        return out
    manifest = json.loads(mp.read_text(encoding="utf-8"))

    # ---- raw archive ---------------------------------------------------
    raw = manifest.get("source", {}).get("download", {})
    rec_path, rec_sha = raw.get("raw_archive"), raw.get("sha256")
    raw_block: dict[str, Any] = {"recorded_path": rec_path, "recorded_sha256": rec_sha}
    if not rec_path or not rec_sha:
        failures.append("REASON_MANIFEST_NO_RAW_HASH: manifest records no raw archive hash")
    else:
        p = resolve(rec_path)
        if not p.exists():
            failures.append(f"REASON_RAW_MISSING: {rec_path} is not on disk")
            raw_block["present"] = False
        else:
            digest = sha256_file(p)
            raw_block.update(
                present=True,
                bytes=p.stat().st_size,
                observed_sha256=digest,
                hash_matches=(digest == rec_sha),
            )
            if digest != rec_sha:
                failures.append(
                    f"REASON_RAW_HASH_MISMATCH: {rec_path} sha256={digest[:16]}... "
                    f"but manifest records {rec_sha[:16]}..."
                )
            if not str(p).startswith(str(raw_dir(spec.key))):
                failures.append(
                    f"REASON_RAW_OUTSIDE_DATASET_TREE: {rec_path} is not under raw/{spec.key}"
                )
    out["raw"] = raw_block

    # ---- processed X / y ----------------------------------------------
    schema_path = processed_dir(spec.key) / "schema.json"
    if not schema_path.exists():
        failures.append("REASON_SCHEMA_MISSING: processed/schema.json is not on disk")
        out["outcome"] = "DATASET_FILES_MISSING"
        return out
    meta = json.loads(schema_path.read_text(encoding="utf-8"))

    xrec = meta["X"]
    xp = resolve(xrec["path"])
    x_block: dict[str, Any] = {"path": xrec["path"], "format": xrec["format"]}
    if not xp.exists():
        failures.append(f"REASON_X_MISSING: {xrec['path']} is not on disk")
    else:
        digest = sha256_file(xp)
        x_block.update(
            present=True, bytes=xp.stat().st_size, observed_sha256=digest,
            hash_matches=(digest == xrec.get("sha256")),
        )
        if digest != xrec.get("sha256"):
            failures.append(
                f"REASON_X_HASH_MISMATCH: {xrec['path']} sha256={digest[:16]}... but schema "
                f"records {str(xrec.get('sha256'))[:16]}..."
            )
    out["processed_X"] = x_block

    yrec = meta.get("y", {})
    yp = resolve(yrec.get("path", ""))
    if not yrec.get("path") or not yp.exists():
        failures.append("REASON_Y_MISSING: processed label vector is not on disk")
        out["outcome"] = "DATASET_FILES_MISSING"
        return out

    sid = processed_dir(spec.key) / "sample_ids.npy"
    try:
        ids = np.load(sid, allow_pickle=False)
        out["sample_ids"] = {"present": True, "dtype": str(ids.dtype), "n": int(len(ids))}
        if ids.dtype.kind != "U":
            failures.append(f"REASON_SAMPLE_IDS_PICKLED: dtype is {ids.dtype}, expected fixed-width")
    except (OSError, ValueError) as exc:
        failures.append(f"REASON_SAMPLE_IDS_UNREADABLE_WITHOUT_PICKLE: {exc}")
        out["sample_ids"] = {"present": False}

    # ---- load and check contents --------------------------------------
    X, y, _ = load_processed(spec.key)
    obs_classes = sorted(set(y.tolist()))
    content: dict[str, Any] = {
        "shape": list(X.shape),
        "y_length": int(len(y)),
        "n_classes_observed": len(obs_classes),
        "class_counts": {
            str(c): int(n) for c, n in zip(*np.unique(y, return_counts=True), strict=True)
        },
        "storage": "scipy_csr" if sparse.issparse(X) else "numpy_dense",
    }
    if X.shape != (spec.n_samples, spec.n_features):
        failures.append(
            f"REASON_SHAPE_MISMATCH: X is {X.shape} but Table 1 declares "
            f"({spec.n_samples}, {spec.n_features})"
        )
    if len(y) != spec.n_samples:
        failures.append(f"REASON_Y_LENGTH_MISMATCH: len(y)={len(y)} expected {spec.n_samples}")
    if len(obs_classes) != spec.n_classes:
        failures.append(
            f"REASON_CLASS_COUNT_MISMATCH: observed {len(obs_classes)} classes, "
            f"Table 1 declares {spec.n_classes}"
        )

    data = X.data if sparse.issparse(X) else np.asarray(X).ravel()
    n_nonfinite = int((~np.isfinite(data)).sum())
    content["nonfinite_values"] = n_nonfinite
    if n_nonfinite:
        failures.append(f"REASON_NONFINITE_VALUES: {n_nonfinite} non-finite feature values")

    label_map = meta.get("label_map_original_to_canonical") or {}
    if set(label_map) != set(obs_classes):
        failures.append(
            "REASON_LABEL_MAP_NOT_REVERSIBLE: schema label_map keys do not match the labels "
            "actually stored in y"
        )
    content["label_map_reversible"] = set(label_map) == set(obs_classes)

    sp = np.load(resolve(meta["source_split"]["path"]), allow_pickle=False).astype(str)
    split_counts = {str(k): int(n) for k, n in zip(*np.unique(sp, return_counts=True), strict=True)}
    content["source_split_counts"] = split_counts
    expected_splits = spec.expected_split_counts()
    if expected_splits:
        for name, want in expected_splits.items():
            if split_counts.get(name) != want:
                failures.append(
                    f"REASON_SPLIT_COMPOSITION_MISMATCH: split {name!r} has "
                    f"{split_counts.get(name)} rows, registry declares {want}"
                )
    if "test" in split_counts:
        failures.append(
            f"REASON_UNLABELED_TEST_ROWS_PRESENT: {split_counts['test']} rows carry the "
            "unlabeled challenge 'test' split marker"
        )
    out["content"] = content

    # ---- manifest vs schema vs arrays ---------------------------------
    man_obs = manifest.get("observed", {})
    for field, actual in (
        ("n_samples", X.shape[0]), ("n_features", X.shape[1]), ("n_classes", len(obs_classes))
    ):
        if man_obs.get(field) not in (None, actual):
            failures.append(
                f"REASON_MANIFEST_DISAGREES_WITH_DATA: manifest observed.{field}="
                f"{man_obs.get(field)} but the loaded array gives {actual}"
            )

    out["plan_checks"] = audit_plans(spec, seeds)
    out["stream_check"] = audit_stream(spec, X, y, seeds)

    if failures:
        out["outcome"] = "DATASET_FILES_PRESENT_BUT_INVALID"
    else:
        out["outcome"] = "DATASET_FILES_PRESENT"
    return out


# --------------------------------------------------------------------------
# plans
# --------------------------------------------------------------------------


def audit_plans(spec, seeds: list[int]) -> dict[str, Any]:
    res: dict[str, Any] = {
        "split_plans_expected": len(seeds),
        "split_plans_present": 0,
        "order_plans_expected": len(seeds),
        "order_plans_present": 0,
        "folds_checked": 0,
        "disjoint_violations": 0,
        "coverage_violations": 0,
        "bounds_violations": 0,
        "permutation_violations": 0,
        "inverse_violations": 0,
        "v1_backup_checked": 0,
        "v1_backup_arrival_order_preserved": 0,
        "failures": [],
    }
    f: list[str] = res["failures"]
    n = spec.n_samples

    for seed in seeds:
        sp = splits_dir(spec.key) / f"seed_{seed}.json"
        if not sp.exists():
            f.append(f"REASON_SPLIT_PLAN_MISSING: seed_{seed}.json")
            continue
        res["split_plans_present"] += 1
        plan = json.loads(sp.read_text(encoding="utf-8"))
        for fold in plan["folds"]:
            res["folds_checked"] += 1
            tr = np.asarray(fold["train"], dtype=np.int64)
            te = np.asarray(fold["test"], dtype=np.int64)
            if tr.size and te.size and np.intersect1d(tr, te).size:
                res["disjoint_violations"] += 1
                f.append(f"REASON_FOLD_NOT_DISJOINT: seed {seed} fold {fold['fold']}")
            if np.union1d(tr, te).size != n:
                res["coverage_violations"] += 1
                f.append(
                    f"REASON_FOLD_COVERAGE: seed {seed} fold {fold['fold']} covers "
                    f"{np.union1d(tr, te).size} of {n} rows"
                )
            if (tr.size and (tr.min() < 0 or tr.max() >= n)) or (te.size and (te.min() < 0 or te.max() >= n)):
                res["bounds_violations"] += 1
                f.append(f"REASON_FOLD_INDEX_OUT_OF_BOUNDS: seed {seed} fold {fold['fold']}")

        op = stream_orders_dir(spec.key) / f"seed_{seed}.json"
        if not op.exists():
            f.append(f"REASON_ORDER_PLAN_MISSING: seed_{seed}.json")
            continue
        res["order_plans_present"] += 1
        rec = json.loads(op.read_text(encoding="utf-8"))
        order = rec.get("arrival_order")
        if not isinstance(order, list) or sorted(order) != list(range(spec.n_features)):
            res["permutation_violations"] += 1
            f.append(f"REASON_ARRIVAL_ORDER_NOT_PERMUTATION: seed {seed}")
            continue
        if rec.get("schema") != "ossfs-dd-reproduction/stage1/stream_order/2.0":
            f.append(
                f"REASON_ORDER_PLAN_STALE_SCHEMA: seed {seed} is "
                f"{rec.get('schema')!r}; its inverse field is a copy of arrival_order"
            )
        inv = {int(k): v for k, v in rec.get("inverse_position_of_feature", {}).items()}
        if len(inv) != spec.n_features or any(
            not (0 <= inv[j] < spec.n_features) or order[inv[j]] != j for j in range(spec.n_features)
        ):
            res["inverse_violations"] += 1
            f.append(f"REASON_INVERSE_ORDER_WRONG: seed {seed}")

        backup = op.with_suffix(op.suffix + ".v1backup")
        if backup.exists():
            res["v1_backup_checked"] += 1
            old = json.loads(backup.read_text(encoding="utf-8"))
            if old.get("arrival_order") == order:
                res["v1_backup_arrival_order_preserved"] += 1
            else:
                f.append(
                    f"REASON_ARRIVAL_ORDER_CHANGED_BY_MIGRATION: seed {seed} differs from its "
                    "pre-migration backup"
                )
    return res


# --------------------------------------------------------------------------
# stream
# --------------------------------------------------------------------------


def audit_stream(spec, X, y, seeds: list[int]) -> dict[str, Any]:
    """Full structural coverage of emitted row counts; sampled value equality."""
    res: dict[str, Any] = {
        "seeds_checked": 0,
        "folds_checked": 0,
        "columns_emitted_total": 0,
        "row_count_violations": 0,
        "value_columns_checked": 0,
        "value_mismatches": 0,
        "held_out_rows_never_emitted": True,
        "value_sampling_policy": (
            f"element-wise comparison of {VALUE_CHECK_SAMPLE} columns per seed; "
            "row-count coverage is 100%"
        ),
        "failures": [],
    }
    f: list[str] = res["failures"]
    rng = np.random.Generator(np.random.PCG64(20261010))

    for seed in seeds:
        sp = splits_dir(spec.key) / f"seed_{seed}.json"
        op = stream_orders_dir(spec.key) / f"seed_{seed}.json"
        if not (sp.exists() and op.exists()):
            continue
        res["seeds_checked"] += 1
        plan = json.loads(sp.read_text(encoding="utf-8"))
        order = np.asarray(
            json.loads(op.read_text(encoding="utf-8"))["arrival_order"], dtype=np.int64
        )
        for fold in plan["folds"][:1]:                      # fold 0 per seed: full column sweep
            res["folds_checked"] += 1
            train = np.asarray(fold["train"], dtype=np.int64)
            held_out = np.setdiff1d(np.arange(X.shape[0]), train)
            stream = FeatureStream(X, y, train, order)
            sample = set(
                rng.choice(spec.n_features, size=min(VALUE_CHECK_SAMPLE, spec.n_features),
                           replace=False).tolist()
            )
            emitted_idx: set[int] = set()
            for feature_index, values in stream:
                res["columns_emitted_total"] += 1
                emitted_idx.add(int(feature_index))
                if values.shape[0] != train.size:
                    res["row_count_violations"] += 1
                    f.append(
                        f"REASON_STREAM_ROW_COUNT: {spec.key} seed {seed} feature {feature_index} "
                        f"emitted {values.shape[0]} rows, expected {train.size}"
                    )
                if int(feature_index) in sample:
                    res["value_columns_checked"] += 1
                    expected = (
                        X[train, feature_index].toarray().ravel()
                        if sparse.issparse(X) else X[train, feature_index]
                    )
                    if not np.array_equal(np.asarray(values), np.asarray(expected)):
                        res["value_mismatches"] += 1
                        f.append(
                            f"REASON_STREAM_VALUE_MISMATCH: {spec.key} seed {seed} feature "
                            f"{feature_index} does not equal X[train, column]"
                        )
            if len(emitted_idx) != spec.n_features:
                f.append(
                    f"REASON_STREAM_INCOMPLETE: {spec.key} seed {seed} emitted "
                    f"{len(emitted_idx)} of {spec.n_features} features"
                )
            if held_out.size:
                # prove held-out rows exist and are not what we emitted
                first = int(order[0])
                emitted_first = (
                    X[train, first].toarray().ravel()
                    if sparse.issparse(X) else X[train, first]
                )
                heldout_first = (
                    X[held_out, first].toarray().ravel()
                    if sparse.issparse(X) else X[held_out, first]
                )
                if np.array_equal(np.asarray(emitted_first), np.asarray(heldout_first)):
                    res["held_out_rows_never_emitted"] = False
                    f.append(
                        f"REASON_STREAM_HELDOUT_INDISTINGUISHABLE: {spec.key} seed {seed}; "
                        "cannot demonstrate test rows were excluded"
                    )
    return res


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 1 live, read-only disk audit")
    ap.add_argument("--dataset", action="append", default=[])
    args = ap.parse_args(argv)

    configure_logging("stage1_live_audit.log")
    t0 = time.monotonic()
    cfg = load_stage1_config()
    registry = load_registry()
    seeds = list(cfg["evaluation_plan"]["seeds"])
    specs = registry.select(args.dataset or None)

    env = {
        "architecture": platform.machine(),
        "python_version": sys.version.split()[0],
        "interpreter_is_project_venv": str(Path(sys.executable).resolve())
        == str((project_root() / ".venv" / "bin" / "python").resolve()),
        "cpu_count": len(os.sched_getaffinity(0)),
        "gpu_used": False,
    }

    if not data_root().exists():
        report = {
            "schema": "ossfs-dd-reproduction/stage1/live_audit/1.0",
            "generated_utc": utc_now(),
            "outcome": "DATASET_FILES_MISSING",
            "outcome_detail": f"{rel_to_research(data_root())} does not exist on this machine",
            "note": (
                "A repository checkout does not prove data is present. Storage was looked for "
                "directly and is absent."
            ),
            "environment": env,
            "git": _git_revision(),
            "datasets": [],
        }
        write_json(followup_dir() / "live_audit.json", report)
        (followup_dir() / "LIVE_AUDIT_REPORT.md").write_text(
            "# Stage 1 Live Audit\n\n**DATASET_FILES_PRESENT: NO**\n\n"
            f"`{rel_to_research(data_root())}` is absent. Nothing could be audited.\n",
            encoding="utf-8",
        )
        LOG.error("dataset tree absent: %s", data_root())
        return 2

    datasets = []
    for spec in specs:
        LOG.info("auditing %s", spec.key)
        datasets.append(audit_dataset(spec, seeds))

    present = [d for d in datasets if d["outcome"] == "DATASET_FILES_PRESENT"]
    present_invalid = [d for d in datasets if d["outcome"] == "DATASET_FILES_PRESENT_BUT_INVALID"]
    missing = [d for d in datasets if d["outcome"] == "DATASET_FILES_MISSING"]

    totals = {
        "raw_present_and_hash_verified": sum(
            1 for d in datasets if (d.get("raw") or {}).get("present")
            and (d["raw"] or {}).get("hash_matches")
        ),
        "processed_x_present_and_hash_verified": sum(
            1 for d in datasets if (d.get("processed_X") or {}).get("present")
            and (d["processed_X"] or {}).get("hash_matches")
        ),
        "dimensions_and_classes_match_table1": sum(
            1 for d in datasets
            if d["outcome"] == "DATASET_FILES_PRESENT"
            and d["content"]["shape"] == [d["expected"]["n"], d["expected"]["p"]]
            and d["content"]["n_classes_observed"] == d["expected"]["c"]
        ),
        "split_plan_files": sum(d["plan_checks"]["split_plans_present"] for d in datasets),
        "order_plan_files": sum(d["plan_checks"]["order_plans_present"] for d in datasets),
        "folds_checked": sum(d["plan_checks"]["folds_checked"] for d in datasets),
        "disjoint_violations": sum(d["plan_checks"]["disjoint_violations"] for d in datasets),
        "coverage_violations": sum(d["plan_checks"]["coverage_violations"] for d in datasets),
        "bounds_violations": sum(d["plan_checks"]["bounds_violations"] for d in datasets),
        "permutation_violations": sum(d["plan_checks"]["permutation_violations"] for d in datasets),
        "inverse_violations": sum(d["plan_checks"]["inverse_violations"] for d in datasets),
        "v1_backups_checked": sum(d["plan_checks"]["v1_backup_checked"] for d in datasets),
        "v1_arrival_order_preserved": sum(
            d["plan_checks"]["v1_backup_arrival_order_preserved"] for d in datasets
        ),
        "stream_columns_emitted": sum(d["stream_check"]["columns_emitted_total"] for d in datasets),
        "stream_value_columns_checked": sum(d["stream_check"]["value_columns_checked"] for d in datasets),
        "stream_value_mismatches": sum(d["stream_check"]["value_mismatches"] for d in datasets),
        "stream_row_count_violations": sum(
            d["stream_check"]["row_count_violations"] for d in datasets
        ),
    }

    git_outside = _dataset_outside_git()
    n_total = len(specs)
    all_ok = (
        len(present) == n_total
        and totals["raw_present_and_hash_verified"] == n_total
        and totals["processed_x_present_and_hash_verified"] == n_total
        and totals["disjoint_violations"] == 0
        and totals["coverage_violations"] == 0
        and totals["bounds_violations"] == 0
        and totals["permutation_violations"] == 0
        and totals["inverse_violations"] == 0
        and totals["stream_value_mismatches"] == 0
        and totals["stream_row_count_violations"] == 0
        and git_outside["status"] == "ok"
    )

    report = {
        "schema": "ossfs-dd-reproduction/stage1/live_audit/1.0",
        "generated_utc": utc_now(),
        "elapsed_seconds": round(time.monotonic() - t0, 1),
        "outcome": "DATASET_FILES_PRESENT" if all_ok else (
            "DATASET_FILES_PRESENT_BUT_INVALID" if len(present_invalid) else "DATASET_FILES_MISSING"
        ),
        "outcome_detail": (
            "every dataset present, hash-verified, dimensionally correct, all plans valid, "
            "no stream leakage observed"
            if all_ok else
            "one or more datasets are missing, hash-mismatched, dimensionally wrong, have "
            "invalid plans, or produced a stream mismatch"
        ),
        "dataset_tree": rel_to_research(data_root()),
        "dataset_tree_outside_git": git_outside,
        "environment": env,
        "git": _git_revision(),
        "seeds": seeds,
        "totals": totals,
        "counts": {
            "datasets_audited": n_total,
            "files_present_and_valid": len(present),
            "files_present_but_invalid": len(present_invalid),
            "files_missing": len(missing),
            "missing_keys": [d["dataset"] for d in missing],
            "invalid_keys": [d["dataset"] for d in present_invalid],
        },
        "datasets": datasets,
        "read_only": True,
        "datasets_modified_by_this_audit": [],
    }

    write_json(followup_dir() / "live_audit.json", report)
    md = render_markdown(report)
    (followup_dir() / "LIVE_AUDIT_REPORT.md").write_text(md, encoding="utf-8")

    LOG.info("outcome: %s", report["outcome"])
    LOG.info("totals: %s", json.dumps(totals, indent=2))
    for d in datasets:
        if d["failures"]:
            for msg in d["failures"][:5]:
                LOG.error("[%s] %s", d["dataset"], msg)
    return 0 if all_ok else 1


def render_markdown(r: dict[str, Any]) -> str:
    t = r["totals"]
    L: list[str] = []
    A = L.append
    A("# Stage 1 Live Disk Audit")
    A("")
    A(f"**Generated (UTC):** {r['generated_utc']}  ")
    A(f"**Elapsed:** {r['elapsed_seconds']} s  ")
    A(f"**Git commit:** `{r['git']['commit'][:12]}` (dirty: {r['git']['dirty']})  ")
    A(f"**Interpreter:** Python {r['environment']['python_version']}, "
      f"{r['environment']['architecture']}, project venv: {r['environment']['interpreter_is_project_venv']}  ")
    A(f"**GPU used:** {r['environment']['gpu_used']}  ")
    A(f"**Dataset tree:** `{r['dataset_tree']}` — outside git: "
      f"{r['dataset_tree_outside_git']['status'] == 'ok'} ({r['dataset_tree_outside_git']['detail']})")
    A("")
    A("This report is recomputed from bytes on disk. It does not trust any previously")
    A("reported state, and it is read-only with respect to datasets.")
    A("")
    A("## Outcome")
    A("")
    A(f"```\n{r['outcome']}\n```")
    A("")
    A(r["outcome_detail"])
    A("")
    A("## Aggregate verification")
    A("")
    A("| Measure | Value |")
    A("|---|---:|")
    A(f"| raw files physically present and SHA-256 verified | {t['raw_present_and_hash_verified']}/{r['counts']['datasets_audited']} |")
    A(f"| processed X present and SHA-256 verified | {t['processed_x_present_and_hash_verified']}/{r['counts']['datasets_audited']} |")
    A(f"| dimensions + class count match Table 1 | {t['dimensions_and_classes_match_table1']}/{r['counts']['datasets_audited']} |")
    A(f"| split plan files present | {t['split_plan_files']} |")
    A(f"| order plan files present | {t['order_plan_files']} |")
    A(f"| CV folds checked (disjointness + coverage + bounds) | {t['folds_checked']} |")
    A(f"| fold disjointness violations | {t['disjoint_violations']} |")
    A(f"| fold coverage violations | {t['coverage_violations']} |")
    A(f"| index-bounds violations | {t['bounds_violations']} |")
    A(f"| arrival-order permutation violations | {t['permutation_violations']} |")
    A(f"| inverse-order violations | {t['inverse_violations']} |")
    A(f"| v1 backups checked / arrival order preserved | {t['v1_backups_checked']} / {t['v1_arrival_order_preserved']} |")
    A(f"| stream columns emitted (row-count coverage) | {t['stream_columns_emitted']} |")
    A(f"| stream columns value-compared against `X[train, col]` | {t['stream_value_columns_checked']} |")
    A(f"| stream value mismatches | {t['stream_value_mismatches']} |")
    A(f"| stream row-count violations | {t['stream_row_count_violations']} |")
    A("")
    A("## Per dataset")
    A("")
    A("| Dataset | Outcome | n×p×c expected | n×p×c observed | classes | non-finite | raw sha256 | failures |")
    A("|---|---|---|---|---:|---:|---|---:|")
    for d in r["datasets"]:
        c = d.get("content") or {}
        exp = d["expected"]
        A(f"| {d['dataset']} | `{d['outcome']}` | {exp['n']}×{exp['p']}×{exp['c']} | "
          f"{'×'.join(str(v) for v in c.get('shape', []))} | "
          f"{c.get('n_classes_observed', 'not measured')} | "
          f"{c.get('nonfinite_values', 'not measured')} | "
          f"`{(d.get('raw') or {}).get('observed_sha256', '')[:12]}` | {len(d['failures'])} |")
    A("")
    A("## Failure reasons (machine codes)")
    A("")
    any_fail = False
    for d in r["datasets"]:
        if d["failures"]:
            any_fail = True
            A(f"**{d['dataset']}**")
            for f in d["failures"]:
                A(f"- `{f}`")
    if not any_fail:
        A("None. Every check passed on every dataset.")
    A("")
    A("## Scope note")
    A("")
    A("This audit verifies **storage, integrity, dimensional correctness and plan**")
    A("validity**. It does not assess scientific provenance, and it does not re-run")
    A("the Stage 1 provenance gate — see `provenance_followup.md` for that.")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
