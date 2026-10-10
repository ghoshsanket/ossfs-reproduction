"""Stage 1b — convert raw bytes into canonical X/y, and build split/order plans.

A blocked dataset never aborts the pass: every remaining dataset is still
attempted and each keeps its own status.

Idempotent: on rerun with unchanged inputs the processed arrays, split plans and
feature orders are byte-identical and are not rewritten.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from scipy import sparse

from ossfs_dd_data import parsers
from ossfs_dd_data.fetch import configure_logging, sha256_file
from ossfs_dd_data.parsers import ParsedDataset, ParseError
from ossfs_dd_data.registry import DatasetSpec, load_registry, load_stage1_config
from ossfs_dd_data.splits import (
    build_feature_orders,
    build_split_plan,
    feature_order_record,
    validate_plan,
)
from ossfs_dd_data.validate import assign_status, audit_dataset
from ossfs_dd_data.workspace import (
    ensure_data_dirs,
    processed_dir,
    raw_dir,
    rel_to_research,
    resolve,
    splits_dir,
    stream_orders_dir,
    write_json,
)

LOG = logging.getLogger("ossfs_dd_data.stage1.prepare")


def parse_dataset(spec: DatasetSpec, cfg: dict) -> ParsedDataset:
    """Dispatch to the adapter named by the registry for this dataset."""
    source = spec.primary_source
    root = raw_dir(spec.key)
    extracted = root / "extracted"
    density = float(cfg["parsing"]["sparse_if_density_below"])
    dtype = str(cfg["parsing"]["float_dtype"])

    if source.fmt == "arff":
        if source.filename_in_archive.lower().endswith(".zip"):
            path = parsers.resolve_member(extracted, [source.filename_in_archive])
        else:
            path = root / source.filename_in_archive
        if not path.exists():
            raise ParseError(f"{spec.key}: expected ARFF at {rel_to_research(path)}")
        return parsers.parse_arff(path, density_threshold=density, dtype=dtype)

    if source.fmt == "csv_label_last":
        if source.filename_in_archive.lower().endswith(".zip"):
            path = parsers.resolve_member(extracted, [source.filename_in_archive])
        else:
            path = root / source.filename_in_archive
        if not path.exists():
            raise ParseError(f"{spec.key}: expected CSV at {rel_to_research(path)}")
        return parsers.parse_csv_label_last(
            path,
            label_column=-1,
            expected_columns=spec.total_attributes,
            density_threshold=density,
            dtype=dtype,
            split_name=spec.combine_splits()[0] if spec.combine_splits() else "original_labeled",
        )

    if source.fmt == "uci_data_labels_pair":
        data_paths, label_paths = {}, {}
        for role in ("train", "valid"):
            d_key, l_key = f"data_{role}", f"labels_{role}"
            if source.members.get(d_key):
                data_paths[role] = parsers.resolve_member(extracted, source.members[d_key])
            if source.members.get(l_key):
                label_paths[role] = parsers.resolve_member(extracted, source.members[l_key])
        if not data_paths:
            raise ParseError(f"{spec.key}: registry lists no data members for {source.url}")
        param_path = None
        if source.members.get("source_param"):
            try:
                param_path = parsers.resolve_member(extracted, source.members["source_param"])
            except parsers.ParseError:
                LOG.warning("[%s] no .param file found; checksum verification unavailable", spec.key)
        return parsers.parse_uci_data_labels_pair(
            data_paths,
            label_paths,
            n_columns=spec.n_features,
            density_threshold=density,
            dtype=dtype,
            strict_order=tuple(spec.combine_splits()) or ("train", "valid"),
            param_path=param_path,
        )

    if source.fmt in {"rdata", "rda"}:
        path = root / source.filename_in_archive
        return parsers.parse_rdata(path, dtype=dtype)

    raise ParseError(f"{spec.key}: unsupported source format {source.fmt!r}")


def save_processed(spec: DatasetSpec, parsed: ParsedDataset, cfg: dict) -> dict:
    """Persist X, y, row ids and feature ids. No training features are derived."""
    out = processed_dir(spec.key)
    out.mkdir(parents=True, exist_ok=True)

    common = {
        "dataset": spec.key,
        "schema": "ossfs-dd-reproduction/stage1/processed/2.0",
        "n_samples": parsed.n_samples,
        "n_features": parsed.n_features,
        "n_classes": len(np.unique(parsed.y)),
        "feature_dtype": parsed.feature_dtype,
        "label_map_original_to_canonical": parsed.label_map,
        "feature_name_scheme": (
            "source_attribute_names"
            if parsed.feature_names and not parsed.feature_names[0].startswith("feature_")
            else "positional_identity_0_to_p_minus_1"
        ),
        "features_exclude_class_column": True,
        "label_name": parsed.label_name,
        "label_position": parsed.label_position,
        "orientation": parsed.orientation,
        "notes": parsed.notes,
        "parsing_extra": parsed.extra,
    }

    if sparse.issparse(parsed.X):
        xp = out / "X_csr.npz"
        sparse.save_npz(str(xp), parsed.X.tocsr(), compressed=True)
        x_record = {"format": "scipy_csr", "path": rel_to_research(xp)}
    else:
        xp = out / "X.npy"
        np.save(xp, np.ascontiguousarray(parsed.X), allow_pickle=False)
        x_record = {"format": "numpy_npy", "path": rel_to_research(xp)}

    yp = out / "y.npy"
    np.save(yp, parsed.y.astype(str), allow_pickle=False)

    sp = out / "source_split.npy"
    np.save(sp, parsed.source_split.astype(str), allow_pickle=False)

    ri = out / "source_row_index.npy"
    np.save(ri, parsed.source_row_index.astype(np.int64), allow_pickle=False)

    # Fixed-width Unicode, written without pickle. The previous object-dtype form
    # could only be read with allow_pickle=True; existing artifacts are migrated
    # non-lossfully by ossfs_dd_data.migrations.migrate_sample_ids.
    sid = out / "sample_ids.npy"
    np.save(sid, np.array([f"{spec.key}:{i}" for i in range(parsed.n_samples)], dtype=np.str_),
            allow_pickle=False)

    fid = out / "feature_ids.npy"
    np.save(fid, np.arange(parsed.n_features, dtype=np.int64), allow_pickle=False)

    fn = out / "feature_names.json"
    write_json(fn, parsed.feature_names)

    return {
        **common,
        "X": {**x_record, "sha256": sha256_file(xp), "bytes": xp.stat().st_size},
        "y": {"path": rel_to_research(yp), "sha256": sha256_file(yp)},
        "source_split": {"path": rel_to_research(sp)},
        "source_row_index": {"path": rel_to_research(ri)},
        "sample_ids": {"path": rel_to_research(sid)},
        "feature_ids": {"path": rel_to_research(fid)},
        "feature_names": {"path": rel_to_research(fn)},
    }


def load_processed(spec: DatasetSpec) -> tuple[Any, np.ndarray, dict]:
    """Reload the canonical artifacts, proving they round-trip."""
    out = processed_dir(spec.key)
    schema = out / "schema.json"
    if not schema.exists():
        raise FileNotFoundError(f"{spec.key}: processed artifacts missing at {rel_to_research(out)}")
    import json

    meta = json.loads(schema.read_text(encoding="utf-8"))
    xrec = meta["X"]
    if xrec["format"] == "scipy_csr":
        X = sparse.load_npz(str(resolve(xrec["path"])))
    else:
        X = np.load(resolve(xrec["path"]), allow_pickle=False)
    y = np.load(out / "y.npy", allow_pickle=False).astype(str)
    return X, y, meta


def write_schema(spec: DatasetSpec, record: dict) -> Path:
    return write_json(processed_dir(spec.key) / "schema.json", record)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 1: build canonical datasets and plans")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dataset", action="append", default=[])
    args = ap.parse_args(argv)

    logfile = configure_logging("stage1_prepare.log")
    LOG.info("stage1_prepare starting; log also at %s", rel_to_research(logfile))

    cfg = load_stage1_config()
    registry = load_registry()
    specs = registry.datasets if (args.all or not args.dataset) else registry.select(args.dataset)
    seeds = list(cfg["evaluation_plan"]["seeds"])
    n_splits = int(cfg["evaluation_plan"]["n_splits"])

    summary: list[dict] = []
    for spec in specs:
        ensure_data_dirs(spec.key)
        source = spec.primary_source
        try:
            raw_ok = (raw_dir(spec.key) / source.filename_in_archive).exists() or (
                raw_dir(spec.key) / "extracted"
            ).exists()
        except Exception:
            raw_ok = False

        if not raw_ok:
            status, reason = "DOWNLOAD_FAILED", "no raw bytes present; run stage1_fetch first"
            LOG.error("[%s] %s", spec.key, status)
            summary.append({"dataset": spec.key, "status": status, "reason": reason})
            continue

        try:
            parsed = parse_dataset(spec, cfg)
        except ParseError as exc:
            status, reason = "PARSE_FAILED", str(exc)
            LOG.error("[%s] PARSE_FAILED %s", spec.key, exc)
            summary.append({"dataset": spec.key, "status": status, "reason": reason})
            continue
        except Exception as exc:
            status, reason = "PARSE_FAILED", f"{type(exc).__name__}: {exc}"
            LOG.exception("[%s] PARSE_FAILED: %s", spec.key, reason)
            summary.append({"dataset": spec.key, "status": status, "reason": reason})
            continue

        LOG.info("[%s] parsed %d x %d, %d classes", spec.key, parsed.n_samples, parsed.n_features,
                 len(set(parsed.y.tolist())))

        record = save_processed(spec, parsed, cfg)
        write_schema(spec, record)

        audit = audit_dataset(parsed, spec)
        status, reason = assign_status(
            spec,
            parsed,
            audit,
            download_ok=True,
            parse_ok=True,
            provenance_tier=source.tier,
        )

        # ---- preregistered plans (index/ordering only; no model fitting) --
        plans = build_split_plan(spec, parsed.y, seeds, n_splits)
        orders = build_feature_orders(parsed.n_features, seeds)
        check = validate_plan(plans, orders, parsed.n_samples)
        if not check["ok"]:
            status = "DATA_QUALITY_BLOCKED"
            reason = f"generated plan failed its own structural check: {check['problems'][:3]}"

        if status in {"EXACT_VERIFIED", "SHAPE_MATCH_PROVENANCE_UNVERIFIED"}:
            sdir, odir = splits_dir(spec.key), stream_orders_dir(spec.key)
            sdir.mkdir(parents=True, exist_ok=True)
            odir.mkdir(parents=True, exist_ok=True)
            for plan in plans:
                write_json(sdir / f"seed_{plan.seed}.json", plan.to_dict())
            for seed, order in orders.items():
                write_json(odir / f"seed_{seed}.json", feature_order_record(spec.key, parsed.n_features, seed, order))
            LOG.info("[%s] wrote %d split plans and %d feature orders", spec.key, len(plans), len(orders))

        # ---- reload and prove the round trip -----------------------------
        X2, y2, meta2 = load_processed(spec)
        rt = _roundtrip_check(parsed, X2, y2, meta2)
        if not rt["ok"] and status in {"EXACT_VERIFIED", "SHAPE_MATCH_PROVENANCE_UNVERIFIED"}:
            status = "DATA_QUALITY_BLOCKED"
            reason = f"round-trip check failed: {rt['problems']}"

        summary.append(
            {
                "dataset": spec.key,
                "status": status,
                "reason": reason,
                "n_samples": parsed.n_samples,
                "n_features": parsed.n_features,
                "n_classes": len(set(parsed.y.tolist())),
                "split_plans": len(plans),
                "feature_orders": len(orders),
                "roundtrip_ok": rt["ok"],
            }
        )
        LOG.info("[%s] %s — %s", spec.key, status, reason)

    from ossfs_dd_data.workspace import utc_now

    out = write_json(
        _results() / "prepare_summary.json",
        {"generated_utc": utc_now(), "datasets": summary},
    )
    LOG.info("prepare summary: %s", out)
    return 0


def _roundtrip_check(parsed: ParsedDataset, X2: Any, y2: np.ndarray, meta: dict) -> dict:
    problems: list[str] = []
    if X2.shape != parsed.X.shape:
        problems.append(f"X shape {X2.shape} != {parsed.X.shape}")
    if not np.array_equal(np.asarray(parsed.y).astype(str), y2):
        problems.append("labels differ after reload")
    if X2.shape[1] != meta["n_features"]:
        problems.append("reloaded X width disagrees with the schema")
    if meta.get("n_features") is not None and meta["n_features"] != int(parsed.X.shape[1]):
        problems.append("schema feature count disagrees with X")
    if int(meta.get("n_samples", -1)) != int(parsed.n_samples):
        problems.append("schema sample count disagrees with X")
    if sparse.issparse(parsed.X) and sparse.issparse(X2):
        diff = (parsed.X != X2).nnz
        if diff:
            problems.append(f"{diff} differing stored values after CSR round trip")
    elif not sparse.issparse(parsed.X) and not sparse.issparse(X2):
        if not np.array_equal(np.asarray(parsed.X), np.asarray(X2)):
            problems.append("dense X differs after reload")
    return {"ok": not problems, "problems": problems}


def _results() -> Path:
    from ossfs_dd_data.workspace import results_dir

    return results_dir()


if __name__ == "__main__":
    raise SystemExit(main())
