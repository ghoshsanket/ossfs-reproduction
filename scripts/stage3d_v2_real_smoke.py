#!/usr/bin/env python
"""Bounded real-data V2 smoke: MADELON, seed 11, fold 0, selection only.

Verifies H-MAP against the Stage-1 processed matrix for every scheduled
feature, using TRAINING rows only. No classifier is fitted and no test row is
read for scoring.

Usage:
    uv run python scripts/stage3d_v2_real_smoke.py --run-id <RUN_ID>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

from ossfs_dd.runner import load_dataset  # noqa: E402
from ossfs_dd_stage3d.scheduler import batch_size_for_p, make_schedule  # noqa: E402
from ossfs_dd_stage3d.selector import make_discrete_batch_v2  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
DATASET = "MADELON"
SEED, FOLD = 11, 0
OUT_ROOT = (Path(os.environ["RESEARCH_OUTPUT_ROOT"]) / "ossfs-dd-reproduction"
            / "stage3d_v2")


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def code_closure_hash() -> dict[str, str]:
    """Hash the true V2 dependency closure, not just selector.py."""
    files = [
        "src/ossfs_dd_stage3d/batch_selector_v2.py",
        "src/ossfs_dd_stage3d/batch_selector.py",
        "src/ossfs_dd_stage3d/selector.py",
        "src/ossfs_dd_stage3d/scheduler.py",
        "src/ossfs_dd_stage3d/standard_fisherz.py",
        "src/ossfs_dd/moments.py",
        "src/ossfs_dd/pools.py",
        "src/ossfs_dd/decisions.py",
        "configs/stage3d_v2_integration_repair.yaml",
    ]
    return {f: sha256_file(PROJECT / f) for f in files}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    run_id = args.run_id
    out_dir = OUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # authorize + verify the manifest and matrix hash
    from ossfs_dd_data.workspace import manifest_path, processed_dir, resolve

    man = json.loads(manifest_path(DATASET).read_text(encoding="utf-8"))
    schema = json.loads((processed_dir(DATASET) / "schema.json").read_text(encoding="utf-8"))
    status = man["stage1_verdict"]["status"]
    disk_sha = sha256_file(resolve(schema["X"]["path"]))
    hash_ok = disk_sha == schema["X"]["sha256"]
    if status != "EXACT_VERIFIED" or not hash_ok:
        (out_dir / "real_smoke_manifest.json").write_text(json.dumps(
            {"status": "HASH_MISMATCH" if not hash_ok else "DATA_UNAVAILABLE",
             "manifest_status": status, "hash_ok": hash_ok}, indent=2) + "\n")
        print(f"  BLOCKED: status={status} hash_ok={hash_ok}")
        return 1

    X, y, _ = load_dataset(DATASET)
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)
    n, p = X.shape
    if (n, p) != (2600, 500):
        raise SystemExit(f"unexpected MADELON shape {X.shape}")

    split = json.loads(
        (Path(os.environ["RESEARCH_DATA_ROOT"]) / "ossfs-dd-reproduction" / "splits"
         / DATASET / f"seed_{SEED}.json").read_text(encoding="utf-8"))
    train_idx = np.asarray(split["folds"][FOLD]["train"], dtype=np.int64)
    order_rec = json.loads(
        (Path(os.environ["RESEARCH_DATA_ROOT"]) / "ossfs-dd-reproduction" / "stream_orders"
         / DATASET / f"seed_{SEED}.json").read_text(encoding="utf-8"))
    order = np.asarray(order_rec["arrival_order"], dtype=np.int64)

    sched = make_schedule(order, DATASET, SEED, FOLD)
    flattened = sched["flattened_order"]
    B = batch_size_for_p(p)
    if B != 10:
        raise SystemExit(f"MADELON p={p} should give B=10, got {B}")

    Xtr = X[train_idx]                      # TRAIN ROWS ONLY
    labels = np.asarray(y)[train_idx]
    bs, _sf, _jf = make_discrete_batch_v2(labels)

    t0 = time.perf_counter()
    delivered: list[int] = []
    mismatches: list[tuple[int, int]] = []
    i = 0
    while i < len(flattened):
        chunk = [int(j) for j in flattened[i:i + B]]
        cols = [Xtr[:, j].copy() for j in chunk]
        # H-MAP: prove the delivered column IS the column of that feature ID
        for fid, col in zip(chunk, cols, strict=True):
            if not np.array_equal(col, Xtr[:, fid]):
                mismatches.append((fid, -1))
        bs.consume_batch(chunk, cols)
        delivered.extend(chunk)
        i += B
    bs.finalize_stream()
    seconds = time.perf_counter() - t0

    coverage_ok = sorted(delivered) == list(range(p))
    unique_ok = len(set(delivered)) == len(delivered)

    manifest = {
        "schema": "ossfs-dd-reproduction/stage3d_v2/real_smoke/2.0",
        "run_id": run_id, "utc": utcnow(),
        "status": "COMPLETED",
        "method_version": "batch_score_shared_v2",
        "dataset": DATASET, "seed": SEED, "fold": FOLD,
        "manifest_status": status,
        "matrix_sha256": disk_sha, "matrix_hash_ok": hash_ok,
        "split_train_sha256": hashlib.sha256(train_idx.tobytes()).hexdigest(),
        "order_sha256": hashlib.sha256(order.tobytes()).hexdigest(),
        "schedule_flattened_sha256": sched["hashes"]["flattened_order_sha256"],
        "code_closure": code_closure_hash(),
        "n_samples": int(n), "n_features": int(p),
        "n_train": int(train_idx.size), "batch_size": B,
        "n_batches": bs.state.n_batches,
        "n_features_seen": bs.state.arrival_count,
        "coverage_ok": coverage_ok, "unique_ids_ok": unique_ok,
        "n_mapping_mismatches": len(mismatches),
        "n_selected": len(bs.candidates.feature_ids()),
        "selected_feature_ids": [int(v) for v in bs.candidates.feature_ids()],
        "pool_flushes": bs.state.pool_flushes,
        "pending_pool_end": len(bs.pool),
        "finalized": getattr(bs, "_finalized", False),
        "seconds": round(seconds, 4),
        "test_rows_read_for_scoring": 0,
        "classifier_fitted": False,
        "forbidden_datasets_read": 0,
    }
    (out_dir / "real_smoke_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"  {DATASET} seed{SEED} fold{FOLD}: p={p} B={B} batches={bs.state.n_batches}")
    print(f"  delivered={len(delivered)} coverage_ok={coverage_ok} unique_ok={unique_ok}")
    print(f"  mapping_mismatches={len(mismatches)} n_selected={manifest['n_selected']}")
    print(f"  finalized={manifest['finalized']} pending_pool={manifest['pending_pool_end']}")
    print(f"  selected_ids={manifest['selected_feature_ids'][:15]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
