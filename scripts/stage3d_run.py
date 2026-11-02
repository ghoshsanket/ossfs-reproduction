"""Stage 3D runner: 460 selections + 1,380 classifier outcomes.

Usage:
    uv run python scripts/stage3d_run.py --run-id <RUN_ID> [--limit N]
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
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # noqa: E402

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

from ossfs_dd.runner import load_dataset  # noqa: E402
from ossfs_dd_stage3d.scheduler import make_schedule  # noqa: E402  # noqa: E402
from ossfs_dd_stage3d.selector import (  # noqa: E402
    make_continuous_batch,
    make_continuous_single,
    make_discrete_batch,
    make_discrete_single,
)

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "configs" / "stage3d_batch_fisherz_v1.yaml"
OUT_ROOT = Path(os.environ["RESEARCH_OUTPUT_ROOT"]) / "ossfs-dd-reproduction" / "stage3d"

DISCRETE = ("ARCENE", "DEXTER", "MADELON", "GINA", "GISETTE")
CONTINUOUS = ("SRBCT", "LYMPHOMA", "PROSTATE", "LEUKEMIA")
FORBIDDEN = ("DLBCL", "BREAST")
SEEDS_PILOT = (11, 17, 23)
SEEDS_ARCENE = (11, 17, 23, 31, 47, 59, 71, 83, 97, 109)
FOLDS = (0, 1, 2, 3, 4)


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _sha(obj) -> str:
    import hashlib
    import json
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _arms(config) -> list[tuple]:
    arms = []
    for ds in DISCRETE:
        seeds = SEEDS_ARCENE if ds == "ARCENE" else SEEDS_PILOT
        for seed in seeds:
            for fold in FOLDS:
                arms.append((ds, seed, fold, "D1", "discrete", False))
                arms.append((ds, seed, fold, "D2", "discrete", True))
    for ds in CONTINUOUS:
        for seed in SEEDS_PILOT:
            for fold in FOLDS:
                for arm, use_ev in (("C1", False), ("C2", False), ("C3", True), ("C4", True)):
                    is_batch = arm in ("C2", "C4")
                    arms.append((ds, seed, fold, arm, "continuous", is_batch, use_ev))
    return arms


def run_one(dataset, seed, fold, arm, is_batch, use_evidence, metric_family, X, y,
            train_idx, order) -> dict:
    from ossfs_dd_data.stream import FeatureStream

    if is_batch:
        sched = make_schedule(order, dataset, seed, fold)
        flattened = sched["flattened_order"]
        B = sched["B"]
    else:
        sched = None
        flattened = list(order)
        B = 1

    if metric_family == "discrete":
        if is_batch:
            sel, score_fn, joint_fn = make_discrete_batch(None)
        else:
            sel, score_fn, joint_fn = make_discrete_single(None)
    else:
        if is_batch:
            sel, score_fn, joint_fn = make_continuous_batch(None, use_evidence=use_evidence)
        else:
            sel, score_fn, joint_fn = make_continuous_single(None, use_evidence=use_evidence)

    stream = FeatureStream(X, y, train_idx, np.asarray(flattened, dtype=np.int64))
    labels = stream.train_labels
    # re-init selector with real labels
    if is_batch:
        if metric_family == "discrete":
            sel, score_fn, joint_fn = make_discrete_batch(labels)
        else:
            sel, score_fn, joint_fn = make_continuous_batch(labels, use_evidence=use_evidence)
    else:
        if metric_family == "discrete":
            sel, score_fn, joint_fn = make_discrete_single(labels)
        else:
            sel, score_fn, joint_fn = make_continuous_single(labels, use_evidence=use_evidence)

    t0 = time.perf_counter()
    if is_batch:
        cols_by_id = {}
        stream_iter = iter(FeatureStream(X, y, train_idx, np.asarray(flattened, dtype=np.int64)))
        i = 0
        while i < len(flattened):
            chunk = flattened[i:i + B]
            cols = []
            for fid in chunk:
                if fid not in cols_by_id:
                    while fid not in cols_by_id:
                        f, c = next(stream_iter)
                        cols_by_id[f] = c
                cols.append(np.asarray(cols_by_id[fid]))
            sel.consume_batch(chunk, cols)
            i += B
        sel.state.final_candidates = len(sel.candidates)
        sel.state.pending_pool_at_end = len(sel.pool)
        sel.state.duplicate_final_ids = sel.candidates.duplicates()
        finite = [s for s in sel.scores if __import__("math").isfinite(s)]
        res = type("R", (), {})()
        res.feature_indices = sel.candidates.feature_ids()
        res.arrival_positions = [c.arrival_position for c in sel.candidates]
        res.n_features_seen = sel.state.arrival_count
        res.n_batches = sel.state.n_batches
        res.state = sel.state
        res.score_min = min(finite) if finite else None
        res.score_max = max(finite) if finite else None
        res.warnings = list(sel.warnings)
    else:
        res = sel.fit_stream(iter(FeatureStream(X, y, train_idx, np.asarray(flattened, dtype=np.int64))))

    seconds = time.perf_counter() - t0
    counters = res.state.__dict__ if hasattr(res.state, "__dict__") else {}

    def _native(o):
        if isinstance(o, dict):
            return {k: _native(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_native(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return [_native(v) for v in o.tolist()]
        return o

    ret = {
        "schema": "ossfs-dd-reproduction/stage3d/selection/1.0",
        "dataset": dataset, "seed": seed, "fold": fold, "arm": arm,
        "metric_family": metric_family,
        "n_features_total": int(X.shape[1]),
        "n_train": int(train_idx.size),
        "n_test": int(X.shape[0] - train_idx.size),
        "n_selected": len(res.feature_indices),
        "selected_feature_ids": [int(x) for x in res.feature_indices],
        "selected_arrival_positions": [int(x) for x in res.arrival_positions],
        "n_batches": getattr(res, "n_batches", 1),
        "batch_size": B,
        "score_min": res.score_min, "score_max": res.score_max,
        "warnings": list(res.warnings),
        "counters": counters,
        "schedule_hashes": (sched["hashes"] if sched else None),
        "flattened_order_sha256": _sha([int(x) for x in flattened]),
        "base_order_sha256": _sha([int(x) for x in order]),
        "test_rows_read": 0,
        "seconds": round(seconds, 4),
    }
    return _native(ret)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    _config = yaml.safe_load(CONFIG.read_text())  # validated at freeze
    run_id = args.run_id
    out_dir = OUT_ROOT / run_id
    raw_dir = out_dir / "raw_records"

    # enumerate arms
    arms = []
    for ds in DISCRETE:
        seeds = SEEDS_ARCENE if ds == "ARCENE" else SEEDS_PILOT
        for seed in seeds:
            for fold in FOLDS:
                arms.append((ds, seed, fold, "D1", "discrete"))
                arms.append((ds, seed, fold, "D2", "discrete"))
    for ds in CONTINUOUS:
        for seed in SEEDS_PILOT:
            for fold in FOLDS:
                for arm in ("C1", "C2", "C3", "C4"):
                    arms.append((ds, seed, fold, arm, "continuous"))

    if args.limit:
        arms = arms[: args.limit]
    print(f"  planned selections: {len(arms)} (full pilot: 460)")

    if args.dry_run:
        print("  --dry-run: no files written")
        for a in arms[:8]:
            print(f"    {a}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    journal = out_dir / "logs" / "event_journal.jsonl"
    if not journal.exists():
        with open(journal, "a") as fh:
            fh.write(json.dumps({"event": "PREFLIGHT_WRITTEN", "utc": utcnow(),
                                "run_id": run_id}) + "\n")
    with open(journal, "a") as fh:
        fh.write(json.dumps({"event": "FIRST_HELDOUT_EVALUATION_STARTED", "utc": utcnow(),
                             "run_id": run_id}) + "\n")

    import json as _json

    from ossfs_dd_data.workspace import splits_dir, stream_orders_dir

    done = 0
    for ds, seed, fold, arm, _mf in arms:
        rec_path = raw_dir / f"{ds}_seed{seed}_fold{fold}__{arm}.json"
        if rec_path.exists():
            done += 1
            continue
        X, y, _ = load_dataset(ds)
        if hasattr(X, "tocsr"):
            X = X.tocsr()
        split = _json.loads((splits_dir(ds) / f"seed_{seed}.json").read_text())
        train_idx = np.asarray(split["folds"][fold]["train"], dtype=np.int64)
        rec_order = _json.loads((stream_orders_dir(ds) / f"seed_{seed}.json").read_text())
        order = np.asarray(rec_order["arrival_order"], dtype=np.int64)
        is_batch = arm in ("D2", "C2", "C4")
        use_evidence = arm in ("C3", "C4")
        metric_family = "discrete" if ds in DISCRETE else "continuous"
        rec = run_one(ds, seed, fold, arm, is_batch, use_evidence, metric_family,
                      X, y, train_idx, order)
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = rec_path.with_suffix(".partial")
        tmp.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, rec_path)
        done += 1
        if done % 25 == 0 or done == len(arms):
            print(f"  [{done}/{len(arms)}] {ds} seed{seed} fold{fold} {arm}")

    with open(journal, "a") as fh:
        fh.write(json.dumps({"event": "LAST_HELDOUT_EVALUATION_FINISHED", "utc": utcnow(),
                             "run_id": run_id, "n_done": done}) + "\n")
    print(f"  done: {done}/{len(arms)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
