#!/usr/bin/env python
"""Stage 3D classifier runner: 1,380 subset-based + 15 all-feature baseline fits.

Usage:
    uv run python scripts/stage3d_classify.py --run-id <RUN_ID>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

from ossfs_dd.runner import load_dataset  # noqa: E402
from ossfs_dd_stage3b_followup.experiment_b import (  # noqa: E402
    evaluate_classifier,
    extract_columns,
)

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "configs" / "stage3d_batch_fisherz_v1.yaml"
OUT_ROOT = Path(os.environ["RESEARCH_OUTPUT_ROOT"]) / "ossfs-dd-reproduction" / "stage3d"

DISCRETE = ("ARCENE", "DEXTER", "MADELON", "GINA", "GISETTE")
CONTINUOUS = ("SRBCT", "LYMPHOMA", "PROSTATE", "LEUKEMIA")


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def subset_hash(cols):
    import hashlib
    import json
    return hashlib.sha256(json.dumps([int(c) for c in cols], sort_keys=True).encode()).hexdigest()


def run_one(task: dict, out_path: Path) -> None:
    if out_path.exists():
        return
    import json as _j

    from ossfs_dd_data.workspace import splits_dir

    ds = task["dataset"]
    seed = task["seed"]
    fold = task["fold"]
    X, y, _ = load_dataset(ds)
    if hasattr(X, "tocsr"):
        X = X.tocsr()
    _split = _j.loads((splits_dir(ds) / f"seed_{seed}.json").read_text(encoding="utf-8"))
    train_idx = np.asarray(_split["folds"][fold]["train"], dtype=np.int64)
    test_idx = np.asarray(_split["folds"][fold]["test"], dtype=np.int64)
    cols = np.asarray(task["cols"], dtype=np.int64)
    Xtr = extract_columns(X, train_idx, cols)
    Xte = extract_columns(X, test_idx, cols)
    rec = evaluate_classifier(task["classifier"], Xtr, y[train_idx], Xte, y[test_idx])
    rec.update({"record_id": task["record_id"], "dataset": ds, "seed": seed,
                "fold": fold, "arm": task["arm"], "classifier": task["classifier"],
                "n_features_used": int(cols.size), "baseline": bool(task["baseline"]),
                "subset_ids_sha256": subset_hash(task["cols"]),
                "train_index_sha256": task["train_sha"],
                "test_index_sha256": task["test_sha"]})
    if task.get("n_selected") is not None:
        rec["n_selected"] = task["n_selected"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".partial")
    tmp.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    run_id = args.run_id
    out_dir = OUT_ROOT / run_id
    raw_dir = out_dir / "raw_records"
    clf_dir = out_dir / "classifiers"
    base_dir = out_dir / "baselines"
    for d in (clf_dir, base_dir):
        d.mkdir(parents=True, exist_ok=True)

    # build task list from selection records
    tasks = []
    for p in sorted(raw_dir.glob("*.json")):
        rec = json.loads(p.read_text())
        ds = rec["dataset"]
        seed = rec["seed"]
        fold = rec["fold"]
        arm = rec["arm"]
        cols = rec["selected_feature_ids"]
        import hashlib
        import json as _j

        from ossfs_dd_data.workspace import splits_dir

        def _h(arr):
            return hashlib.sha256(json.dumps([int(v) for v in arr], sort_keys=True).encode()).hexdigest()

        _split = _j.loads((splits_dir(ds) / f"seed_{seed}.json").read_text(encoding="utf-8"))
        train_idx = np.asarray(_split["folds"][fold]["train"], dtype=np.int64)
        test_idx = np.asarray(_split["folds"][fold]["test"], dtype=np.int64)
        train_sha = _h([int(v) for v in train_idx])
        test_sha = _h([int(v) for v in test_idx])
        for clf in ("knn", "cart", "svm"):
            tasks.append({"record_id": f"{rec['dataset']}_seed{seed}_fold{fold}__{arm}__{clf}",
                          "dataset": ds, "seed": seed, "fold": fold, "arm": arm,
                          "classifier": clf, "cols": cols, "baseline": False,
                          "n_selected": rec["n_selected"],
                          "train_sha": train_sha, "test_sha": test_sha})

    # all-feature baselines: ARCENE only, 5 folds x 3 classifiers
    for fold in range(5):
        X, y, _ = load_dataset("ARCENE")
        if hasattr(X, "tocsr"):
            X = X.tocsr()
        _split = _j.loads((splits_dir("ARCENE") / "seed_11.json").read_text(encoding="utf-8"))
        train_idx = np.asarray(_split["folds"][fold]["train"], dtype=np.int64)
        test_idx = np.asarray(_split["folds"][fold]["test"], dtype=np.int64)
        all_cols = list(range(X.shape[1]))
        for clf in ("knn", "cart", "svm"):
            tasks.append({"record_id": f"ARCENE_all_features_fold{fold}__{clf}",
                          "dataset": "ARCENE", "seed": 11, "fold": fold, "arm": "baseline",
                          "classifier": clf, "cols": all_cols, "baseline": True,
                          "n_selected": None,
                          "train_sha": _h([int(v) for v in train_idx]),
                          "test_sha": _h([int(v) for v in test_idx])})

    # run
    todo = []
    for t in tasks:
        out_path = (base_dir if t["baseline"] else clf_dir) / f"{t['record_id']}.json"
        if not out_path.exists():
            t["_out_path"] = out_path
            todo.append(t)
    print(f"  {len(tasks)} planned, {len(todo)} to run")
    for i, t in enumerate(todo):
        run_one(t, t["_out_path"])
        if (i + 1) % 50 == 0 or i + 1 == len(todo):
            print(f"  [{i+1}/{len(todo)}]")
    print(f"  done: classifiers={len(list(clf_dir.glob('*.json')))} "
          f"baselines={len(list(base_dir.glob('*.json')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
