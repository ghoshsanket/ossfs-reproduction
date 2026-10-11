#!/usr/bin/env python
"""Stage 3B follow-up Experiment B: ARCENE trajectory + classifier accuracy.

50 repaired selections (5 folds x 10 orders, split seed 11 fixed), 150 subset
classifier evaluations and 15 all-feature baselines. Each classifier fit runs
in a killable worker with a 10-minute limit.

Usage:
    uv run python scripts/stage3b_followup_run_b.py --run-id <RUN_ID>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

from ossfs_dd_stage3b_followup.experiment_b import (  # noqa: E402
    CART_SEED,
    evaluate_classifier,
    extract_columns,
    subset_feature_hash,
)
from ossfs_dd_stage3b_followup.instrument import run_cell  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "configs" / "stage3b_followup_v1.yaml"
OUT_ROOT = (Path(os.environ["RESEARCH_OUTPUT_ROOT"]) / "ossfs-dd-reproduction"
            / "stage3b_followup")
BASE_RUN = "20261011T153549Z"


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_journal(path: Path, event: str, run_id: str, cfg_sha: str, extra: dict) -> None:
    row = {"event": event, "utc": utcnow(), "run_id": run_id, "config_sha256": cfg_sha}
    row.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _guard(task: dict) -> None:
    """Worker body: fit one classifier and write its own record atomically."""
    out = Path(task["out_dir"]) / f"{task['record_id']}.json"
    if out.exists():
        return
    tmp = out.with_suffix(".partial")
    try:
        from ossfs_dd.runner import load_dataset
        from ossfs_dd_stage3b.provenance import load_split

        X, y, _ = load_dataset("ARCENE")
        if hasattr(X, "tocsr"):
            X = X.tocsr()
        train_idx, test_idx = load_split("ARCENE", 11, task["fold"])
        cols = np.asarray(task["cols"], dtype=np.int64)
        Xtr = extract_columns(X, train_idx, cols)
        Xte = extract_columns(X, test_idx, cols)
        rec = evaluate_classifier(task["classifier"], Xtr, y[train_idx], Xte, y[test_idx])
        rec.update({"record_id": task["record_id"], "fold": task["fold"],
                    "order_seed": task.get("order_seed"),
                    "n_features_used": int(cols.size),
                    "baseline": bool(task.get("baseline")), "cart_seed": CART_SEED,
                    "subset_ids_sha256": subset_feature_hash(task["cols"]),
                    "train_index_sha256": task["train_index_sha256"],
                    "test_index_sha256": task["test_index_sha256"]})
        if task.get("n_selected") is not None:
            rec["n_selected"] = task["n_selected"]
    except Exception as exc:  # noqa: BLE001 - an honest terminal status, never silent
        rec = {"record_id": task["record_id"], "classifier": task["classifier"],
               "status": "FIT_ERROR", "accuracy": None, "balanced_accuracy": None,
               "converged": None, "fold": task["fold"], "order_seed": task.get("order_seed"),
               "baseline": bool(task.get("baseline")), "n_features_used": len(task["cols"]),
               "error": f"{type(exc).__name__}: {exc}"}
    tmp.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out)


def run_pool(tasks: list[dict], label: str) -> tuple[int, int]:
    """Run classifier fits in killable child processes, at most 2 concurrent.

    Each child writes its own record, so a timeout only needs the child
    terminated; there is no result to lose across a process boundary.
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    n_timeout = 0
    n_error = 0
    pending = list(tasks)
    idx = 0
    active: list = []
    while idx < len(pending) or active:
        while idx < len(pending) and len(active) < 2:
            proc = ctx.Process(target=_guard, args=(pending[idx],), daemon=False)
            proc.start()
            active.append((proc, pending[idx], time.perf_counter()))
            idx += 1
        time.sleep(0.2)
        still = []
        for proc, task, started in active:
            if proc.is_alive():
                if time.perf_counter() - started > 600:
                    proc.terminate()
                    try:
                        proc.join(timeout=10)
                    except Exception:  # noqa: BLE001
                        proc.kill()
                    n_timeout += 1
                    out = Path(task["out_dir"]) / f"{task['record_id']}.json"
                    out.write_text(json.dumps(
                        {"record_id": task["record_id"], "classifier": task["classifier"],
                         "status": "TIMEOUT", "accuracy": None, "balanced_accuracy": None,
                         "converged": None, "fold": task["fold"],
                         "order_seed": task.get("order_seed"),
                         "baseline": bool(task.get("baseline")),
                         "n_features_used": len(task["cols"]),
                         "error": "exceeded the 600s hard wall-clock limit"},
                        indent=2, sort_keys=True) + "\n", encoding="utf-8")
                    continue
                still.append((proc, task, started))
            else:
                proc.join(timeout=5)
        active = still
    for task in pending:
        p = Path(task["out_dir"]) / f"{task['record_id']}.json"
        if p.exists() and json.loads(p.read_text()).get("status") == "FIT_ERROR":
            n_error += 1
    print(f"    [{label}] {len(pending)} tasks, timeouts={n_timeout}, errors={n_error}")
    return n_timeout, n_error


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    run_id = args.run_id

    from ossfs_dd_stage3b.provenance import load_order, load_split, sha256_file

    cfg_sha = sha256_file(CONFIG)
    preflight = json.loads(
        (PROJECT / "results" / "stage3b_followup" / run_id / "preflight.json").read_text()
    )
    if preflight["config_sha256"] != cfg_sha:
        raise SystemExit("config changed since freeze; start a NEW run/version")

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    exp_b = config["experiment_b"]
    folds = list(exp_b["folds"])
    order_seeds = list(exp_b["order_seeds"])
    split_seed = int(exp_b["split_seed"])

    out_dir = OUT_ROOT / run_id
    arc_dir = out_dir / "arcene" / "runs"
    clf_dir = out_dir / "classifiers"
    base_dir = out_dir / "baselines"
    for d in (arc_dir, clf_dir, base_dir, out_dir / "traces", out_dir / "logs"):
        d.mkdir(parents=True, exist_ok=True)
    journal = out_dir / "logs" / "event_journal.jsonl"
    append_journal(journal, "FIRST_DATA_EVALUATION_STARTED", run_id, cfg_sha,
                   {"experiment": "B"})

    from ossfs_dd.runner import load_dataset

    X, y, _ = load_dataset("ARCENE")
    if hasattr(X, "tocsr"):
        X = X.tocsr()
    if X.shape != (200, 10000):
        raise SystemExit(f"ARCENE shape {X.shape} != (200, 10000)")

    baseline_src = OUT_ROOT.parent / "stage3b" / BASE_RUN / "arcene" / "runs"
    import gzip

    clf_tasks: list[dict] = []
    invariance: list[dict] = []
    ledger_ok = 0
    trace_threshold = {"n_checked": 0, "n_mismatch": 0, "ok": True}
    t0 = time.perf_counter()

    for fold in folds:
        train_idx, _ = load_split("ARCENE", split_seed, fold)
        train_sha = subset_feature_hash([int(v) for v in train_idx])
        for oseed in order_seeds:
            rid = f"ARCENE_split{split_seed}_fold{fold}_order{oseed}"
            rec_path = arc_dir / f"{rid}.json"
            if rec_path.exists():
                rec = json.loads(rec_path.read_text())
            else:
                order = load_order("ARCENE", oseed)
                rec = run_cell(X=X, y=y, train_idx=train_idx, order=order,
                               fold=fold, order_seed=oseed, split_seed=split_seed)
                rec["config_sha256"] = cfg_sha
                rec["selected_feature_ids_sha256"] = subset_feature_hash(
                    rec["selected_feature_ids"])
                rec["train_index_sha256"] = train_sha
                rec_path.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n",
                                    encoding="utf-8")
                # write the scalar-only trace
                tpath = out_dir / "traces" / f"{rid}.jsonl.gz"
                if not tpath.exists():
                    with gzip.open(tpath, "wt", encoding="utf-8") as fh:
                        for row in rec.get("_events", []):
                            fh.write(json.dumps(row, sort_keys=True) + "\n")
            old = json.loads((baseline_src / f"{rid}.json").read_text())
            ids_same = list(rec["selected_feature_ids"]) == old["selected_feature_ids"]
            pos_same = list(rec["selected_arrival_positions"]) == old["selected_arrival_positions"]
            invariance.append({"run_id": rid, "fold": fold, "order_seed": oseed,
                               "selected_ids_identical": ids_same,
                               "arrival_positions_identical": pos_same,
                               "n_selected_new": rec["n_selected"],
                               "n_selected_stage3b": old["n_selected"]})
            ledger_ok += int(bool(rec["ledger"]["ledger_identity_holds"]
                                  and rec["ledger"]["state_matches_ledger"]))
            trace_threshold["n_checked"] += rec["trace_summary"]["n_checkpoints"]
            cols = rec["selected_feature_ids"]
            _tr, test_idx = load_split("ARCENE", split_seed, fold)
            test_sha = subset_feature_hash([int(v) for v in test_idx])
            for clf in ("knn", "cart", "svm"):
                clf_tasks.append({
                    "record_id": f"{rid}__{clf}", "classifier": clf, "fold": fold,
                    "order_seed": oseed, "cols": cols, "baseline": False,
                    "n_selected": rec["n_selected"],
                    "out_dir": str(out_dir / "classifiers"),
                    "train_index_sha256": train_sha, "test_index_sha256": test_sha,
                })
        print(f"    fold{fold}: {len(order_seeds)} orders done "
              f"({(time.perf_counter()-t0)/60:.1f} min)")

    all_cols = list(range(X.shape[1]))
    for fold in folds:
        train_idx, test_idx = load_split("ARCENE", split_seed, fold)
        for clf in ("knn", "cart", "svm"):
            clf_tasks.append({
                "record_id": f"ARCENE_all_features_fold{fold}__{clf}",
                "classifier": clf, "fold": fold, "order_seed": None,
                "cols": all_cols, "baseline": True, "n_selected": None,
                "out_dir": str(out_dir / "baselines"),
                "train_index_sha256": subset_feature_hash([int(v) for v in train_idx]),
                "test_index_sha256": subset_feature_hash([int(v) for v in test_idx]),
            })

    print(f"  Experiment B: {len(clf_tasks)} classifier tasks "
          f"(150 subset + 15 baseline)")
    n_timeout, n_error = run_pool(clf_tasks, "classifiers")
    n_clf = sum(1 for t in clf_tasks if not t["baseline"])
    n_base = sum(1 for t in clf_tasks if t["baseline"])

    append_journal(journal, "EXPERIMENT_B_FINISHED", run_id, cfg_sha,
                   {"n_invariant": sum(1 for x in invariance if x["selected_ids_identical"]),
                    "n_ledger_ok": ledger_ok, "n_subset_classifiers": n_clf,
                    "n_baselines": n_base})
    print(f"  Experiment B finished: invariance "
          f"{sum(1 for x in invariance if x['selected_ids_identical'])}/50, "
          f"ledger {ledger_ok}/50, classifiers {n_clf}, baselines {n_base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
