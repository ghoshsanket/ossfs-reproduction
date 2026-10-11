#!/usr/bin/env python
"""Stage 3B follow-up Experiment A: fixed-standardization SVM C/tol study.

270 fits = 3 datasets x 3 split seeds x 5 folds x 3 C values x 2 tolerances.

Standardization is FROZEN on for every cell, fitted on the selected TRAINING
rows only. Subsets come from the raw Stage 3A records and are never reselected.
The control cell (C=1, tol=1e-3) is compared against the saved Stage 3B
standardized highest-cap result for the same subset.

Usage:
    uv run python scripts/stage3b_followup_run_a.py --run-id <RUN_ID> [--workers N]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

from ossfs_dd_stage3b_followup.experiment_a import (  # noqa: E402
    fit_ids,
    run_one_fit,
)

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "configs" / "stage3b_followup_v1.yaml"
OUT_ROOT = (Path(os.environ["RESEARCH_OUTPUT_ROOT"]) / "ossfs-dd-reproduction"
            / "stage3b_followup")

_CACHE: dict = {}


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


def _dataset(key: str):
    from ossfs_dd.runner import load_dataset

    if key not in _CACHE:
        X, y, _ = load_dataset(key)
        if hasattr(X, "tocsr"):
            X = X.tocsr()
        _CACHE[key] = (X, y)
    return _CACHE[key]


def worker(task_q, result_q) -> None:
    while True:
        task = task_q.get()
        if task is None:
            return
        fit_id = task
        try:
            parts = fit_id.split("__")
            key = parts[0]
            dataset, seed_part, fold_part = key.split("_")
            seed = int(seed_part.removeprefix("seed"))
            fold = int(fold_part.removeprefix("fold"))
            c_val = float(parts[1].removeprefix("C"))
            tol = float(parts[2].removeprefix("tol"))
            X, y = _dataset(dataset)
            from ossfs_dd_stage3b.protocol import SubsetID
            from ossfs_dd_stage3b.provenance import load_split, load_stage3a_record

            rec = load_stage3a_record(SubsetID(dataset, seed, fold))
            train_idx, test_idx = load_split(dataset, seed, fold)
            out = run_one_fit(
                fit_id=fit_id, dataset=dataset, seed=seed, fold=fold,
                subset_record=rec, X=X, y=y, train_idx=train_idx, test_idx=test_idx,
                C=c_val, tol=tol,
            )
        except Exception as exc:  # noqa: BLE001 - recorded honestly as a terminal status
            out = {"fit_id": fit_id, "status": "FIT_ERROR", "accuracy": None,
                   "converged": None, "error": f"{type(exc).__name__}: {exc}"}
        result_q.put(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run_id = args.run_id
    workers = max(1, min(args.workers, 2))

    from ossfs_dd_stage3b.provenance import sha256_file

    cfg_sha = sha256_file(CONFIG)
    preflight = json.loads(
        (PROJECT / "results" / "stage3b_followup" / run_id / "preflight.json").read_text()
    )
    if preflight["config_sha256"] != cfg_sha:
        raise SystemExit("config changed since freeze; start a NEW run/version")

    out_dir = OUT_ROOT / run_id
    runs_dir = out_dir / "svm" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    journal = out_dir / "logs" / "event_journal.jsonl"
    append_journal(journal, "FIRST_DATA_EVALUATION_STARTED", run_id, cfg_sha,
                   {"experiment": "A"})

    planned = fit_ids()
    if args.limit:
        planned = planned[: args.limit]
    todo = [f for f in planned if not (runs_dir / f"{f}.json").exists()]
    print(f"  Experiment A: {len(planned)} planned, {len(todo)} to run "
          f"({workers} worker(s), 600s timeout/fit)")

    ctx = mp.get_context("spawn")
    task_q, result_q = ctx.Queue(), ctx.Queue()
    procs = []
    for _ in range(workers):
        p = ctx.Process(target=worker, args=(task_q, result_q), daemon=False)
        p.start()
        procs.append(p)
    append_journal(journal, "EXPERIMENT_A_STARTED", run_id, cfg_sha,
                   {"n_planned": len(planned), "n_todo": len(todo), "workers": workers})

    t0 = time.perf_counter()
    done = timeouts = errors = 0
    nxt = 0
    inflight: dict = {}
    active = list(procs)
    try:
        while nxt < len(todo) or inflight:
            for proc in list(active):
                if nxt < len(todo) and not any(ip is proc for _, ip in inflight.values()):
                    fid = todo[nxt]
                    task_q.put(fid)
                    inflight[fid] = (time.perf_counter(), proc)
                    nxt += 1
            if not inflight:
                break
            try:
                out = result_q.get(timeout=1.0)
            except Exception:
                fid, (start, proc) = min(inflight.items(), key=lambda kv: kv[1][0])
                if time.perf_counter() - start > 600:
                    proc.terminate()
                    try:
                        proc.join(timeout=10)
                    except Exception:  # noqa: BLE001
                        proc.kill()
                    timeouts += 1
                    rec = {"fit_id": fid, "status": "TIMEOUT", "accuracy": None,
                           "converged": None, "n_iter_vector": None,
                           "error": "exceeded the 600s hard wall-clock limit"}
                    (runs_dir / f"{fid}.json").write_text(
                        json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                    inflight.pop(fid, None)
                    active = [p for p in active if p is not proc]
                    np_ = ctx.Process(target=worker, args=(task_q, result_q), daemon=False)
                    np_.start()
                    active.append(np_)
                continue
            out.setdefault("config_sha256", cfg_sha)
            (runs_dir / f"{out['fit_id']}.json").write_text(
                json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            inflight.pop(out["fit_id"], None)
            done += 1
            errors += int(out.get("status") == "FIT_ERROR")
            if done % 30 == 0 or done == len(todo):
                el = (time.perf_counter() - t0) / 60
                rate = done / max(el, 1e-9)
                print(f"    [{done}/{len(todo)}] {el:.1f}min "
                      f"eta={(len(todo)-done)/rate if rate else 0:.1f}min "
                      f"timeouts={timeouts} errors={errors}")
    finally:
        for _ in active:
            task_q.put(None)
        for p in active:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()

    el = (time.perf_counter() - t0) / 60
    append_journal(journal, "EXPERIMENT_A_FINISHED", run_id, cfg_sha,
                   {"n_done": done, "n_timeouts": timeouts, "n_errors": errors,
                    "elapsed_minutes": round(el, 2)})
    print(f"  Experiment A finished: {done} fits in {el:.1f} min "
          f"(timeouts={timeouts}, errors={errors})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
