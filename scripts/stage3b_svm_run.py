#!/usr/bin/env python
"""Stage 3B Experiment A: SVM conditioning on the already-selected subsets.

Each fit runs in a dedicated worker process with a hard wall-clock timeout; on
timeout the worker is killed and replaced, so no orphan process survives and no
fit can silently run forever.

Usage:
    uv run python scripts/stage3b_svm_run.py --run-id <RUN_ID> [--workers N] [--limit N]
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

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

from ossfs_dd_stage3b.protocol import (  # noqa: E402
    all_fit_ids,
)
from ossfs_dd_stage3b.provenance import (  # noqa: E402
    BlockedInput,
    load_split,
    load_stage3a_record,
    sha256_file,
)

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "configs" / "stage3b.yaml"
OUT_ROOT = Path(os.environ["RESEARCH_OUTPUT_ROOT"]) / "ossfs-dd-reproduction" / "stage3b"

DATA_CACHE: dict = {}


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_journal(journal: Path, event: str, run_id: str, cfg_sha: str, extra: dict) -> None:
    row = {"event": event, "utc": utcnow(), "run_id": run_id, "config_sha256": cfg_sha}
    row.update(extra)
    journal.parent.mkdir(parents=True, exist_ok=True)
    with open(journal, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _dataset(key: str):
    from ossfs_dd.runner import load_dataset

    if key not in DATA_CACHE:
        X, y, _ = load_dataset(key)
        if hasattr(X, "tocsr"):
            X = X.tocsr()
        DATA_CACHE[key] = (X, y)
    return DATA_CACHE[key]


def worker(task_q, result_q) -> None:
    """One fit per task, then keep looping until asked to stop."""
    from ossfs_dd_stage3b.protocol import FitID, SubsetID
    from ossfs_dd_stage3b.svm_experiment import run_one_fit

    while True:
        task = task_q.get()
        if task is None:
            return
        fit_id_str, subset_key, scale_variant, max_iter = task
        try:
            dataset, seed_part, fold_part = subset_key.split("_")
            seed = int(seed_part.removeprefix("seed"))
            fold = int(fold_part.removeprefix("fold"))
            X, y = _dataset(dataset)
            rec = load_stage3a_record(SubsetID(dataset, seed, fold))
            train_idx, test_idx = load_split(dataset, seed, fold)
            fit_id = FitID(SubsetID(dataset, seed, fold), scale_variant, max_iter)
            out = run_one_fit(fit_id, X, y, train_idx, test_idx,
                              rec["selected_feature_ids"], rec)
        except BlockedInput as exc:
            out = {"fit_id": fit_id_str, "status": "BLOCKED_INPUT", "accuracy": None,
                   "error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            out = {"fit_id": fit_id_str, "status": "FIT_ERROR", "accuracy": None,
                   "error": f"{type(exc).__name__}: {exc}"}
        result_q.put(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run_id = args.run_id
    workers = max(1, min(args.workers, 2))          # hard cap of 2 per the budget
    timeout_s = 10 * 60

    _config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))  # validated at freeze
    cfg_sha = sha256_file(CONFIG)
    preflight = json.loads(
        (PROJECT / "results" / "stage3b" / run_id / "stage3b_preflight.json").read_text()
    )
    if preflight["config_sha256"] != cfg_sha:
        raise SystemExit("config digest changed since the freeze; start a new run id")

    out_dir = OUT_ROOT / run_id
    runs_dir = out_dir / "svm" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    journal = out_dir / "logs" / "event_journal.jsonl"
    append_journal(journal, "FIRST_DATA_EVALUATION_STARTED", run_id, cfg_sha,
                   {"experiment": "A"})

    fits = all_fit_ids()
    if args.limit:
        fits = fits[: args.limit]
    todo = []
    for f in fits:
        path = runs_dir / f"{f.fit_id}.json"
        if not path.exists():
            todo.append(f)
    print(f"  Experiment A: {len(fits)} planned fits, {len(todo)} to run "
          f"({workers} worker(s), {timeout_s}s timeout each)")

    ctx = mp.get_context("spawn")
    task_q = ctx.Queue()
    result_q = ctx.Queue()
    procs = []
    for _ in range(workers):
        p = ctx.Process(target=worker, args=(task_q, result_q), daemon=False)
        p.start()
        procs.append(p)
    append_journal(journal, "EXPERIMENT_A_STARTED", run_id, cfg_sha,
                   {"n_planned": len(fits), "n_todo": len(todo), "workers": workers})

    t_start = time.perf_counter()
    done = 0
    timeouts = 0
    errors = 0
    next_idx = 0
    inflight: dict = {}
    active = list(procs)
    try:
        while next_idx < len(todo) or inflight:
            # fill free workers
            for p in list(active):
                if next_idx < len(todo) and not any(ip is p for _, ip, _ in inflight.values()):
                    f = todo[next_idx]
                    task_q.put((f.fit_id, f.subset.key, f.scale_variant, f.max_iter))
                    inflight[f.fit_id] = (time.perf_counter(), p, f)
                    next_idx += 1
            if not inflight:
                break
            try:
                out = result_q.get(timeout=1.0)
            except Exception:
                # check for a timeout on the oldest in-flight task
                oldest = min(inflight.items(), key=lambda kv: kv[1][0])
                fid, (t0, proc, fspec) = oldest
                if time.perf_counter() - t0 > timeout_s:
                    proc.terminate()
                    try:
                        proc.join(timeout=10)
                    except Exception:  # noqa: BLE001
                        proc.kill()
                    timeouts += 1
                    rec = {"fit_id": fid, "status": "TIMEOUT", "accuracy": None,
                           "error": f"exceeded the {timeout_s}s hard wall-clock limit",
                           "dataset": fspec.subset.dataset,
                           "split_seed": fspec.subset.seed, "fold": fspec.subset.fold,
                           "scale_variant": fspec.scale_variant,
                           "max_iter": fspec.max_iter,
                           "timeout_seconds": timeout_s,
                           "n_selected": None, "n_test": None, "n_correct": None,
                           "converged": None}
                    (runs_dir / f"{fid}.json").write_text(
                        json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                    inflight.pop(fid, None)
                    active = [p for p in active if p is not proc]
                    if next_idx < len(todo) or inflight:
                        np_ = ctx.Process(target=worker, args=(task_q, result_q), daemon=False)
                        np_.start()
                        active.append(np_)
                continue
            out.setdefault("config_sha256", cfg_sha)
            out.setdefault("protocol_digest", preflight["protocol_digest"])
            (runs_dir / f"{out['fit_id']}.json").write_text(
                json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            inflight.pop(out["fit_id"], None)
            done += 1
            if out.get("status") == "FIT_ERROR":
                errors += 1
            if done % 25 == 0 or done == len(todo):
                el = (time.perf_counter() - t_start) / 60
                rate = done / max(el, 1e-9)
                eta = (len(todo) - done) / rate if rate > 0 else float("nan")
                print(f"    [{done}/{len(todo)}] elapsed={el:.1f}min eta={eta:.1f}min "
                      f"timeouts={timeouts} errors={errors}")
    finally:
        for _ in active:
            task_q.put(None)
        for p in active:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()

    elapsed = (time.perf_counter() - t_start) / 60
    append_journal(journal, "LAST_EVALUATION_FINISHED", run_id, cfg_sha,
                   {"experiment": "A", "n_done": done, "n_timeouts": timeouts,
                    "n_errors": errors, "elapsed_minutes": round(elapsed, 2)})
    print(f"  Experiment A finished: {done} fits in {elapsed:.1f} min "
          f"(timeouts={timeouts}, errors={errors})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
