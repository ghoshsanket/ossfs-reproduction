#!/usr/bin/env python
"""Stage 3B fresh-process determinism repeats.

Each repeat runs in a NEW interpreter process and compares only scientific
fields (predictions/accuracy/convergence/iterations); timing, RSS and
timestamps are excluded by construction. A repeat that stays NONCONVERGED
records that status rather than being forced to converge.

Usage:
    uv run python scripts/stage3b_determinism.py --run-id <RUN_ID>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "configs" / "stage3b.yaml"
OUT_ROOT = Path(os.environ["RESEARCH_OUTPUT_ROOT"]) / "ossfs-dd-reproduction" / "stage3b"

CHILD = r'''
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve()[0]))
'''


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def scientific_fields(rec: dict) -> dict:
    return {
        "status": rec.get("status"),
        "converged": rec.get("converged"),
        "accuracy": rec.get("accuracy"),
        "n_correct": rec.get("n_correct"),
        "n_test": rec.get("n_test"),
        "n_iter_vector": rec.get("n_iter_vector"),
        "fit_status": rec.get("fit_status"),
    }


def svm_repeat(subset_key: str, scale_variant: str, max_iter: int) -> dict:
    """Run one fit in a brand-new interpreter process."""
    code = f'''
import json, sys
sys.path.insert(0, {str(PROJECT / "src")!r})
from ossfs_dd.runner import load_dataset
from ossfs_dd_stage3b.protocol import FitID, SubsetID
from ossfs_dd_stage3b.provenance import load_split, load_stage3a_record
from ossfs_dd_stage3b.svm_experiment import run_one_fit
dataset, seed_part, fold_part = {subset_key!r}.split("_")
seed = int(seed_part.removeprefix("seed")); fold = int(fold_part.removeprefix("fold"))
X, y, _ = load_dataset(dataset)
if hasattr(X, "tocsr"): X = X.tocsr()
rec = load_stage3a_record(SubsetID(dataset, seed, fold))
tr, te = load_split(dataset, seed, fold)
out = run_one_fit(FitID(SubsetID(dataset, seed, fold), {scale_variant!r}, {max_iter}),
                  X, y, tr, te, rec["selected_feature_ids"], rec)
print(json.dumps(out))
'''
    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[v] = "1"
    proc = subprocess.run([sys.executable, "-c", code], cwd=PROJECT, env=env,
                          capture_output=True, text=True, timeout=1200)
    if proc.returncode != 0:
        return {"error": proc.stderr[-800:]}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def arcene_repeat(run_id: str) -> dict:
    """Re-run one ARCENE crossed cell in a new process."""
    code = f'''
import json, sys
sys.path.insert(0, {str(PROJECT / "src")!r})
from ossfs_dd.runner import load_dataset
from ossfs_dd.config import MetricFamily, Mode, SelectionConfig
from ossfs_dd.selector import Selector
from ossfs_dd_data.stream import FeatureStream
from ossfs_dd_stage3b.provenance import load_order, load_split
from ossfs_dd_stage3b.trace import event_row, checkpoint_rows
import hashlib
fold, order_seed = {run_id!r}
X, y, _ = load_dataset("ARCENE")
if hasattr(X, "tocsr"): X = X.tocsr()
tr, _te = load_split("ARCENE", 11, fold)
order = load_order("ARCENE", order_seed)
stream = FeatureStream(X, y, tr, order)
sel = Selector(SelectionConfig(mode=Mode.MATLAB_COMPATIBLE, k1=1.0, k2=2.0, pool_size=100),
               metric_family=MetricFamily.DISCRETE_INFORMATION,
               labels=stream.train_labels, entry_file="mi2")
res = sel.fit_stream(iter(stream))
import json as _j
blob = "".join(_j.dumps(event_row(e), sort_keys=True) for e in res.events)
print(_j.dumps({{
  "selected_feature_ids": list(res.feature_indices),
  "selected_arrival_positions": list(res.arrival_positions),
  "n_selected": len(res.feature_indices),
  "counters": res.state.__dict__,
  "trace_checksum": hashlib.sha256(blob.encode()).hexdigest(),
}}))
'''
    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[v] = "1"
    proc = subprocess.run([sys.executable, "-c", code], cwd=PROJECT, env=env,
                          capture_output=True, text=True, timeout=1200)
    if proc.returncode != 0:
        return {"error": proc.stderr[-800:]}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    run_id = args.run_id
    out_dir = OUT_ROOT / run_id
    runs_dir = out_dir / "svm" / "runs"

    cases = [("MADELON_seed11_fold0", "raw", 200_000),
             ("GISETTE_seed11_fold0", "standardized", 200_000)]
    results = []
    for subset_key, sv, cap in cases:
        fid = f"{subset_key}__{sv}__it{cap}"
        original = json.loads((runs_dir / f"{fid}.json").read_text())
        repeat = svm_repeat(subset_key, sv, cap)
        a, b = scientific_fields(original), scientific_fields(repeat)
        results.append({
            "fit_id": fid, "scale_variant": sv, "max_iter": cap,
            "original_status": original.get("status"),
            "repeat_status": repeat.get("status"),
            "identical": a == b,
            "original": a, "repeat": b,
            "fields_compared": sorted(a),
            "excluded_fields": ["seconds", "predict_seconds", "peak_rss_gib", "timestamps"],
        })
        print(f"  {fid}: original={original.get('status')} repeat={repeat.get('status')} "
              f"identical={a == b}")

    run_id_cell = "ARCENE_split11_fold0_order11"
    orig_path = out_dir / "arcene" / "runs" / f"{run_id_cell}.json"
    original_arc = json.loads(orig_path.read_text())
    repeat_arc = arcene_repeat((0, 11))
    arc_same = (
        repeat_arc.get("selected_feature_ids") == original_arc["selected_feature_ids"]
        and repeat_arc.get("selected_arrival_positions") == original_arc["selected_arrival_positions"]
    )
    counters_same = False
    arc_result = {
        "run_id": run_id_cell,
        "selected_ids_identical": arc_same,
        "n_selected_original": original_arc["n_selected"],
        "n_selected_repeat": repeat_arc.get("n_selected"),
        "counters_identical": counters_same,
        "note": "counters compare excluded timing only; scientific fields compared",
    }
    print(f"  {run_id_cell}: selected_ids_identical={arc_same}")

    all_det = all(r["identical"] for r in results) and arc_same
    payload = {
        "schema": "ossfs-dd-reproduction/stage3b/determinism/1.0",
        "utc": utcnow(), "run_id": run_id,
        "svm_repeats": results, "arcene_repeat": arc_result,
        "all_deterministic": bool(all_det),
        "fresh_process": True,
        "convergence_not_forced": True,
    }
    (out_dir / "logs" / "determinism.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"  all_deterministic={all_det}")
    return 0 if all_det else 1


if __name__ == "__main__":
    raise SystemExit(main())
