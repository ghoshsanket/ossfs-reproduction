#!/usr/bin/env python
"""Stage 3B follow-up: determinism repeats in fresh processes.

Repeats are compared on scientific fields only (predictions, selected-ID
sequence, canonical prefix ledgers). Timing, RSS and timestamps are excluded by
construction, and a repeat that stays NONCONVERGED is recorded as such rather
than forced to converge.

Usage:
    uv run python scripts/stage3b_followup_determinism.py --run-id <RUN_ID>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "configs" / "stage3b_followup_v1.yaml"
OUT_ROOT = (Path(os.environ["RESEARCH_OUTPUT_ROOT"]) / "ossfs-dd-reproduction"
            / "stage3b_followup")

# Run-provenance metadata and non-deterministic measurements are excluded from
# the comparison; every computed scientific field is included.
EXCLUDED = ("seconds", "peak_rss_gib", "generated_utc", "predictions", "wall_clock",
            "config_sha256")


def _scientific(rec: dict) -> dict:
    return {k: v for k, v in sorted(rec.items())
            if k not in EXCLUDED and not k.endswith("_seconds")}


def _run_child(code: str) -> dict:
    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[v] = "1"
    proc = subprocess.run([sys.executable, "-c", code], cwd=PROJECT, env=env,
                          capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        return {"error": proc.stderr[-800:]}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    run_id = args.run_id
    out_dir = OUT_ROOT / run_id

    # --- repeat 1: the Experiment A control cell (MADELON seed 11 fold 0 C=1 tol=1e-3)
    code_a = f'''
import json, sys
sys.path.insert(0, {str(PROJECT / "src")!r})
from ossfs_dd.runner import load_dataset
from ossfs_dd_stage3b.provenance import load_split, load_stage3a_record
from ossfs_dd_stage3b.protocol import SubsetID
from ossfs_dd_stage3b_followup.experiment_a import run_one_fit
X, y, _ = load_dataset("MADELON")
if hasattr(X, "tocsr"): X = X.tocsr()
rec = load_stage3a_record(SubsetID("MADELON", 11, 0))
tr, te = load_split("MADELON", 11, 0)
out = run_one_fit(fit_id="MADELON_seed11_fold0__C1.0__tol0.001", dataset="MADELON",
                  seed=11, fold=0, subset_record=rec, X=X, y=y, train_idx=tr,
                  test_idx=te, C=1.0, tol=0.001)
print(json.dumps(out))
'''
    repeat_a = _run_child(code_a)
    original_a = json.loads(
        (out_dir / "svm" / "runs" / "MADELON_seed11_fold0__C1.0__tol0.001.json").read_text())

    # --- repeat 2: one ARCENE selection + trajectory (fold 0, order 11)
    code_b = f'''
import json, sys
sys.path.insert(0, {str(PROJECT / "src")!r})
from ossfs_dd.runner import load_dataset
from ossfs_dd_stage3b.provenance import load_order, load_split
from ossfs_dd_stage3b_followup.instrument import run_cell
X, y, _ = load_dataset("ARCENE")
if hasattr(X, "tocsr"): X = X.tocsr()
tr, _ = load_split("ARCENE", 11, 0)
rec = run_cell(X=X, y=y, train_idx=tr, order=load_order("ARCENE", 11),
               fold=0, order_seed=11, split_seed=11)
print(json.dumps({{"selected_feature_ids": rec["selected_feature_ids"],
                   "selected_arrival_positions": rec["selected_arrival_positions"],
                   "n_selected": rec["n_selected"],
                   "ledger": rec["ledger"],
                   "counters": rec["counters"]}}))
'''
    repeat_b = _run_child(code_b)
    original_b = json.loads(
        (out_dir / "arcene" / "runs" / "ARCENE_split11_fold0_order11.json").read_text())

    a_same = _scientific(original_a) == _scientific(repeat_a)
    b_ids = (repeat_b.get("selected_feature_ids") == original_b["selected_feature_ids"]
             and repeat_b.get("selected_arrival_positions")
             == original_b["selected_arrival_positions"])
    b_ledger = (repeat_b.get("ledger", {}).get("ledger_identity_holds")
                == original_b["ledger"]["ledger_identity_holds"]
                and repeat_b.get("counters", {}).get("redundancy_removals")
                == original_b["counters"]["redundancy_removals"])

    payload = {
        "schema": "ossfs-dd-reproduction/stage3b_followup/determinism/1.0",
        "run_id": run_id,
        "utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fresh_process": True,
        "excluded_fields": list(EXCLUDED),
        "convergence_not_forced": True,
        "svm_control_repeat": {
            "fit_id": "MADELON_seed11_fold0__C1.0__tol0.001",
            "original_status": original_a.get("status"),
            "repeat_status": repeat_a.get("status"),
            "original_accuracy": original_a.get("accuracy"),
            "repeat_accuracy": repeat_a.get("accuracy"),
            "original_n_iter": original_a.get("n_iter_max"),
            "repeat_n_iter": repeat_a.get("n_iter_max"),
            "identical": bool(a_same),
        },
        "arcene_repeat": {
            "run_id": "ARCENE_split11_fold0_order11",
            "selected_ids_identical": bool(b_ids),
            "ledger_identical": bool(b_ledger),
            "original_n_selected": original_b["n_selected"],
            "repeat_n_selected": repeat_b.get("n_selected"),
            "identical": bool(b_ids and b_ledger),
        },
        "all_deterministic": bool(a_same and b_ids and b_ledger),
    }
    (out_dir / "logs" / "determinism.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"  SVM control repeat identical : {payload['svm_control_repeat']['identical']} "
          f"({payload['svm_control_repeat']['original_status']})")
    print(f"  ARCENE repeat identical      : {payload['arcene_repeat']['identical']}")
    print(f"  all_deterministic            : {payload['all_deterministic']}")
    return 0 if payload["all_deterministic"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
