#!/usr/bin/env python
"""Stage 3B Experiment B: ARCENE crossed (fold x order) selection with traces.

The CV partition is ALWAYS Stage 1 ``seed_11.json`` fold ``f``; only the
arrival permutation varies across the ten order seeds. No classifier is fitted
and no test row is read.

Usage:
    uv run python scripts/stage3b_arcene_run.py --run-id <RUN_ID> [--workers N]
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ossfs_dd.config import MetricFamily, Mode, SelectionConfig  # noqa: E402
from ossfs_dd.runner import load_dataset  # noqa: E402
from ossfs_dd.selector import Selector  # noqa: E402
from ossfs_dd_data.stream import FeatureStream  # noqa: E402
from ossfs_dd_stage3b.arcene_experiment import (  # noqa: E402
    verify_event_positions,
    verify_selected_ids,
)
from ossfs_dd_stage3b.protocol import (  # noqa: E402
    B_SPLIT_SEED,
    ArceneRunID,
    SubsetID,  # noqa: E402
    all_arcene_run_ids,
)
from ossfs_dd_stage3b.provenance import (  # noqa: E402
    BlockedInput,
    load_order,
    load_split,
    load_stage3a_record,
    sha256_file,
)
from ossfs_dd_stage3b.trace import checkpoint_rows, event_row  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "configs" / "stage3b.yaml"
OUT_ROOT = Path(os.environ["RESEARCH_OUTPUT_ROOT"]) / "ossfs-dd-reproduction" / "stage3b"


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


def write_json_once(path: Path, payload: dict) -> bool:
    """Write a record unless an identical one already exists (resumable)."""
    blob = json.dumps(payload, indent=2, sort_keys=True)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = None
        if existing is not None:
            keep = {k: v for k, v in existing.items()
                    if k not in ("seconds", "peak_rss_gib", "generated_utc")}
            new = {k: v for k, v in payload.items()
                   if k not in ("seconds", "peak_rss_gib", "generated_utc")}
            if keep == new:
                return False
            raise SystemExit(f"STALE_CACHE: {path.name} exists with different content")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(blob + "\n", encoding="utf-8")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    run_id = args.run_id

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg_sha = sha256_file(CONFIG)
    preflight = json.loads(
        (PROJECT / "results" / "stage3b" / run_id / "stage3b_preflight.json").read_text()
    )
    if preflight["config_sha256"] != cfg_sha:
        raise SystemExit("config digest changed since the freeze; start a new run id")

    out_dir = OUT_ROOT / run_id
    journal = out_dir / "logs" / "event_journal.jsonl"
    append_journal(journal, "FIRST_DATA_EVALUATION_STARTED", run_id, cfg_sha,
                   {"experiment": "B"})

    exp_b = config["experiment_b"]
    sel_cfg = config["selector"]
    k1, k2 = float(sel_cfg["k1"]), float(sel_cfg["k2"])
    pool_size = int(sel_cfg["pool_size"])

    X, y, _schema = load_dataset("ARCENE")
    if X.shape != (200, 10000):
        raise SystemExit(f"ARCENE shape {X.shape} != expected (200, 10000)")
    if len(set(y.tolist())) != 2:
        raise SystemExit("ARCENE is not 2-class as expected")

    order_cache = {o: load_order("ARCENE", o) for o in exp_b["order_seeds"]}
    runs = all_arcene_run_ids()
    print(f"  Experiment B: {len(runs)} crossed runs (split seed {B_SPLIT_SEED} fixed)")

    n_written = 0
    for i, rid in enumerate(runs, start=1):
        rec_path = out_dir / "arcene" / "runs" / f"{rid.run_id}.json"
        if rec_path.exists():
            print(f"  [{i}/{len(runs)}] {rid.run_id} already present, skipping")
            continue
        train_idx, _test_idx = load_split("ARCENE", B_SPLIT_SEED, rid.fold)
        order = order_cache[rid.order_seed]

        stream = FeatureStream(X, y, train_idx, order)
        selector = Selector(
            SelectionConfig(mode=Mode.MATLAB_COMPATIBLE, k1=k1, k2=k2, pool_size=pool_size),
            metric_family=MetricFamily.DISCRETE_INFORMATION,
            labels=stream.train_labels,
            entry_file="mi2",
        )
        t0 = time.perf_counter()
        result = selector.fit_stream(iter(stream))
        seconds = time.perf_counter() - t0

        counters = result.state.__dict__
        record = {
            "schema": "ossfs-dd-reproduction/stage3b/arcene_run/1.0",
            "run_id": rid.run_id,
            "config_sha256": cfg_sha,
            "protocol_digest": preflight["protocol_digest"],
            "dataset": "ARCENE",
            "split_seed": B_SPLIT_SEED,
            "fold": rid.fold,
            "order_seed": rid.order_seed,
            "is_anchor": rid.order_seed == exp_b["anchor_order_seed"],
            "mode": sel_cfg["mode"],
            "entry_file": sel_cfg["entry_file"],
            "k1": k1, "k2": k2, "pool_size": pool_size,
            "n_train": int(train_idx.size),
            "n_features_total": int(X.shape[1]),
            "n_features_seen": int(result.n_features_seen),
            "n_selected": int(len(result.feature_indices)),
            "selected_feature_ids": list(result.feature_indices),
            "selected_arrival_positions": list(result.arrival_positions),
            "score_min": result.score_min,
            "score_max": result.score_max,
            "counters": {
                "direct_accepts": counters["direct_accepts"],
                "immediate_rejects": counters["immediate_rejects"],
                "deferred": counters["deferred"],
                "redundancy_removals": counters["redundancy_removals"],
                "promoted_from_pool": counters["promoted_from_pool"],
                "pool_flushes": counters["pool_flushes"],
                "joint_evaluations": counters["joint_evaluations"],
                "peak_candidates": counters["peak_candidates"],
                "peak_pool": counters["peak_pool"],
                "pending_pool_at_end": counters["pending_pool_at_end"],
            },
            "warnings": list(result.warnings),
            "seconds": round(seconds, 4),
            "test_rows_read": 0,
            "classifier_fitted": False,
        }
        record["id_verification"] = verify_selected_ids(record, X.shape[1])
        record["event_verification"] = verify_event_positions(result.events, order)

        if write_json_once(rec_path, record):
            n_written += 1

        # trace -> gzipped JSONL under $RESEARCH_OUTPUT_ROOT only
        trace_path = out_dir / "traces" / f"{rid.run_id}.jsonl.gz"
        if not trace_path.exists():
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(trace_path, "wt", encoding="utf-8") as fh:
                for ev in result.events:
                    fh.write(json.dumps(event_row(ev), sort_keys=True) + "\n")
                for cp in checkpoint_rows(result.events, list(selector.scores), k1, k2):
                    fh.write(json.dumps({"kind": "checkpoint", **cp}, sort_keys=True) + "\n")

        if i % 10 == 0 or i == len(runs):
            print(f"  [{i}/{len(runs)}] fold{rid.fold} order{rid.order_seed} "
                  f"n_selected={record['n_selected']} ({seconds:.1f}s)")

    append_journal(journal, "EXPERIMENT_B_FINISHED", run_id, cfg_sha,
                   {"n_runs": len(runs), "n_newly_written": n_written})

    # anchor comparison against existing Stage 3A records
    anchors = []
    for fold in exp_b["folds"]:
        rid = ArceneRunID(fold, exp_b["anchor_order_seed"])
        rec = json.loads((out_dir / "arcene" / "runs" / f"{rid.run_id}.json").read_text())
        try:
            ref = load_stage3a_record(SubsetID("ARCENE", 11, fold))
        except BlockedInput as exc:
            anchors.append({"fold": fold, "match": False, "reason": str(exc)})
            continue
        same_ids = rec["selected_feature_ids"] == ref["selected_feature_ids"]
        same_n = rec["n_selected"] == ref["n_selected"]
        same_pos = rec["selected_arrival_positions"] == ref.get("selected_arrival_positions", rec["selected_arrival_positions"])
        anchors.append({
            "fold": fold,
            "run_id": rid.run_id,
            "stage3a_run_id": ref["run_id"],
            "n_selected_stage3b": rec["n_selected"],
            "n_selected_stage3a": ref["n_selected"],
            "selected_ids_identical": same_ids,
            "n_selected_identical": same_n,
            "arrival_positions_identical": same_pos,
            "match": bool(same_ids and same_n),
        })
    print(f"  anchors matched: {sum(1 for a in anchors if a['match'])}/5")
    (out_dir / "logs" / "anchor_comparison.json").write_text(
        json.dumps(anchors, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
