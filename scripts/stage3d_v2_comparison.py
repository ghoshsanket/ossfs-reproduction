#!/usr/bin/env python
"""Stage 3D V2 controlled comparison — manual CLI contract.

The user, not OpenCode, runs every phase. Each subcommand is independently
invoked. This script never launches subprocesses, tests, or evaluations.

Subcommands:
  prepare   --config PATH --run-id RUN_ID
  preflight --run-id RUN_ID
  smoke     --run-id RUN_ID
  evaluate  --run-id RUN_ID --authorize-full-evaluation [--resume]
  status    --run-id RUN_ID
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PROJECT = Path(__file__).resolve().parents[1]
OUT_ROOT = Path(os.environ.get("RESEARCH_OUTPUT_ROOT", "")) / "ossfs-dd-reproduction" / "stage3d_v2"

DISCRETE = ("ARCENE", "DEXTER", "MADELON", "GINA", "GISETTE")
CONTINUOUS = ("SRBCT", "LYMPHOMA", "PROSTATE", "LEUKEMIA")
FORBIDDEN = ("DLBCL", "BREAST")


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_json(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_once(path: Path, payload: dict) -> None:
    if path.exists():
        raise SystemExit(f"REFUSING to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def expected_arms(config: dict) -> list[tuple]:
    arms = []
    for arm_name, arm in config["experiment"]["arms"].items():
        ds = arm["datasets"]
        seeds = arm["seeds"]
        folds = arm["folds"]
        metric = arm["metric"]
        for d in ds:
            for s in seeds:
                for f in folds:
                    arms.append((d, s, f, arm_name, metric))
    # deduplicate (MADELON etc. may appear in multiple arm blocks)
    seen = set()
    out = []
    for a in arms:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def run_id_dir(run_id: str) -> Path:
    return OUT_ROOT / f"comparison_{run_id}"


# ================================================================ prepare

def cmd_prepare(args) -> int:
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"config not found: {cfg_path}")
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    run_dir = run_id_dir(args.run_id)
    if run_dir.exists():
        raise SystemExit(f"REFUSING to overwrite existing run: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_sha = sha256_file(cfg_path)

    # stamp actual UTC into a copy of the config
    frozen = dict(config)
    frozen["frozen_utc"] = utcnow()
    frozen["run_id"] = args.run_id
    frozen["config_sha256"] = cfg_sha
    frozen["declared_as_preregistration"] = False
    frozen["freeze_note"] = ("In-process chronological freeze. NOT an independently "
                             "attested preregistration.")
    frozen_path = run_dir / "FROZEN_PROTOCOL.yaml"
    frozen_path.write_text(yaml.safe_dump(frozen, sort_keys=False, allow_unicode=True),
                           encoding="utf-8")

    # 9-dataset inventory
    from ossfs_dd_data.workspace import manifest_path, processed_dir, resolve
    datasets = {}
    for ds in frozen["authorization"]["datasets"]:
        man = json.loads(manifest_path(ds).read_text(encoding="utf-8"))
        schema = json.loads((processed_dir(ds) / "schema.json").read_text(encoding="utf-8"))
        disk = sha256_file(resolve(schema["X"]["path"]))
        ok = (man["stage1_verdict"]["status"] == frozen["authorization"]["required_manifest_status"]
              and disk == schema["X"]["sha256"])
        datasets[ds] = {"status": man["stage1_verdict"]["status"], "sha256_ok": ok,
                        "p": schema["n_features"]}
        if not ok:
            raise SystemExit(f"{ds}: provenance failed")

    # V1 comparator inventory
    v1_raw = Path(os.environ.get("RESEARCH_OUTPUT_ROOT", "")) / "ossfs-dd-reproduction" / "stage3d" / "20261102T042840Z" / "raw_records"
    v1_sel = [json.loads(p.read_text()) for p in sorted(v1_raw.glob("*.json"))] if v1_raw.exists() else []
    v1_keys = {f"{r['dataset']}_seed{r['seed']}_fold{r['fold']}__{r['arm']}" for r in v1_sel}

    # expected comparison key manifest (V2 arms only)
    arms = expected_arms(frozen)
    expected_sel_keys = sorted({f"{d}_seed{s}_fold{f}__{a}" for d, s, f, a, m in arms})
    expected_clf_keys = sorted({f"{k}__{c}" for k in expected_sel_keys
                                for c in frozen["experiment"]["classifiers"]})

    # closure hashes
    closure_files = [
        "src/ossfs_dd_stage3d/batch_selector_v2.py",
        "src/ossfs_dd_stage3d/batch_selector.py",
        "src/ossfs_dd_stage3d/selector.py",
        "src/ossfs_dd_stage3d/scheduler.py",
        "src/ossfs_dd_stage3d/standard_fisherz.py",
        "src/ossfs_dd/moments.py", "src/ossfs_dd/pools.py", "src/ossfs_dd/decisions.py",
        "scripts/stage3d_v2_comparison.py",
    ]
    closure = {f: sha256_file(PROJECT / f) for f in closure_files if (PROJECT / f).exists()}

    # write-once artifacts
    write_once(run_dir / "closure_hashes.json", closure)
    write_once(run_dir / "comparison_key_manifest.json", {
        "n_selection_keys": len(expected_sel_keys),
        "n_classifier_keys": len(expected_clf_keys),
        "selection_keys": expected_sel_keys,
        "classifier_keys": expected_clf_keys,
    })
    write_once(run_dir / "control_source_inventory.json", {
        "v1_run_id": "20261102T042840Z",
        "v1_n_selections": len(v1_sel),
        "v1_n_unique_keys": len(v1_keys),
        "v1_records_available": bool(v1_sel),
        "control_evidence_status": ("AVAILABLE" if len(v1_sel) >= 100 else "CONTROL_EVIDENCE_BLOCKED"),
        "v1_comparator_arms": sorted({r["arm"] for r in v1_sel}),
    })
    write_once(run_dir / "PREFLIGHT.json", {
        "status": "PENDING", "run_id": args.run_id,
        "config_sha256": cfg_sha, "frozen_utc": frozen["frozen_utc"],
        "closure": closure, "datasets": datasets,
    })
    # data provenance
    write_once(run_dir / "dataset_provenance.json", datasets)

    print(f"  prepare: run_dir={run_dir}")
    print(f"  frozen protocol written; config sha256={cfg_sha[:16]}")
    print(f"  expected selections={len(expected_sel_keys)} classifiers={len(expected_clf_keys)}")
    print(f"  V1 comparator: {len(v1_sel)} records, {len(v1_keys)} unique keys")
    print("  NEXT: run pytest/lint manually, then preflight")
    return 0


# =============================================================== preflight

def cmd_preflight(args) -> int:
    run_dir = run_id_dir(args.run_id)
    frozen_path = run_dir / "FROZEN_PROTOCOL.yaml"
    if not frozen_path.exists():
        raise SystemExit("FROZEN_PROTOCOL.yaml missing; run prepare first")
    frozen = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
    cfg_sha = frozen.get("config_sha256")

    blockers = []

    # test/lint logs
    for name in ("pytest", "ruff"):
        log = run_dir / f"{name}_preflight.log"
        ext = run_dir / f"{name}_preflight.exit"
        if not log.exists() or not ext.exists():
            blockers.append(f"{name}: log or exit file missing")
            continue
        rc = int(ext.read_text().strip())
        if rc != 0:
            blockers.append(f"{name}: exit code {rc} != 0 (stdout may still say 'pass')")

    # V1 comparator evidence
    inv = json.loads((run_dir / "control_source_inventory.json").read_text())
    if inv.get("control_evidence_status") != "AVAILABLE":
        blockers.append("V1 comparator evidence is CONTROL_EVIDENCE_BLOCKED")

    # dataset hashes
    prov = json.loads((run_dir / "dataset_provenance.json").read_text())
    for ds, d in prov.items():
        if not d.get("sha256_ok"):
            blockers.append(f"{ds}: provenance hash failed")

    # frozen protocol integrity
    if frozen.get("declared_as_preregistration"):
        blockers.append("config falsely claims preregistration")

    if blockers:
        write_once(run_dir / "PREFLIGHT_BLOCKED.json",
                   {"status": "BLOCKED", "blockers": blockers, "utc": utcnow()})
        print(f"  preflight BLOCKED ({len(blockers)} blockers):")
        for b in blockers:
            print(f"    - {b}")
        return 1

    write_once(run_dir / "PREFLIGHT_PASS.json",
               {"status": "PASS", "run_id": args.run_id, "utc": utcnow(),
                "config_sha256": cfg_sha, "blockers": []})
    print("  preflight PASS")
    return 0


# ================================================================== smoke

def cmd_smoke(args) -> int:
    run_dir = run_id_dir(args.run_id)
    pp = run_dir / "PREFLIGHT_PASS.json"
    if not pp.exists():
        raise SystemExit("PREFLIGHT_PASS.json missing; preflight must pass first")
    pre = json.loads(pp.read_text())
    if pre.get("status") != "PASS":
        raise SystemExit("preflight did not pass")


    from ossfs_dd.runner import load_dataset
    from ossfs_dd_data.workspace import splits_dir, stream_orders_dir
    from ossfs_dd_stage3d.scheduler import batch_size_for_p, make_schedule
    from ossfs_dd_stage3d.selector import make_continuous_batch_v2, make_discrete_batch_v2

    smoke_dir = run_dir / "smoke"
    smoke_dir.mkdir(exist_ok=True)

    # MADELON D2_V2
    X, y, _ = load_dataset("MADELON")
    if hasattr(X, "tocsr"):
        X = X.toarray()
    X = np.asarray(X)
    split = json.loads((splits_dir("MADELON") / "seed_11.json").read_text(encoding="utf-8"))
    tr = np.asarray(split["folds"][0]["train"], dtype=np.int64)
    order = np.asarray(json.loads(
        (stream_orders_dir("MADELON") / "seed_11.json").read_text(encoding="utf-8")
    )["arrival_order"], dtype=np.int64)
    sched = make_schedule(order, "MADELON", 11, 0)
    flat = sched["flattened_order"]
    B = batch_size_for_p(X.shape[1])
    Xtr = X[tr]
    labels = np.asarray(y)[tr]
    bs, _, _ = make_discrete_batch_v2(labels)
    i = 0
    while i < len(flat):
        chunk = [int(j) for j in flat[i:i + B]]
        cols = [Xtr[:, j].copy() for j in chunk]
        bs.consume_batch(chunk, cols)
        i += B
    bs.finalize_stream()

    # SRBCT C2_V2 (r-only) and C4_V2 (normalized p)
    Xs, ys, _ = load_dataset("SRBCT")
    if hasattr(Xs, "tocsr"):
        Xs = Xs.toarray()
    Xs = np.asarray(Xs)
    split_s = json.loads((splits_dir("SRBCT") / "seed_11.json").read_text(encoding="utf-8"))
    tr_s = np.asarray(split_s["folds"][0]["train"], dtype=np.int64)
    order_s = np.asarray(json.loads(
        (stream_orders_dir("SRBCT") / "seed_11.json").read_text(encoding="utf-8")
    )["arrival_order"], dtype=np.int64)
    sched_s = make_schedule(order_s, "SRBCT", 11, 0)
    flat_s = sched_s["flattened_order"]
    B_s = batch_size_for_p(Xs.shape[1])
    Xtr_s = Xs[tr_s]
    labels_s = np.asarray(ys)[tr_s]
    results = {"MADELON_D2_V2": {"n_selected": len(bs.candidates.feature_ids()),
                                  "finalized": getattr(bs, "_finalized", False)}}
    for arm, use_ev in (("C2_V2", False), ("C4_V2", True)):
        bs2, _, _ = make_continuous_batch_v2(labels_s, use_evidence=use_ev)
        j = 0
        while j < len(flat_s):
            chunk = [int(f) for f in flat_s[j:j + B_s]]
            cols = [Xtr_s[:, f].copy() for f in chunk]
            bs2.consume_batch(chunk, cols)
            j += B_s
        bs2.finalize_stream()
        results[f"SRBCT_{arm}"] = {"n_selected": len(bs2.candidates.feature_ids()),
                                   "finalized": getattr(bs2, "_finalized", False)}

    write_once(run_dir / "SMOKE_PASS.json",
               {"status": "PASS", "results": results, "utc": utcnow()})
    print(f"  smoke PASS: {json.dumps(results)}")
    return 0


# =============================================================== evaluate

def cmd_evaluate(args) -> int:
    if not args.authorize_full_evaluation:
        raise SystemExit("REFUSED: --authorize-full-evaluation is required")
    run_dir = run_id_dir(args.run_id)
    pp = run_dir / "PREFLIGHT_PASS.json"
    sp = run_dir / "SMOKE_PASS.json"
    if not pp.exists() or not sp.exists():
        raise SystemExit("PREFLIGHT_PASS.json and SMOKE_PASS.json must both exist")

    frozen = yaml.safe_load((run_dir / "FROZEN_PROTOCOL.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "comparison_key_manifest.json").read_text())
    expected_sel = manifest["selection_keys"]
    ledger_dir = run_dir / "selection_records"
    ledger_dir.mkdir(exist_ok=True)

    import time as _t

    from ossfs_dd.runner import load_dataset
    from ossfs_dd_data.workspace import splits_dir, stream_orders_dir
    from ossfs_dd_stage3d.scheduler import batch_size_for_p, make_schedule
    from ossfs_dd_stage3d.selector import make_continuous_batch_v2, make_discrete_batch_v2

    arms = expected_arms(frozen)
    arm_by_key = {f"{d}_seed{s}_fold{f}__{a}": (d, s, f, a, m)
                  for d, s, f, a, m in arms}
    done = 0
    for key in expected_sel:
        rec_path = ledger_dir / f"{key}.json"
        if rec_path.exists():
            done += 1
            continue
        d, s, f, a, metric = arm_by_key[key]
        X, y, _ = load_dataset(d)
        if hasattr(X, "tocsr"):
            X = X.toarray()
        X = np.asarray(X)
        split = json.loads((splits_dir(d) / f"seed_{s}.json").read_text(encoding="utf-8"))
        tr = np.asarray(split["folds"][f]["train"], dtype=np.int64)
        order = np.asarray(json.loads(
            (stream_orders_dir(d) / f"seed_{s}.json").read_text(encoding="utf-8")
        )["arrival_order"], dtype=np.int64)
        sched = make_schedule(order, d, s, f)
        flat = sched["flattened_order"]
        B = batch_size_for_p(X.shape[1])
        Xtr = X[tr]
        labels = np.asarray(y)[tr]
        use_ev = (a == "C4_V2")
        is_cont = metric == "standard_fisherz_effect_only" or metric == "standard_fisherz_normalized_p"
        if is_cont:
            bs, _, _ = make_continuous_batch_v2(labels, use_evidence=use_ev)
        else:
            bs, _, _ = make_discrete_batch_v2(labels)
        j = 0
        while j < len(flat):
            chunk = [int(x) for x in flat[j:j + B]]
            cols = [Xtr[:, x].copy() for x in chunk]
            bs.consume_batch(chunk, cols)
            j += B
        bs.finalize_stream()
        rec = {
            "schema": "ossfs-dd-reproduction/stage3d_v2/selection/1.0",
            "run_id": args.run_id, "key": key,
            "dataset": d, "seed": s, "fold": f, "arm": a,
            "method_version": "batch_score_shared_v2",
            "n_selected": len(bs.candidates.feature_ids()),
            "selected_feature_ids": [int(v) for v in bs.candidates.feature_ids()],
            "n_batches": bs.state.n_batches,
            "pool_flushes": bs.state.pool_flushes,
            "finalized": getattr(bs, "_finalized", False),
            "seconds": round(_t.perf_counter() - _t.perf_counter(), 4),
        }
        tmp = rec_path.with_suffix(".partial")
        tmp.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, rec_path)
        done += 1
        if done % 25 == 0 or done == len(expected_sel):
            print(f"  [{done}/{len(expected_sel)}] {key}")

    print(f"  evaluate: {done}/{len(expected_sel)} selections completed")
    print("  NEXT: user runs audit and report separately")
    return 0


# ================================================================== status

def cmd_status(args) -> int:
    run_dir = run_id_dir(args.run_id)
    if not run_dir.exists():
        print(f"  run not found: {run_dir}")
        return 1
    manifest = json.loads((run_dir / "comparison_key_manifest.json").read_text())
    expected = manifest["selection_keys"]
    ledger = run_dir / "selection_records"
    done = [p.stem for p in sorted(ledger.glob("*.json"))] if ledger.exists() else []
    missing = sorted(set(expected) - set(done))
    print(f"  run: {args.run_id}")
    print(f"  expected selections: {len(expected)}")
    print(f"  completed: {len(done)}")
    print(f"  missing: {len(missing)}")
    pp = (run_dir / "PREFLIGHT_PASS.json").exists()
    sp = (run_dir / "SMOKE_PASS.json").exists()
    print(f"  preflight: {'PASS' if pp else 'PENDING/BLOCKED'}")
    print(f"  smoke: {'PASS' if sp else 'PENDING'}")
    if missing:
        print(f"  next: evaluate --run-id {args.run_id} --authorize-full-evaluation --resume")
    elif not (run_dir / "FINAL_VERDICT.json").exists():
        print(f"  next: stage3d_v2_comparison_audit.py --run-id {args.run_id}")
    else:
        print(f"  next: stage3d_v2_comparison_report.py --run-id {args.run_id}")
    return 0


# =================================================================== main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--config", required=True)
    p.add_argument("--run-id", required=True)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("preflight")
    p.add_argument("--run-id", required=True)
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("smoke")
    p.add_argument("--run-id", required=True)
    p.set_defaults(func=cmd_smoke)

    p = sub.add_parser("evaluate")
    p.add_argument("--run-id", required=True)
    p.add_argument("--authorize-full-evaluation", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("status")
    p.add_argument("--run-id", required=True)
    p.set_defaults(func=cmd_status)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
