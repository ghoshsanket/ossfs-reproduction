#!/usr/bin/env python
"""Stage 3D V2 controlled comparison — manual CLI contract (corrected).

The user runs every phase. This script never launches subprocesses.

Subcommands:
  prepare   --config PATH --run-id RUN_ID
  preflight --run-id RUN_ID
  smoke     --run-id RUN_ID
  evaluate  --run-id RUN_ID --authorize-full-evaluation [--resume]
  classify  --run-id RUN_ID --authorize-classification [--resume]
  status    --run-id RUN_ID
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PROJECT = Path(__file__).resolve().parents[1]
OUT_ROOT = Path(os.environ.get("RESEARCH_OUTPUT_ROOT", ""))
if not OUT_ROOT or not OUT_ROOT.is_absolute():
    raise RuntimeError("RESEARCH_OUTPUT_ROOT must be set to an absolute path")
OUT_ROOT = OUT_ROOT / "ossfs-dd-reproduction" / "stage3d_v2"
V1_ROOT = Path(os.environ.get("RESEARCH_OUTPUT_ROOT", "")) / "ossfs-dd-reproduction" / "stage3d" / "20261102T042840Z"

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


def write_once(path: Path, payload: dict) -> None:
    if path.exists():
        raise SystemExit(f"REFUSING to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def run_id_dir(run_id: str) -> Path:
    return OUT_ROOT / f"comparison_{run_id}"


def load_frozen(run_id: str) -> dict:
    fp = run_id_dir(run_id) / "FROZEN_PROTOCOL.yaml"
    if not fp.exists():
        raise SystemExit("FROZEN_PROTOCOL.yaml missing; run prepare first")
    return yaml.safe_load(fp.read_text(encoding="utf-8"))


def expected_selection_keys(config: dict) -> set[str]:
    """Exact 230-key selection set from the frozen config."""
    keys = set()
    for arm_name, arm in config["experiment"]["arms"].items():
        for ds in arm["datasets"]:
            for s in arm["seeds"]:
                for f in arm["folds"]:
                    keys.add(f"{ds}_seed{s}_fold{f}__{arm_name}")
    return keys


def expected_classifier_keys(sel_keys: set[str], classifiers: list[str]) -> set[str]:
    return {f"{k}__{c}" for k in sel_keys for c in classifiers}


# ================================================================ prepare

def cmd_prepare(args) -> int:
    cfg_path = Path(args.config)
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    run_dir = run_id_dir(args.run_id)
    if run_dir.exists():
        raise SystemExit(f"REFUSING to overwrite existing run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_sha = sha256_file(cfg_path)

    frozen = dict(config)
    frozen["frozen_utc"] = utcnow()
    frozen["run_id"] = args.run_id
    frozen["config_sha256"] = cfg_sha
    frozen["declared_as_preregistration"] = False
    (run_dir / "FROZEN_PROTOCOL.yaml").write_text(
        yaml.safe_dump(frozen, sort_keys=False, allow_unicode=True), encoding="utf-8")

    from ossfs_dd_data.workspace import manifest_path, processed_dir, resolve
    datasets = {}
    for ds in frozen["authorization"]["datasets"]:
        man = json.loads(manifest_path(ds).read_text(encoding="utf-8"))
        schema = json.loads((processed_dir(ds) / "schema.json").read_text(encoding="utf-8"))
        disk = sha256_file(resolve(schema["X"]["path"]))
        ok = (man["stage1_verdict"]["status"] == "EXACT_VERIFIED" and disk == schema["X"]["sha256"])
        datasets[ds] = {"status": man["stage1_verdict"]["status"], "sha256_ok": ok}
        if not ok:
            raise SystemExit(f"{ds}: provenance failed")

    sel_keys = sorted(expected_selection_keys(frozen))
    clf_keys = sorted(expected_classifier_keys(set(sel_keys), frozen["experiment"]["classifiers"]))

    closure_files = [
        "src/ossfs_dd_stage3d/batch_selector_v2.py",
        "src/ossfs_dd_stage3d/batch_selector.py",
        "src/ossfs_dd_stage3d/selector.py",
        "src/ossfs_dd_stage3d/scheduler.py",
        "src/ossfs_dd_stage3d/standard_fisherz.py",
        "src/ossfs_dd/moments.py", "src/ossfs_dd/pools.py", "src/ossfs_dd/decisions.py",
    ]
    closure = {f: sha256_file(PROJECT / f) for f in closure_files if (PROJECT / f).exists()}

    # V1 comparator inventory
    v1_raw = V1_ROOT / "raw_records"
    v1_clf_dir = V1_ROOT / "classifiers"
    v1_sel = [json.loads(p.read_text()) for p in sorted(v1_raw.glob("*.json"))] if v1_raw.exists() else []
    v1_clf = [json.loads(p.read_text()) for p in sorted(v1_clf_dir.glob("*.json"))] if v1_clf_dir.exists() else []
    # V1 batch-arm keys that match V2 canonical arms
    v1_batch_keys = {f"{r['dataset']}_seed{r['seed']}_fold{r['fold']}__{r['arm']}"
                     for r in v1_sel if r["arm"] in ("D2", "C2", "C4")}
    # Exact arm mapping for the classifier join: D2_V2->D2, D2_V2_pilot->D2,
    # C2_V2->C2, C4_V2->C4. Never a naive replace.
    _v2_to_v1_arm = {"D2_V2": "D2", "D2_V2_pilot": "D2", "C2_V2": "C2", "C4_V2": "C4"}
    v1_batch_clf_keys = set()
    for c in v1_clf:
        _p = None
        if "__" in c["record_id"]:
            _sel, _clf = c["record_id"].rsplit("__", 1)
            if "__" in _sel:
                _head, _arm = _sel.rsplit("__", 1)
                if _arm in ("D2", "C2", "C4"):
                    v1_batch_clf_keys.add(c["record_id"])

    write_once(run_dir / "closure_hashes.json", closure)
    write_once(run_dir / "comparison_key_manifest.json", {
        "n_selection_keys": len(sel_keys), "n_classifier_keys": len(clf_keys),
        "selection_keys": sel_keys, "classifier_keys": clf_keys,
    })
    write_once(run_dir / "control_source_inventory.json", {
        "v1_run_id": "20261102T042840Z",
        "v1_n_selections": len(v1_sel), "v1_n_classifiers": len(v1_clf),
        "v1_batch_arm_selection_keys": sorted(v1_batch_keys),
        "v1_batch_arm_classifier_keys": sorted(v1_batch_clf_keys),
        "v1_batch_arm_count": len(v1_batch_keys),
        "v1_batch_arm_classifier_count": len(v1_batch_clf_keys),
        "control_evidence_status": ("AVAILABLE" if len(v1_batch_keys) >= 230
                                    and len(v1_batch_clf_keys) >= 690
                                    else "CONTROL_EVIDENCE_BLOCKED"),
    })
    write_once(run_dir / "PREFLIGHT.json", {
        "status": "PENDING", "run_id": args.run_id,
        "config_sha256": cfg_sha, "frozen_utc": frozen["frozen_utc"],
        "closure": closure, "datasets": datasets,
    })
    write_once(run_dir / "dataset_provenance.json", datasets)

    print(f"  prepare: run_dir={run_dir}")
    print(f"  config sha256={cfg_sha[:16]}")
    print(f"  expected selections={len(sel_keys)} classifiers={len(clf_keys)}")
    print(f"  V1 batch-arm keys: {len(v1_batch_keys)} classifiers: {len(v1_clf_keys)}")
    print(f"  V1 evidence: {json.loads((run_dir / 'control_source_inventory.json').read_text())['control_evidence_status']}")
    print("  NEXT: run pytest/lint manually, then preflight")
    return 0


# =============================================================== preflight

def cmd_preflight(args) -> int:
    run_dir = run_id_dir(args.run_id)
    frozen = load_frozen(args.run_id)
    blockers = []

    for name in ("pytest", "ruff"):
        log = run_dir / f"{name}_preflight.log"
        ext = run_dir / f"{name}_preflight.exit"
        if not log.exists() or not ext.exists():
            blockers.append(f"{name}: log or exit file missing")
            continue
        rc = int(ext.read_text().strip())
        if rc != 0:
            blockers.append(f"{name}: exit code {rc} != 0")

    inv = json.loads((run_dir / "control_source_inventory.json").read_text())
    if inv.get("control_evidence_status") != "AVAILABLE":
        blockers.append("V1 comparator evidence is CONTROL_EVIDENCE_BLOCKED")
    else:
        # exact V1 batch-arm key matching
        want_v1 = set(inv["v1_batch_arm_selection_keys"])
        if len(want_v1) < 230:
            blockers.append(f"V1 has {len(want_v1)} batch-arm keys < 230")
        if len(set(inv["v1_batch_arm_classifier_keys"])) < 690:
            blockers.append(f"V1 has {len(inv['v1_batch_arm_classifier_keys'])} classifier keys < 690")

    prov = json.loads((run_dir / "dataset_provenance.json").read_text())
    for ds, d in prov.items():
        if not d.get("sha256_ok"):
            blockers.append(f"{ds}: provenance hash failed")

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
                "config_sha256": frozen.get("config_sha256"), "blockers": []})
    print("  preflight PASS")
    return 0


# ================================================================== smoke

def cmd_smoke(args) -> int:
    run_dir = run_id_dir(args.run_id)
    pp = run_dir / "PREFLIGHT_PASS.json"
    if not pp.exists():
        raise SystemExit("PREFLIGHT_PASS.json missing")
    import time

    from sklearn.neighbors import KNeighborsClassifier

    from ossfs_dd.runner import load_dataset
    from ossfs_dd_data.workspace import splits_dir, stream_orders_dir
    from ossfs_dd_stage3d.scheduler import batch_size_for_p, make_schedule
    from ossfs_dd_stage3d.selector import make_continuous_batch_v2, make_discrete_batch_v2

    smoke_dir = run_dir / "smoke"
    smoke_dir.mkdir(exist_ok=True)

    # MADELON D2_V2: selection + one KNN held-out classification
    X, y, _ = load_dataset("MADELON")
    if hasattr(X, "tocsr"):
        X = X.toarray()
    X = np.asarray(X)
    split = json.loads((splits_dir("MADELON") / "seed_11.json").read_text(encoding="utf-8"))
    tr = np.asarray(split["folds"][0]["train"], dtype=np.int64)
    te = np.asarray(split["folds"][0]["test"], dtype=np.int64)
    order = np.asarray(json.loads(
        (stream_orders_dir("MADELON") / "seed_11.json").read_text(encoding="utf-8")
    )["arrival_order"], dtype=np.int64)
    sched = make_schedule(order, "MADELON", 11, 0)
    flat = sched["flattened_order"]
    B = batch_size_for_p(X.shape[1])
    Xtr, Xte = X[tr], X[te]
    ytr, yte = np.asarray(y)[tr], np.asarray(y)[te]
    bs, _, _ = make_discrete_batch_v2(ytr)
    i = 0
    t0 = time.perf_counter()
    delivered: list[int] = []
    while i < len(flat):
        chunk = [int(j) for j in flat[i:i + B]]
        cols = [Xtr[:, j].copy() for j in chunk]
        bs.consume_batch(chunk, cols)
        delivered.extend(chunk)
        i += B
    bs.finalize_stream()
    sel_seconds = time.perf_counter() - t0
    sel_ids = [int(v) for v in bs.candidates.feature_ids()]

    # validate mapping: every scheduled feature delivered exactly once
    smoke_ok = True
    blockers = []
    if sorted(delivered) != list(range(X.shape[1])):
        smoke_ok = False
        blockers.append(f"MADELON mapping: coverage {len(delivered)}/{X.shape[1]}")
    if len(set(delivered)) != len(delivered):
        smoke_ok = False
        blockers.append("MADELON mapping: duplicate delivered IDs")
    if not sel_ids:
        smoke_ok = False
        blockers.append("MADELON D2_V2 selected empty subset (valid outcome, but smoke needs a non-empty check)")
    if not getattr(bs, "_finalized", False):
        smoke_ok = False
        blockers.append("MADELON D2_V2 not finalized")

    # KNN held-out classification on selected features
    acc = None
    if sel_ids and smoke_ok:
        knn = KNeighborsClassifier(n_neighbors=5, weights="uniform", metric="minkowski", p=2)
        try:
            knn.fit(Xtr[:, sel_ids], ytr)
            acc = float(np.mean(knn.predict(Xte[:, sel_ids]) == yte))
            if not np.isfinite(acc):
                smoke_ok = False
                blockers.append("MADELON KNN accuracy not finite")
        except Exception as exc:
            smoke_ok = False
            blockers.append(f"MADELON KNN fit/predict failed: {exc}")
    results = {"MADELON_D2_V2": {
        "n_selected": len(sel_ids), "finalized": getattr(bs, "_finalized", False),
        "selection_seconds": round(sel_seconds, 4),
        "knn_accuracy": acc, "n_test": len(te),
    }}

    # SRBCT C2_V2 and C4_V2: selection only
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
    for arm, use_ev in (("C2_V2", False), ("C4_V2", True)):
        bs2, _, _ = make_continuous_batch_v2(labels_s, use_evidence=use_ev)
        j = 0
        delivered_s: list[int] = []
        while j < len(flat_s):
            chunk = [int(f) for f in flat_s[j:j + B_s]]
            cols = [Xtr_s[:, f].copy() for f in chunk]
            bs2.consume_batch(chunk, cols)
            delivered_s.extend(chunk)
            j += B_s
        bs2.finalize_stream()
        if sorted(delivered_s) != list(range(Xs.shape[1])):
            smoke_ok = False
            blockers.append(f"SRBCT {arm}: coverage {len(delivered_s)}/{Xs.shape[1]}")
        if not getattr(bs2, "_finalized", False):
            smoke_ok = False
            blockers.append(f"SRBCT {arm}: not finalized")
        results[f"SRBCT_{arm}"] = {"n_selected": len(bs2.candidates.feature_ids()),
                                   "finalized": getattr(bs2, "_finalized", False)}

    if not smoke_ok:
        write_once(run_dir / "SMOKE_BLOCKED.json",
                   {"status": "BLOCKED", "blockers": blockers, "results": results, "utc": utcnow()})
        print(f"  smoke BLOCKED: {blockers}")
        return 1

    write_once(run_dir / "SMOKE_PASS.json",
               {"status": "PASS", "results": results, "utc": utcnow()})
    print(f"  smoke PASS: {json.dumps(results)}")
    return 0


# =============================================================== evaluate

def cmd_evaluate(args) -> int:
    if not args.authorize_full_evaluation:
        raise SystemExit("REFUSED: --authorize-full-evaluation is required")
    run_dir = run_id_dir(args.run_id)
    # D: evaluation is gated on BOTH the preflight and the smoke pass.
    pp = run_dir / "PREFLIGHT_PASS.json"
    sp = run_dir / "SMOKE_PASS.json"
    if not pp.exists() or not sp.exists():
        raise SystemExit("REFUSED: PREFLIGHT_PASS.json and SMOKE_PASS.json must both exist "
                         "(run preflight, then smoke)")
    for label, path in (("preflight", pp), ("smoke", sp)):
        try:
            if json.loads(path.read_text()).get("status") != "PASS":
                raise SystemExit(f"REFUSED: {label} did not pass")
        except json.JSONDecodeError:
            raise SystemExit(f"REFUSED: {label} file is corrupt")
    frozen = load_frozen(args.run_id)
    manifest = json.loads((run_dir / "comparison_key_manifest.json").read_text())
    expected_sel = manifest["selection_keys"]
    ledger_dir = run_dir / "selection_records"
    ledger_dir.mkdir(exist_ok=True)

    from ossfs_dd.runner import load_dataset
    from ossfs_dd_data.workspace import splits_dir, stream_orders_dir
    from ossfs_dd_stage3d.scheduler import batch_size_for_p, make_schedule
    from ossfs_dd_stage3d.selector import make_continuous_batch_v2, make_discrete_batch_v2

    arm_by_key = {}
    for arm_name, arm in frozen["experiment"]["arms"].items():
        for d in arm["datasets"]:
            for s in arm["seeds"]:
                for f in arm["folds"]:
                    arm_by_key[f"{d}_seed{s}_fold{f}__{arm_name}"] = (d, s, f, arm_name, arm["metric"])

    done = 0
    for key in expected_sel:
        rec_path = ledger_dir / f"{key}.json"
        if rec_path.exists():
            # D: resume must VERIFY the existing record, never blindly skip it.
            try:
                existing = json.loads(rec_path.read_text())
            except json.JSONDecodeError:
                raise SystemExit(f"CORRUPT_RECORD: {key} is not valid JSON; refusing to resume")
            if (existing.get("key") != key
                    or existing.get("method_version") != "batch_score_shared_v2"
                    or existing.get("finalized") is not True
                    or not existing.get("split_train_sha256")
                    or not existing.get("schedule_hash")):
                raise SystemExit(f"STALE_CACHE: {key} fails integrity checks; "
                                 "start a NEW run ID rather than resuming")
            ids = existing.get("selected_feature_ids") or []
            if len(set(ids)) != len(ids):
                raise SystemExit(f"CORRUPT_RECORD: {key} has duplicate selected IDs")
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
        is_cont = metric in ("standard_fisherz_effect_only", "standard_fisherz_normalized_p")
        if is_cont:
            bs, _, _ = make_continuous_batch_v2(labels, use_evidence=use_ev)
        else:
            bs, _, _ = make_discrete_batch_v2(labels)
        j = 0
        t0 = time.perf_counter()
        while j < len(flat):
            chunk = [int(x) for x in flat[j:j + B]]
            cols = [Xtr[:, x].copy() for x in chunk]
            bs.consume_batch(chunk, cols)
            j += B
        bs.finalize_stream()
        elapsed = time.perf_counter() - t0
        rec = {
            "schema": "ossfs-dd-reproduction/stage3d_v2/selection/1.0",
            "run_id": args.run_id, "key": key,
            "dataset": d, "seed": s, "fold": f, "arm": a,
            "method_version": "batch_score_shared_v2",
            "n_features_total": int(X.shape[1]),
            "n_selected": len(bs.candidates.feature_ids()),
            "selected_feature_ids": [int(v) for v in bs.candidates.feature_ids()],
            "n_batches": bs.state.n_batches,
            "pool_flushes": bs.state.pool_flushes,
            "finalized": getattr(bs, "_finalized", False),
            "selection_seconds": round(elapsed, 4),
            "split_train_sha256": hashlib.sha256(tr.tobytes()).hexdigest(),
            "schedule_hash": sched["hashes"]["flattened_order_sha256"],
        }
        tmp = rec_path.with_suffix(".partial")
        tmp.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, rec_path)
        done += 1
        if done % 25 == 0 or done == len(expected_sel):
            print(f"  [{done}/{len(expected_sel)}] {key}")

    print(f"  evaluate: {done}/{len(expected_sel)} selections completed")
    print(f"  NEXT: python scripts/stage3d_v2_comparison.py classify --run-id {args.run_id} --authorize-classification")
    return 0


# =============================================================== classify

def cmd_classify(args) -> int:
    if not args.authorize_classification:
        raise SystemExit("REFUSED: --authorize-classification is required")
    run_dir = run_id_dir(args.run_id)
    # D: classification is gated on BOTH the preflight and the smoke pass.
    pp = run_dir / "PREFLIGHT_PASS.json"
    sp = run_dir / "SMOKE_PASS.json"
    if not pp.exists() or not sp.exists():
        raise SystemExit("REFUSED: PREFLIGHT_PASS.json and SMOKE_PASS.json must both exist")
    for label, path in (("preflight", pp), ("smoke", sp)):
        try:
            if json.loads(path.read_text()).get("status") != "PASS":
                raise SystemExit(f"REFUSED: {label} did not pass")
        except json.JSONDecodeError:
            raise SystemExit(f"REFUSED: {label} file is corrupt")
    frozen = load_frozen(args.run_id)
    manifest = json.loads((run_dir / "comparison_key_manifest.json").read_text())
    expected_clf = manifest["classifier_keys"]
    sel_dir = run_dir / "selection_records"
    clf_dir = run_dir / "classification_records"
    clf_dir.mkdir(exist_ok=True)

    # verify all 230 selections exist AND are valid before classifying
    sel_keys = {p.stem for p in sorted(sel_dir.glob("*.json"))}
    expected_sel = set(manifest["selection_keys"])
    missing = expected_sel - sel_keys
    extra = sel_keys - expected_sel
    if missing:
        raise SystemExit(f"REFUSED: {len(missing)} selection records missing, e.g. {sorted(missing)[:5]}")
    if extra:
        raise SystemExit(f"REFUSED: {len(extra)} unexpected selection records, e.g. {sorted(extra)[:5]}")

    # validate every selection record before trusting it
    for key in sorted(expected_sel):
        p = sel_dir / f"{key}.json"
        try:
            r = json.loads(p.read_text())
        except json.JSONDecodeError:
            raise SystemExit(f"CORRUPT_RECORD: {key}")
        if (r.get("key") != key
                or r.get("method_version") != "batch_score_shared_v2"
                or r.get("finalized") is not True
                or not r.get("split_train_sha256")
                or not r.get("schedule_hash")):
            raise SystemExit(f"STALE_CACHE: {key} fails integrity checks")
        ids = r.get("selected_feature_ids") or []
        if len(set(ids)) != len(ids):
            raise SystemExit(f"CORRUPT_RECORD: {key} has duplicate selected IDs")
        if any((not isinstance(i, int)) or i < 0 or i >= 10**9 for i in ids):
            raise SystemExit(f"CORRUPT_RECORD: {key} has an out-of-range ID")

    import time

    from sklearn.metrics import balanced_accuracy_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    from ossfs_dd.runner import load_dataset
    from ossfs_dd_data.workspace import splits_dir

    def build(clf, seed=11):
        if clf == "knn":
            return KNeighborsClassifier(n_neighbors=5, weights="uniform", metric="minkowski", p=2)
        if clf == "svm":
            return SVC(kernel="linear", C=1.0, tol=0.001, max_iter=2000000)
        if clf == "cart":
            return DecisionTreeClassifier(criterion="gini", splitter="best", random_state=11)
        raise ValueError(clf)

    done = 0
    for key in expected_clf:
        rec_path = clf_dir / f"{key}.json"
        if rec_path.exists():
            # validate existing record before skipping
            try:
                existing = json.loads(rec_path.read_text())
            except json.JSONDecodeError:
                raise SystemExit(f"CORRUPT_RECORD: {key} is not valid JSON")
            if (existing.get("record_id") != key
                    or existing.get("method_version") != "batch_score_shared_v2"
                    or existing.get("status") not in ("COMPLETED", "EMPTY_SELECTION", "FIT_ERROR")
                    or existing.get("classifier") not in ("knn", "svm", "cart")):
                raise SystemExit(f"STALE_CACHE: {key} fails integrity checks")
            done += 1
            continue
        sel_key, clf_name = key.rsplit("__", 1)
        sel_rec = json.loads((sel_dir / f"{sel_key}.json").read_text())
        d, s, f = sel_rec["dataset"], sel_rec["seed"], sel_rec["fold"]
        sel_ids = sel_rec["selected_feature_ids"]
        X, y, _ = load_dataset(d)
        if hasattr(X, "tocsr"):
            X = X.toarray()
        X = np.asarray(X)
        split = json.loads((splits_dir(d) / f"seed_{s}.json").read_text(encoding="utf-8"))
        tr = np.asarray(split["folds"][f]["train"], dtype=np.int64)
        te = np.asarray(split["folds"][f]["test"], dtype=np.int64)
        ytr, yte = np.asarray(y)[tr], np.asarray(y)[te]
        if not sel_ids:
            rec = {"record_id": key, "classifier": clf_name, "status": "EMPTY_SELECTION",
                   "accuracy": None, "balanced_accuracy": None, "n_test": len(te),
                   "n_correct": None, "n_selected": 0}
        else:
            Xtr = X[tr][:, sel_ids]
            Xte = X[te][:, sel_ids]
            est = build(clf_name)
            t0 = time.perf_counter()
            try:
                # SINGLE fit: capture convergence warnings around the one and
                # only .fit(), never re-fitting the same task a second time.
                import warnings as _w
                with _w.catch_warnings(record=True) as caught:
                    _w.simplefilter("always")
                    est.fit(Xtr, ytr)
                yhat = est.predict(Xte)
                n_test = len(yte)
                n_correct = int(np.sum(yhat == yte))
                early = [str(w.message) for w in caught
                         if "Convergence" in str(w.message)
                         or "terminated early" in str(w.message)]
                n_iter_vec = None
                if clf_name == "svm":
                    ni = getattr(est, "n_iter_", None)
                    n_iter_vec = [int(v) for v in np.ravel(ni)] if ni is not None else None
                rec = {
                    "record_id": key, "classifier": clf_name, "status": "COMPLETED",
                    "dataset": d, "seed": s, "fold": f, "arm": sel_rec["arm"],
                    "method_version": "batch_score_shared_v2",
                    "accuracy": n_correct / n_test if n_test else None,
                    "balanced_accuracy": float(balanced_accuracy_score(yte, yhat)),
                    "n_correct": n_correct, "n_test": n_test,
                    "n_selected": len(sel_ids),
                    "converged": not bool(early),
                    "early_termination_warnings": early,
                    "n_iter_vector": n_iter_vec,
                    "estimator_params": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                                         for k, v in sorted(est.get_params(deep=False).items())},
                    "fit_seconds": round(time.perf_counter() - t0, 4),
                    "subset_ids_sha256": hashlib.sha256(
                        json.dumps([int(x) for x in sel_ids], sort_keys=True).encode()).hexdigest(),
                    "split_train_sha256": sel_rec.get("split_train_sha256"),
                    "schedule_hash": sel_rec.get("schedule_hash"),
                }
            except Exception as exc:
                rec = {"record_id": key, "classifier": clf_name, "status": "FIT_ERROR",
                       "dataset": d, "seed": s, "fold": f, "arm": sel_rec["arm"],
                       "method_version": "batch_score_shared_v2",
                       "accuracy": None, "balanced_accuracy": None,
                       "n_test": len(te), "n_correct": None, "n_selected": len(sel_ids),
                       "converged": None, "error": f"{type(exc).__name__}: {exc}"}
        tmp = rec_path.with_suffix(".partial")
        tmp.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, rec_path)
        done += 1
        if done % 50 == 0 or done == len(expected_clf):
            print(f"  [{done}/{len(expected_clf)}] {key}")

    print(f"  classify: {done}/{len(expected_clf)} classifier outcomes completed")
    print(f"  NEXT: python scripts/stage3d_v2_comparison_audit.py --run-id {args.run_id}")
    return 0


# ================================================================== status

def cmd_status(args) -> int:
    run_dir = run_id_dir(args.run_id)
    if not run_dir.exists():
        print(f"  run not found: {run_dir}")
        return 1
    manifest = json.loads((run_dir / "comparison_key_manifest.json").read_text())
    expected_sel = set(manifest["selection_keys"])
    expected_clf = set(manifest["classifier_keys"])
    sel_dir = run_dir / "selection_records"
    clf_dir = run_dir / "classification_records"
    done_sel = {p.stem for p in sorted(sel_dir.glob("*.json"))} if sel_dir.exists() else set()
    done_clf = {p.stem for p in sorted(clf_dir.glob("*.json"))} if clf_dir.exists() else set()
    pp = (run_dir / "PREFLIGHT_PASS.json").exists()
    sp = (run_dir / "SMOKE_PASS.json").exists()
    av = (run_dir / "FINAL_VERDICT.json").exists()
    print(f"  run: {args.run_id}")
    print(f"  preflight: {'PASS' if pp else 'PENDING'}")
    print(f"  smoke: {'PASS' if sp else 'PENDING'}")
    print(f"  selections: {len(done_sel & expected_sel)}/{len(expected_sel)}")
    print(f"  classifiers: {len(done_clf & expected_clf)}/{len(expected_clf)}")
    print(f"  audit: {'DONE' if av else 'PENDING'}")
    if len(done_sel) < len(expected_sel):
        print(f"  next: evaluate --run-id {args.run_id} --authorize-full-evaluation --resume")
    elif len(done_clf) < len(expected_clf):
        print(f"  next: classify --run-id {args.run_id} --authorize-classification --resume")
    elif not av:
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
    p = sub.add_parser("classify")
    p.add_argument("--run-id", required=True)
    p.add_argument("--authorize-classification", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.set_defaults(func=cmd_classify)
    p = sub.add_parser("status")
    p.add_argument("--run-id", required=True)
    p.set_defaults(func=cmd_status)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
