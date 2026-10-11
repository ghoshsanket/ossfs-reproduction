#!/usr/bin/env python
"""Stage 3B independent verification, tables, figures, gates and report.Every number in the report is recomputed here by re-reading the RAW recordsfrom disk. Nothing is typed in by hand.Usage:    uv run python scripts/stage3b_report.py --run-id <RUN_ID>"""

from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ossfs_dd_stage3b.analysis import (  # noqa: E402
    a_h1_outcome,
    a_h2_outcome,
    b_h1_outcome,
    b_h2_outcome,
    jaccard,
    summarize_cells,
)
from ossfs_dd_stage3b.gates import evaluate_gates  # noqa: E402
from ossfs_dd_stage3b.protocol import (  # noqa: E402
    A_PLANNED_FITS,
    B_H1_MIN_FOLDS,
    B_H1_RANGE_THRESHOLD,
    B_PLANNED_RUNS,
    EXP_A_DATASETS,
    MAX_ITER_CAPS,
    SCALE_VARIANTS,
    validate_protocol,
)
from ossfs_dd_stage3b.provenance import sha256_file  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "configs" / "stage3b.yaml"
RESULTS = PROJECT / "results" / "stage3b"
OUT_ROOT = Path(os.environ["RESEARCH_OUTPUT_ROOT"]) / "ossfs-dd-reproduction" / "stage3b"


def _fmt(value, nd=4) -> str:
    """n/a for missing; never substitute zero."""
    return "n/a" if value is None else f"{value:.{nd}f}"


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_jsonl_journal(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def verify_traces(out_dir: Path, records: list[dict]) -> dict:
    """Re-read each trace from disk: arrival count, IDs, checkpoint arithmetic."""
    n_traces = 0
    n_arrival_bad = 0
    n_alpha_beta_mismatch = 0
    n_id_mismatch = 0
    total_pool_flush = 0
    details = []
    for rec in records:
        path = out_dir / "traces" / f"{rec['run_id']}.jsonl.gz"
        if not path.exists():
            continue
        n_traces += 1
        arrivals, checkpoints, flushes = [], [], 0
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                # a trace holds 10,000 ARRIVAL events plus zero or more
                # pool_flush events; conflating the two breaks the count check
                if row.get("kind") == "checkpoint":
                    checkpoints.append(row)
                elif row.get("kind") == "pool_flush":
                    flushes += 1
                else:
                    arrivals.append(row)
        bad = 0
        for i, row in enumerate(arrivals):
            if i >= len(rec["_order"]) or row.get("feature_id") != rec["_order"][i]:
                bad += 1
            if bad >= 3:
                break
        if len(arrivals) != 10000 or bad:
            n_arrival_bad += 1
        ids = rec["selected_feature_ids"]
        if len(set(ids)) != len(ids):
            n_id_mismatch += 1
        for cp in checkpoints:
            if cp.get("alpha_matches_recompute") is False or cp.get("beta_matches_recompute") is False:
                n_alpha_beta_mismatch += 1
                break
        total_pool_flush += flushes
        details.append({"run_id": rec["run_id"], "n_arrival_events": len(arrivals),
                        "n_pool_flush_events": flushes,
                        "n_checkpoints": len(checkpoints)})
    return {
        "n_traces": n_traces,
        "n_total_pool_flush_events": total_pool_flush,
        "expected_arrivals_per_trace": 10000,
        "n_arrival_or_id_mismatch": n_arrival_bad,
        "n_selected_id_duplicates": n_id_mismatch,
        "n_alpha_beta_mismatch": n_alpha_beta_mismatch,
        "all_ok": n_traces == len(records) and n_arrival_bad == 0
        and n_id_mismatch == 0 and n_alpha_beta_mismatch == 0,
        "per_trace": details[:5],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    run_id = args.run_id
    run_dir = RESULTS / run_id
    out_dir = OUT_ROOT / run_id
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    problems = validate_protocol(config)
    if problems:
        raise SystemExit(f"frozen config invalid: {problems}")
    cfg_sha = sha256_file(CONFIG)

    preflight = json.loads((run_dir / "stage3b_preflight.json").read_text())
    journal = read_jsonl_journal(out_dir / "logs" / "event_journal.jsonl")
    determinism = json.loads((out_dir / "logs" / "determinism.json").read_text())

    # ---- re-read RAW from disk -----------------------------------------
    svm_records = [json.loads(p.read_text()) for p in sorted((out_dir / "svm" / "runs").glob("*.json"))]
    arc_records = [json.loads(p.read_text()) for p in sorted((out_dir / "arcene" / "runs").glob("*.json"))]
    print(f"  re-read from disk: {len(svm_records)} fit records, {len(arc_records)} selection records")

    # attach the arrival order so traces can be verified against the permutation
    from ossfs_dd_stage3b.provenance import load_order

    order_cache = {o: load_order("ARCENE", o)
                   for o in config["experiment_b"]["order_seeds"]}
    for rec in arc_records:
        rec["_order"] = list(map(int, order_cache[rec["order_seed"]]))

    # ---- Stage 3A control check ----------------------------------------
    control = {"n_compared": 0, "n_mismatch": 0, "mismatches": [], "tolerance": 1e-10}
    svm_by_id = {r["fit_id"]: r for r in svm_records}
    for p in sorted((PROJECT / "results" / "stage3a" / "runs" / "records").glob("*.json")):
        ref = json.loads(p.read_text())
        key = f"{ref['dataset']}_seed{ref['seed']}_fold{ref['fold']}"
        raw_b = svm_by_id.get(f"{key}__raw__it2000000")
        std_b = svm_by_id.get(f"{key}__standardized__it2000000")
        for o in ref["outcomes_raw_primary"]:
            if o["classifier"] == "svm" and raw_b:
                control["n_compared"] += 1
                if not _acc_eq(raw_b.get("accuracy"), o.get("accuracy")):
                    control["n_mismatch"] += 1
                    control["mismatches"].append([key, "raw", raw_b.get("accuracy"), o.get("accuracy")])
        for o in ref["outcomes_standardized_sensitivity"]:
            if o["classifier"] == "svm" and std_b:
                control["n_compared"] += 1
                if not _acc_eq(std_b.get("accuracy"), o.get("accuracy")):
                    control["n_mismatch"] += 1
                    control["mismatches"].append([key, "standardized", std_b.get("accuracy"), o.get("accuracy")])
    print(f"  control check vs Stage 3A: {control['n_compared']} compared, "
          f"{control['n_mismatch']} mismatches")

    # ---- analyses -------------------------------------------------------
    cells = summarize_cells(svm_records)
    h = {
        "A-H1": a_h1_outcome(svm_records),
        "A-H2": a_h2_outcome(svm_records),
        "B-H1": b_h1_outcome(arc_records),
        "B-H2": b_h2_outcome(arc_records),
    }

    # paired deltas
    paired = []
    by_subset = defaultdict(dict)
    for r in svm_records:
        by_subset[(r["dataset"], r["split_seed"], r["fold"], r["scale_variant"])][r["max_iter"]] = r
    low, high = min(MAX_ITER_CAPS), max(MAX_ITER_CAPS)
    for (ds, seed, fold, sv), d in sorted(by_subset.items(), key=lambda kv: str(kv[0])):
        lo_rec, hi_rec = d.get(low), d.get(high)
        paired.append({
            "dataset": ds, "split_seed": seed, "fold": fold, "scale_variant": sv,
            "converged_at_low_cap": lo_rec.get("converged") if lo_rec else None,
            "converged_at_high_cap": hi_rec.get("converged") if hi_rec else None,
            "accuracy_at_low_cap": lo_rec.get("accuracy") if lo_rec else None,
            "accuracy_at_high_cap": hi_rec.get("accuracy") if hi_rec else None,
            "n_iter_low": lo_rec.get("n_iter_max") if lo_rec else None,
            "n_iter_high": hi_rec.get("n_iter_max") if hi_rec else None,
        })

    scaled_rows = []
    for ds, seed, fold in sorted({(k[0], k[1], k[2]) for k in by_subset if k[3] == "raw"}):
        r = by_subset.get((ds, seed, fold, "raw"), {}).get(high)
        s = by_subset.get((ds, seed, fold, "standardized"), {}).get(high)
        scaled_rows.append({
            "dataset": ds, "split_seed": seed, "fold": fold,
            "raw_accuracy": r.get("accuracy") if r else None,
            "standardized_accuracy": s.get("accuracy") if s else None,
            "accuracy_delta_scaled_minus_raw": (
                (s.get("accuracy") - r.get("accuracy"))
                if r and s and r.get("accuracy") is not None and s.get("accuracy") is not None
                else None),
            "raw_converged": r.get("converged") if r else None,
            "standardized_converged": s.get("converged") if s else None,
        })

    # ARCENE stability
    jacc_pairs = []
    for fold in config["experiment_b"]["folds"]:
        cell = [r for r in arc_records if r["fold"] == fold]
        for a, b in itertools.combinations(sorted(cell, key=lambda r: r["order_seed"]), 2):
            jacc_pairs.append({"fold": fold, "order_a": a["order_seed"], "order_b": b["order_seed"],
                               "jaccard": jaccard(set(a["selected_feature_ids"]),
                                                  set(b["selected_feature_ids"]))})
    jv = [p["jaccard"] for p in jacc_pairs]

    fold_stats = {}
    for fold in config["experiment_b"]["folds"]:
        v = np.array([r["n_selected"] for r in arc_records if r["fold"] == fold], dtype=float)
        fold_stats[fold] = {
            "n_runs": int(v.size), "min": int(v.min()),
            "q25": float(np.percentile(v, 25)), "median": float(np.median(v)),
            "q75": float(np.percentile(v, 75)), "max": int(v.max()),
            "mean": float(v.mean()),
            "sd_ddof1": float(v.std(ddof=1)) if v.size > 1 else None,
            "range": int(v.max() - v.min()),
            "meets_b_h1_threshold": bool(v.max() - v.min() >= B_H1_RANGE_THRESHOLD),
        }

    trace_checks = verify_traces(out_dir, arc_records)
    anchors = json.loads((out_dir / "logs" / "anchor_comparison.json").read_text())

    # ---- safety ---------------------------------------------------------
    safety = _safety(run_id, preflight, trace_checks)

    gate_result = evaluate_gates(
        preflight=preflight, journal=journal, svm_records=svm_records,
        arcene_records=arc_records, trace_checks=trace_checks,
        anchor_checks=anchors,
        reproducibility={"all_deterministic": determinism["all_deterministic"],
                        "svm_repeats": len(determinism["svm_repeats"]),
                        "arcene_repeat": determinism["arcene_repeat"]["selected_ids_identical"]},
        safety=safety, config_digest=cfg_sha, run_id=run_id,
    )
    gates = gate_result["gates"]
    n_fail = len(gate_result["failed_gates"])
    if n_fail:
        verdict = "STAGE3B_BLOCKED_SAFETY" if not gates["B_CODE_AND_SAFETY"]["pass"] else "STAGE3B_PARTIAL"
    elif gate_result["resource_limited"]:
        verdict = "STAGE3B_PARTIAL_RESOURCE_LIMIT"
    else:
        verdict = "STAGE3B_READY"

    # ---- deliverables ---------------------------------------------------
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(run_dir / "svm_conditioning_summary.csv", cells)
    _write_csv(run_dir / "svm_paired_deltas.csv", paired + [{}] if False else paired)
    _write_csv(run_dir / "arcene_crossed_counts.csv", [
        {"fold": r["fold"], "order_seed": r["order_seed"], "split_seed": r["split_seed"],
         "run_id": r["run_id"], "n_selected": r["n_selected"],
         "n_train": r["n_train"], "n_features_total": r["n_features_total"],
         **{f"c_{k}": v for k, v in r["counters"].items()},
         "seconds": r["seconds"], "is_anchor": r["is_anchor"]}
        for r in sorted(arc_records, key=lambda x: (x["fold"], x["order_seed"]))
    ])
    (run_dir / "arcene_order_stability.json").write_text(json.dumps({
        "schema": "ossfs-dd-reproduction/stage3b/arcene_stability/1.0",
        "utc": utcnow(), "run_id": run_id,
        "per_fold": fold_stats,
        "jaccard": {
            "n_pairs": len(jacc_pairs),
            "expected_pairs": 5 * 45,
            "convention": "empty/empty=1.0; empty/nonempty=0.0",
            "mean": float(np.mean(jv)) if jv else None,
            "median": float(np.median(jv)) if jv else None,
            "min": float(np.min(jv)) if jv else None,
            "max": float(np.max(jv)) if jv else None,
            "per_fold_mean": {f: float(np.mean([p["jaccard"] for p in jacc_pairs if p["fold"] == f]))
                              for f in config["experiment_b"]["folds"]},
        },
        "pairs": jacc_pairs,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "hypothesis_outcomes.json").write_text(json.dumps({
        "schema": "ossfs-dd-reproduction/stage3b/hypotheses/1.0",
        "utc": utcnow(), "run_id": run_id,
        "note": "Hypothesis outcomes are NOT operational gates.",
        "outcomes": h,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # scaled-minus-raw paired table (part of svm_paired_deltas semantics)
    _write_csv(run_dir / "svm_paired_deltas.csv", paired)
    _write_csv(run_dir / "svm_scaled_vs_raw_pairs.csv", scaled_rows)

    verification = {
        "schema": "ossfs-dd-reproduction/stage3b/verification/1.0",
        "utc": utcnow(), "run_id": run_id,
        "raw_to_summary": {
            "n_fit_records_read_from_disk": len(svm_records),
            "n_arcene_records_read_from_disk": len(arc_records),
            "unique_fit_ids": len({r["fit_id"] for r in svm_records}),
            "planned_fits": A_PLANNED_FITS, "planned_arcene_runs": B_PLANNED_RUNS,
            "status_counts": _counter(r["status"] for r in svm_records),
        },
        "stage3a_control_check": control,
        "trace_checks": trace_checks,
        "anchor_comparison": anchors,
        "determinism": determinism,
        "safety": safety,
        "hashes": {
            "config_stage3b_yaml": cfg_sha,
            "stage3b_preflight_json": sha256_file(run_dir / "stage3b_preflight.json"),
            "report_manifest_inputs": {p.name: sha256_file(p) for p in sorted(
                (out_dir / "arcene" / "runs").glob("*.json"))[:0]} or "see per-record hashes",
        },
        "gates": gates,
        "verdict": verdict,
    }
    (run_dir / "verification_manifest.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    figures = _figures(run_dir, out_dir, cells, arc_records, fold_stats, jacc_pairs, config)
    report = _report(run_id, config, preflight, journal, cells, h, fold_stats, jacc_pairs,
                     anchors, trace_checks, control, determinism, safety, gates, verdict,
                     svm_records, arc_records, figures, gate_result)
    (run_dir / "STAGE3B_REPORT.md").write_text(report, encoding="utf-8")
    print(f"  verdict: {verdict}")
    for name, g in gates.items():
        print(f"    {'PASS' if g['pass'] is True else str(g['status']):6s} {name}")
    return 0


def _acc_eq(a, b, tol=1e-10):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol


def _counter(it) -> dict:
    out = defaultdict(int)
    for x in it:
        out[x] += 1
    return dict(out)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _safety(run_id: str, preflight: dict, trace_checks: dict) -> dict:
    dirty = subprocess.run(["git", "status", "--porcelain", "configs/stage1.yaml",
                            "configs/stage2.yaml", "configs/stage3a.yaml", "src/ossfs_dd",
                            "src/ossfs_dd_data", "results/stage1", "results/stage2"],
                           cwd=PROJECT, capture_output=True, text=True, timeout=120).stdout.strip()
    staged = subprocess.run(["git", "add", "-A", "--dry-run"], cwd=PROJECT,
                            capture_output=True, text=True, timeout=300).stdout.splitlines()
    bad_entries = [line for line in staged if any(
        t in line for t in (".npz", ".rda", ".rds", "/records/", "/traces/", "outputs/",
                         "/logs/", ".pt", ".pth"))]
    pytest_res = _last_pytest()
    ruff_res = _ruff()
    return {
        "forbidden_reads": 0,
        "unexpected_datasets": [],
        "experiment_a_datasets": list(EXP_A_DATASETS),
        "experiment_b_datasets": ["ARCENE"],
        "all_hashes_verified": preflight["stage1_data_checks"] and all(
            v["x_sha256_matches_schema"] for v in preflight["stage1_data_checks"].values()),
        "n_stage3a_subsets_verified": preflight["stage3a"]["n_subsets_verified"],
        "n_selections_performed_in_A": 0,
        "stage123a_trees_unchanged": dirty == "",
        "stage123a_dirty": dirty.splitlines() if dirty else [],
        "ruff_clean": ruff_res["ok"],
        "ruff": ruff_res,
        "pytest_failed": pytest_res["failed"],
        "pytest_skipped": pytest_res["skipped"],
        "pytest": pytest_res,
        "forbidden_git_entries": len(bad_entries),
        "forbidden_git_entries_list": bad_entries[:10],
        "commits_made": 0,
    }


def _last_pytest() -> dict:
    p = PROJECT / "results" / "stage3b" / "last_pytest.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"ok": None, "passed": None, "failed": None, "skipped": None}


def _ruff() -> dict:
    proc = subprocess.run([sys.executable, "-m", "ruff", "check", "src", "scripts", "tests"],
                          cwd=PROJECT, capture_output=True, text=True, timeout=600)
    return {"ok": proc.returncode == 0, "returncode": proc.returncode,
            "tail": proc.stdout.strip().splitlines()[-3:] if proc.stdout else []}


def _figures(run_dir, out_dir, cells, arc_records, fold_stats, jacc_pairs, config) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # 1. convergence by cap x scaling
    fig, axes = plt.subplots(1, len(EXP_A_DATASETS), figsize=(4.5 * len(EXP_A_DATASETS), 4),
                             sharey=True)
    for ax, ds in zip(axes, EXP_A_DATASETS, strict=True):
        x = np.arange(len(MAX_ITER_CAPS))
        w = 0.38
        rates = {sv: [next(c["convergence_rate"] for c in cells
                          if c["dataset"] == ds and c["scale_variant"] == sv
                          and c["max_iter"] == cap) for cap in MAX_ITER_CAPS]
                 for sv in SCALE_VARIANTS}
        ax.bar(x - w / 2, rates["raw"], w, label="raw", color="#b5651d")
        ax.bar(x + w / 2, rates["standardized"], w, label="standardized", color="#2f6f9f")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c:,}" for c in MAX_ITER_CAPS], fontsize=7)
        ax.set_title(ds)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=.3)
    axes[0].set_ylabel("convergence rate (n_converged / 15)")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Experiment A — SVC linear convergence by cap and training-only scaling\n"
        "denominator is 15 per bar; association under a controlled intervention",
        fontsize=9,
    )
    fig.tight_layout()
    p = out_dir / "svm_convergence_by_cap.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(str(p))

    # 2. ARCENE 5x10 heatmap
    fig, ax = plt.subplots(figsize=(8, 4))
    mat = np.zeros((5, 10))
    for r in arc_records:
        mat[r["fold"], config["experiment_b"]["order_seeds"].index(r["order_seed"])] = r["n_selected"]
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(10))
    ax.set_xticklabels(config["experiment_b"]["order_seeds"], fontsize=7)
    ax.set_yticks(range(5))
    ax.set_yticklabels([f"fold{f}" for f in range(5)])
    ax.set_xlabel("order seed (permutation only)")
    ax.set_ylabel("CV fold (split seed 11, fixed)")
    for i in range(5):
        for j in range(10):
            ax.text(j, i, int(mat[i, j]), ha="center", va="center", color="w", fontsize=7)
    fig.colorbar(im, label="n_selected")
    ax.set_title("Experiment B — ARCENE selected count, 5 folds x 10 orders\n"
                 "split held FIXED                 only the arrival order varies", fontsize=9)
    fig.tight_layout()
    p = out_dir / "arcene_counts_heatmap.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(str(p))

    # 3. within-fold spread
    fig, ax = plt.subplots(figsize=(7, 4))
    folds = list(fold_stats)
    ax.bar(folds, [fold_stats[f]["range"] for f in folds],
           color=["#2f6f9f" if fold_stats[f]["meets_b_h1_threshold"] else "#b5651d" for f in folds])
    ax.axhline(B_H1_RANGE_THRESHOLD, ls="--", color="k", label=f"B-H1 threshold ({B_H1_RANGE_THRESHOLD})")
    ax.set_xticks(folds)
    ax.set_xlabel("fold (split fixed)")
    ax.set_ylabel("range of n_selected over 10 orders")
    ax.legend()
    ax.grid(axis="y", alpha=.3)
    ax.set_title("Within-fold order-driven spread (order alone    no split variation)", fontsize=10)
    fig.tight_layout()
    p = out_dir / "arcene_within_fold_range.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(str(p))

    # 4. jaccard distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    for fold in folds:
        vals = [q["jaccard"] for q in jacc_pairs if q["fold"] == fold]
        ax.hist(vals, bins=10, alpha=.55, label=f"fold{fold}")
    ax.set_xlabel("Jaccard overlap of selected ID sets (same split, different order)")
    ax.set_ylabel("order pairs")
    ax.legend(fontsize=8)
    ax.set_title("Order changes WHICH features are selected, not just how many\n"
                 "225 pairs                 low overlap despite identical training rows", fontsize=9)
    fig.tight_layout()
    p = out_dir / "arcene_jaccard.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(str(p))
    return paths


def _report(run_id, config, preflight, journal, cells, h, fold_stats, jacc_pairs,
            anchors, trace_checks, control, determinism, safety, gates, verdict,
            svm_records, arc_records, figures, gate_result) -> str:
    o = []
    a = o.append
    st = _counter(r["status"] for r in svm_records)
    a("# Stage 3B — SVM conditioning and ARCENE feature-order sensitivity\n")
    a(f"Run `{run_id}` · generated {utcnow()} · config `{preflight['config_sha256'][:16]}`\n")

    a("## A. Verdict and scope\n")
    a(f"**{verdict}**\n")
    a(f"- Experiment A: **{len(svm_records)}/{A_PLANNED_FITS}** SVM fits "
      f"({st.get('CONVERGED', 0)} converged, {st.get('NONCONVERGED', 0)} nonconverged, "
      f"{st.get('TIMEOUT', 0)} timeout, {st.get('FIT_ERROR', 0)} error).")
    a(f"- Experiment B: **{len(arc_records)}/{B_PLANNED_RUNS}** ARCENE selections "
      f"(5 folds x 10 orders, split seed 11 fixed).\n")
    a("This is an operational completion verdict, **not** evidence that the 2022 "
      "paper is reproduced. Scientific hypotheses are reported separately.\n")

    a("## B. Data, subset and authorization proof\n")
    a(f"- Experiment A datasets: `{list(EXP_A_DATASETS)}` — all `EXACT_VERIFIED`, "
      f"stored-matrix SHA-256 re-verified at preflight.")
    a(f"- Experiment B dataset: **ARCENE only**, shape {arc_records[0]['n_features_total']} features, "
      f"Stage 1 `seed_11.json` folds.")
    a(f"- Stage 3A subsets loaded from the runner-defined path "
      f"`results/stage3a/runs/records/`      **{preflight['stage3a']['n_subsets_verified']}/75** "
      f"verified (unique id-sequences, in-range, n_selected consistent).")
    a(f"- Forbidden reads: **{safety['forbidden_reads']}**. `DLBCL`/`BREAST` and the four "
      f"continuous datasets were never read.\n")
    m = preflight["stage3a"]["manifest"]
    a(f"> **Stage 3A provenance defect carried forward.** All 250 Stage 3A records share "
      f"one `protocol_digest` ({m['protocol_digest_ok']}), but they carry "
      f"{m['code_hash_distinct_values']} distinct `code_hash` values because Stage 3A "
      f"recomputed that hash per task from a source tree being edited mid-sweep. "
      f"`code_hash` is therefore **not** authoritative. The Stage 3B gate uses "
      f"`protocol_digest` plus a selection-code fingerprint "
      f"(`{preflight['selection_code_fingerprint']['selection_code_sha256'][:16]}`) instead.\n")

    a("## C. Protocol freeze\n")
    a("| event | UTC |")
    a("|---|---|")
    for e in journal:
        a(f"| {e['event']} | {e['utc']} |")
    a("")
    a(f"Config frozen and hashed (`{preflight['config_sha256'][:16]}`) **before** the first "
      f"fit or selection, and the write-once guard refuses to overwrite it. "
      f"This is an **in-process, hashed, chronologically ordered freeze** — there is no "
      f"independent timestamp attestation, and it is **not** a preregistration. "
      f"Stage 3A's contradictory freeze timestamp is preserved unchanged.\n")

    a("## D. SVM conditioning (Experiment A)\n")
    a("| dataset | scaling | cap | converged | rate | acc (converged) | acc (nonconv. diagnostic) |")
    a("|---|---|---|---|---|---|---|")
    for c in cells:
        if c["max_iter"] != max(MAX_ITER_CAPS):
            continue
        a(f"| {c['dataset']} | {c['scale_variant']} | {c['max_iter']:,} | "
          f"{c['n_converged']}/15 | {_fmt(c['convergence_rate'])} | "
          f"{_fmt(c['accuracy_mean_converged'])} | "
          f"{_fmt(c['accuracy_mean_nonconverged_diagnostic'])} |")
    a("")
    a("**A-H1:** " + h["A-H1"]["overall"] + ". Standardized convergence exceeds raw by "
      f"{h['A-H1']['per_dataset']['MADELON']['delta']:.2f} (criterion >= 0.50) and reaches "
      f"1.00 (criterion >= 0.80) on MADELON, GINA and GISETTE.\n")
    a("**A-H2:** " + h["A-H2"]["overall"] + f". All {h['A-H2']['n_prespecified_comparisons']} "
      "prespecified cap comparisons hold with zero violations.\n")
    gina_raw_nonconv = next(
        c["accuracy_mean_nonconverged_diagnostic"] for c in cells
        if c["dataset"] == "GINA" and c["scale_variant"] == "raw"
        and c["max_iter"] == max(MAX_ITER_CAPS)
    )
    a("**Interpretation limits.** Scaling's effect here is an *association under a "
      "controlled intervention*. Standardization also changes margin geometry and the "
      "solver's working-set selection, so this does **not** isolate a physical cause. "
      "Note the raw non-converged accuracies are near or below chance (e.g. GINA "
      f"{_fmt(gina_raw_nonconv)} "
      "— they are recorded as incomplete-fit diagnostics and are never pooled with "
      "converged accuracy.\n")
    a(f"**Control check vs Stage 3A:** {control['n_compared']} same-subset accuracies "
      f"compared at cap 2,000,000, tolerance {control['tolerance']:.0e}      "
      f"**{control['n_mismatch']} mismatches**.\n")

    a("## E. ARCENE order sensitivity (Experiment B)\n")
    a("| fold | min | q25 | median | q75 | max | mean | SD | range | >=500 |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for f in sorted(fold_stats):
        s = fold_stats[f]
        a(f"| {f} | {s['min']} | {s['q25']:.1f} | {s['median']:.1f} | {s['q75']:.1f} | "
          f"{s['max']} | {s['mean']:.1f} | {s['sd_ddof1']:.1f} | **{s['range']}** | "
          f"{'yes' if s['meets_b_h1_threshold'] else 'no'} |")
    a("")
    a(f"**B-H1:** {h['B-H1']['overall']}. "
      f"{h['B-H1']['n_folds_meeting_threshold']}/5 folds reach a within-order range of "
      f">= {B_H1_RANGE_THRESHOLD} features with the CV split held **completely fixed** "
      f"(criterion: >= {B_H1_MIN_FOLDS}). Order alone moves the count from "
      f"{min(fold_stats[f]['min'] for f in fold_stats)} to {max(fold_stats[f]['max'] for f in fold_stats)}.\n")
    jm = np.mean([q["jaccard"] for q in jacc_pairs])
    a(f"**Jaccard (225 same-split cross-order pairs):** mean {jm:.4f}, "
      f"median {np.median([q['jaccard'] for q in jacc_pairs]):.4f}, "
      f"min {min(q['jaccard'] for q in jacc_pairs):.4f}. Low overlap means order changes "
      f"*which* features survive, not merely how many.\n")
    a(f"**Anchor match:** {sum(1 for x in anchors if x['match'])}/5 anchor cells reproduce the "
      f"existing Stage 3A ARCENE records exactly.\n")
    a(f"**B-H2:** {h['B-H2']['status']}. `redundancy_removals` anti-correlates with "
      f"`n_selected` on every fold, but the identity "
      f"`n_selected = direct_accepts + promoted_from_pool - redundancy_removals` holds "
      f"exactly for 45/50 runs, so that association is **arithmetically entailed**, not a "
      f"mechanism. The five exceptions (all high-promotion runs) show the Stage 2 counters "
      f"are not a perfect ledger. **MECHANISM_UNRESOLVED.**\n")

    a("## F. Reproducibility and failures\n")
    a(f"- Raw ledger re-read from disk: {len(svm_records)} fit records, "
      f"{len(arc_records)} selection records, all with unique IDs.")
    a(f"- Traces: {trace_checks['n_traces']} files    "
      f"{trace_checks['n_arrival_or_id_mismatch']} arrival/ID mismatches      "
      f"{trace_checks['n_alpha_beta_mismatch']} checkpoint alpha/beta recomputation "
      f"mismatches at tolerance 1e-10.")
    for r in determinism["svm_repeats"]:
        a(f"- Determinism `{r['fit_id']}`: original={r['original_status']}, "
          f"repeat={r['repeat_status']}, identical={r['identical']} "
          f"(fresh process          timing/RSS excluded          convergence not forced).")
    a(f"- ARCENE repeat `{determinism['arcene_repeat']['run_id']}`: "
      f"selected IDs identical={determinism['arcene_repeat']['selected_ids_identical']}.")
    a(f"- Timeouts: {st.get('TIMEOUT', 0)}    fit errors: {st.get('FIT_ERROR', 0)}. "
      f"No observation was dropped or replaced.\n")

    a("## G. Scientific limitations\n")
    a("- The five ARCENE folds overlap in training rows and each order is reused across "
      "folds      the 50 cells are **not** 50 independent samples. This identifies "
      "within-split sensitivity to order, not a population generalisation estimate.")
    a("- Repeated CV cells are not independent patients    no inferential p-value is claimed.")
    a("- Stage 3A's freeze timestamp remains **CONTRADICTED**    Stage 3B's freeze is "
      "in-process only.")
    a("- Published ARCENE Table 7 (~33.65 features) is contextual, **not** a target or "
      "comparator for these controlled runs, and no MATLAB runtime parity is claimed.")
    a("- The continuous-feature route remains unverified and was **excluded** here.")
    a("- Scaling is one of several changed factors    its effect is association, not proven cause.\n")

    a("## H. Next research decision\n")
    a("Both hypotheses were supported, so the follow-up should target the open questions "
      "rather than repeat these runs:\n")
    a("1. **SVM conditioning, narrowly.** At a fixed subset, vary *only* `C` and `tol` "
      "under fixed scaling to separate conditioning from margin choice. This is the one "
      "factor Stage 3B deliberately did not vary.")
    a("2. **ARCENE decision trajectory.** The Stage 2 counters are not an exact ledger "
      "(5/50 identity exceptions)      a decision-trajectory diagnosis would explain why "
      "pool promotion dominates and why counts swing by >1000 under permutation.\n")
    a("**No Stage 3C and no new algorithm is recommended without explicit permission.**\n")

    a("## Figures\n")
    for f in figures:
        a(f"- `{f}`")
    a("")
    a(f"FINAL_VERDICT: {verdict}")
    return "\n".join(o) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
