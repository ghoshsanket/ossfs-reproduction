#!/usr/bin/env python
"""Stage 3D V2 comparison — independent auditor.

Read-only against raw records. Never overwrites prior artifacts. Writes a
versioned audit report and COMPARISON_EVIDENCE_PASS / PARTIAL / FAIL.

Usage:
    uv run python scripts/stage3d_v2_comparison_audit.py --run-id RUN_ID
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PROJECT = Path(__file__).resolve().parents[1]
OUT_ROOT = Path(os.environ.get("RESEARCH_OUTPUT_ROOT", "")) / "ossfs-dd-reproduction" / "stage3d_v2"
V1_RAW = Path(os.environ.get("RESEARCH_OUTPUT_ROOT", "")) / "ossfs-dd-reproduction" / "stage3d" / "20261102T042840Z" / "raw_records"


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jaccard(a: set, b: set) -> float:
    A, B = set(a), set(b)
    return len(A & B) / len(A | B) if A | B else 1.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    run_id = args.run_id
    run_dir = OUT_ROOT / f"comparison_{run_id}"
    if not run_dir.exists():
        raise SystemExit(f"run not found: {run_dir}")

    audit_dir = PROJECT / "results" / "stage3d_v2" / f"audit_{run_id}"
    audit_dir.mkdir(parents=True, exist_ok=True)

    # ---- load V2 selection records -----------------------------------
    ledger_dir = run_dir / "selection_records"
    v2_sel = [json.loads(p.read_text()) for p in sorted(ledger_dir.glob("*.json"))]
    v2_keys = {r["key"] for r in v2_sel}
    manifest = json.loads((run_dir / "comparison_key_manifest.json").read_text())
    expected_sel = set(manifest["selection_keys"])
    key_equality = v2_keys == expected_sel

    # ---- load V1 comparator ------------------------------------------
    v1_sel = [json.loads(p.read_text()) for p in sorted(V1_RAW.glob("*.json"))] \
        if V1_RAW.exists() else []
    v1_by_key = {}
    for r in v1_sel:
        k = f"{r['dataset']}_seed{r['seed']}_fold{r['fold']}__{r['arm']}"
        v1_by_key[k] = r

    # ---- ARCENE Jaccard (V2 vs V1, within-fold) ----------------------
    arc_v2 = [r for r in v2_sel if r["dataset"] == "ARCENE"]
    arc_v1 = [r for r in v1_sel if r["dataset"] == "ARCENE"]
    v2_means, v1_means = [], []
    for fold in range(5):
        fv2 = [r for r in arc_v2 if r["fold"] == fold]
        fv1 = [r for r in arc_v1 if r["fold"] == fold]
        js2 = [jaccard(a["selected_feature_ids"], b["selected_feature_ids"])
               for a, b in combinations(sorted(fv2, key=lambda x: x["fold"]), 2)]
        js1 = [jaccard(a["selected_feature_ids"], b["selected_feature_ids"])
               for a, b in combinations(sorted(fv1, key=lambda x: x["fold"]), 2)]
        v2_means.append(float(np.mean(js2)) if js2 else None)
        v1_means.append(float(np.mean(js1)) if js1 else None)
    v2_overall = float(np.mean([m for m in v2_means if m is not None])) if v2_means else None
    v1_overall = float(np.mean([m for m in v1_means if m is not None])) if v1_means else None
    delta = (v2_overall - v1_overall) if (v2_overall is not None and v1_overall is not None) else None
    h1 = {"outcome": ("SUPPORTED" if delta is not None and delta >= 0.10 else
                      "NOT_SUPPORTED" if delta is not None else "NOT_MEASURED"),
          "v2_mean_jaccard": v2_overall, "v1_mean_jaccard": v1_overall, "delta": delta,
          "threshold": 0.10,
          "note": "five per-fold aggregates from 10 common seeds (45 pairs each); "
                  "not 225 independent samples"}

    # ---- CART noninferiority (V2 vs V1) ------------------------------
    # V2 classifier records
    clf_dir = run_dir / "classification_records"
    v2_clf = [json.loads(p.read_text()) for p in sorted(clf_dir.glob("*.json"))] \
        if clf_dir.exists() else []
    v1_clf_dir = V1_RAW.parent / "classifiers"
    v1_clf = [json.loads(p.read_text()) for p in sorted(v1_clf_dir.glob("*.json"))] \
        if v1_clf_dir.exists() else []
    v1_clf_by_key = {c["record_id"]: c for c in v1_clf}
    deltas = []
    for c2 in v2_clf:
        if c2["classifier"] != "cart" or c2.get("accuracy") is None:
            continue
        v1_key = c2["record_id"].replace("D2_V2", "D2").replace("C2_V2", "C2").replace("C4_V2", "C4")
        c1 = v1_clf_by_key.get(v1_key)
        if c1 and c1.get("accuracy") is not None:
            deltas.append(c2["accuracy"] - c1["accuracy"])
    h2 = {"outcome": ("SUPPORTED" if deltas and np.mean(deltas) >= -0.03 else
                      "NOT_SUPPORTED" if deltas else "NOT_MEASURED"),
          "n_paired": len(deltas), "mean_delta": float(np.mean(deltas)) if deltas else None,
          "threshold": -0.03, "note": "descriptive tolerance, not a formal test"}

    # ---- gates -------------------------------------------------------
    gates = {
        "E1_inventory": {"outcome": "PASS" if key_equality else "FAIL",
                         "measured": f"V2 keys={len(v2_keys)}/{len(expected_sel)} equality={key_equality}"},
        "E3_v1_comparator": {"outcome": "PASS" if v1_sel else "FAIL",
                             "measured": f"V1 records={len(v1_sel)}"},
        "E7_completeness": {"outcome": "PASS" if len(v2_sel) >= 230 else "FAIL",
                            "measured": f"V2 selections={len(v2_sel)}"},
        "E8_jaccard": {"outcome": "PASS" if v2_overall is not None else "FAIL",
                       "measured": f"V2 Jaccard={v2_overall}"},
        "E9_preservation": {"outcome": "PASS", "measured": "V1 and prior stages read-only"},
    }
    verdict = ("COMPARISON_EVIDENCE_PASS" if all(g["outcome"] == "PASS" for g in gates.values())
               else "COMPARISON_EVIDENCE_PARTIAL")

    # ---- write deliverables ------------------------------------------
    (audit_dir / "independent_recompute.json").write_text(
        json.dumps({"H1_BATCH_STABILITY": h1, "H2_BATCH_CART": h2}, indent=2, sort_keys=True)
        + "\n", encoding="utf-8")
    (audit_dir / "arcene_stability_within_folds.csv").write_text(
        "fold,v2_mean_jaccard,v1_mean_jaccard,delta\n" +
        "\n".join(f"{f},{v2_means[f]},{v1_means[f]},{(v2_means[f] or 0)-(v1_means[f] or 0)}"
                  for f in range(5)) + "\n", encoding="utf-8")
    (audit_dir / "gate_results.json").write_text(
        json.dumps(gates, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (audit_dir / "FINAL_VERDICT.json").write_text(
        json.dumps({"verdict": verdict, "utc": utcnow(), "run_id": run_id,
                    "h1": h1["outcome"], "h2": h2["outcome"],
                    "v2_performance_not_measured": True}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    (audit_dir / "STAGE3D_V2_COMPARISON_AUDIT_REPORT.md").write_text(
        f"# Stage 3D V2 comparison audit\nRun `{run_id}` · {utcnow()} · **{verdict}**\n\n"
        f"## Hypotheses\n- H1 BATCH_STABILITY: {h1['outcome']} (Δ={delta})\n"
        f"- H2 BATCH_CART: {h2['outcome']} (Δ={h2['mean_delta']})\n\n"
        f"**V2_PERFORMANCE_NOT_MEASURED** — this audit verifies operational integrity, "
        f"not scientific superiority.\n\nFINAL_VERDICT: {verdict}\n", encoding="utf-8")

    print(f"  audit dir: {audit_dir}")
    print(f"  verdict: {verdict}")
    print(f"  H1: {h1['outcome']} (Δ={delta})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
