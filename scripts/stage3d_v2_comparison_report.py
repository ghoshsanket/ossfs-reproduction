#!/usr/bin/env python
"""Stage 3D V2 comparison — report from raw records only."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PROJECT = Path(__file__).resolve().parents[1]
OUT_ROOT = Path(os.environ.get("RESEARCH_OUTPUT_ROOT", "")) / "ossfs-dd-reproduction" / "stage3d_v2"


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    run_id = args.run_id
    run_dir = OUT_ROOT / f"comparison_{run_id}"
    if not run_dir.exists():
        raise SystemExit(f"run not found: {run_dir}")
    report_dir = PROJECT / "results" / "stage3d_v2" / f"report_{run_id}"
    report_dir.mkdir(parents=True, exist_ok=True)

    audit_dir = PROJECT / "results" / "stage3d_v2" / f"audit_{run_id}"
    audit_verdict = "NOT_AUDITED"
    if (audit_dir / "FINAL_VERDICT.json").exists():
        audit_verdict = json.loads((audit_dir / "FINAL_VERDICT.json").read_text()).get("verdict", "NOT_AUDITED")
    hypotheses = {}
    if (audit_dir / "independent_recompute.json").exists():
        h = json.loads((audit_dir / "independent_recompute.json").read_text())
        hypotheses = {k: v.get("outcome", "NOT_MEASURED") for k, v in h.items()}

    sel_dir = run_dir / "selection_records"
    clf_dir = run_dir / "classification_records"
    v2_sel = [json.loads(p.read_text()) for p in sorted(sel_dir.glob("*.json"))] if sel_dir.exists() else []
    v2_clf = [json.loads(p.read_text()) for p in sorted(clf_dir.glob("*.json"))] if clf_dir.exists() else []

    by_ds = {}
    for r in v2_sel:
        by_ds.setdefault(r["dataset"], []).append(r["n_selected"])
    ds_rows = [(d, len(v), round(float(np.mean(v)), 1) if v else "n/a") for d, v in sorted(by_ds.items())]

    by_clf = {}
    for c in v2_clf:
        ds = c.get("dataset") or c["record_id"].split("_seed")[0]
        cn = c["classifier"]
        acc = c.get("accuracy")
        if acc is not None:
            by_clf.setdefault((ds, cn), []).append(acc)
    clf_rows = [(k[0], k[1], len(v), round(float(np.mean(v)), 4)) for k, v in sorted(by_clf.items())]

    lines = ["# Stage 3D V2 comparison report", "",
             f"Run `{run_id}` · generated {utcnow()} · audit verdict **{audit_verdict}**", "",
             "## Hypotheses", ""]
    lines += [f"- {k}: {v}" for k, v in sorted(hypotheses.items())]
    lines += ["", "## Selection summary", "", "| Dataset | Runs | Mean n_selected |", "|---|---|---|"]
    lines += [f"| {d} | {n} | {m} |" for d, n, m in ds_rows]
    lines += ["", "## Classifier accuracy", "", "| Dataset | Classifier | N | Mean accuracy |", "|---|---|---|---|"]
    lines += [f"| {d} | {c} | {n} | {m} |" for d, c, n, m in clf_rows]
    has_clf = bool(v2_clf)
    lines += ["", "## Limitations", ""]
    lines.append("**V2_PERFORMANCE_MEASURED** — classifier outcomes exist above." if has_clf
                 else "**V2_PERFORMANCE_NOT_MEASURED** — no classifier outcomes.")
    lines += ["", f"FINAL_VERDICT: {audit_verdict}", ""]
    report = "\n".join(lines)

    (report_dir / "STAGE3D_V2_COMPARISON_REPORT.md").write_text(report, encoding="utf-8")
    (report_dir / "FINAL_VERDICT.json").write_text(
        json.dumps({"verdict": audit_verdict, "utc": utcnow(), "run_id": run_id,
                    "hypotheses": hypotheses}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"  report dir: {report_dir}")
    print(f"  audit verdict: {audit_verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
