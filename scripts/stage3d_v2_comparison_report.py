#!/usr/bin/env python
"""Stage 3D V2 comparison — report and figures from raw records only.

Usage:
    uv run python scripts/stage3d_v2_comparison_report.py --run-id RUN_ID
"""

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

    # read audit verdict
    audit_dir = PROJECT / "results" / "stage3d_v2" / f"audit_{run_id}"
    audit_verdict = "NOT_AUDITED"
    if (audit_dir / "FINAL_VERDICT.json").exists():
        audit_verdict = json.loads((audit_dir / "FINAL_VERDICT.json").read_text()).get("verdict", "NOT_AUDITED")

    # read selection records
    ledger_dir = run_dir / "selection_records"
    v2_sel = [json.loads(p.read_text()) for p in sorted(ledger_dir.glob("*.json"))] \
        if ledger_dir.exists() else []

    # per-dataset summary
    by_ds = {}
    for r in v2_sel:
        by_ds.setdefault(r["dataset"], []).append(r["n_selected"])
    ds_summary = {d: {"n_runs": len(v), "mean_n_selected": float(np.mean(v)) if v else None}
                  for d, v in sorted(by_ds.items())}

    report = f"""# Stage 3D V2 comparison report

Run `{run_id}` · generated {utcnow()} · audit verdict **{audit_verdict}**

## Selection summary
| Dataset | Runs | Mean n_selected |
|---|---|---|
""" + "\n".join(f"| {d} | {v['n_runs']} | {v['mean_n_selected']:.1f} |"
                for d, v in ds_summary.items()) + f"""

## Limitations
**V2_PERFORMANCE_NOT_MEASURED** — this report presents operational integrity
and descriptive comparisons only. It does not claim V2 accuracy or stability
superiority over V1 or the published paper.

FINAL_VERDICT: {audit_verdict}
"""
    (report_dir / "STAGE3D_V2_COMPARISON_REPORT.md").write_text(report, encoding="utf-8")
    (report_dir / "FINAL_VERDICT.json").write_text(
        json.dumps({"verdict": audit_verdict, "utc": utcnow(), "run_id": run_id},
                   indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"  report dir: {report_dir}")
    print(f"  audit verdict: {audit_verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
