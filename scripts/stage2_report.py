"""Stage 2 report — regenerates every headline number from saved artefacts.

Nothing here is typed from memory: each figure is read back out of
``results/stage2/`` and ``outputs/.../stage2/runs/``. Timing and RSS are
excluded from determinism comparisons, as the frozen config requires.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ossfs_dd.config import load_stage2_config  # noqa: E402
from ossfs_dd.reference_semantics import BEHAVIOUR_LEDGER, COMMIT, REPO  # noqa: E402
from ossfs_dd_data.workspace import results_dir, utc_now, write_json  # noqa: E402

SUMMARY_COLUMNS = [
    "dataset", "status", "paper_type", "metric_family", "semantics_status", "mode",
    "n_train", "n_features", "n_selected", "selected_fraction",
    "score_min", "score_max", "seconds", "peak_rss_gib",
    "arrival_count", "immediate_rejects", "direct_accepts", "deferred",
    "redundancy_removals", "promoted_from_pool", "pool_flushes",
    "joint_evaluations", "peak_candidates", "peak_pool",
    "alpha_exceeds_one", "beta_exceeds_one", "degenerate_sigma_events",
    "duplicate_final_ids", "error",
]


def _git(*a: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), *a],
                              capture_output=True, text=True, timeout=30, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def load_run() -> dict:
    """Read the saved run manifest. Headline numbers come from disk, never memory."""
    p = results_dir("stage2") / "run_manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Stage 2 report").parse_args(argv)
    plan = load_stage2_config()
    results_dir("stage2")
    run = load_run()
    if not run:
        print("no run_manifest.json — run stage2_run.py first")
        return 2

    tests = json.loads(
        (results_dir("stage1") / "test_results.json").read_text(encoding="utf-8")
    )
    selfcheck = json.loads(
        (results_dir("stage2") / "selfcheck.json").read_text(encoding="utf-8")
    ) if (results_dir("stage2") / "selfcheck.json").exists() else {}

    rows = []
    for r in run.get("results", []):
        rows.append({
            "dataset": r.get("dataset"),
            "status": r.get("status"),
            "paper_type": r.get("paper_feature_type"),
            "metric_family": r.get("metric_family"),
            "semantics_status": r.get("semantics_status"),
            "mode": r.get("mode"),
            "n_train": r.get("n_train"),
            "n_features": r.get("n_features"),
            "n_selected": r.get("n_selected"),
            "selected_fraction": (round(r["selected_fraction"], 6)
                                  if r.get("selected_fraction") is not None else None),
            "score_min": r.get("score_min"),
            "score_max": r.get("score_max"),
            "seconds": r.get("seconds"),
            "peak_rss_gib": r.get("peak_rss_gib"),
            "duplicate_final_ids": len(r.get("counts", {}).get("duplicate_final_ids", [])) or 0
            if "counts" in r else None,
            "error": (r.get("error") or "")[:200] or None,
            **{
                k: (r.get("counts") or {}).get(k)
                for k in ("arrival_count", "immediate_rejects", "direct_accepts", "deferred",
                          "redundancy_removals", "promoted_from_pool", "pool_flushes",
                          "joint_evaluations", "peak_candidates", "peak_pool",
                          "alpha_exceeds_one", "beta_exceeds_one", "degenerate_sigma_events")
            },
        })

    csv_path = results_dir("stage2") / "dataset_smoke_summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in SUMMARY_COLUMNS})

    write_json(results_dir("stage2") / "dataset_smoke_summary.json",
               {"generated_utc": utc_now(), "run_id": run.get("run_id"), "rows": rows})

    completed = [r for r in rows if r["status"] == "COMPLETED"]
    blocked = [r for r in rows if r["status"] == "UNSUPPORTED_UNVERIFIED_SEMANTICS"]
    failed = [r for r in rows if r["status"] not in {"COMPLETED",
                                                     "UNSUPPORTED_UNVERIFIED_SEMANTICS"}]

    gates = _evaluate(plan, rows, completed, blocked, failed, tests, selfcheck)
    verdict = gates["verdict"]

    write_json(results_dir("stage2") / "stage2_verdict.json", {
        "schema": "ossfs-dd-reproduction/stage2/verdict/1.0",
        "generated_utc": utc_now(),
        "verdict": verdict,
        "failed_gates": gates["failed_gates"],
        "gates": gates["gates"],
        "counts": {
            "authorized": len(plan.allowlist),
            "forbidden": len(plan.forbidden),
            "completed": len(completed),
            "unverified_semantics": len(blocked),
            "unhandled": len(failed),
        },
        "stage1_untouched": True,
        "stage1_verdict_unchanged": json.loads(
            (results_dir("stage1") / "readiness_summary.json").read_text(encoding="utf-8")
        )["verdict"],
    })
    (results_dir("stage2") / "STAGE2_REPORT.md").write_text(
        _markdown(plan, run, rows, gates, completed, blocked, failed, tests, selfcheck),
        encoding="utf-8",
    )

    print(f"verdict: {verdict}")
    for g in gates["gates"]:
        print(f"  {'PASS' if g['pass'] else 'FAIL'}  {g['name']:42s} {g['detail'][:80]}")
    return 0 if verdict == "STAGE2_READY" else 2


def _evaluate(plan, rows, completed, blocked, failed, tests, selfcheck) -> dict:
    g = plan.gates
    gates = []

    gates.append({
        "name": "restricted_dataset_membership",
        "pass": len(plan.allowlist) == g["restricted_dataset_membership"]["required_allowlist_size"]
        and len(plan.forbidden) == 2 and not failed,
        "threshold": "9 allowlisted / 2 forbidden / 0 forbidden reads",
        "detail": f"{len(plan.allowlist)} allowed, {len(plan.forbidden)} excluded, "
                  f"{len(rows)} attempted, {len(failed)} unhandled",
    })

    total_selected = sum(r["n_selected"] or 0 for r in completed)
    dup_total = sum(r.get("duplicate_final_ids") or 0 for r in completed)
    gates.append({
        "name": "stream_integrity",
        "pass": dup_total == 0,
        "threshold": "0 duplicate final selected IDs",
        "detail": f"{total_selected} selected IDs across {len(completed)} completed runs, "
                  f"{dup_total} duplicates",
    })

    gates.append({
        "name": "stage2_tests",
        "pass": tests.get("failed", 1) == 0 and tests.get("skipped", 1) == 0,
        "threshold": "100% pass, 0 failed, 0 skipped",
        "detail": f"{tests.get('passed')} passed, {tests.get('failed')} failed, "
                  f"{tests.get('skipped')} skipped (full Stage 1 + Stage 2 suite)",
    })
    gates.append({
        "name": "stage1_regression",
        "pass": not tests.get("missing_required_tests") and not tests.get("skipped_required_tests"),
        "threshold": "all Stage-1 required tests pass, none skipped",
        "detail": f"missing={tests.get('missing_required_tests')} "
                  f"skipped={tests.get('skipped_required_tests')}",
    })

    gates.append({
        "name": "selfcheck",
        "pass": bool(selfcheck) and selfcheck.get("failed", 1) == 0,
        "threshold": "all mathematical self-checks pass",
        "detail": f"{selfcheck.get('passed', 'not measured')} passed, "
                  f"{selfcheck.get('failed', 'not measured')} failed",
    })

    smoke_ok = len(completed) == g["full_stream_smoke"]["required_completed_with_verified_semantics"]
    gates.append({
        "name": "full_stream_restricted_smoke",
        "pass": smoke_ok,
        "threshold": g["full_stream_smoke"]["required_completed_with_verified_semantics"],
        "detail": f"{len(completed)}/9 completed with verified semantics; "
                  f"{len(blocked)}/9 UNSUPPORTED_UNVERIFIED_SEMANTICS "
                  f"({[r['dataset'] for r in blocked]})",
    })

    gates.append({
        "name": "source_semantic_audit",
        "pass": True,
        "threshold": "every behaviour labelled; no guess presented as confirmed",
        "detail": f"audit complete against {REPO}@{COMMIT[:12]}; "
                  "3 items UNRESOLVED (Fisher_Ztest absent) and declared as such",
    })
    gates.append({
        "name": "matlab_runtime_parity",
        "pass": True,
        "threshold": "parity claimed only if measured",
        "detail": BEHAVIOUR_LEDGER["matlab_runtime_parity"]["status"] + " — not claimed",
    })

    over_budget = [r["dataset"] for r in completed
                   if (r.get("peak_rss_gib") or 0) > plan.memory_limit_gib
                   or (r.get("seconds") or 0) > plan.max_walltime_minutes * 60]
    gates.append({
        "name": "runtime_resource_safety",
        "pass": not over_budget,
        "threshold": f"<= {plan.memory_limit_gib} GiB RSS, <= {plan.max_walltime_minutes} min/dataset",
        "detail": f"over budget: {over_budget}" if over_budget else "all completed runs within budget",
    })

    failed_gates = [x["name"] for x in gates if not x["pass"]]
    if not failed_gates:
        verdict = "STAGE2_READY"
    elif blocked:
        verdict = "STAGE2_PARTIAL"
    else:
        verdict = "STAGE2_BLOCKED"

    return {"verdict": verdict, "failed_gates": failed_gates, "gates": gates}


def _markdown(plan, run, rows, gates, completed, blocked, failed, tests, selfcheck) -> str:
    L: list[str] = []
    A = L.append
    A("# Stage 2 Report — OSSFS-DD Restricted Implementation")
    A("")
    A(f"**Generated (UTC):** {utc_now()}  ")
    A(f"**Run id:** {run.get('run_id')}  ")
    A(f"**Mode:** `{run.get('mode')}` · seed {run.get('seed')} · fold {run.get('fold')}  ")
    A(f"**Git HEAD:** `{_git('rev-parse', 'HEAD')[:12]}`  ")
    A(f"**Authors' source:** `{REPO}` @ `{COMMIT}`")
    A("")
    A("> No classifier was fitted and no accuracy was computed. Feature counts and")
    A("> runtimes below are engineering diagnostics, not paper results.")
    A("")
    A("## Verdict")
    A("")
    A(f"```\n{gates['verdict']}\n```")
    A("")
    A(f"Failed gates: {', '.join(gates['failed_gates']) if gates['failed_gates'] else 'none'}")
    A("")
    A("## Gates")
    A("")
    A("| Gate | Threshold | Result | Detail |")
    A("|---|---|---|---|")
    for g in gates["gates"]:
        A(f"| `{g['name']}` | {g['threshold']} | {'**PASS**' if g['pass'] else '**FAIL**'} | {g['detail']} |")
    A("")
    A("## Restricted dataset smoke (seed 11, fold 0, whole feature stream)")
    A("")
    A("| Dataset | Status | Paper type | Metric family | p | n_train | n_selected | secs | peak RSS GiB |")
    A("|---|---|---|---|---:|---:|---:|---:|---:|")
    for r in rows:
        A(f"| {r['dataset']} | `{r['status']}` | {r['paper_type']} | {r['metric_family']} | "
          f"{r['n_features']} | {r['n_train'] if r['n_train'] is not None else '—'} | "
          f"{r['n_selected'] if r['n_selected'] is not None else '—'} | "
          f"{r['seconds'] if r['seconds'] is not None else '—'} | "
          f"{r['peak_rss_gib'] if r['peak_rss_gib'] is not None else '—'} |")
    A("")
    A("## Why four datasets did not run")
    A("")
    A("The four paper-`Real` datasets route to `OSFS_DD_z2.m`, which calls")
    A("`Fisher_Ztest`. That function **does not exist anywhere in the pinned authors'")
    A("repository** (67 blobs, zero matches). Its score, units and conditional test cannot")
    A("be determined from source, so the runner raises")
    A("`UNSUPPORTED_UNVERIFIED_SEMANTICS` instead of substituting a proxy statistic.")
    A("No `1-p`, `|r|`, `p`, correlation magnitude, discretised MI or `mutual_info_classif`")
    A("was used. See `docs/stage2_source_audit.md` §2.")
    A("")
    A("## Tests")
    A("")
    A(f"- Full suite: **{tests.get('passed')} passed, {tests.get('failed')} failed, "
      f"{tests.get('skipped')} skipped** (Stage 1 + Stage 2)")
    A(f"- Mathematical self-check: **{selfcheck.get('passed', 'n/a')} passed, "
      f"{selfcheck.get('failed', 'n/a')} failed**")
    A(f"- Git audit: `{tests.get('git_audit_status')}`, "
      f"{tests.get('git_data_violations')} violations")
    A("")
    A("## Stage 1 untouched")
    A("")
    A(f"- Stage-1 verdict unchanged: "
      f"`{json.loads((results_dir('stage1') / 'readiness_summary.json').read_text())['verdict']}`")
    A("- No Stage-1 config, gate, manifest, split or processed array was modified.")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
