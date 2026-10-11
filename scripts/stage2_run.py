"""Stage 2 run — the restricted nine-dataset smoke, one execution per dataset.

Runs exactly the authorized datasets, seed 11 / fold 0, over the WHOLE feature
stream, in the frozen primary mode. This is 9 executions, not a 9x10x5 sweep.
No classifier is fitted and no accuracy is computed.

Run bundles are written under ``RESEARCH_OUTPUT_ROOT`` (outside Git). Only small
JSON/CSV summaries land in ``results/stage2/``.

Refuses unauthorized names even when the user supplies them explicitly.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ossfs_dd.config import load_stage2_config  # noqa: E402
from ossfs_dd.reference_semantics import (  # noqa: E402
    BEHAVIOUR_LEDGER,
    COMMIT,
    REFERENCE_SEMANTICS,
    REPO,
)
from ossfs_dd.runner import AuthorizationError, run_dataset  # noqa: E402
from ossfs_dd_data.workspace import (  # noqa: E402
    output_root,
    rel_to_research,
    results_dir,
    utc_now,
    write_json,
)

RUNS_DIR = output_root() / "stage2" / "runs"


def run_id() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _git(*a: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), *a],
                              capture_output=True, text=True, timeout=30, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 2 restricted smoke run")
    ap.add_argument("--all", action="store_true", help="run every authorized dataset")
    ap.add_argument("--dataset", action="append", default=[],
                    help="restrict to one authorized key (repeatable)")
    args = ap.parse_args(argv)

    plan = load_stage2_config()
    results_dir("stage2")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    rid = run_id()
    bundle = RUNS_DIR / rid
    bundle.mkdir(parents=True, exist_ok=True)

    targets = args.dataset or plan.allowlist
    for key in targets:
        if key in plan.forbidden or key not in plan.allowlist:
            print(f"REFUSED  {key}: not authorized for Stage 2")
            return 2

    print(f"=== STAGE 2 SMOKE — run_id={rid} mode={plan.primary_mode.value} "
          f"seed={plan.seed} fold={plan.fold} ===")

    records: list[dict] = []
    started = utc_now()
    t_all = time.perf_counter()
    for key in targets:
        t0 = time.perf_counter()
        try:
            rec = run_dataset(key, plan)
        except AuthorizationError as exc:
            rec = {"dataset": key, "status": "AUTHORIZATION_REFUSED", "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 - one dataset must not abort the pass
            rec = {"dataset": key, "status": "UNHANDLED_EXCEPTION",
                   "error": f"{type(exc).__name__}: {exc}"}
        rec["wall_seconds_including_load"] = round(time.perf_counter() - t0, 3)
        records.append(rec)

        write_json(bundle / f"{key}.json", rec)
        status = rec.get("status")
        n_sel = rec.get("n_selected")
        secs = rec.get("seconds")
        print(f"  {key:9s} {status:38s} n_selected={n_sel} secs={secs}")
        if status not in {"COMPLETED", "UNSUPPORTED_UNVERIFIED_SEMANTICS"}:
            print(f"            error: {rec.get('error')}")

    elapsed = round(time.perf_counter() - t_all, 2)
    summary = {
        "schema": "ossfs-dd-reproduction/stage2/run/1.0",
        "run_id": rid,
        "started_utc": started,
        "finished_utc": utc_now(),
        "elapsed_seconds": elapsed,
        "mode": plan.primary_mode.value,
        "seed": plan.seed,
        "fold": plan.fold,
        "whole_feature_stream": True,
        "concurrency": 1,
        "config_path": str(Path(plan.path).name),
        "source": {"repo": REPO, "commit": COMMIT},
        "requested": targets,
        "results": records,
        "counts": _counts(records),
        "no_classifier_fitted": True,
        "no_accuracy_computed": True,
    }
    write_json(bundle / "run_summary.json", summary)
    write_json(results_dir("stage2") / "run_manifest.json", summary)
    write_json(
        results_dir("stage2") / "score_semantics_ledger.json",
        {"schema": "ossfs-dd-reproduction/stage2/score_semantics/1.0",
         "source": {"repo": REPO, "commit": COMMIT},
         "semantics": REFERENCE_SEMANTICS,
         "behaviours": BEHAVIOUR_LEDGER},
    )
    write_json(
        results_dir("stage2") / "implementation_delta_ledger.json",
        {"schema": "ossfs-dd-reproduction/stage2/implementation_delta/1.0",
         "source": {"repo": REPO, "commit": COMMIT},
         "mode": plan.primary_mode.value,
         "deltas": BEHAVIOUR_LEDGER},
    )
    print(f"\n  run bundle: {rel_to_research(bundle)}")
    print(f"  counts: {json.dumps(summary['counts'])}")

    # Exit code reflects whether every REQUESTED dataset finished. The 9/9
    # preregistered gate is evaluated separately by stage2_report.py, so a
    # subset debug run cannot masquerade as satisfying it.
    failed = [r for r in records if r.get("status") not in
              {"COMPLETED", "UNSUPPORTED_UNVERIFIED_SEMANTICS"}]
    if failed:
        print(f"SMOKE RUN INCOMPLETE — {len(failed)} requested dataset(s) did not finish: "
              f"{[r['dataset'] for r in failed]}")
        return 2
    if targets != plan.allowlist:
        print(f"  (subset run: {len(targets)}/{len(plan.allowlist)} authorized datasets; "
              "the 9/9 gate is evaluated by stage2_report.py)")
    return 0


def _counts(records: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in records:
        st = r.get("status", "UNKNOWN")
        out[st] = out.get(st, 0) + 1
    return dict(sorted(out.items()))


if __name__ == "__main__":
    raise SystemExit(main())
