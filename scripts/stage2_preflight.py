"""Stage 2 preflight — snapshot the environment and fail closed on violations.

Read-only. Prints sanitised paths (never secrets), records Git state and
Stage-1 hashes, and refuses to proceed if an unauthorized or unverified dataset
would be touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ossfs_dd.config import load_stage2_config  # noqa: E402
from ossfs_dd.runner import TABLE1, AuthorizationError, authorize  # noqa: E402
from ossfs_dd_data.workspace import (  # noqa: E402
    data_root,
    project_root,
    rel_to_research,
    research_root,
    results_dir,
    utc_now,
    write_json,
)


def _git(*args: str) -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "-C", str(project_root()), *args],
            capture_output=True, text=True, timeout=30, check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def snapshot_stage1_data() -> dict[str, str]:
    """Hash every Stage-1 artifact Stage 2 is allowed to read. Read-only."""
    out: dict[str, str] = {}
    root = data_root()
    if not root.exists():
        return out
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root))] = _sha256(p)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 2 preflight")
    ap.add_argument("--write", action="store_true", help="persist preflight snapshot")
    args = ap.parse_args(argv)

    plan = load_stage2_config()

    print("=== STAGE 2 PREFLIGHT ===")
    print(f"  workspace root : {research_root()}")
    print(f"  project root   : {project_root()}")
    print(f"  data root      : {data_root()}")
    print(f"  architecture   : {platform.machine()} / {platform.system()}")
    print(f"  python         : {sys.version.split()[0]} ({platform.python_implementation()})")
    print(f"  in project venv: "
          f"{str(Path(sys.executable).resolve()) == str((project_root()/'.venv'/'bin'/'python').resolve())}")
    print("  GPU used       : False (CPU-only by configuration)")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"  git HEAD       : {_git('rev-parse', 'HEAD') or 'not a repository'}")
    print(f"  git branch     : {_git('rev-parse', '--abbrev-ref', 'HEAD')}")
    print(f"  git dirty files: {len(_git('status', '--porcelain').splitlines())}")
    print(f"  primary mode   : {plan.primary_mode.value}")
    print(f"  seed / fold    : {plan.seed} / {plan.fold}")

    print("\n=== AUTHORIZED DATASETS (fail closed) ===")
    failures: list[str] = []
    rows = []
    for key in plan.allowlist:
        try:
            a = authorize(key, plan)
            rows.append({
                "dataset": key, "status": "AUTHORIZED",
                "manifest_status": a.manifest_status,
                "paper_feature_type": a.paper_feature_type,
                "metric_family": a.metric_family.value,
                "semantics_status": a.semantics_status,
                "expected": f"{a.n_samples}x{a.n_features}x{a.n_classes}",
            })
            print(f"  {key:9s} OK  {a.manifest_status:14s} type={a.paper_feature_type:8s} "
                  f"route={a.metric_family.value:22s} semantics={a.semantics_status}")
        except AuthorizationError as exc:
            failures.append(str(exc))
            print(f"  {key:9s} DENIED: {exc}")

    for key in plan.forbidden:
        try:
            authorize(key, plan)
            failures.append(f"forbidden dataset {key} was NOT rejected")
            print(f"  {key:9s} *** FORBIDDEN DATASET WAS NOT REJECTED ***")
        except AuthorizationError:
            print(f"  {key:9s} correctly refused (forbidden)")

    results_dir("stage2")
    stage1 = json.loads(
        (results_dir("stage1") / "readiness_summary.json").read_text(encoding="utf-8")
    )
    print("\n=== STAGE 1 STATE (must remain unchanged by Stage 2) ===")
    print(f"  verdict  : {stage1['verdict']}")
    print(f"  failed   : {stage1['failed_gates']}")

    snap = {
        "schema": "ossfs-dd-reproduction/stage2/preflight/1.0",
        "captured_utc": utc_now(),
        "environment": {
            "workspace_root": str(research_root()),
            "project_root": str(project_root()),
            "data_root": str(data_root()),
            "architecture": platform.machine(),
            "python": sys.version.split()[0],
            "gpu_used": False,
        },
        "git": {
            "head": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_lines": len(_git("status", "--porcelain").splitlines()),
            "diff_stat": _git("diff", "--stat"),
        },
        "stage1": {"verdict": stage1["verdict"], "failed_gates": stage1["failed_gates"]},
        "plan": {"primary_mode": plan.primary_mode.value, "seed": plan.seed, "fold": plan.fold},
        "authorization": rows,
        "authorization_failures": failures,
        "table1_reference": {k: list(v) for k, v in TABLE1.items()},
        "stage1_data_sha256": snapshot_stage1_data(),
    }
    if args.write:
        out = write_json(results_dir("stage2") / "preflight.json", snap)
        print(f"\n  wrote {rel_to_research(out)}")

    if failures:
        print(f"\nPREFLIGHT FAILED: {len(failures)} authorization problem(s)")
        return 2
    print("\nPREFLIGHT OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
