#!/usr/bin/env python
"""Run the Stage-3A Workstream-A sweep: 250 selections, 750 classifier outcomes.

The sweep is resumable. Each completed (dataset, seed, fold) selection is
written to its own immutable JSON record keyed by the run ID; a resumed sweep
reuses a record only when the frozen protocol digest and the source code hash
still match, so changing the code invalidates cached results rather than
silently mixing generations.

No test-fold evaluation happens anywhere in this script beyond the single
classifier step that the Stage-3A protocol requires; selection itself only
ever sees training rows.

Usage:
    uv run python scripts/stage3a_run.py [--workers N] [--limit N] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ossfs_dd_eval.evaluate import evaluate_run  # noqa: E402
from ossfs_dd_eval.ledger import Ledger  # noqa: E402
from ossfs_dd_eval.protocol import (  # noqa: E402
    PLANNED_OUTCOMES,
    PLANNED_SELECTIONS,
    STAGE3A_DATASETS,
    RunSpec,
    all_runs,
    assert_protocol,
    protocol_digest,
)
from ossfs_dd_eval.provenance import (  # noqa: E402
    code_hash,
    dataset_fingerprints,
    package_versions,
)

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "stage3a.yaml"
OUT_ROOT = Path(__file__).resolve().parents[1] / "results" / "stage3a"


def _worker_init() -> None:
    """Pin BLAS threads per process so N workers do not oversubscribe cores."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        import os

        os.environ[var] = "1"


def _run_one(spec_tuple: tuple[str, int, int]) -> tuple[str, str | None, float]:
    """One sweep task, fully self-contained so it is picklable across processes.

    The config, protocol digest, code hash and ledger are rebuilt inside the
    worker rather than inherited from the parent, because a
    ``ProcessPoolExecutor`` worker does not share module globals with it.
    """
    dataset, seed, fold = spec_tuple
    spec = RunSpec(dataset, seed, fold)
    t0 = time.perf_counter()
    try:
        config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        assert_protocol(config)
        ledger = Ledger(
            OUT_ROOT / "runs",
            protocol_digest=protocol_digest(config),
            code_hash=code_hash(),
        )
        record = evaluate_run(spec, config)
        ledger.save(record)
    except Exception as exc:  # noqa: BLE001 - a failed run is recorded, not hidden
        return spec.run_id, f"{type(exc).__name__}: {exc}", time.perf_counter() - t0
    return spec.run_id, None, time.perf_counter() - t0


def run_sweep(*, workers: int, limit: int | None, dry_run: bool) -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert_protocol(config)
    digest = protocol_digest(config)
    chash = code_hash()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(OUT_ROOT / "runs", protocol_digest=digest, code_hash=chash)

    manifest = {
        "schema": "ossfs-dd-reproduction/stage3a/manifest/1.0",
        "protocol_digest": digest,
        "code_hash": chash,
        "planned_selections": PLANNED_SELECTIONS,
        "planned_outcomes": PLANNED_OUTCOMES,
        "datasets": list(STAGE3A_DATASETS),
        "versions": package_versions(),
        "dataset_fingerprints": dataset_fingerprints(STAGE3A_DATASETS),
        "workers": workers,
    }
    (OUT_ROOT / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    specs = all_runs()
    if limit:
        specs = specs[:limit]
    todo = [s for s in specs if ledger.load(s.run_id) is None]

    print(f"  protocol_digest={digest[:16]} code_hash={chash[:16]}")
    print(f"  ledger already holds {ledger.count()} run(s); {len(todo)} to compute of {len(specs)}")
    if dry_run:
        for s in todo[:5]:
            print(f"    would run {s.run_id}")
        return 0
    if not todo:
        print("  nothing to do")
        return 0

    t0 = time.perf_counter()
    done = 0
    failures: list[tuple[str, str]] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
        futures = {pool.submit(_run_one, (s.dataset, s.seed, s.fold)): s for s in todo}
        for fut in as_completed(futures):
            run_id, err, dt = fut.result()
            done += 1
            if err:
                failures.append((run_id, err))
                print(f"  [{done}/{len(todo)}] {run_id} FAILED: {err[:120]}")
            else:
                print(f"  [{done}/{len(todo)}] {run_id} ok in {dt:.1f}s")
            if done % 10 == 0 or done == len(todo):
                el = time.perf_counter() - t0
                print(
                    f"    progress {done}/{len(todo)} elapsed={el/60:.1f}min "
                    f"eta={(el/done*(len(todo)-done))/60:.1f}min"
                )

    print(f"  finished {done} run(s) in {(time.perf_counter()-t0)/60:.1f} min")
    if failures:
        print(f"  FAILURES ({len(failures)}):")
        for run_id, err in failures[:20]:
            print(f"    {run_id}: {err}")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    return run_sweep(workers=args.workers, limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
