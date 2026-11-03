#!/usr/bin/env python
"""Stage 3D V2 runner: explicitly versioned, bounded synthetic smoke.

Uses V2 factories and finalize_stream() exactly once. Never changes V1 defaults.
Writes versioned V2 records under $RESEARCH_OUTPUT_ROOT/.../stage3d_v2/.

Usage:
    uv run python scripts/stage3d_v2_run.py --smoke --run-id <RUN_ID>
    uv run python scripts/stage3d_v2_run.py --smoke --run-id <RUN_ID> --repeat  (determinism)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ossfs_dd_stage3d.scheduler import batch_size_for_p, make_schedule  # noqa: E402
from ossfs_dd_stage3d.selector import (  # noqa: E402
    make_discrete_batch_v2,
)

PROJECT = Path(__file__).resolve().parents[1]
OUT_ROOT = Path(os.environ["RESEARCH_OUTPUT_ROOT"]) / "ossfs-dd-reproduction" / "stage3d_v2"


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def deliver_batches(X, flattened, B, labels):
    """PRODUCTION mapping: original feature ID -> its own train-only column.

    Returns (selector, delivered_ids). Exposed so tests exercise this exact
    code path rather than a re-implementation.
    """
    bs, _sf, _jf = make_discrete_batch_v2(np.asarray(labels))
    i = 0
    delivered: list[int] = []
    while i < len(flattened):
        chunk = [int(j) for j in flattened[i:i + B]]
        cols = [np.asarray(X)[:, j].copy() for j in chunk]
        bs.consume_batch(chunk, cols)
        delivered.extend(chunk)
        i += B
    bs.finalize_stream()
    return bs, delivered


def run_smoke(run_id: str, *, repeat: bool = False, X=None, y=None,
              dataset: str = "SYNTHETIC", n_features: int = 30) -> dict:
    """Run a bounded synthetic V2 smoke. Returns a manifest dict.

    ``X``/``y`` may be injected so tests exercise the production path with an
    encodable matrix; defaults reproduce the original synthetic fixture.
    """
    out_dir = OUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)

    if X is None:
        # Synthetic data: 200 samples, 30 features, binary
        rng = np.random.default_rng(42)
        n = 200
        y = np.where(np.arange(n) < n // 2, "a", "b")
        sig = (y == "a").astype(float)
        X = np.column_stack([sig * 2 + rng.normal(0, 0.5, n) for _ in range(5)]
                            + [rng.normal(0, 1, n) for _ in range(25)])
    if y is None:
        n = X.shape[0]
        y = np.where(np.arange(n) < n // 2, "a", "b")
    n_features = int(X.shape[1])
    order = np.arange(n_features)
    sched = make_schedule(order, dataset, 11, 0)
    flattened = sched["flattened_order"]
    B = batch_size_for_p(30)

    # discrete V2 smoke via the production delivery path
    import time as _t
    t0 = _t.perf_counter()
    bs, delivered = deliver_batches(X, flattened, B, np.asarray(y))
    elapsed = _t.perf_counter() - t0
    if sorted(delivered) != list(range(n_features)):
        raise RuntimeError("H-MAP: coverage/duplicate failure")

    manifest = {
        "schema": "ossfs-dd-reproduction/stage3d_v2/smoke/1.0",
        "run_id": run_id,
        "utc": utcnow(),
        "method_version": "batch_score_shared_v2",
        "source_fingerprint": sha256_file(PROJECT / "src" / "ossfs_dd_stage3d" / "selector.py"),
        "config_hash": None,
        "dataset": "SYNTHETIC",
        "n_features": 30,
        "batch_size": B,
        "n_batches": bs.state.n_batches,
        "n_features_seen": bs.state.arrival_count,
        "n_selected": len(bs.candidates.feature_ids()),
        "selected_feature_ids": [int(x) for x in bs.candidates.feature_ids()],
        "pool_flushes": bs.state.pool_flushes,
        "pending_pool_end": len(bs.pool),
        "finalized": getattr(bs, "_finalized", False),
        "seconds": round(elapsed, 4),
        "repeat": repeat,
        "test_rows_read": 0,
        "classifier_fitted": False,
    }
    (out_dir / "v2_smoke_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--repeat", action="store_true")
    args = ap.parse_args()

    manifest = run_smoke(args.run_id, repeat=args.repeat)
    print(f"  V2 smoke: method_version={manifest['method_version']}")
    print(f"  n_features={manifest['n_features']} n_selected={manifest['n_selected']} "
          f"n_batches={manifest['n_batches']} pool_flushes={manifest['pool_flushes']}")
    print(f"  finalized={manifest['finalized']} pending_pool={manifest['pending_pool_end']}")
    print(f"  selected_ids={manifest['selected_feature_ids']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
