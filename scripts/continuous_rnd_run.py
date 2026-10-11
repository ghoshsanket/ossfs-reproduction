#!/usr/bin/env python
"""Workstream B: one train-only fold-0 diagnostic per continuous dataset.

This is RESEARCH, NOT reproduction. The authors' ``Fisher_Ztest`` helper is
absent from their repository, so ``configs/continuous_rnd_v1.yaml`` declares our
own reconstruction. Everything this script writes is labelled
``continuous_reconstructed_fisherz_v1`` and
``RECONSTRUCTED_NOT_SOURCE_VERIFIED``.

What this script deliberately does NOT do, per the frozen config:

* it never reads a test row (the stream is built over training indices only);
* it never fits a classifier or reports accuracy;
* it never sweeps seeds or folds (seed 11, fold 0, one run per dataset);
* it never tunes anything after seeing the result.

Usage:
    uv run python scripts/continuous_rnd_run.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ossfs_dd.config import Mode, SelectionConfig  # noqa: E402
from ossfs_dd.runner import load_dataset  # noqa: E402
from ossfs_dd_continuous import ALLOWLIST, MODE_ID, SEMANTICS_STATUS  # noqa: E402
from ossfs_dd_continuous.authorization import authorize_continuous  # noqa: E402
from ossfs_dd_continuous.selector import (  # noqa: E402
    CONTINUOUS_METRIC_FAMILY,
    PAIR_LABEL,
    ContinuousReconstructedSelector,
)
from ossfs_dd_data.stream import FeatureStream  # noqa: E402
from ossfs_dd_data.workspace import splits_dir, stream_orders_dir  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "configs" / "continuous_rnd_v1.yaml"
OUT_ROOT = PROJECT / "results" / "continuous_rnd"
SEED, FOLD = 11, 0


def diagnose(dataset: str, config: dict) -> dict:
    """One train-only selection with the reconstructed continuous estimator."""
    auth = authorize_continuous(dataset, seed=SEED, fold=FOLD)
    X, y, _ = load_dataset(dataset)
    split = json.loads(
        (splits_dir(dataset) / f"seed_{SEED}.json").read_text(encoding="utf-8")
    )
    train_idx = np.asarray(split["folds"][FOLD]["train"], dtype=np.int64)
    rec = json.loads(
        (stream_orders_dir(dataset) / f"seed_{SEED}.json").read_text(encoding="utf-8")
    )
    if rec.get("schema") != "ossfs-dd-reproduction/stage1/stream_order/2.0":
        raise ValueError(f"{dataset}: unexpected stream-order schema {rec.get('schema')!r}")
    order = np.asarray(rec["arrival_order"], dtype=np.int64)

    stream = FeatureStream(X, y, train_idx, order)
    sel_cfg = SelectionConfig(
        mode=Mode.MATLAB_COMPATIBLE,
        k1=float(config["selection_integration"]["k1"]),
        k2=float(config["selection_integration"]["k2"]),
        pool_size=int(config["selection_integration"]["pool_size"]),
    )
    meta = {
        "mode_id": MODE_ID,
        "semantics_status": SEMANTICS_STATUS,
        "metric_family": CONTINUOUS_METRIC_FAMILY,
        "pair_estimator": PAIR_LABEL,
        "source_reference": "OSFS_DD_z2.m (structure only); Fisher_Ztest ABSENT upstream",
        "structure_verified_from_source": [
            "z2.m:41-46 warm-up ACCEPTS the first two features",
            "z2.m:35-37 NaN relevance score -> 0",
            "z2.m:124 / z2.m:152 skip a pair when the helper reports CI==1 or NaN",
        ],
        "scores_reconstructed": True,
    }

    t0 = time.perf_counter()
    selector = ContinuousReconstructedSelector(
        sel_cfg, labels=stream.train_labels, entry_file="z2", mode_metadata=meta
    )
    result = selector.fit_stream(iter(stream))
    seconds = time.perf_counter() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)

    return {
        "dataset": dataset,
        "mode_id": MODE_ID,
        "semantics_status": SEMANTICS_STATUS,
        "pair_estimator": PAIR_LABEL,
        "seed": SEED,
        "fold": FOLD,
        "status": "DIAGNOSTIC_COMPLETED",
        "training_rows_only": True,
        "test_rows_read": 0,
        "classifier_evaluated": False,
        "n_samples": int(auth.n_samples),
        "n_features": int(auth.n_features),
        "n_train_rows": int(train_idx.size),
        "n_features_seen": result.n_features_seen,
        "n_selected": len(result.feature_indices),
        "selected_fraction": len(result.feature_indices) / auth.n_features,
        "selected_feature_ids": result.feature_indices,
        "seconds": round(seconds, 3),
        "peak_rss_gib": round(rss, 3),
        "pair_skips": selector.pair_skips,
        "selection_summary": result.summary(),
        "warnings": result.warnings,
        "mode_metadata": meta,
        "not_a_reproduction_note": (
            "This is a reconstruction experiment. It does not reproduce the "
            "authors' continuous route and does not change Stage 2's UNRESOLVED "
            "verdict for these datasets."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if tuple(config["authorization"]["allowlist"]) != ALLOWLIST:
        raise SystemExit("frozen allowlist disagrees with the package allowlist")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    keys = list(ALLOWLIST)
    if args.limit:
        keys = keys[: args.limit]
    if args.dry_run:
        print(f"  would diagnose {len(keys)} dataset(s), seed {SEED} fold {FOLD}, train-only:")
        for k in keys:
            print(f"    {k}")
        return 0

    results = []
    for k in keys:
        t0 = time.perf_counter()
        try:
            rec = diagnose(k, config)
        except Exception as exc:  # noqa: BLE001 - recorded, never hidden
            rec = {
                "dataset": k, "status": "DIAGNOSTIC_FAILED", "mode_id": MODE_ID,
                "error": f"{type(exc).__name__}: {exc}",
            }
        (OUT_ROOT / f"{k}_seed{SEED}_fold{FOLD}.json").write_text(
            json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        results.append(rec)
        extra = (
            f"n_sel={rec['n_selected']}/{rec['n_features']} skips={rec['pair_skips']}"
            if rec["status"] == "DIAGNOSTIC_COMPLETED"
            else rec.get("error", "")[:100]
        )
        print(f"  {k:9s} {rec['status']:22s} {time.perf_counter()-t0:7.1f}s  {extra}")
    return 0 if all(r["status"] == "DIAGNOSTIC_COMPLETED" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
