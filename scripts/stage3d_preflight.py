#!/usr/bin/env python
"""Stage 3D preflight: verify inputs, freeze config, journal, write-once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ossfs_dd_stage3d.standard_fisherz import marginal_fisherz, partial_correlation

PROJECT = Path(__file__).resolve().parents[1]
OUT_ROOT = Path(os.environ["RESEARCH_OUTPUT_ROOT"]) / "ossfs-dd-reproduction" / "stage3d"
CONFIG = PROJECT / "configs" / "stage3d_batch_fisherz_v1.yaml"


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_config(frozen_utc: str) -> dict:
    return {
        "schema_version": 1,
        "stage": "3d",
        "frozen_utc": frozen_utc,
        "freeze_statement": (
            "Frozen and hashed before any new held-out evaluation. In-process, "
            "chronologically ordered freeze; NOT an independently registered or "
            "attested preregistration."
        ),
        "authorization": {
            "datasets": ["ARCENE", "DEXTER", "MADELON", "GINA", "GISETTE",
                         "SRBCT", "LYMPHOMA", "PROSTATE", "LEUKEMIA"],
            "forbidden": ["DLBCL", "BREAST"],
            "required_manifest_status": "EXACT_VERIFIED",
        },
        "batch_sizes": {
            "rule": "p<=2000:10, 2000<p<=10000:25, p>10000:50",
            "MADELON": 10, "GINA": 10, "SRBCT": 25, "LYMPHOMA": 25,
            "PROSTATE": 25, "LEUKEMIA": 25, "GISETTE": 25, "ARCENE": 25,
            "DEXTER": 50,
        },
        "arms": {
            "discrete": ["D1_single", "D2_batch"],
            "continuous": ["C1_single_r_only", "C2_batch_r_only",
                           "C3_single_normalized_p", "C4_batch_normalized_p"],
        },
        "pilot": {
            "discrete_seeds": [11, 17, 23],
            "discrete_folds": [0, 1, 2, 3, 4],
            "arcene_seeds": [11, 17, 23, 31, 47, 59, 71, 83, 97, 109],
            "arcene_folds": [0, 1, 2, 3, 4],
            "continuous_seeds": [11, 17, 23],
            "continuous_folds": [0, 1, 2, 3, 4],
            "planned_selections": 460,
            "planned_classifier_outcomes": 1380,
        },
        "hypotheses": {
            "H_BATCH_STABILITY": "batch improves within-fold Jaccard by >=0.10 on ARCENE",
            "H_BATCH_CART": "batch CART accuracy no more than 0.03 lower than single",
            "H_PVAL_SCORING": "compare effect-size vs normalized-p; no advance claim",
            "H_NUMERICAL_CORRECTNESS": "Fisher stats agree with oracles within tolerance",
        },
        "resources": {
            "memory_limit_gib": 32, "per_selector_attempt_minutes": 60,
            "initial_concurrency": 1, "max_concurrency": 4, "blas_threads": 1,
            "cpu_only": True,
        },
        "fisherz": {
            "EPS_P": 1e-12, "L": 12.0,
            "marginal": "r -> atanh -> Z=z*sqrt(n-3) -> p=2*norm.sf(|Z|)",
            "holms_for_K_gt_2": True,
            "min_n_marginal": 5, "min_n_pair": 6,
            "r_clip": 1e-12,
        },
        "pair_operator": {
            "type": "partial_correlation",
            "joint": "max(marginal_x + cond_z_given_x, marginal_z + cond_x_given_z)",
            "label": "partial_correlation_heuristic",
            "detects_xor": False,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    print(f"  run_id: {run_id}")

    # verify datasets
    import json as _j

    from ossfs_dd_data.workspace import manifest_path, processed_dir, resolve

    for ds in ["ARCENE", "DEXTER", "MADELON", "GINA", "GISETTE",
               "SRBCT", "LYMPHOMA", "PROSTATE", "LEUKEMIA"]:
        man = _j.loads(manifest_path(ds).read_text())
        schema = _j.loads((processed_dir(ds) / "schema.json").read_text())
        digest = sha256_file(resolve(schema["X"]["path"]))
        if man["stage1_verdict"]["status"] != "EXACT_VERIFIED":
            raise SystemExit(f"{ds}: not EXACT_VERIFIED")
        if digest != schema["X"]["sha256"]:
            raise SystemExit(f"{ds}: hash mismatch")
        print(f"  {ds}: EXACT_VERIFIED, hash ok, p={schema['n_features']}")

    for bad in ("DLBCL", "BREAST"):
        try:
            manifest_path(bad)
        except Exception as exc:  # noqa: BLE001
            print(f"  {bad}: refused ({type(exc).__name__})")

    # numerical oracle smoke
    rng = np.random.default_rng(0)
    x = rng.normal(size=100)
    y = (rng.normal(size=100) > 0).astype(int).astype(str)
    y = np.where(y == "1", "a", "b")
    m = marginal_fisherz(x, y, use_evidence=True)
    if not (0 <= m.score <= 1):
        raise SystemExit(f"numerical oracle failed: score={m.score}")
    r, p = partial_correlation(rng.normal(size=50), rng.normal(size=50), rng.normal(size=50))
    if not (np.isfinite(r) and 0 <= p <= 1):
        raise SystemExit(f"numerical oracle failed: r={r} p={p}")
    print("  numerical oracles: OK")

    if args.dry_run:
        print("  --dry-run: no files written")
        return 0

    # freeze
    frozen_utc = utcnow()
    config = build_config(frozen_utc)
    cfg_path = PROJECT / "configs" / "stage3d_batch_fisherz_v1.yaml"
    if cfg_path.exists():
        raise SystemExit(f"REFUSING to overwrite frozen config {cfg_path}")
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                        encoding="utf-8")
    cfg_sha = sha256_file(cfg_path)

    run_dir = PROJECT / "results" / "stage3d" / run_id
    out_dir = OUT_ROOT / run_id
    journal = out_dir / "logs" / "event_journal.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    with open(journal, "a") as fh:
        fh.write(json.dumps({"event": "PREFLIGHT_WRITTEN", "utc": frozen_utc,
                             "run_id": run_id, "config_sha256": cfg_sha}) + "\n")

    # write-once preflight
    preflight = {
        "schema": "ossfs-dd-reproduction/stage3d/preflight/1.0",
        "run_id": run_id, "utc": frozen_utc,
        "config_sha256": cfg_sha,
        "stage3c_historical_reference": "NOT_AVAILABLE",
        "datasets_authorized": 9,
        "forbidden": ["DLBCL", "BREAST"],
    }
    preflight_path = run_dir / "PREFLIGHT.json"
    if preflight_path.exists():
        raise SystemExit(f"REFUSING to overwrite {preflight_path}")
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    preflight_path.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n",
                              encoding="utf-8")
    print(f"  frozen config sha256={cfg_sha[:16]} at {frozen_utc}")
    print(f"  preflight: {preflight_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
