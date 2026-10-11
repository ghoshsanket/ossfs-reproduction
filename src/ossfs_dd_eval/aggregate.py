"""Aggregate the Stage-3A sweep into per-dataset, per-classifier summaries.

The aggregation order matters and is fixed by the frozen protocol: accuracy is
averaged WITHIN a seed across that seed's five folds first, and the
per-seed means are then averaged across the ten seeds. Averaging all 250 runs
directly would let a dataset with a different fold count weigh more heavily,
though here every dataset has exactly 50 so the two coincide numerically; the
nested form is used because it is the correct one and is checked.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .protocol import CLASSIFIERS, SEEDS, STAGE3A_DATASETS

SCHEMA = "ossfs-dd-reproduction/stage3a/aggregate/1.0"


def load_records(runs_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((runs_dir / "records").glob("*.json"))
    ]


def _finite(values: list[float | None]) -> list[float]:
    return [v for v in values if v is not None and np.isfinite(v)]


def per_seed_means(
    records: list[dict[str, Any]], variant: str
) -> dict[tuple[str, str], dict[str, float | None]]:
    """(dataset, classifier) -> per-seed mean over that seed's folds."""
    acc: dict[tuple[str, str], dict[int, list[float]]] = {}
    for rec in records:
        for out in rec.get(variant, []):
            key = (rec["dataset"], out["classifier"])
            if out["accuracy"] is None:
                continue
            acc.setdefault(key, {}).setdefault(rec["seed"], []).append(float(out["accuracy"]))
    result: dict[tuple[str, str], dict[str, float | None]] = {}
    for key, by_seed in acc.items():
        result[key] = {
            seed: (float(np.mean(v)) if v else None) for seed, v in sorted(by_seed.items())
        }
    return result


def dataset_summary(
    records: list[dict[str, Any]], variant: str
) -> dict[str, Any]:
    """Per dataset: nested seed-mean accuracy, feature counts, non-convergence."""
    psm = per_seed_means(records, variant)
    out: dict[str, Any] = {}
    for dataset in STAGE3A_DATASETS:
        drecs = [r for r in records if r["dataset"] == dataset]
        if not drecs:
            continue
        n_sel = [r["n_selected"] for r in drecs]
        secs = [r["selection_seconds"] for r in drecs]
        entry: dict[str, Any] = {
            "n_runs": len(drecs),
            "n_features_total": drecs[0]["n_features_total"],
            "selected_fraction_mean": float(np.mean(n_sel)) / drecs[0]["n_features_total"],
            "selected_count_mean": float(np.mean(n_sel)),
            "selection_seconds_mean": float(np.mean(secs)),
            "selection_seconds_total": float(np.sum(secs)),
            "peak_rss_gib_max": max(r["peak_rss_gib"] for r in drecs),
            "classifiers": {},
        }
        for clf in CLASSIFIERS:
            seeds = psm.get((dataset, clf), {})
            vals = _finite(list(seeds.values()))
            nonconv = sum(
                1
                for r in drecs
                for o in r.get(variant, [])
                if o["classifier"] == clf and o.get("converged") is False
            )
            entry["classifiers"][clf] = {
                "per_seed_mean": seeds,
                "n_seeds_evaluated": len(vals),
                "mean_accuracy": float(np.mean(vals)) if vals else None,
                "sd_across_seed_means": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                "min_seed_mean": float(np.min(vals)) if vals else None,
                "max_seed_mean": float(np.max(vals)) if vals else None,
                "n_nonconverged_runs": nonconv,
            }
        out[dataset] = entry
    return out


def seed_blocked_bootstrap(
    values: list[float], *, resamples: int = 10000, seed: int = 20261011
) -> dict[str, Any]:
    """Bootstrap over SEED-level means. DESCRIPTIVE ONLY.

    Repeated cross-validation over one dataset is not an independent population,
    so this interval is not a generalisation confidence interval.
    """
    vals = np.asarray(_finite(values), dtype=float)
    if vals.size == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0, "label": "DESCRIPTIVE_ONLY"}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(resamples, vals.size))
    means = vals[idx].mean(axis=1)
    return {
        "mean": float(vals.mean()),
        "lo": float(np.percentile(means, 2.5)),
        "hi": float(np.percentile(means, 97.5)),
        "n": int(vals.size),
        "resamples": resamples,
        "label": "DESCRIPTIVE_ONLY",
        "caution": (
            "Repeated CV over the same dataset is not an independent population; "
            "this is not a generalisation confidence interval."
        ),
    }


def paired_holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (step-down)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (key, p) in enumerate(items):
        val = min(1.0, (m - i) * p)
        running = max(running, val)
        adjusted[key] = running
    return adjusted


def build(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "n_records": len(records),
        "variants": {},
        "seeds": list(SEEDS),
    }
    for variant, label in (
        ("outcomes_raw_primary", "RAW_PRIMARY"),
        ("outcomes_standardized_sensitivity", "STANDARDIZED_SENSITIVITY"),
    ):
        summary = dataset_summary(records, variant)
        boot: dict[str, Any] = {}
        for dataset, entry in summary.items():
            for clf, stats in entry["classifiers"].items():
                key = f"{dataset}:{clf}"
                boot[key] = seed_blocked_bootstrap(
                    list(stats["per_seed_mean"].values()),
                    resamples=int(config["statistics"]["seed_blocked_bootstrap"]["resamples"]),
                    seed=int(config["statistics"]["seed_blocked_bootstrap"]["bootstrap_seed"]),
                )
        payload["variants"][variant] = {
            "label": label,
            "datasets": summary,
            "seed_blocked_bootstrap": boot,
        }
    return payload
