"""Compare Stage-3A results with the published discrete reference values.

The published numbers come from Zhou et al. (TKDD 2022, DOI 10.1145/3502737),
Tables 3-7, transcribed into ``configs/paper_reference_discrete.csv`` before the
sweep. They are a DESCRIPTIVE comparison target only.

Three things this module refuses to do:

* it never runs a significance test of our accuracy against a published mean --
  a published point estimate from another machine, library and protocol is not
  a sample we can test against;
* it never calls the published runtimes comparable (different hardware, MATLAB
  vs scikit-learn, no shared timing protocol);
* it never reports a "speedup" or "parity" number.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

PAPER_DOI = "10.1145/3502737"


def load_reference(path: Path) -> dict[str, dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return {row["dataset"]: row for row in csv.DictReader(fh)}


def _f(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    return float(value) if value not in (None, "") else None


def compare(
    aggregate: dict[str, Any], reference: dict[str, dict[str, Any]], variant: str
) -> dict[str, Any]:
    """One row per (dataset, classifier) with the published mean alongside."""
    rows: list[dict[str, Any]] = []
    block = aggregate["variants"][variant]
    for dataset, entry in block["datasets"].items():
        ref = reference.get(dataset)
        for clf, stats in entry["classifiers"].items():
            ours = stats["mean_accuracy"]
            paper_key = f"{clf}_accuracy_paper"
            paper = _f(ref, paper_key) if ref else None
            diff = (ours - paper) if (ours is not None and paper is not None) else None
            rows.append(
                {
                    "dataset": dataset,
                    "classifier": clf,
                    "ours_mean_accuracy": ours,
                    "paper_accuracy": paper,
                    "absolute_difference": diff,
                    "ours_sd_across_seed_means": stats["sd_across_seed_means"],
                    "n_seeds_evaluated": stats["n_seeds_evaluated"],
                    "ours_mean_features": entry["selected_count_mean"],
                    "paper_mean_features": _f(ref, "paper_mean_features") if ref else None,
                    "feature_count_difference": (
                        (entry["selected_count_mean"] - float(ref["paper_mean_features"]))
                        if ref and ref.get("paper_mean_features")
                        else None
                    ),
                    "n_nonconverged_runs": stats["n_nonconverged_runs"],
                }
            )
    return {
        "source": f"DOI {PAPER_DOI}, Tables 3-7",
        "status": "DESCRIPTIVE_ONLY",
        "statistical_test_against_published_mean": "FORBIDDEN_AND_NOT_PERFORMED",
        "runtime_comparability": "NONE",
        "runtime_note": (
            "Published timings were produced on an AMD 3700X / 32 GB machine using "
            "MATLAB, with no shared timing protocol. They are displayed as context "
            "only and are NOT evidence of speedup, parity or resource efficiency."
        ),
        "our_runtime_seconds_mean": {
            dataset: entry["selection_seconds_mean"]
            for dataset, entry in block["datasets"].items()
        },
        "paper_runtime_seconds": {
            dataset: _f(ref, "paper_runtime_seconds") for dataset, ref in reference.items()
        },
        "rows": rows,
    }
