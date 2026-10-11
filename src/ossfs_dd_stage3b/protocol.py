"""The frozen Stage 3B factorial designs and their identifiers.

Everything here is *derived* from the preregistered config so the two cannot
drift: the run-ID lists are generated, never hand-written.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from . import EXP_A_DATASETS, EXP_B_DATASET

SCALE_VARIANTS = ("raw", "standardized")
MAX_ITER_CAPS = (20_000, 200_000, 2_000_000)

# Experiment A
A_SEEDS = (11, 17, 23)
A_FOLDS = (0, 1, 2, 3, 4)
A_N_SUBSETS = len(EXP_A_DATASETS) * len(A_SEEDS) * len(A_FOLDS)          # 75
A_FITS_PER_SUBSET = len(SCALE_VARIANTS) * len(MAX_ITER_CAPS)             # 6
A_PLANNED_FITS = A_N_SUBSETS * A_FITS_PER_SUBSET                         # 450

# Experiment B -- the CV split seed is FIXED; order seeds vary independently.
B_SPLIT_SEED = 11
B_FOLDS = (0, 1, 2, 3, 4)
B_ORDER_SEEDS = (11, 17, 23, 31, 47, 59, 71, 83, 97, 109)
B_PLANNED_RUNS = len(B_FOLDS) * len(B_ORDER_SEEDS)                       # 50

# B-H1 preregistered effect criterion
B_H1_RANGE_THRESHOLD = 500
B_H1_MIN_FOLDS = 3


@dataclass(frozen=True)
class SubsetID:
    dataset: str
    seed: int
    fold: int

    @property
    def key(self) -> str:
        return f"{self.dataset}_seed{self.seed}_fold{self.fold}"

    @property
    def stage3a_run_id(self) -> str:
        return self.key


@dataclass(frozen=True)
class FitID:
    subset: SubsetID
    scale_variant: str
    max_iter: int

    @property
    def fit_id(self) -> str:
        return f"{self.subset.key}__{self.scale_variant}__it{self.max_iter}"


@dataclass(frozen=True)
class ArceneRunID:
    fold: int
    order_seed: int

    @property
    def run_id(self) -> str:
        return f"ARCENE_split{B_SPLIT_SEED}_fold{self.fold}_order{self.order_seed}"


def all_subsets() -> list[SubsetID]:
    return [SubsetID(d, s, f) for d in EXP_A_DATASETS for s in A_SEEDS for f in A_FOLDS]


def all_fit_ids() -> list[FitID]:
    return [
        FitID(sub, sv, cap)
        for sub in all_subsets()
        for sv in SCALE_VARIANTS
        for cap in MAX_ITER_CAPS
    ]


def all_arcene_run_ids() -> list[ArceneRunID]:
    return [ArceneRunID(f, o) for f in B_FOLDS for o in B_ORDER_SEEDS]


def validate_protocol(config: dict[str, Any]) -> list[str]:
    """Fail closed if the frozen config disagrees with this module."""
    problems: list[str] = []
    exp_a = config["experiment_a"]
    exp_b = config["experiment_b"]
    if tuple(exp_a["datasets"]) != EXP_A_DATASETS:
        problems.append(f"experiment_a.datasets {exp_a['datasets']} != {list(EXP_A_DATASETS)}")
    if tuple(exp_a["seeds"]) != A_SEEDS:
        problems.append(f"experiment_a.seeds {exp_a['seeds']} != {list(A_SEEDS)}")
    if tuple(exp_a["folds"]) != A_FOLDS:
        problems.append(f"experiment_a.folds {exp_a['folds']} != {list(A_FOLDS)}")
    if tuple(exp_a["scale_variants"]) != SCALE_VARIANTS:
        problems.append("experiment_a.scale_variants mismatch")
    if tuple(exp_a["max_iter_caps"]) != MAX_ITER_CAPS:
        problems.append("experiment_a.max_iter_caps mismatch")
    if int(exp_a["planned_subsets"]) != A_N_SUBSETS:
        problems.append(f"planned_subsets != {A_N_SUBSETS}")
    if int(exp_a["planned_fits"]) != A_PLANNED_FITS:
        problems.append(f"planned_fits != {A_PLANNED_FITS}")
    if str(exp_b["dataset"]) != EXP_B_DATASET:
        problems.append(
            f"experiment_b.dataset must be {EXP_B_DATASET!r}, got {exp_b['dataset']!r}"
        )
    if int(exp_b["split_seed"]) != B_SPLIT_SEED:
        problems.append("experiment_b.split_seed must be 11 (split is held fixed)")
    if tuple(exp_b["order_seeds"]) != B_ORDER_SEEDS:
        problems.append("experiment_b.order_seeds mismatch")
    if tuple(exp_b["folds"]) != B_FOLDS:
        problems.append("experiment_b.folds mismatch")
    if int(exp_b["planned_runs"]) != B_PLANNED_RUNS:
        problems.append(f"planned_runs != {B_PLANNED_RUNS}")
    if -1 in MAX_ITER_CAPS:
        problems.append("max_iter=-1 is explicitly outside the authorized budget")
    for key in ("DLBCL", "BREAST"):
        if key in exp_a["datasets"] or key == exp_b["dataset"]:
            problems.append(f"forbidden dataset {key} appears in the design")
    return problems


def protocol_digest(config: dict[str, Any]) -> str:
    """Stable digest of the design+config that every record references."""
    payload = {
        "datasets": list(EXP_A_DATASETS),
        "a_seeds": list(A_SEEDS), "a_folds": list(A_FOLDS),
        "scale_variants": list(SCALE_VARIANTS), "max_iter_caps": list(MAX_ITER_CAPS),
        "b_split_seed": B_SPLIT_SEED, "b_folds": list(B_FOLDS),
        "b_order_seeds": list(B_ORDER_SEEDS),
        "b_h1_range_threshold": B_H1_RANGE_THRESHOLD,
        "b_h1_min_folds": B_H1_MIN_FOLDS,
        "estimator": config["estimator"],
        "scaler": config["scaler"],
    }
    blob = repr(sorted(payload.items(), key=lambda kv: kv[0])).encode()
    return hashlib.sha256(blob).hexdigest()


def enumerate_run_ids() -> dict[str, list[str]]:
    return {
        "experiment_a_fit_ids": [f.fit_id for f in all_fit_ids()],
        "experiment_a_subset_ids": [s.key for s in all_subsets()],
        "experiment_b_run_ids": [r.run_id for r in all_arcene_run_ids()],
    }
