"""The frozen Stage-3A protocol: which selections and which outcomes exist.

Three invariants are enforced here rather than trusted to the runner:

* exactly the five Stage-3A datasets, and never ``DLBCL``/``BREAST``;
* one ``arrival_order`` per dataset/seed, re-used by that seed's five folds;
* the planned 250 selections and 750 classifier outcomes are derived from the
  protocol, not hard-coded, and every derived run ID is unique.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

STAGE3A_DATASETS = ("ARCENE", "DEXTER", "MADELON", "GINA", "GISETTE")
STAGE3A_FORBIDDEN = ("DLBCL", "BREAST")
CONTINUOUS_DATASETS = ("SRBCT", "LYMPHOMA", "PROSTATE", "LEUKEMIA")

SEEDS = (11, 17, 23, 31, 47, 59, 71, 83, 97, 109)
FOLDS = (0, 1, 2, 3, 4)
CLASSIFIERS = ("knn", "svm", "cart")

PLANNED_SELECTIONS = len(STAGE3A_DATASETS) * len(SEEDS) * len(FOLDS)
PLANNED_OUTCOMES = PLANNED_SELECTIONS * len(CLASSIFIERS)


@dataclass(frozen=True)
class RunSpec:
    """One (dataset, seed, fold) selection plus its three classifier outcomes."""

    dataset: str
    seed: int
    fold: int

    @property
    def run_id(self) -> str:
        return f"{self.dataset}_seed{self.seed}_fold{self.fold}"

    @property
    def order_key(self) -> tuple[str, int]:
        """The stream order a run consumes: one order per dataset and seed.

        The five folds of a seed therefore share a feature order and are *not*
        five independent permutations.
        """
        return (self.dataset, self.seed)

    @property
    def record_id(self) -> str:
        return f"{self.run_id}__{MODE_TAG}"

    def as_dict(self) -> dict[str, Any]:
        return {"dataset": self.dataset, "seed": self.seed, "fold": self.fold, "run_id": self.run_id}


MODE_TAG = "matlab_compatible_v1"
MODE_ID = MODE_TAG


def all_runs() -> list[RunSpec]:
    """The 250 planned selections, in a deterministic order."""
    return [
        RunSpec(dataset, seed, fold)
        for dataset in STAGE3A_DATASETS
        for seed in SEEDS
        for fold in FOLDS
    ]


def assert_protocol(config: dict[str, Any]) -> None:
    """Fail closed when the frozen config and this module disagree."""
    protocol = config["protocol"]
    allow = tuple(config["authorization"]["allowlist"])
    if allow != STAGE3A_DATASETS:
        raise ValueError(f"allowlist {allow} != frozen Stage-3A set {STAGE3A_DATASETS}")
    forbidden = set(config["authorization"]["forbidden"])
    if not forbidden >= set(STAGE3A_FORBIDDEN):
        raise ValueError(f"forbidden set {forbidden} must include {STAGE3A_FORBIDDEN}")
    overlap = set(allow) & set(CONTINUOUS_DATASETS)
    if overlap:
        raise ValueError(f"continuous datasets must not appear in Workstream A: {overlap}")
    if set(protocol["seeds"]) != set(SEEDS) or set(protocol["folds"]) != set(FOLDS):
        raise ValueError("protocol seeds/folds differ from the frozen Stage-1 protocol")
    if int(protocol["planned_selector_evaluations"]) != PLANNED_SELECTIONS:
        raise ValueError(
            f"planned selections {protocol['planned_selector_evaluations']} != {PLANNED_SELECTIONS}"
        )
    if int(protocol["planned_classifier_outcomes"]) != PLANNED_OUTCOMES:
        raise ValueError(
            f"planned outcomes {protocol['planned_classifier_outcomes']} != {PLANNED_OUTCOMES}"
        )
    if int(config["selection"]["pool_size"]) <= 0:
        raise ValueError("pool_size must be positive")


def assert_run_allowed(spec: RunSpec, config: dict[str, Any]) -> None:
    """Per-run authorization, including the continuous-dataset separation."""
    if spec.dataset in config["authorization"]["forbidden"]:
        raise PermissionError(f"{spec.dataset} is forbidden")
    if spec.dataset in CONTINUOUS_DATASETS:
        raise PermissionError(
            f"{spec.dataset} is a continuous Workstream-B dataset and must not be read by Stage 3A"
        )
    if spec.dataset not in config["authorization"]["allowlist"]:
        raise PermissionError(f"{spec.dataset} is not in the Stage-3A allowlist")
    if spec.seed not in SEEDS or spec.fold not in FOLDS:
        raise ValueError(f"({spec.seed},{spec.fold}) is outside the frozen protocol")


def protocol_digest(config: dict[str, Any]) -> str:
    """Stable digest of the frozen protocol, recorded in every artifact."""
    payload = {
        "datasets": list(STAGE3A_DATASETS),
        "seeds": list(SEEDS),
        "folds": list(FOLDS),
        "classifier_names": list(CLASSIFIERS),
        "mode": MODE_ID,
        "selection": config["selection"],
        "classifier_config": config["classifiers"],
        "metrics": config["metrics"],
    }
    blob = repr(sorted(payload.items(), key=lambda kv: kv[0])).encode()
    return hashlib.sha256(blob).hexdigest()
