"""Preregistered cross-validation and feature-arrival-order plans.

No model is fitted here. These are index plans for a later stage.

Important scientific caveat, recorded in every plan file: the paper describes
5-fold CV with 10 repetitions of randomized feature arrival orders, but does NOT
disclose its splits or seeds. The seeds below are therefore an explicit
reproduction assumption, not a recovery of the original experiment.

No scaling, discretisation, selection or class-conditional processing happens
here. A later stage must fit any preprocessing on training rows only, separately
within each fold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .registry import DatasetSpec

LOG = logging.getLogger("ossfs_dd_data.splits")

GENERATOR_NAME = "numpy.random.Generator"
BIT_GENERATOR_NAME = "PCG64"


@dataclass
class SplitPlan:
    dataset: str
    seed: int
    n_splits: int
    train_indices: list[np.ndarray]
    test_indices: list[np.ndarray]
    n_features: int
    stratified: bool
    stratified_skip_reason: str | None
    class_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "seed": self.seed,
            "n_splits": self.n_splits,
            "n_samples": int(sum(len(t) for t in self.test_indices)),
            "n_features": self.n_features,
            "splitter": "sklearn.model_selection.StratifiedKFold",
            "shuffle": True,
            "random_state": self.seed,
            "stratified": self.stratified,
            "stratified_skip_reason": self.stratified_skip_reason,
            "class_counts": self.class_counts,
            "folds": [
                {
                    "fold": i,
                    "train": t.tolist(),
                    "test": v.tolist(),
                    "n_train": len(t),
                    "n_test": len(v),
                }
                for i, (t, v) in enumerate(zip(self.train_indices, self.test_indices, strict=False))
            ],
            "assumption_notice": (
                "The paper does not disclose its splits or seeds. These indices are a fixed, "
                "preregistered reproduction assumption, not a recovery of the original experiment."
            ),
            "original_source_split_membership_retained": True,
            "note": "Index plan only. No model is fitted and no preprocessing is applied.",
        }


def is_stratification_feasible(labels: np.ndarray, n_splits: int) -> tuple[bool, str | None]:
    """Stratification needs every class to have at least ``n_splits`` members."""
    _, counts = np.unique(labels.astype(str), return_counts=True)
    small = {int(c) for c in counts if c < n_splits}
    if small:
        return False, (
            f"class counts below n_splits={n_splits} ({sorted(small)}); StratifiedKFold "
            "cannot place every class in every fold, so KFold is used instead and the "
            "substitution is recorded rather than hidden"
        )
    return True, None


def build_split_plan(
    spec: DatasetSpec,
    y: np.ndarray,
    seeds: list[int],
    n_splits: int,
) -> list[SplitPlan]:
    plans: list[SplitPlan] = []
    labels = np.asarray(y).astype(str)
    counts = {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True), strict=False)}
    feasible, reason = is_stratification_feasible(labels, n_splits)

    for seed in seeds:
        if feasible:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            train_idx, test_idx = [], []
            for tr, te in splitter.split(np.zeros(labels.shape[0]), labels):
                train_idx.append(np.sort(tr))
                test_idx.append(np.sort(te))
            strat, why = True, None
        else:
            from sklearn.model_selection import KFold

            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            train_idx, test_idx = [], []
            for tr, te in splitter.split(np.zeros(labels.shape[0])):
                train_idx.append(np.sort(tr))
                test_idx.append(np.sort(te))
            strat, why = False, reason
        plans.append(
            SplitPlan(
                dataset=spec.key,
                seed=seed,
                n_splits=n_splits,
                train_indices=train_idx,
                test_indices=test_idx,
                n_features=0,
                stratified=strat,
                stratified_skip_reason=why,
                class_counts=counts,
            )
        )
    LOG.info(
        "%s: built %d split plans (n_splits=%d, stratified=%s)", spec.key, len(plans), n_splits, feasible
    )
    return plans


def build_feature_orders(n_features: int, seeds: list[int]) -> dict[int, np.ndarray]:
    """One independent deterministic permutation of 0..p-1 per seed.

    The permutation is over ORIGINAL feature indices, so the mapping from
    arrival position to source feature identity stays reversible.
    """
    orders: dict[int, np.ndarray] = {}
    for seed in seeds:
        rng = np.random.Generator(np.random.PCG64(seed))
        orders[seed] = rng.permutation(n_features)
    return orders


def feature_order_record(dataset: str, n_features: int, seed: int, order: np.ndarray) -> dict[str, Any]:
    """Build one feature-arrival-order plan record.

    Two distinct mappings are stored and must not be confused:

    ``arrival_order``
        ``arrival_order[position] -> original_feature_index``. The order in
        which features reach the selector.

    ``inverse_position_of_feature``
        ``str(feature_index) -> arrival_position``. The true mathematical
        inverse, so ``inverse_position_of_feature[str(f)] == position`` and
        ``arrival_order[inverse_position_of_feature[str(f)]] == f``.

    A v1 of this record shipped ``arrival_order`` duplicated under the inverse
    key, which carried no additional information and could not be inverted. v2
    stores the real inverse.
    """
    return {
        "schema": "ossfs-dd-reproduction/stage1/stream_order/2.0",
        "dataset": dataset,
        "seed": seed,
        "n_features": int(n_features),
        "generator": GENERATOR_NAME,
        "bit_generator": BIT_GENERATOR_NAME,
        "generator_api": f"numpy.random.Generator(numpy.random.PCG64({seed})).permutation({n_features})",
        "space": "original feature indices 0..p-1",
        "arrival_order": order.tolist(),
        "arrival_order_meaning": "arrival_order[position] -> original_feature_index",
        "inverse_position_of_feature": _inverse(order),
        "inverse_position_of_feature_meaning": (
            "str(feature_index) -> arrival_position; "
            "inverse_position_of_feature[str(f)] == position and "
            "arrival_order[inverse_position_of_feature[str(f)]] == f"
        ),
        "reversible": True,
        "assumption_notice": (
            "The paper does not disclose its feature arrival orders or seeds. This permutation "
            "is a fixed, preregistered reproduction assumption."
        ),
        "note": "Ordering plan only. No selection policy is implemented here.",
    }


def _inverse(order: np.ndarray) -> dict[str, int]:
    """feature_index -> arrival position.

    This is the genuine inverse of ``arrival_order``; the earlier v1 stored the
    forward mapping here instead, which made the two fields indistinguishable.
    """
    return {str(feature_index): int(position) for position, feature_index in enumerate(order.tolist())}


def validate_plan(plans: list[SplitPlan], orders: dict[int, np.ndarray], n_samples: int) -> dict[str, Any]:
    """Structural self-check on the generated plans."""
    problems: list[str] = []
    n_folds = 0
    for plan in plans:
        for tr, te in zip(plan.train_indices, plan.test_indices, strict=False):
            n_folds += 1
            if set(tr.tolist()) & set(te.tolist()):
                problems.append(f"{plan.dataset} seed {plan.seed}: train/test overlap")
            if sorted(tr.tolist() + te.tolist()) != list(range(n_samples)):
                problems.append(
                    f"{plan.dataset} seed {plan.seed}: fold does not partition all {n_samples} rows"
                )
    for seed, order in orders.items():
        if sorted(order.tolist()) != list(range(len(order))):
            problems.append(f"feature order for seed {seed} is not a permutation of 0..p-1")
    return {"ok": not problems, "n_test_folds": n_folds, "problems": problems}
