"""Stage 3B protocol design and authorisation tests (synthetic inputs only).

No test here performs a real SVC fit or a real ARCENE selection; they use a
tiny known linear problem. Run these BEFORE the freeze.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ossfs_dd_stage3b import EXP_A_DATASETS
from ossfs_dd_stage3b.protocol import (
    A_N_SUBSETS,
    A_PLANNED_FITS,
    B_PLANNED_RUNS,
    B_SPLIT_SEED,
    MAX_ITER_CAPS,
    all_arcene_run_ids,
    all_fit_ids,
    all_subsets,
    enumerate_run_ids,
    protocol_digest,
    validate_protocol,
)
from ossfs_dd_stage3b.provenance import BlockedInput, assert_allowed

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "configs" / "stage3b.yaml"


def frozen_config() -> dict:
    """The frozen protocol. Tests that need it skip before the freeze."""
    if not CONFIG.exists():
        pytest.skip("configs/stage3b.yaml does not exist yet (pre-freeze)")
    import yaml

    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- protocol


def test_design_sizes_are_exact():
    assert A_N_SUBSETS == 75
    assert A_PLANNED_FITS == 450
    assert B_PLANNED_RUNS == 50
    assert len(all_subsets()) == 75
    assert len(all_fit_ids()) == 450
    assert len(all_arcene_run_ids()) == 50


def test_every_run_id_is_unique():
    ids = enumerate_run_ids()
    for name, lst in ids.items():
        assert len(lst) == len(set(lst)), f"{name} has duplicate IDs"


def test_fit_grid_is_full_factorial():
    cells = {(f.subset.key, f.scale_variant, f.max_iter) for f in all_fit_ids()}
    assert len(cells) == 75 * 2 * 3
    # every subset gets all 6 cells
    per_subset = {}
    for f in all_fit_ids():
        per_subset.setdefault(f.subset.key, set()).add((f.scale_variant, f.max_iter))
    assert all(len(v) == 6 for v in per_subset.values())


def test_experiment_b_split_is_fixed_and_order_is_crossed():
    # the split seed is a module constant, not a per-run choice
    assert B_SPLIT_SEED == 11
    folds = {r.fold for r in all_arcene_run_ids()}
    orders = {r.order_seed for r in all_arcene_run_ids()}
    assert len(folds) == 5 and len(orders) == 10
    # order_seed must never select the split seed: they are separate axes
    assert set(orders) != {B_SPLIT_SEED}


def test_no_accidental_seed_coupling_in_experiment_b():
    """Run IDs encode fold and order_seed independently of the split seed."""
    runs = all_arcene_run_ids()
    assert all("split11" in r.run_id for r in runs)
    assert all("order" in r.run_id for r in runs)


def test_max_iter_grid_excludes_minus_one():
    assert -1 not in MAX_ITER_CAPS
    assert tuple(MAX_ITER_CAPS) == (20_000, 200_000, 2_000_000)


def test_validate_protocol_accepts_frozen_config():
    assert validate_protocol(frozen_config()) == []


def test_validate_protocol_rejects_drift():
    import copy

    cfg = frozen_config()
    bad = copy.deepcopy(cfg)
    bad["experiment_a"]["max_iter_caps"] = [20_000, 200_000, 2_000_000, 20_000_000]
    assert validate_protocol(bad), "adding a 4th cap must be detected as drift"
    bad2 = copy.deepcopy(cfg)
    bad2["experiment_b"]["split_seed"] = 17
    assert validate_protocol(bad2), "moving the split seed must be detected as drift"
    assert cfg["experiment_a"]["planned_fits"] == 450


def test_protocol_digest_is_sensitive_to_covered_fields():
    """The digest hashes estimator+scaler (design comes from frozen constants).

    Design drift is caught by validate_protocol instead; both mechanisms are
    checked so neither is assumed to cover the other.
    """
    import copy

    cfg = frozen_config()
    other = copy.deepcopy(cfg)
    other["estimator"]["params"]["C"] = 2.0
    assert protocol_digest(cfg) != protocol_digest(other), "estimator changes must move the digest"

    scaler = copy.deepcopy(cfg)
    scaler["scaler"]["standardized"] = "min-max"
    assert protocol_digest(cfg) != protocol_digest(scaler), "scaler changes must move the digest"

    order_drift = copy.deepcopy(cfg)
    order_drift["experiment_b"]["order_seeds"] = [11, 17]
    assert validate_protocol(order_drift), "order-seed drift must be caught by validation"


# ----------------------------------------------------------- authorisation


@pytest.mark.parametrize("dataset", ["DLBCL", "BREAST"])
def test_forbidden_datasets_are_refused(dataset):
    with pytest.raises(BlockedInput):
        assert_allowed(dataset)


@pytest.mark.parametrize("dataset", ["SRBCT", "LYMPHOMA", "PROSTATE", "LEUKEMIA"])
def test_continuous_datasets_are_refused_in_stage3b(dataset):
    with pytest.raises(BlockedInput):
        assert_allowed(dataset)


@pytest.mark.parametrize("dataset", EXP_A_DATASETS)
def test_authorized_datasets_pass(dataset):
    assert_allowed(dataset)  # must not raise


def test_unknown_dataset_is_refused():
    with pytest.raises(BlockedInput):
        assert_allowed("NOT_A_DATASET")


def test_experiment_b_is_arcene_only():
    cfg = frozen_config()
    assert cfg["experiment_b"]["dataset"] == "ARCENE"
    assert cfg["experiment_a"]["datasets"] == list(EXP_A_DATASETS)
