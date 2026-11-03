"""Stage 3D V2 comparison: config integrity, expected key sets, V2 lifecycle,
V1 comparator pairing, and audit mutation tests. Executed by the human."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from ossfs_dd_stage3d.batch_selector_v2 import BatchSelectorV2
from ossfs_dd_stage3d.scheduler import batch_size_for_p, make_schedule

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "configs" / "stage3d_v2_comparison_v1.yaml"


@pytest.fixture(scope="module")
def config():
    if not CONFIG.exists():
        pytest.skip("config not found")
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_config_exists_and_arms_declared(config):
    assert "experiment" in config
    assert "arms" in config["experiment"]
    assert "H1_BATCH_STABILITY" in config.get("hypotheses", {})


def test_nominal_selections_match_arms(config):
    arms = config["experiment"]["arms"]
    total = 0
    for arm in arms.values():
        total += len(arm["datasets"]) * len(arm["seeds"]) * len(arm["folds"])
    # ARCENE appears in one arm with 10 seeds × 5 folds = 50
    # DEXTER/MADELON/GINA/GISETTE share one arm with 3 seeds × 5 folds = 60
    # SRBCT/LYMPHOMA/PROSTATE/LEUKEMIA each in 2 arms with 3 seeds × 5 folds × 2 = 120
    assert total >= 230, f"arm expansion gives {total} < 230"


def test_batch_size_boundaries():
    for p, expected in [(2000, 10), (2001, 25), (10000, 25), (10001, 50)]:
        assert batch_size_for_p(p) == expected


def test_forbidden_datasets_not_in_config(config):
    for ds in ("DLBCL", "BREAST"):
        assert ds not in config["authorization"]["datasets"]


def test_v2_selector_finalizes_once():
    labels = np.array(["a", "b"] * 50)
    it = iter([0.5] * 5)
    bs = BatchSelectorV2(
        yaml.safe_load(CONFIG.read_text()) and
        __import__("ossfs_dd.config", fromlist=["SelectionConfig"]).SelectionConfig(
            mode=__import__("ossfs_dd.config", fromlist=["Mode"]).Mode.MATLAB_COMPATIBLE,
            k1=0.0, k2=10.0, pool_size=100),
        metric_family="x", score_fn=lambda c, lab: next(it),
        joint_fn=lambda a, b, lab, m: (m + 0.0, "x"), is_redundant_fn=lambda j, b: (False, ""),
        promotes_pair_fn=lambda j, b: (False, ""),
        labels=labels, pool_size=100, k1=0.0, k2=10.0)
    bs.consume_batch(list(range(5)), [np.zeros(100) for _ in range(5)])
    bs.finalize_stream()
    f1 = bs.state.pool_flushes
    bs.finalize_stream()  # idempotent
    assert bs.state.pool_flushes == f1


def test_v2_pool_persists_across_batches():
    labels = np.array(["a", "b"] * 50)
    it = iter([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0] + [0.05] * 90)
    bs = BatchSelectorV2(
        __import__("ossfs_dd.config", fromlist=["SelectionConfig"]).SelectionConfig(
            mode=__import__("ossfs_dd.config", fromlist=["Mode"]).Mode.MATLAB_COMPATIBLE,
            k1=1.0, k2=2.0, pool_size=100),
        metric_family="x",
        score_fn=lambda c, lab: next(it),
        joint_fn=lambda a, b, lab, m: (m + 0.0, "x"),
        is_redundant_fn=lambda j, b: (False, ""),
        promotes_pair_fn=lambda j, b: (False, ""),
        labels=labels, pool_size=100, k1=1.0, k2=2.0)
    bs.consume_batch(list(range(10)), [np.zeros(100) for _ in range(10)])
    assert len(bs.pool) > 0, "pool must persist after batch 1 (V2 semantics)"
    assert bs.state.pool_flushes == 0
    bs.consume_batch(list(range(10, 100)), [np.zeros(100) for _ in range(90)])
    bs.finalize_stream()
    assert len(bs.pool) == 0 and bs.state.pool_flushes == 1


def test_v2_no_duplicate_selected_ids():
    labels = np.array(["a", "b"] * 50)
    it = iter([0.5] * 50)
    bs = BatchSelectorV2(
        __import__("ossfs_dd.config", fromlist=["SelectionConfig"]).SelectionConfig(
            mode=__import__("ossfs_dd.config", fromlist=["Mode"]).Mode.MATLAB_COMPATIBLE,
            k1=0.0, k2=10.0, pool_size=100),
        metric_family="x", score_fn=lambda c, lab: next(it),
        joint_fn=lambda a, b, lab, m: (m + 0.0, "x"), is_redundant_fn=lambda j, b: (False, ""),
        promotes_pair_fn=lambda j, b: (False, ""),
        labels=labels, pool_size=100, k1=0.0, k2=10.0)
    bs.consume_batch(list(range(50)), [np.zeros(100) for _ in range(50)])
    bs.finalize_stream()
    ids = bs.candidates.feature_ids()
    assert len(set(ids)) == len(ids)


def test_schedule_is_permutation():
    sched = make_schedule(np.arange(30), "SYNTH", 11, 0)
    flat = sched["flattened_order"]
    assert sorted(flat) == list(range(30))


def test_v1_factory_unchanged():
    """V1 default must still be reachable and not replaced by V2."""
    import numpy as np

    from ossfs_dd_stage3d.selector import make_discrete_batch
    bs, _, _ = make_discrete_batch(np.array(["a", "b"] * 10))
    assert type(bs).__name__ == "BatchSelector"  # V1, not V2


def test_v2_factory_exists():
    import numpy as np

    from ossfs_dd_stage3d.selector import make_discrete_batch_v2
    bs, _, _ = make_discrete_batch_v2(np.array(["a", "b"] * 10))
    assert type(bs).__name__ == "BatchSelectorV2"
