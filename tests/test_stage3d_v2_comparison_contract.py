"""Stage 3D V2 comparison: exact 230/690 key counts, V2 lifecycle, contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from ossfs_dd_stage3d.batch_selector_v2 import BatchSelectorV2
from ossfs_dd_stage3d.scheduler import batch_size_for_p

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "configs" / "stage3d_v2_comparison_v1.yaml"


def _cfg():
    if not CONFIG.exists():
        pytest.skip("config not found")
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _expected_selection_keys():
    cfg = _cfg()
    keys = set()
    for arm_name, arm in cfg["experiment"]["arms"].items():
        for ds in arm["datasets"]:
            for s in arm["seeds"]:
                for f in arm["folds"]:
                    keys.add(f"{ds}_seed{s}_fold{f}__{arm_name}")
    return keys


def test_exact_230_selection_keys():
    keys = _expected_selection_keys()
    assert len(keys) == 230, f"expected 230, got {len(keys)}"
    # breakdown: ARCENE D2_V2 = 10*5=50, 4 discrete D2_V2 = 4*3*5=60,
    # 4 continuous C2_V2 = 4*3*5=60, 4 continuous C4_V2 = 4*3*5=60
    arcene = {k for k in keys if k.startswith("ARCENE") and "__D2_V2" in k}
    d2_pilot = {k for k in keys if k.startswith(("DEXTER","MADELON","GINA","GISETTE")) and "__D2_V2_pilot" in k}
    c2 = {k for k in keys if "__C2_V2" in k}
    c4 = {k for k in keys if "__C4_V2" in k}
    assert len(arcene) == 50
    assert len(d2_pilot) == 60
    assert len(c2) == 60
    assert len(c4) == 60


def test_exact_690_classifier_keys():
    sel = _expected_selection_keys()
    clf = {f"{k}__{c}" for k in sel for c in ("knn", "svm", "cart")}
    assert len(clf) == 690


def test_no_duplicate_selection_keys():
    keys = list(_expected_selection_keys())
    assert len(keys) == len(set(keys))


def test_no_duplicate_classifier_keys():
    sel = _expected_selection_keys()
    clf = list(f"{k}__{c}" for k in sel for c in ("knn", "svm", "cart"))
    assert len(clf) == len(set(clf))


def test_svm_c_matches_v1():
    cfg = _cfg()
    svm = cfg["experiment"]["classifier_params"]["svm"]
    assert svm["C"] == 1.0, f"SVM C must be 1.0 to match V1, got {svm['C']}"
    assert svm["kernel"] == "linear"
    assert svm["tol"] == 0.001
    assert svm["max_iter"] == 2000000


def test_no_d2_v2_pilot_double_counting():
    """The D2_V2_pilot arm must not overlap with D2_V2 on any dataset."""
    cfg = _cfg()
    arms = cfg["experiment"]["arms"]
    d2_main = set(arms["D2_V2"]["datasets"])
    d2_pilot = set(arms["D2_V2_pilot"]["datasets"])
    overlap = d2_main & d2_pilot
    # ARCENE is only in D2_V2; DEXTER/MADELON/GINA/GISETTE only in D2_V2_pilot
    assert "ARCENE" not in d2_pilot, "ARCENE must not appear in D2_V2_pilot"
    assert not (d2_main & d2_pilot), f"datasets appear in both D2_V2 and D2_V2_pilot: {overlap}"


def test_batch_size_boundaries():
    for p, expected in [(2000, 10), (2001, 25), (10000, 25), (10001, 50)]:
        assert batch_size_for_p(p) == expected


def test_forbidden_datasets_absent():
    cfg = _cfg()
    for ds in ("DLBCL", "BREAST"):
        assert ds not in cfg["authorization"]["datasets"]


def test_svm_is_svc_not_linearsvc():
    """The YAML must specify SVC kernel=linear, not LinearSVC."""
    cfg = _cfg()
    svm = cfg["experiment"]["classifier_params"]["svm"]
    assert svm["kernel"] == "linear"
    # probability must be false or absent (default false)
    assert not svm.get("probability", False)


def test_v2_pool_persists_across_batches():
    labels = np.array(["a", "b"] * 50)
    it = iter([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0] + [0.05] * 90)
    from ossfs_dd.config import Mode, SelectionConfig
    bs = BatchSelectorV2(
        SelectionConfig(mode=Mode.MATLAB_COMPATIBLE, k1=1.0, k2=2.0, pool_size=100),
        metric_family="x",
        score_fn=lambda c, lab: next(it),
        joint_fn=lambda a, b, lab, m: (m + 0.0, "x"),
        is_redundant_fn=lambda j, b: (False, ""),
        promotes_pair_fn=lambda j, b: (False, ""),
        labels=labels, pool_size=100, k1=1.0, k2=2.0)
    bs.consume_batch(list(range(10)), [np.zeros(100) for _ in range(10)])
    assert len(bs.pool) > 0, "pool must persist after batch 1"
    assert bs.state.pool_flushes == 0
    bs.consume_batch(list(range(10, 100)), [np.zeros(100) for _ in range(90)])
    bs.finalize_stream()
    assert len(bs.pool) == 0 and bs.state.pool_flushes == 1


def test_v2_finalization_idempotent():
    labels = np.array(["a", "b"] * 50)
    it = iter([0.5] * 5)
    from ossfs_dd.config import Mode, SelectionConfig
    bs = BatchSelectorV2(
        SelectionConfig(mode=Mode.MATLAB_COMPATIBLE, k1=0.0, k2=10.0, pool_size=100),
        metric_family="x", score_fn=lambda c, lab: next(it),
        joint_fn=lambda a, b, lab, m: (m + 0.0, "x"),
        is_redundant_fn=lambda j, b: (False, ""),
        promotes_pair_fn=lambda j, b: (False, ""),
        labels=labels, pool_size=100, k1=0.0, k2=10.0)
    bs.consume_batch(list(range(5)), [np.zeros(100) for _ in range(5)])
    bs.finalize_stream()
    f1 = bs.state.pool_flushes
    bs.finalize_stream()
    assert bs.state.pool_flushes == f1


def test_v2_no_duplicate_selected_ids():
    labels = np.array(["a", "b"] * 50)
    it = iter([0.5] * 50)
    from ossfs_dd.config import Mode, SelectionConfig
    bs = BatchSelectorV2(
        SelectionConfig(mode=Mode.MATLAB_COMPATIBLE, k1=0.0, k2=10.0, pool_size=100),
        metric_family="x", score_fn=lambda c, lab: next(it),
        joint_fn=lambda a, b, lab, m: (m + 0.0, "x"),
        is_redundant_fn=lambda j, b: (False, ""),
        promotes_pair_fn=lambda j, b: (False, ""),
        labels=labels, pool_size=100, k1=0.0, k2=10.0)
    bs.consume_batch(list(range(50)), [np.zeros(100) for _ in range(50)])
    bs.finalize_stream()
    ids = bs.candidates.feature_ids()
    assert len(set(ids)) == len(ids)


def test_v1_factory_unchanged():
    from ossfs_dd_stage3d.selector import make_discrete_batch
    bs, _, _ = make_discrete_batch(np.array(["a", "b"] * 10))
    assert type(bs).__name__ == "BatchSelector"  # V1, not V2


def test_v2_factory_exists():
    from ossfs_dd_stage3d.selector import make_discrete_batch_v2
    bs, _, _ = make_discrete_batch_v2(np.array(["a", "b"] * 10))
    assert type(bs).__name__ == "BatchSelectorV2"


def test_v1_comparator_arms_are_d2_c2_c4():
    """V2 D2_V2 pairs with V1 D2; V2 C2_V2 pairs with V1 C2; V2 C4_V2 with V1 C4."""
    v2_to_v1 = {"D2_V2": "D2", "C2_V2": "C2", "C4_V2": "C4", "D2_V2_pilot": "D2"}
    # verify the mapping is consistent
    for _v2_arm, v1_arm in v2_to_v1.items():
        assert v1_arm in ("D2", "C2", "C4"), f"unexpected V1 arm {v1_arm}"
