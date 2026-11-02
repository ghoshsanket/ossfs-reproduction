"""Stage 3D selectors: single-stream and atomic-batch variants.

Both discrete and continuous (standard Fisher-Z) routes. The batch variants
make genuine group-level decisions (one alpha/beta per batch, batch-level sort),
not `consume()` in a loop.
"""

from __future__ import annotations

import numpy as np

from ossfs_dd.config import Mode, SelectionConfig
from ossfs_dd.decisions import is_redundant, promotes_pair
from ossfs_dd.metrics import symmetrical_uncertainty
from ossfs_dd.selector import Selector
from ossfs_dd_stage3d.batch_selector import BatchSelector
from ossfs_dd_stage3d.standard_fisherz import (
    marginal_fisherz,
    pair_score_standard,
)


def make_discrete_single(labels):
    from ossfs_dd.metrics import conditional_mutual_information
    def score_fn(col, lab):
        return symmetrical_uncertainty(np.asarray(col), np.asarray(lab))
    def joint_fn(a, b, lab, ma):
        cmi = conditional_mutual_information(a, lab, b).value
        return ma + cmi, "SU+cmi"
    sel = Selector(SelectionConfig(mode=Mode.MATLAB_COMPATIBLE, k1=1.0, k2=2.0,
                                  pool_size=100),
                   metric_family=__import__("ossfs_dd.config", fromlist=["MetricFamily"]).MetricFamily.DISCRETE_INFORMATION,
                   labels=labels, entry_file="mi2", score_fn=score_fn)
    sel._joint = lambda a, b, lab, ma: joint_fn(a, b, lab, ma)
    return sel, score_fn, joint_fn


def make_discrete_batch(labels):
    def score_fn(col, lab):
        return symmetrical_uncertainty(np.asarray(col), np.asarray(lab))
    def joint_fn(a, b, lab, ma):
        from ossfs_dd.metrics import conditional_mutual_information
        cmi = conditional_mutual_information(a, lab, b).value
        return ma + cmi, "SU+cmi"
    bs = BatchSelector(
        SelectionConfig(mode=Mode.MATLAB_COMPATIBLE, k1=1.0, k2=2.0, pool_size=100),
        metric_family="discrete",
        score_fn=score_fn, joint_fn=joint_fn,
        is_redundant_fn=lambda j, beta: is_redundant(j, beta, mode="matlab_compatible_v1"),
        promotes_pair_fn=lambda j, beta: promotes_pair(j, beta, mode="matlab_compatible_v1"),
        labels=labels, pool_size=100, k1=1.0, k2=2.0)
    return bs, score_fn, joint_fn


def make_continuous_single(labels, *, use_evidence):
    def score_fn(col, lab):
        return marginal_fisherz(np.asarray(col), np.asarray(lab), use_evidence=use_evidence).score
    def joint_fn(a, b, lab, ma):
        ps = pair_score_standard(np.asarray(a), np.asarray(b), np.asarray(lab), use_evidence=use_evidence)
        return ma + ps["conditional_z_given_x"], "marginal+partial"
    sel = Selector(SelectionConfig(mode=Mode.MATLAB_COMPATIBLE, k1=1.0, k2=2.0,
                                  pool_size=100),
                   metric_family=__import__("ossfs_dd.config", fromlist=["MetricFamily"]).MetricFamily.DISCRETE_INFORMATION,
                   labels=labels, entry_file="mi2", score_fn=score_fn)
    sel._joint = lambda a, b, lab, ma: joint_fn(a, b, lab, ma)
    return sel, score_fn, joint_fn


def make_continuous_batch(labels, *, use_evidence):
    def score_fn(col, lab):
        return marginal_fisherz(np.asarray(col), np.asarray(lab), use_evidence=use_evidence).score
    def joint_fn(a, b, lab, ma):
        ps = pair_score_standard(np.asarray(a), np.asarray(b), np.asarray(lab), use_evidence=use_evidence)
        return ma + ps["conditional_z_given_x"], "marginal+partial"
    bs = BatchSelector(
        SelectionConfig(mode=Mode.MATLAB_COMPATIBLE, k1=1.0, k2=2.0, pool_size=100),
        metric_family="continuous",
        score_fn=score_fn, joint_fn=joint_fn,
        is_redundant_fn=lambda j, beta: is_redundant(j, beta, mode="matlab_compatible_v1"),
        promotes_pair_fn=lambda j, beta: promotes_pair(j, beta, mode="matlab_compatible_v1"),
        labels=labels, pool_size=100, k1=1.0, k2=2.0)
    return bs, score_fn, joint_fn
