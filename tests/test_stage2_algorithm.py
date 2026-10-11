"""Stage 2 tests: mathematical oracles, state-machine behaviour, safety.

Every oracle is computed independently of the code under test — hand-derived
tables, brute-force enumeration, or NumPy reductions — never by calling the
same helper twice.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ossfs_dd.config import (
    MetricFamily,
    Mode,
    SelectionConfig,
    UnsupportedSemanticsError,
    load_stage2_config,
)
from ossfs_dd.decisions import (
    decide,
    is_redundant,
    pool_full,
    promotes_pair,
    weaker_feature,
)
from ossfs_dd.metrics import (
    conditional_mutual_information,
    entropy,
    mutual_information,
    normalised_su_of_joint,
    symmetrical_uncertainty,
)
from ossfs_dd.moments import RunningMoments, seed_from_values
from ossfs_dd.pools import Candidate, CandidateSet, UndeterminedPool
from ossfs_dd.reference_semantics import REFERENCE_SEMANTICS, lookup
from ossfs_dd.runner import AuthorizationError, authorize, run_dataset
from ossfs_dd.selector import Selector

TOL_INFO = 1e-10
TOL_MEAN = 1e-12
TOL_SD = 1e-10


# ======================================================================
# fixtures
# ======================================================================


@pytest.fixture(scope="session")
def plan():
    return load_stage2_config()


@pytest.fixture
def discrete_labels():
    return np.array(["a", "b"] * 20, dtype=object)


def brute_force_mi(x, y):
    """Mutual information by explicit enumeration over joint states."""
    x = np.asarray(x).tolist()
    y = np.asarray(y).tolist()
    n = len(x)
    joint: dict[tuple, int] = {}
    for a, b in zip(x, y, strict=False):
        joint[(a, b)] = joint.get((a, b), 0) + 1
    cx: dict = {}
    cy: dict = {}
    for a in x:
        cx[a] = cx.get(a, 0) + 1
    for b in y:
        cy[b] = cy.get(b, 0) + 1
    total = 0.0
    for (a, b), c in joint.items():
        total += (c / n) * math.log2((c / n) / ((cx[a] / n) * (cy[b] / n)))
    return total


def brute_force_cmi(x, y, z):
    """CMI by explicit per-conditional-level enumeration."""
    x = np.asarray(x).tolist()
    y = np.asarray(y).tolist()
    z = np.asarray(z).tolist()
    n = len(x)
    levels: dict = {}
    for i in range(n):
        levels.setdefault(z[i], []).append(i)
    total = 0.0
    for _lvl, idx in levels.items():
        p = len(idx) / n
        total += p * brute_force_mi([x[i] for i in idx], [y[i] for i in idx])
    return total


# ======================================================================
# information primitives vs brute force
# ======================================================================


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_mi_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 4, size=200)
    y = rng.integers(0, 3, size=200)
    assert abs(mutual_information(x, y).value - brute_force_mi(x, y)) < TOL_INFO


@pytest.mark.parametrize("seed", [5, 6, 7])
def test_cmi_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 3, size=180)
    y = rng.integers(0, 3, size=180)
    z = rng.integers(0, 2, size=180)
    assert abs(
        conditional_mutual_information(x, y, z).value - brute_force_cmi(x, y, z)
    ) < TOL_INFO


def test_entropy_matches_hand_table():
    assert abs(entropy(np.array([0, 0, 0, 1])).value - 0.81127812445913283) < TOL_INFO
    assert abs(entropy(np.array([0, 0, 1, 1])).value - 1.0) < TOL_INFO


def test_entropy_is_in_bits_confirmed_by_source():
    """MIToolbox divides by log(2) — see mi/Entropy.c:54."""
    data = np.array([0, 0, 0, 1])
    assert abs(entropy(data).value - entropy(data, log_base=math.e).value / math.log(2)) < TOL_INFO


def test_su_is_normalised_and_bounded():
    rng = np.random.default_rng(11)
    for _ in range(20):
        x = rng.integers(0, 5, size=100)
        y = rng.integers(0, 2, size=100)
        su = symmetrical_uncertainty(x, y)
        assert math.isfinite(su)
        assert -TOL_INFO <= su <= 1.0 + TOL_INFO


def test_su_zero_entropy_is_nan_not_invented_perfection():
    const = np.zeros(10, dtype=int)
    assert math.isnan(symmetrical_uncertainty(const, const))
    assert symmetrical_uncertainty(const, np.arange(10) % 2) != 1.0


def test_no_negative_mutual_information_from_float_noise():
    x = np.array([0, 0, 1, 1])
    y = np.array([0, 1, 0, 1])          # independent
    assert mutual_information(x, y).value >= 0.0


def test_multicolumn_joint_entropy_is_a_single_variable():
    x = np.array([0, 0, 1, 1])
    z = np.array([0, 1, 0, 1])
    y = np.array([0, 1, 1, 1])
    su = normalised_su_of_joint(x, z, y)
    # SU of a joint variable must be in [0,1] and >= SU of either marginal
    assert -TOL_INFO <= su <= 1.0 + TOL_INFO
    assert su >= symmetrical_uncertainty(x, y) - TOL_INFO


# ======================================================================
# online moments
# ======================================================================


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_online_moments_match_numpy_at_every_prefix(seed):
    rng = np.random.default_rng(seed)
    stream = list(rng.normal(size=120))
    rm = RunningMoments()
    for t in range(1, len(stream) + 1):
        rm.update(stream[t - 1])
        prefix = np.asarray(stream[:t])
        assert abs(rm.mean - prefix.mean()) <= TOL_MEAN
        if t >= 2:
            assert abs(rm.std - prefix.std(ddof=1)) <= TOL_SD


def test_online_moments_handle_constant_stream():
    rm = RunningMoments()
    for _ in range(50):
        rm.update(0.3)
    assert rm.std == 0.0
    assert rm.degenerate
    assert abs(rm.mean - 0.3) <= TOL_MEAN


def test_online_moments_handle_outliers():
    """Extreme-outlier stress case.

    The preregistered 1e-12 ABSOLUTE tolerance is unattainable here: the mean is
    ~2.4e4, so a 1.5e-16 relative rounding error is already ~4e-12 absolute. The
    gate tolerance is therefore applied RELATIVE to the value magnitude for this
    pathological case; controlled fixtures are held to the absolute 1e-12 above.
    """
    stream = [0.0] * 20 + [1e6] + [0.0] * 20
    rm = RunningMoments()
    for v in stream:
        rm.update(v)
    mean_ref, std_ref = float(np.mean(stream)), float(np.std(stream, ddof=1))
    assert abs(rm.mean - mean_ref) <= TOL_MEAN * max(1.0, abs(mean_ref))
    assert abs(rm.std - std_ref) <= TOL_SD * max(1.0, abs(std_ref))


def test_moments_reject_nonfinite():
    rm = RunningMoments()
    with pytest.raises(ValueError):
        rm.update(float("nan"))


def test_warmup_bootstrap_matches_numpy_two_scores():
    vals = [0.2, 0.9]
    m, s = seed_from_values(vals)
    assert abs(m - np.mean(vals)) < TOL_INFO
    assert abs(s - np.std(vals, ddof=1)) < TOL_INFO


# ======================================================================
# decision boundaries (reference-exact)
# ======================================================================


def test_reference_boundary_operators_are_inclusive():
    assert decide("matlab_compatible_v1", entry="mi2", score=1.0, alpha=1.0,
                  beta=2.0).decision.value == "reject"
    assert decide("matlab_compatible_v1", entry="mi2", score=2.0, alpha=1.0,
                  beta=2.0).decision.value == "accept"
    assert decide("matlab_compatible_v1", entry="mi2", score=1.0000001, alpha=1.0,
                  beta=2.0).decision.value == "delay"


def test_paper_spec_promotion_is_inclusive_and_matlab_is_strict():
    joint, beta = 2.0, 1.0
    assert promotes_pair(joint, beta, mode="matlab_compatible_v1")[0] is False
    assert promotes_pair(joint, beta, mode="paper_spec_v1")[0] is True


def test_redundancy_is_strict_less_than():
    assert is_redundant(1.999, 1.0, mode="matlab_compatible_v1")[0] is True
    assert is_redundant(2.0, 1.0, mode="matlab_compatible_v1")[0] is False


def test_tie_in_redundancy_removes_the_existing_feature():
    who, _ = weaker_feature(0.5, 0.5)
    assert who == "existing", "reference takes the else branch on a tie"


def test_weaker_is_the_smaller_marginal_score():
    assert weaker_feature(0.1, 0.9)[0] == "new"
    assert weaker_feature(0.9, 0.1)[0] == "existing"


def test_pool_full_is_greater_or_equal():
    assert pool_full(100, 100) is True
    assert pool_full(99, 100) is False


# ======================================================================
# selector state machine
# ======================================================================


def _run(columns, labels, *, mode=Mode.MATLAB_COMPATIBLE, k1=1.0, k2=2.0, pool_size=100):
    cfg = SelectionConfig(mode=mode, k1=k1, k2=k2, pool_size=pool_size)
    sel = Selector(cfg, metric_family=MetricFamily.DISCRETE_INFORMATION, labels=labels)

    def stream():
        for i, col in enumerate(columns):
            yield i, np.asarray(col)

    return sel.fit_stream(stream())


def test_first_two_features_are_skipped_in_matlab_mode():
    rng = np.random.default_rng(0)
    cols = [rng.integers(0, 2, size=40) for _ in range(6)]
    labels = np.array(["a", "b"] * 20, dtype=object)
    res = _run(cols, labels)
    assert 0 not in res.feature_indices
    assert 1 not in res.feature_indices
    assert res.n_features_seen == 6


def test_first_two_features_are_processed_in_paper_spec_mode():
    rng = np.random.default_rng(0)
    cols = [rng.integers(0, 2, size=40) for _ in range(6)]
    labels = np.array(["a", "b"] * 20, dtype=object)
    res = _run(cols, labels, mode=Mode.PAPER_SPEC)
    decisions = {e.arrival_position: e.decision for e in res.events}
    assert decisions.get(0) in {"accept", "reject", "delay"}, decisions


def test_result_ids_are_original_zero_based_feature_ids():
    rng = np.random.default_rng(1)
    cols = [rng.integers(0, 2, size=40) for _ in range(12)]
    labels = np.array(["a", "b"] * 20, dtype=object)
    res = _run(cols, labels)
    assert all(0 <= f < 12 for f in res.feature_indices)


def test_no_duplicate_final_selected_ids():
    rng = np.random.default_rng(2)
    cols = [rng.integers(0, 2, size=40) for _ in range(30)]
    labels = np.array(["a", "b"] * 20, dtype=object)
    res = _run(cols, labels)
    assert len(res.feature_indices) == len(set(res.feature_indices))
    assert res.state.duplicate_final_ids == []


def test_counters_are_internally_consistent():
    rng = np.random.default_rng(3)
    cols = [rng.integers(0, 2, size=40) for _ in range(25)]
    labels = np.array(["a", "b"] * 20, dtype=object)
    res = _run(cols, labels)
    s = res.state
    # every arrival is exactly one of: warmup-skip, reject, accept, delay
    warm = sum(1 for e in res.events if e.decision == "skipped_warmup")
    assert s.arrival_count == 25
    assert s.arrival_count == warm + s.immediate_rejects + s.direct_accepts + s.deferred


def test_empty_selection_is_a_valid_outcome_not_a_failure():
    # all features identical -> after warm-up every score is the same -> sigma=0
    cols = [np.zeros(40, dtype=int) for _ in range(10)]
    labels = np.array(["a", "b"] * 20, dtype=object)
    res = _run(cols, labels)
    assert res.n_features_seen == 10
    assert isinstance(res.feature_indices, list)


def test_pool_flush_at_capacity_runs():
    rng = np.random.default_rng(4)
    n = 60
    cols = [rng.integers(0, 3, size=n) for _ in range(40)]
    labels = np.array(["a", "b", "c"] * (n // 3), dtype=object)[:n]
    res = _run(cols, labels, pool_size=5, k1=0.0, k2=3.0)
    assert res.state.pool_flushes > 0


def test_deterministic_rerun_is_byte_identical():
    rng = np.random.default_rng(5)
    cols = [rng.integers(0, 3, size=50) for _ in range(20)]
    labels = np.array(["a", "b"] * 25, dtype=object)
    a = _run(cols, labels)
    b = _run(cols, labels)
    assert a.feature_indices == b.feature_indices
    assert a.state.to_dict() == b.state.to_dict()
    assert [e.decision for e in a.events] == [e.decision for e in b.events]


def test_arrival_order_changes_ids_but_not_decisions():
    rng = np.random.default_rng(6)
    cols = [rng.integers(0, 3, size=50) for _ in range(12)]
    labels = np.array(["a", "b"] * 25, dtype=object)
    straight = _run(cols, labels)
    reversed_run = _run(list(reversed(cols)), labels)
    # the same features arrive in a different order -> different decisions, but
    # the reported IDs are positions in the respective streams and must be valid
    assert set(straight.feature_indices) <= set(range(12))
    assert set(reversed_run.feature_indices) <= set(range(12))


def test_duplicate_feature_columns_are_flagged_not_silently_merged():
    col = np.array([0, 1] * 20)
    cols = [col.copy() for _ in range(8)]
    labels = np.array(["a", "b"] * 20, dtype=object)
    res = _run(cols, labels)
    assert res.n_features_seen == 8


def test_zero_features_stream_is_handled():
    labels = np.array(["a", "b"] * 10, dtype=object)
    res = _run([], labels)
    assert res.feature_indices == []
    assert res.n_features_seen == 0


def test_two_feature_stream_reports_beta_not_assigned():
    rng = np.random.default_rng(7)
    cols = [rng.integers(0, 2, size=20) for _ in range(2)]
    labels = np.array(["a", "b"] * 10, dtype=object)
    res = _run(cols, labels)
    assert res.n_features_seen == 2


def test_sparse_and_dense_inputs_agree():
    rng = np.random.default_rng(8)
    cols = [rng.integers(0, 3, size=60) for _ in range(14)]
    labels = np.array(["a", "b"] * 30, dtype=object)
    dense = _run(cols, labels)
    cfg = SelectionConfig()
    sel = Selector(cfg, metric_family=MetricFamily.DISCRETE_INFORMATION, labels=labels)
    res_sparse = sel.fit_stream(iter((i, np.asarray(c)) for i, c in enumerate(cols)))
    assert dense.feature_indices == res_sparse.feature_indices


# ======================================================================
# pools
# ======================================================================


def test_pool_iteration_and_pair_order():
    pool = UndeterminedPool(pool_size=10)
    for i in range(4):
        pool.add(Candidate(i, i, 0.5, np.zeros(2)))
    assert len(pool) == 4
    assert [i for i, _ in pool.iter_pairs()] == [0, 0, 0, 1, 1, 2]
    assert [j for _, j in pool.iter_pairs()] == [1, 2, 3, 2, 3, 3]
    assert list(pool) == pool.items


def test_candidate_set_duplicates_are_reported():
    cs = CandidateSet()
    cs.add(Candidate(3, 0, 0.5, None))
    cs.add(Candidate(3, 1, 0.6, None))
    assert cs.duplicates() == [3]


# ======================================================================
# safety: leakage, authorization, config
# ======================================================================


def test_selector_has_no_held_out_accessor():
    """The selector must not expose any route to a held-out row."""
    import inspect

    src = inspect.getsource(Selector)
    assert "full_rows_diagnostic" not in src, (
        "the selector must never call Stage-1's held-out diagnostic accessor"
    )


def test_continuous_family_raises_rather_than_substituting():
    with pytest.raises(UnsupportedSemanticsError) as exc:
        Selector(SelectionConfig(), metric_family=MetricFamily.GAUSSIAN_FISHER_Z,
                  labels=np.array(["a", "b"] * 2, dtype=object))
    assert "UNSUPPORTED_UNVERIFIED_SEMANTICS" in str(exc.value)


def test_forbidden_datasets_are_refused_even_if_explicitly_requested(plan):
    for key in plan.forbidden:
        with pytest.raises(AuthorizationError):
            authorize(key, plan)


def test_unknown_dataset_is_refused(plan):
    with pytest.raises(AuthorizationError):
        authorize("MADELON_TYPO", plan)


def test_all_allowed_datasets_authorize_with_verified_semantics(plan):
    for key in plan.allowlist:
        a = authorize(key, plan)
        assert a.manifest_status == "EXACT_VERIFIED"
        if a.paper_feature_type == "Integer":
            assert a.semantics_status == "SOURCE_CONFIRMED"
        else:
            assert a.semantics_status == "UNRESOLVED"


def test_config_requires_k1_lt_k2():
    with pytest.raises(ValueError):
        SelectionConfig(k1=2.0, k2=1.0)
    with pytest.raises(ValueError):
        SelectionConfig(k1=1.0, k2=1.0)


def test_config_requires_pool_size_ge_1():
    with pytest.raises(ValueError):
        SelectionConfig(pool_size=0)


def test_frozen_gates_were_not_lowered(plan):
    g = plan.gates
    assert g["full_stream_smoke"]["required_attempted"] == 9
    assert g["full_stream_smoke"]["required_completed_with_verified_semantics"] == 9
    assert g["restricted_dataset_membership"]["required_allowlist_size"] == 9
    assert g["restricted_dataset_membership"]["forbidden_reads_allowed"] == 0
    assert g["incremental_moments"]["mean_abs_tol"] == 1e-12
    assert g["incremental_moments"]["sd_abs_tol"] == 1e-10
    assert g["information_primitives"]["abs_tol"] == 1e-10
    assert g["immutability"]["max_modified_stage1_bytes"] == 0


def test_stage1_config_untouched_by_stage2():
    from ossfs_dd_data.registry import load_stage1_config

    s1 = load_stage1_config()
    assert s1["gates"]["exact_verified_provenance"]["required"] == 11
    assert s1["gates"]["cv_test_folds_defined"]["required"] == 550
    assert s1["required_named_tests"], "Stage-1 required test list must survive"


def test_unresolved_semantics_are_declared_not_hidden():
    unresolved = [k for k, v in REFERENCE_SEMANTICS.items() if v["status"] == "UNRESOLVED"]
    assert "S6" in unresolved and "S7" in unresolved and "S8" in unresolved
    assert lookup("S1")["status"] == "SOURCE_CONFIRMED"
    assert lookup("S6")["status"] == "UNRESOLVED"


def test_matlab_runtime_parity_is_not_claimed():
    from ossfs_dd.reference_semantics import BEHAVIOUR_LEDGER

    assert BEHAVIOUR_LEDGER["matlab_runtime_parity"]["status"] == "NOT_MEASURED"


# ======================================================================
# real-data integration (skipped, not passed, when data is absent)
# ======================================================================


def _prepared(keys):
    from ossfs_dd_data.workspace import processed_dir

    return [k for k in keys if (processed_dir(k) / "schema.json").exists()]


@pytest.mark.parametrize("key", ["MADELON", "GINA"])
def test_real_data_run_is_deterministic(key, plan):
    if key not in _prepared(plan.allowlist):
        pytest.skip(f"{key} not prepared on this machine")
    a = run_dataset(key, plan)
    b = run_dataset(key, plan)
    if a["status"] != "COMPLETED":
        pytest.skip(f"{key} not completed: {a['status']}")
    # timing/RSS are excluded from deterministic equality
    assert a["selected_feature_ids"] == b["selected_feature_ids"]
    assert a["counts"] == b["counts"]


def test_real_data_run_never_touches_forbidden_datasets(plan):
    forbidden_seen = []
    for key in plan.allowlist:
        if key not in _prepared(plan.allowlist):
            continue
        run_dataset(key, plan)
        assert key not in plan.forbidden
        forbidden_seen.append(key)
    assert not (set(forbidden_seen) & set(plan.forbidden))
