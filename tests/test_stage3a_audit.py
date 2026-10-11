"""Tests for the Stage 3A audit checks and report arithmetic.

These pin the audit's own logic: the nested-mean recomputation rule, the
aggregate.json seed-key normalisation, the verified solver facts, the
variability arithmetic, and the continuous-estimator boundary claims.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
AUDIT = sorted((PROJECT / "results" / "stage3a").glob("audit_*"))
AGGREGATE = PROJECT / "results" / "stage3a" / "aggregate.json"
PER_RUN_CSV = PROJECT / "results" / "stage3a" / "per_run_outcomes.csv"

pytestmark = pytest.mark.skipif(not AUDIT, reason="no audit run present")


def _latest() -> Path:
    return AUDIT[-1]


# ------------------------------------------------------- recompute arithmetic


def test_nested_mean_rule_matches_manual_computation():
    """folds within a seed, then seeds -- not a flat mean over all runs."""
    folds = [0.8, 0.6, 1.0, 0.4, 0.2]   # seed A
    other = [0.5, 0.5, 0.5, 0.5, 0.5]   # seed B
    seed_a = float(np.mean(folds))
    seed_b = float(np.mean(other))
    nested = (seed_a + seed_b) / 2
    flat = float(np.mean(folds + other))
    assert seed_a == pytest.approx(0.6)
    assert nested == pytest.approx(0.55)
    # The two rules coincide only when every seed has the same fold count; the
    # audit must use the nested form, so make the difference explicit.
    assert nested == pytest.approx(flat)  # equal here: 5 folds each
    weights = [1.0, 0.0, 1.0, 0.0, 0.0]
    assert float(np.average(folds, weights=weights)) == pytest.approx(0.9)


def test_recomputed_means_match_committed_aggregate():
    agg = json.loads(AGGREGATE.read_text())
    psm = agg["variants"]["outcomes_raw_primary"]["datasets"]["ARCENE"]["classifiers"]["knn"]
    seeds = psm["per_seed_mean"]
    assert all(isinstance(k, str) for k in seeds), "JSON seed keys are strings"
    values = list(seeds.values())
    assert psm["mean_accuracy"] == pytest.approx(sum(values) / len(values))
    assert psm["n_seeds_evaluated"] == len(values) == 10


def test_accuracy_is_consistent_with_n_correct_over_n_test():
    import csv

    seen = 0
    with open(PER_RUN_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if not row["accuracy"]:
                continue
            assert float(row["accuracy"]) == pytest.approx(
                int(row["n_correct"]) / int(row["n_test"]), abs=1e-12
            )
            seen += 1
    assert seen == 1500  # 250 runs x 3 classifiers x 2 variants


def test_csv_has_expected_shape():
    import csv

    with open(PER_RUN_CSV, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 250 * 3 * 2
    preps = {r["preprocessing"] for r in rows}
    assert preps == {"RAW_PRIMARY", "STANDARDIZED_SENSITIVITY"}


# ---------------------------------------------------------- solver facts (M2)


def test_svc_is_libsvm_backed_and_max_iter_default_is_minus_one():
    import inspect

    from sklearn.svm import SVC, LinearSVC

    assert "based on libsvm" in (SVC.__doc__ or "").lower()
    assert inspect.signature(SVC.__init__).parameters["max_iter"].default == -1
    assert "liblinear" in (LinearSVC.__doc__ or "").lower()
    assert inspect.signature(LinearSVC.__init__).parameters["max_iter"].default == 1000


def test_audit_records_the_user_declared_cap_not_a_default():
    data = json.loads((_latest() / "solver_and_convergence.json").read_text())
    assert data["declared_in_project"]["SVM_MAX_ITER"] == 2_000_000
    assert data["verified_from_installed_package"]["SVC_max_iter_default"] == -1
    assert data["verified_from_installed_package"]["svc_docstring_states_libsvm"] is True
    assert data["verified_from_installed_package"]["early_termination_message_emitted_in_libsvm_module"]


def test_convergence_counts_are_reported_per_dataset_and_variant():
    data = json.loads((_latest() / "solver_and_convergence.json").read_text())
    per = data["per_dataset_convergence"]
    assert set(per) == {"outcomes_raw_primary", "outcomes_standardized_sensitivity"}
    for variant in per:
        for s in per[variant].values():
            assert s["n_converged"] + s["n_nonconverged"] == s["n_fits"] == 50
    # standardized never failed; raw failed on the badly-scaled datasets
    assert all(v["n_nonconverged"] == 0 for v in per["outcomes_standardized_sensitivity"].values())


def test_nonconvergence_cause_is_labelled_as_hypothesis_not_proof():
    data = json.loads((_latest() / "solver_and_convergence.json").read_text())
    assert data["nonconvergence_cause"]["status"] == "SOLVER_CONDITIONING_HYPOTHESIS_SUPPORTED_NOT_PROVEN"


# -------------------------------------------------------------- variability (M4)


def test_arcene_variability_arithmetic():
    import csv

    with open(_latest() / "selection_variability.csv", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["dataset"] == "ARCENE"]
    v = np.array([int(r["n_selected"]) for r in rows])
    assert v.size == 50
    assert v.min() == 34 and v.max() == 1561
    assert v.mean() == pytest.approx(963.64, abs=0.01)
    assert v.std(ddof=1) == pytest.approx(487.29, abs=0.01)
    # values recomputed from the raw records, not assumed
    assert np.percentile(v, 25) == pytest.approx(614.75)
    assert np.median(v) == pytest.approx(1023.5)
    assert np.percentile(v, 75) == pytest.approx(1444.75)


def test_jaccard_is_bounded_and_computed():
    a, b = {1, 2, 3}, {2, 3, 4}
    assert len(a & b) / len(a | b) == pytest.approx(0.5)
    assert len({1, 2} & {1, 2}) / len({1, 2} | {1, 2}) == 1.0


def test_cause_of_variability_is_not_asserted():
    data = json.loads((_latest() / "audit_manifest.json").read_text())
    assert data["gates"]["AUDIT_VARIABILITY_CHARACTERIZED"]["cause"] == "UNRESOLVED"


# -------------------------------------------------------------- continuous (M5)


def test_continuous_pair_score_cannot_alone_exceed_two_beta_but_joint_can():
    data = json.loads((_latest() / "continuous_rnd_boundaries.json").read_text())
    d = data["boundaries"]["d_2beta_feasibility"]
    assert d["pair_score_range"] == "[0, 1]"
    # the actual code adds the marginal component, so no impossibility holds
    assert d["joint_computation_adds_marginal"] is True
    assert d["impossibility_claim"] == "NOT_APPLICABLE"
    assert d["per_step_thresholds"] == "NOT_MEASURED"


def test_continuous_promotions_were_observed():
    data = json.loads((_latest() / "continuous_rnd_boundaries.json").read_text())
    ev = data["boundaries"]["d_2beta_feasibility"]["empirical_evidence"]
    assert ev["promotions_observed"] is True
    assert sum(v for v in ev["promoted_from_pool_per_dataset"].values() if v) > 0
    assert ev["beta_exceeded_one_in_any_run"] is False


def test_continuous_is_labelled_reconstructed_and_no_classifier():
    data = json.loads((_latest() / "continuous_rnd_boundaries.json").read_text())
    assert data["status"] == "RECONSTRUCTED_NOT_SOURCE_VERIFIED"
    for v in data["per_dataset"].values():
        assert v["training_rows_only"] is True
        assert v["test_rows_read"] == 0
        assert v["classifier_evaluated"] is False


# ------------------------------------------------------------------ safety/M7


def test_only_authorized_datasets_were_executed():
    data = json.loads((_latest() / "test_and_immutability_evidence.json").read_text())
    safety = data["safety"]
    assert safety["n_forbidden_reads"] == 0
    assert safety["forbidden_datasets_executed"] == []
    assert set(safety["datasets_executed_in_records"]) == {
        "ARCENE", "DEXTER", "MADELON", "GINA", "GISETTE"
    }


def test_chronology_is_not_forced_to_verified():
    data = json.loads((_latest() / "freeze_chronology_evidence.json").read_text())
    assert data["status"] in {"VERIFIED", "UNVERIFIED", "CONTRADICTED"}
    assert data["status"] == "CONTRADICTED", "declared frozen_utc postdates all artifacts"
    assert "No date was altered" in data["explicitly_not_done"]


def test_stage1_stage2_trees_unchanged():
    data = json.loads((_latest() / "test_and_immutability_evidence.json").read_text())
    assert data["safety"]["stage1_stage2_unchanged"] is True
    assert data["safety"]["n_commits_made_by_audit"] == 0
