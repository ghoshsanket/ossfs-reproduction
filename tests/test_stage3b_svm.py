"""Experiment A SVM tests — SYNTHETIC inputs only, no real dataset fits."""

from __future__ import annotations

import numpy as np
import pytest

from ossfs_dd_stage3b.analysis import a_h1_outcome, a_h2_outcome, convergence_rate, summarize_cells
from ossfs_dd_stage3b.protocol import FitID, SubsetID
from ossfs_dd_stage3b.svm_experiment import (
    STATUS_CONVERGED,
    STATUS_NONCONVERGED,
    extract_columns,
    prepare_inputs,
    run_one_fit,
)


def _separable(n=60, seed=0):
    """A tiny known linear classification problem."""
    rng = np.random.default_rng(seed)
    y = np.where(np.arange(n) < n // 2, "a", "b")
    sig = (y == "a").astype(float)
    X = np.column_stack([
        sig * 3 + rng.normal(0, 0.3, n),
        sig * 3 + rng.normal(0, 0.3, n),
        rng.normal(0, 1, n),
        rng.normal(0, 1, n),
    ])
    return X, y


def test_standardization_fits_training_rows_only():
    X, y = _separable()
    train, test = np.arange(40), np.arange(40, 60)
    # make the held-out rows wildly larger
    X[test] *= 1000.0
    prep = prepare_inputs(X, train, test, [0, 1], "standardized")
    # the scaler must be driven by TRAIN statistics only
    assert abs(float(prep["X_train"].mean())) < 1e-10
    assert prep["meta"]["scaler_fitted_on"] == "selected_training_rows_only"


def test_raw_variant_is_identity():
    X, y = _separable()
    train, test = np.arange(40), np.arange(40, 60)
    prep = prepare_inputs(X, train, test, [0, 1], "raw")
    assert np.allclose(prep["X_train"], extract_columns(X, train, [0, 1]))


def test_unknown_scale_variant_rejected():
    X, y = _separable()
    with pytest.raises(ValueError):
        prepare_inputs(X, np.arange(40), np.arange(40, 60), [0], "bogus")


def test_scaler_does_not_see_test_statistics():
    X, y = _separable()
    train, test = np.arange(40), np.arange(40, 60)
    a = prepare_inputs(X, train, test, [0], "standardized")
    X2 = X.copy()
    X2[test] += 1e6
    b = prepare_inputs(X2, train, test, [0], "standardized")
    # identical training transform regardless of how the test rows changed
    assert np.allclose(a["X_train"], b["X_train"])


def test_extract_columns_matches_dense_truth():
    X, y = _separable()
    rows, cols = np.array([1, 3, 5]), np.array([2, 0])
    assert np.array_equal(extract_columns(X, rows, cols), X[np.ix_(rows, cols)])


def test_fit_on_separable_synthetic_data_converges():
    X, y = _separable()
    train, test = np.arange(40), np.arange(40, 60)
    subset = SubsetID("SYNTH", 11, 0)
    fit = FitID(subset, "standardized", 2_000_000)
    rec = run_one_fit(fit, X, y, train, test, [0, 1], {"n_train": 40, "n_test": 20})
    assert rec["status"] in (STATUS_CONVERGED, STATUS_NONCONVERGED)
    assert rec["accuracy"] is not None and 0.0 <= rec["accuracy"] <= 1.0
    assert rec["n_correct"] == int(rec["accuracy"] * rec["n_test"])


def test_fit_records_full_n_iter_vector_and_params():
    X, y = _separable()
    train, test = np.arange(40), np.arange(40, 60)
    subset = SubsetID("SYNTH", 11, 0)
    rec = run_one_fit(FitID(subset, "raw", 2_000_000), X, y, train, test, [0, 1],
                      {"n_train": 40, "n_test": 20})
    assert rec["estimator"]["class_name"] == "SVC"
    assert rec["estimator"]["params"]["kernel"] == "linear"
    assert rec["estimator"]["params"]["C"] == 1.0
    assert rec["estimator"]["params"]["max_iter"] == 2_000_000
    assert isinstance(rec["n_iter_vector"], list) and rec["n_iter_vector"]
    assert rec["n_iter_max"] == max(rec["n_iter_vector"])


def test_nonconverged_fit_is_flagged_and_accuracy_marked_diagnostic():
    X, y = _separable(n=120, seed=1)
    # badly scaled so a tiny cap forces NONCONVERGED
    X[:, 0] *= 1e4
    train, test = np.arange(80), np.arange(80, 120)
    subset = SubsetID("SYNTH", 11, 0)
    rec = run_one_fit(FitID(subset, "raw", 1), X, y, train, test, [0, 1],
                      {"n_train": 80, "n_test": 40})
    assert rec["status"] == STATUS_NONCONVERGED
    assert rec["converged"] is False
    assert rec["hit_iteration_cap"] is True
    # accuracy may exist but MUST be marked as an incomplete-fit diagnostic
    if rec["accuracy"] is not None:
        assert rec["accuracy_is_incomplete_fit_diagnostic"] is True


def test_accuracy_denominator_recorded():
    X, y = _separable()
    train, test = np.arange(40), np.arange(40, 60)
    subset = SubsetID("SYNTH", 11, 0)
    rec = run_one_fit(FitID(subset, "raw", 2_000_000), X, y, train, test, [0, 1],
                      {"n_train": 40, "n_test": 20})
    assert rec["n_test"] == 20 and 0 <= rec["n_correct"] <= rec["n_test"]


# ------------------------------------------------------------- analysis


def test_convergence_rate_needs_full_denominator():
    assert convergence_rate(15, 15) == 1.0
    assert convergence_rate(0, 15) == 0.0
    # a single missing/censored fit must NOT yield a rate
    assert convergence_rate(10, 14) is None
    assert convergence_rate(10, 13) is None


def _synthetic_records(converged_map):
    recs = []
    for ds in ["ARCENE", "DEXTER", "MADELON", "GINA", "GISETTE"]:
        for seed in (11, 17, 23):
            for fold in range(5):
                for sv in ("raw", "standardized"):
                    for cap in (20_000, 200_000, 2_000_000):
                        conv = converged_map(ds, sv, cap, seed, fold)
                        recs.append({
                            "fit_id": f"{ds}_seed{seed}_fold{fold}__{sv}__it{cap}",
                            "dataset": ds, "split_seed": seed, "fold": fold,
                            "scale_variant": sv, "max_iter": cap,
                            "status": STATUS_CONVERGED if conv else STATUS_NONCONVERGED,
                            "converged": conv, "accuracy": 0.8 if conv else 0.5,
                            "n_test": 40, "n_correct": 32 if conv else 20,
                            "seconds": 1.0, "n_iter_max": 1000,
                        })
    return recs


def test_a_h1_supported_when_standardized_dominates():
    recs = _synthetic_records(
        lambda ds, sv, cap, s, f: (sv == "standardized" and cap == 2_000_000)
    )
    out = a_h1_outcome(recs)
    assert out["overall"] == "SUPPORTED"
    assert out["per_dataset"]["GINA"]["delta"] == 1.0


def test_a_h1_refuted_when_no_difference():
    recs = _synthetic_records(lambda ds, sv, cap, s, f: False)
    out = a_h1_outcome(recs)
    assert out["overall"] == "REFUTED"


def test_a_h1_inconclusive_when_a_fit_is_missing():
    recs = _synthetic_records(lambda ds, sv, cap, s, f: sv == "standardized")
    # drop one terminal record so MADELON raw lacks a full denominator
    recs = [r for r in recs if not (r["dataset"] == "MADELON" and r["split_seed"] == 11
                                    and r["fold"] == 0 and r["scale_variant"] == "raw"
                                    and r["max_iter"] == 2_000_000)]
    out = a_h1_outcome(recs)
    assert out["per_dataset"]["MADELON"]["outcome"] == "INCONCLUSIVE_MISSING_FITS"
    assert out["overall"] == "INCONCLUSIVE"


def test_a_h2_has_ten_prespecified_comparisons():
    recs = _synthetic_records(lambda ds, sv, cap, s, f: cap == 2_000_000)
    out = a_h2_outcome(recs)
    assert out["n_prespecified_comparisons"] == 10
    assert out["overall"] == "SUPPORTED"
    for c in out["comparisons"]:
        assert c["n_pairs"] == 15 and c["n_violations"] == 0


def test_a_h2_refuted_on_a_monotonicity_violation():
    def conv(ds, sv, cap, s, f):
        # one dataset converges better at the SMALL cap
        return (cap == 20_000) if ds == "GINA" else (cap == 2_000_000)

    out = a_h2_outcome(_synthetic_records(conv))
    assert out["overall"] == "REFUTED"
    gina = next(c for c in out["comparisons"] if c["dataset"] == "GINA")
    assert gina["n_violations"] > 0


def test_a_h2_reports_missing_pairs_as_inconclusive():
    recs = _synthetic_records(lambda ds, sv, cap, s, f: cap == 2_000_000)
    recs = [r for r in recs if not (r["dataset"] == "DEXTER" and r["max_iter"] == 20_000
                                    and r["split_seed"] == 17 and r["fold"] == 2)]
    out = a_h2_outcome(recs)
    dexter = next(c for c in out["comparisons"] if c["dataset"] == "DEXTER")
    assert dexter["n_pairs"] == 14 and dexter["outcome"] == "INCONCLUSIVE"


def test_summarize_cells_separates_converged_from_nonconverged_accuracy():
    recs = _synthetic_records(lambda ds, sv, cap, s, f: sv == "standardized")
    cells = summarize_cells(recs)
    cell = next(c for c in cells
                if c["dataset"] == "GINA" and c["scale_variant"] == "raw"
                and c["max_iter"] == 2_000_000)
    assert cell["n_converged"] == 0
    assert cell["accuracy_mean_converged"] is None
    assert cell["accuracy_mean_nonconverged_diagnostic"] is not None
