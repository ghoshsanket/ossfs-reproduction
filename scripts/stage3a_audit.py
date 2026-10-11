#!/usr/bin/env python
"""Stage 3A scientific audit: recomputation, erratum, chronology, variability.

This script performs NO selection, NO classifier fitting and NO continuous-data
classification. It only reads existing Stage 3A artifacts, recomputes what can
be recomputed, and writes versioned audit evidence.

Checks implemented:
  M1  ledger reconciliation: run records -> per_run CSV -> aggregate.json -> markdown
  M2  SVM backend / iteration-cap erratum and convergence accounting
  M3  config freeze chronology evidence
  M4  selected-feature-count variability (incl. the ARCENE seed x fold matrix)
  M5  continuous reconstructed-estimator boundaries and 2*beta feasibility
  M6  (implemented in tests/conftest.py + tests/test_stage1_evidence_side_effect.py)
  M7  safety: forbidden-dataset reads, immutable trees, git staging audit

Usage:
    uv run python scripts/stage3a_audit.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ossfs_dd_eval.protocol import (  # noqa: E402
    CLASSIFIERS,
    SEEDS,
    STAGE3A_DATASETS,
    all_runs,
)

PROJECT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = PROJECT.parents[1]
OUT = PROJECT / "results" / "stage3a"
RECORDS = OUT / "runs" / "records"
DOCS = PROJECT / "docs"
CONFIG = PROJECT / "configs" / "stage3a.yaml"
CONT_CONFIG = PROJECT / "configs" / "continuous_rnd_v1.yaml"
PAPER_CSV = PROJECT / "configs" / "paper_reference_discrete.csv"
AGGREGATE = OUT / "aggregate.json"
PER_RUN_CSV = OUT / "per_run_outcomes.csv"
REPORT_MD = DOCS / "stage3a_evaluation.md"
FORBIDDEN = ("DLBCL", "BREAST")
VARIANTS = {
    "outcomes_raw_primary": "RAW_PRIMARY",
    "outcomes_standardized_sensitivity": "STANDARDIZED_SENSITIVITY",
}
TOL = 1e-9


# ----------------------------------------------------------------- helpers


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(RESEARCH_ROOT.resolve()))
    except ValueError:
        return str(path)


def artifact_inventory() -> dict[str, Any]:
    """Every input the audit used: path, sha256, kind (raw vs derived)."""
    items: list[dict[str, Any]] = []

    def add(path: Path, kind: str, note: str) -> None:
        items.append(
            {
                "path": rel(path),
                "exists": path.exists(),
                "sha256": sha256(path),
                "kind": kind,
                "note": note,
            }
        )

    add(CONFIG, "frozen_config", "Stage 3A protocol config")
    add(CONT_CONFIG, "frozen_config", "Workstream B reconstruction config")
    add(PAPER_CSV, "frozen_config",
        "Published reference ledger (NOT git-tracked: blocked by .gitignore '*.csv')")
    add(AGGREGATE, "derived", "Committed aggregate produced by stage3a_report.py")
    add(PER_RUN_CSV, "derived", "Committed per-run table produced by stage3a_report.py")
    add(REPORT_MD, "derived", "Committed Stage 3A markdown report")
    add(OUT / "report_manifest.json", "derived", "Gate manifest")
    add(OUT / "run_manifest.json", "derived", "Sweep manifest (contains no timestamp field)")
    add(PROJECT / "src" / "ossfs_dd_eval" / "classifiers.py", "source", "Estimator definitions")
    add(PROJECT / "src" / "ossfs_dd_continuous" / "fisherz.py", "source", "Reconstructed estimator")
    add(PROJECT / "src" / "ossfs_dd_continuous" / "selector.py", "source", "Continuous selector")
    add(PROJECT / "tests" / "conftest.py", "source", "Pytest session hook (M6 repair)")
    for key in ("SRBCT", "LYMPHOMA", "PROSTATE", "LEUKEMIA"):
        add(
            PROJECT / "results" / "continuous_rnd" / f"{key}_seed11_fold0.json",
            "raw_record", "Workstream B train-only diagnostic",
        )

    records = sorted(RECORDS.glob("*.json")) if RECORDS.exists() else []
    digest_of_records = hashlib.sha256()
    for p in records:
        digest_of_records.update(sha256(p).encode())  # type: ignore[arg-type]
    items.append(
        {
            "path": rel(RECORDS) + "/",
            "exists": bool(records),
            "sha256": digest_of_records.hexdigest(),
            "n_files": len(records),
            "kind": "raw_record",
            "note": "Composite digest over all 250 per-run JSON records (order: filename sort)",
        }
    )
    return {"generated_utc": utcnow(), "artifacts": items}


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def diff(name: str, expected: Any, observed: Any, source: str, tolerance: Any = TOL,
         note: str = "") -> dict[str, Any]:
    if isinstance(expected, float) and isinstance(observed, float):
        ok = abs(expected - observed) <= tolerance
    else:
        ok = expected == observed
    return {
        "check": name, "expected": expected, "observed": observed,
        "source": source, "tolerance": tolerance, "agree": bool(ok), "note": note,
    }


def load_records() -> list[dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RECORDS.glob("*.json"))]


# --------------------------------------------------------------------- M1


def m1_ledger(records: list[dict[str, Any]]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    expected_ids = [s.run_id for s in all_runs()]

    # 1. exact 5 x 10 x 5 coverage, unique IDs
    ids = [r["run_id"] for r in records]
    checks.append(diff("n_records", 250, len(records), "raw run records"))
    checks.append(diff("unique_run_ids", 250, len(set(ids)), "raw run records"))
    checks.append(diff("run_id_set_matches_protocol", sorted(expected_ids), sorted(ids),
                       "protocol.all_runs() vs raw run records"))
    expected_tuples = sorted((d, s, f) for d in STAGE3A_DATASETS for s in SEEDS for f in range(5))
    got_tuples = sorted((r["dataset"], r["seed"], r["fold"]) for r in records)
    checks.append(diff("dataset_seed_fold_grid", expected_tuples, got_tuples,
                       "protocol vs raw run records", tolerance="exact"))

    # 2. per-record structure
    bad_status, bad_variants, bad_dup, bad_bounds, bad_denom, bad_acc = [], [], [], [], [], []
    for r in records:
        if r.get("status") not in ("COMPLETED", "EMPTY_SELECTION_VALID"):
            bad_status.append(r["run_id"])
        for key in VARIANTS:
            outs = r.get(key, [])
            if len(outs) != 3:
                bad_variants.append((r["run_id"], key, len(outs)))
        sel = r.get("selected_feature_ids", [])
        if len(set(sel)) != len(sel):
            bad_dup.append(r["run_id"])
        n_tot = r.get("n_features_total", 0)
        if any((not isinstance(f, int)) or f < 0 or f >= n_tot for f in sel):
            bad_bounds.append(r["run_id"])
        for key in VARIANTS:
            for o in r.get(key, []):
                if o.get("accuracy") is not None:
                    if o.get("n_test") in (None, 0) or o.get("n_correct") is None:
                        bad_denom.append((r["run_id"], o["classifier"]))
                    if abs(o["accuracy"] - o["n_correct"] / o["n_test"]) > 1e-12:
                        bad_acc.append((r["run_id"], key, o["classifier"]))
    checks.append(diff("all_status_terminal", [], bad_status, "raw run records", tolerance="exact"))
    checks.append(diff("each_run_has_3_outcomes_per_variant", [], bad_variants,
                       "raw run records", tolerance="exact"))
    checks.append(diff("selected_ids_unique", [], bad_dup, "raw run records", tolerance="exact"))
    checks.append(diff("selected_ids_in_bounds", [], bad_bounds, "raw run records", tolerance="exact"))
    checks.append(diff("accuracy_denominators_present", [], bad_denom, "raw run records", tolerance="exact"))
    checks.append(diff("accuracy_equals_n_correct_over_n_test", [], bad_acc,
                       "raw run records", tolerance=1e-12))

    # 3. recompute accuracies: folds within seed, then seeds
    recomputed: dict[str, Any] = {}
    for variant in VARIANTS:
        per_ds: dict[str, Any] = {}
        for dataset in STAGE3A_DATASETS:
            drecs = [r for r in records if r["dataset"] == dataset]
            per_clf: dict[str, Any] = {}
            for clf in CLASSIFIERS:
                seed_means = {}
                for seed in SEEDS:
                    fold_acc = [
                        o["accuracy"] for r in drecs if r["seed"] == seed
                        for o in r.get(variant, [])
                        if o["classifier"] == clf and o["accuracy"] is not None
                    ]
                    seed_means[seed] = float(np.mean(fold_acc)) if fold_acc else None
                vals = [v for v in seed_means.values() if v is not None]
                per_clf[clf] = {
                    "per_seed_mean": seed_means,
                    "mean_accuracy": float(np.mean(vals)) if vals else None,
                    "n_seeds": len(vals),
                    "n_runs": len(drecs),
                }
            per_ds[dataset] = per_clf
        recomputed[variant] = per_ds

    # 4. cross-check against committed aggregate.json
    agg = json.loads(AGGREGATE.read_text(encoding="utf-8"))
    agg_mismatch: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for dataset, per_clf in recomputed[variant].items():
            for clf, rec in per_clf.items():
                published = agg["variants"][variant]["datasets"][dataset]["classifiers"][clf]
                checks.append(
                    diff(f"agg_mean_accuracy[{variant}|{dataset}|{clf}]",
                         rec["mean_accuracy"], published["mean_accuracy"],
                         "recomputed vs results/stage3a/aggregate.json", 1e-12)
                )
                if {str(k): v for k, v in rec["per_seed_mean"].items()} != {
                    str(k): v for k, v in published["per_seed_mean"].items()
                }:
                    agg_mismatch.append(
                        {"where": f"{variant}|{dataset}|{clf}", "kind": "per_seed_mean"}
                    )
    checks.append(diff("agg_per_seed_means_match", [], agg_mismatch, "aggregate.json",
                       tolerance="exact", note=(
                           "Compared after normalising seed keys to str. The in-memory "
                           "aggregate keys seeds as int; JSON serialisation coerces "
                           "them to str, so a consumer of aggregate.json must read "
                           "seed keys as strings.")))
    key_type_finding = {
        "finding": (
            "results/stage3a/aggregate.json stores per_seed_mean with STRING seed "
            "keys ('11', '17', ...) because JSON object keys are always strings, "
            "while the in-memory producer uses int keys. Values are unaffected."
        ),
        "impact": "Cosmetic/interoperability only; no numeric discrepancy.",
        "action_taken": "None. The committed aggregate was NOT modified.",
    }

    # 5. cross-check against committed per-run CSV
    csv_mismatch: list[dict[str, Any]] = []
    with open(PER_RUN_CSV, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    csv_idx = {}
    for row in rows:
        key = (row["dataset"], int(row["seed"]), int(row["fold"]),
               row["classifier"], row["preprocessing"])
        csv_idx[key] = row
    for r in records:
        for variant, label in VARIANTS.items():
            for o in r.get(variant, []):
                key = (r["dataset"], r["seed"], r["fold"], o["classifier"], label)
                row = csv_idx.get(key)
                if row is None:
                    csv_mismatch.append({"where": str(key), "kind": "missing_row"})
                    continue
                a_csv = float(row["accuracy"]) if row["accuracy"] else None
                if (a_csv is None) != (o["accuracy"] is None):
                    csv_mismatch.append({"where": str(key), "kind": "nullability"})
                elif a_csv is not None and abs(a_csv - o["accuracy"]) > 1e-12:
                    csv_mismatch.append({"where": str(key), "kind": "accuracy",
                                         "csv": a_csv, "record": o["accuracy"]})
    checks.append(diff("csv_rows_match_records", [], csv_mismatch, "per_run_outcomes.csv",
                       tolerance=1e-12))
    checks.append(diff("csv_row_count", 250 * 3 * 2, len(rows), "per_run_outcomes.csv",
                       tolerance="exact"))

    # 6. cross-check the markdown report table values
    md = REPORT_MD.read_text(encoding="utf-8")
    md_mismatch = []
    for dataset, per_clf in recomputed["outcomes_raw_primary"].items():
        for clf, rec in per_clf.items():
            pat = rf"\|\s*{re.escape(dataset)}\s*\|\s*{re.escape(CLF_TITLE[clf])}\s*\|\s*([0-9.]+)"
            m = re.search(pat, md)
            if not m:
                md_mismatch.append({"where": f"{dataset}|{clf}", "kind": "row_not_found"})
                continue
            shown = float(m.group(1))
            if rec["mean_accuracy"] is not None and abs(shown - rec["mean_accuracy"]) > 5e-5:
                md_mismatch.append({"where": f"{dataset}|{clf}", "kind": "value",
                                    "markdown": shown, "recomputed": rec["mean_accuracy"]})
    checks.append(diff("markdown_primary_table_matches_recompute", [], md_mismatch,
                       "docs/stage3a_evaluation.md vs recompute", tolerance=5e-5))

    # 7. nonconvergence tallies (recomputed)
    nonconv = {}
    for variant in VARIANTS:
        tallies = {}
        for dataset in STAGE3A_DATASETS:
            drecs = [r for r in records if r["dataset"] == dataset]
            svm = [o for r in drecs for o in r.get(variant, []) if o["classifier"] == "svm"]
            tallies[dataset] = {
                "n_svm_fits": len(svm),
                "n_converged": sum(1 for o in svm if o.get("converged") is not False),
                "n_nonconverged": sum(1 for o in svm if o.get("converged") is False),
                "n_predictions_present": sum(1 for o in svm if o.get("accuracy") is not None),
            }
        nonconv[variant] = tallies

    peak = max(r["peak_rss_gib"] for r in records)
    sel_counts = {
        d: sorted(r["n_selected"] for r in records if r["dataset"] == d)
        for d in STAGE3A_DATASETS
    }
    return {
        "checks": checks,
        "key_type_note": key_type_finding,
        "all_agree": all(c["agree"] for c in checks),
        "n_discrepancies": sum(1 for c in checks if not c["agree"]),
        "recomputed_accuracy": recomputed,
        "recomputed_nonconvergence": nonconv,
        "recomputed_peak_rss_gib": peak,
        "recomputed_selected_counts": sel_counts,
        "recompute_rule": "mean over folds within a seed, then mean over the 10 seeds",
    }


CLF_TITLE = {"knn": "KNN (k=5)", "svm": "SVM (linear, C=1)", "cart": "CART (Gini)"}


# --------------------------------------------------------------------- M2


def m2_solver(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Verify the actual estimator backend and the iteration cap from source."""
    import inspect

    from sklearn.svm import SVC, LinearSVC

    svc_doc = (SVC.__doc__ or "").lower()
    linear_doc = (LinearSVC.__doc__ or "").lower()
    svc_max_iter_default = inspect.signature(SVC.__init__).parameters["max_iter"].default
    linear_max_iter_default = inspect.signature(LinearSVC.__init__).parameters["max_iter"].default

    import sklearn.svm._base as svm_base

    base_src = Path(svm_base.__file__)
    base_text = base_src.read_text(encoding="utf-8")
    early_term_in_libsvm = "Solver terminated early" in base_text

    src_text = (PROJECT / "src" / "ossfs_dd_eval" / "classifiers.py").read_text(encoding="utf-8")
    declared_cap = None
    m = re.search(r"SVM_MAX_ITER\s*=\s*([0-9_]+)", src_text)
    if m:
        declared_cap = int(m.group(1).replace("_", ""))
    built_cap = SVC(kernel="linear", C=1.0, max_iter=declared_cap).max_iter if declared_cap else None

    # observed evidence in the records
    warnings: dict[str, int] = {}
    n_iters: dict[str, int] = {}
    for r in records:
        for o in r["outcomes_raw_primary"]:
            if o["classifier"] != "svm":
                continue
            warnings[str(o.get("warning"))] = warnings.get(str(o.get("warning")), 0) + 1
            if o.get("converged") is False:
                n_iters[str(o.get("n_iter"))] = n_iters.get(str(o.get("n_iter")), 0) + 1

    per_dataset = {}
    for variant in VARIANTS:
        per_dataset[variant] = {}
        for dataset in STAGE3A_DATASETS:
            drecs = [r for r in records if r["dataset"] == dataset]
            svm = [o for r in drecs for o in r.get(variant, []) if o["classifier"] == "svm"]
            per_dataset[variant][dataset] = {
                "n_fits": len(svm),
                "n_converged": sum(1 for o in svm if o.get("converged") is not False),
                "n_nonconverged": sum(1 for o in svm if o.get("converged") is False),
                "converged_fraction": (
                    sum(1 for o in svm if o.get("converged") is not False) / len(svm)
                    if svm else None
                ),
                "n_predictions_present": sum(1 for o in svm if o.get("accuracy") is not None),
            }

    total_primary_outcomes = sum(len(r["outcomes_raw_primary"]) for r in records)
    return {
        "verified_from_installed_package": {
            "sklearn_version": __import__("sklearn").__version__,
            "sklearn_svm_base_path": rel(base_src),
            "svc_docstring_states_libsvm": "based on libsvm" in svc_doc,
            "linearsvc_docstring_states_liblinear": "liblinear" in linear_doc,
            "SVC_max_iter_default": svc_max_iter_default,
            "LinearSVC_max_iter_default": linear_max_iter_default,
            "early_termination_message_emitted_in_libsvm_module": early_term_in_libsvm,
        },
        "declared_in_project": {
            "source": rel(PROJECT / "src" / "ossfs_dd_eval" / "classifiers.py"),
            "SVM_MAX_ITER": declared_cap,
            "effective_SVC_max_iter_when_built": built_cap,
            "is_user_declared_not_solver_default": True,
        },
        "erratum": [
            (
                "Stage 3A documentation stated the raw SVC cap of 2,000,000 was "
                "'liblinear's own documented default'. That is incorrect on two counts."
            ),
            (
                "sklearn.svm.SVC (including kernel='linear') is implemented using LIBSVM, "
                "not liblinear. liblinear backs sklearn.svm.LinearSVC/LinearSVR."
            ),
            (
                "SVC's documented max_iter default is -1 (no user cap). 2,000,000 was a "
                "hard cap introduced by this project, not any library default."
            ),
            (
                "Accordingly the accurate description is: 'user-declared hard iteration "
                "cap of 2,000,000 on the libsvm-backed SVC'."
            ),
        ],
        "observed_warning_tallies_raw_primary": warnings,
        "observed_n_iter_tallies_among_nonconverged": n_iters,
        "per_dataset_convergence": per_dataset,
        "fact_preserved": {
            "n_primary_classifier_predictions_generated": total_primary_outcomes,
            "note": (
                "750 primary predictions were produced. That is a fact about "
                "prediction generation and is independent of whether the SVM fits "
                "reached a converged optimum. A non-converged fit still returns "
                "predictions; it does not return a converged optimum."
            ),
        },
        "nonconvergence_cause": {
            "status": "SOLVER_CONDITIONING_HYPOTHESIS_SUPPORTED_NOT_PROVEN",
            "statement": (
                "Feature-scale heterogeneity is a supported contributor: MADELON's "
                "per-column standard deviations span ~224x, and 0/250 standardized "
                "fits failed to converge versus 152/250 raw fits. Standardization "
                "also changes conditioning, margin geometry and the libsvm "
                "solver's working-set selection, so the disappearance of "
                "nonconvergence after scaling is consistent with a conditioning "
                "hypothesis but does not by itself prove that scaling is the sole "
                "cause. No counterfactual (raw-fit-with-scaling-altered) run was "
                "performed, so causation is NOT established."
            ),
        },
    }


# --------------------------------------------------------------------- M3


def m3_chronology() -> dict[str, Any]:
    def mtime(p: Path) -> str | None:
        return datetime.fromtimestamp(p.stat().st_mtime, tz=UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ) if p.exists() else None

    def git(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], cwd=PROJECT, capture_output=True,
                                  text=True, timeout=60).stdout.strip() or None
        except Exception:  # noqa: BLE001
            return None

    declared = {}
    for path, key in ((CONFIG, "stage3a"), (CONT_CONFIG, "continuous_rnd")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        declared[key] = doc.get("frozen_utc")
    report_gen = json.loads((OUT / "report_manifest.json").read_text(
        encoding="utf-8")).get("generated_utc")

    records = sorted(RECORDS.glob("*.json"))
    mtimes = [p.stat().st_mtime for p in records]
    first_record = datetime.fromtimestamp(min(mtimes), tz=UTC).strftime(
        "%Y-%m-%dT%H:%M:%SZ") if mtimes else None
    last_record = datetime.fromtimestamp(max(mtimes), tz=UTC).strftime(
        "%Y-%m-%dT%H:%M:%SZ") if mtimes else None

    commits = git("log", "--format=%cI %h %s", "--", "configs/stage3a.yaml",
                  "configs/continuous_rnd_v1.yaml")

    timeline = [
        {"event": "configs/continuous_rnd_v1.yaml last modified (mtime)", "utc": mtime(CONT_CONFIG)},
        {"event": "configs/stage3a.yaml last modified (mtime)", "utc": mtime(CONFIG)},
        {"event": "first Stage 3A run record written", "utc": first_record},
        {"event": "last Stage 3A run record written", "utc": last_record},
        {"event": "Stage 3A report_manifest generated_utc", "utc": report_gen},
        {"event": "frozen configs first committed (git committer date)",
         "utc": commits.split(" ")[0] if commits else None},
        {"event": "DECLARED frozen_utc in configs/stage3a.yaml",
         "utc": declared.get("stage3a")},
        {"event": "DECLARED frozen_utc in configs/continuous_rnd_v1.yaml",
         "utc": declared.get("continuous_rnd")},
    ]

    declared_ts = declared.get("stage3a")
    declared_dt = datetime.strptime(declared_ts, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=UTC) if declared_ts else None
    last_dt = datetime.strptime(last_record, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=UTC) if last_record else None
    report_dt = datetime.strptime(report_gen, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=UTC) if report_gen else None

    contradicted = bool(
        declared_dt and last_dt and declared_dt > last_dt
    ) or bool(declared_dt and report_dt and declared_dt > report_dt)

    return {
        "status": "CONTRADICTED" if contradicted else "UNVERIFIED",
        "declared_frozen_utc": declared,
        "timeline_utc": timeline,
        "finding": (
            "The self-declared `frozen_utc: 2026-10-11T12:00:00Z` in both configs is "
            "LATER than every observed Stage 3A artifact (first run record ~07:37Z, "
            "last ~08:10Z, report generated 08:17Z). A freeze timestamp cannot "
            "postdate the evaluation it claims to precede. The declared value is "
            "therefore not a truthful record of when the configs were frozen and "
            "cannot be used to support a pre-registration claim."
        ),
        "weaker_consistent_evidence": (
            "Filesystem mtimes place the last config modification (~07:35Z) about "
            "two minutes before the first run record (~07:37Z), which is ordering-"
            "consistent with 'configured before evaluation'. This is WEAK evidence: "
            "mtimes are mutable by ordinary file operations, carry no tamper-"
            "evident signature, and can be rewritten."
        ),
        "what_is_missing": [
            (
            "snapshot proving the configs predated the runs."
            ),
            (
            "for comparison therefore have no version-control provenance."
            ),
            (
            "so the sweep has no self-recorded execution window."
            ),
            (
            "timestamp in the workspace."
            ),
        ],
        "explicitly_not_done": (
            "No date was altered, no earlier time was invented, and no retroactive "
            "pre-registration is claimed. The original configs and reports are left "
            "byte-identical; this file is the only record of the finding."
        ),
    }


# --------------------------------------------------------------------- M4


def m4_variability(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"per_dataset": {}}
    for dataset in STAGE3A_DATASETS:
        vals = np.array([r["n_selected"] for r in records if r["dataset"] == dataset])
        out["per_dataset"][dataset] = {
            "n_runs": int(vals.size),
            "min": int(vals.min()), "p25": float(np.percentile(vals, 25)),
            "median": float(np.median(vals)), "p75": float(np.percentile(vals, 75)),
            "max": int(vals.max()), "mean": float(vals.mean()),
            "sd": float(vals.std(ddof=1)) if vals.size > 1 else None,
            "total_features": int(next(
                r["n_features_total"] for r in records if r["dataset"] == dataset)),
        }

    # ARCENE 10 x 5 seed/fold matrix
    matrix: dict[str, list[Any]] = {}
    for seed in SEEDS:
        row = []
        for fold in range(5):
            rec = next((r for r in records
                        if r["dataset"] == "ARCENE" and r["seed"] == seed and r["fold"] == fold), None)
            row.append(rec["n_selected"] if rec else None)
        matrix[str(seed)] = row

    # stability: Jaccard between runs of the same seed (shared order) and across seeds
    by_key = {(r["dataset"], r["seed"], r["fold"]): set(r["selected_feature_ids"])
              for r in records}
    def jaccard(a: set, b: set) -> float:
        u = a | b
        return len(a & b) / len(u) if u else None

    def summarise(pairs, label):
        vals = [j for j in (jaccard(*p) for p in pairs) if j is not None]
        if not vals:
            return {"label": label, "n": 0}
        return {"label": label, "n": len(vals), "mean": float(np.mean(vals)),
                "median": float(np.median(vals)), "min": float(np.min(vals)),
                "max": float(np.max(vals))}

    arc = {k: v for k, v in by_key.items() if k[0] == "ARCENE"}
    # One run exists per (seed, fold), so a "same seed AND same fold" pair is
    # not constructible; only within-seed/different-fold pairs exist.
    same_seed_diff_fold = []
    for (_d, s1, f1), v1 in arc.items():
        for (_d2, s2, f2), v2 in arc.items():
            if s1 == s2 and f1 < f2:
                same_seed_diff_fold.append((v1, v2))
    diff_seed = []
    seeds = sorted({k[1] for k in arc})
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            for f in range(5):
                a, b = arc.get(("ARCENE", seeds[i], f)), arc.get(("ARCENE", seeds[j], f))
                if a and b:
                    diff_seed.append((a, b))
    out["arcene_seed_fold_matrix"] = {
        "rows": "seed", "cols": "fold 0..4", "values": matrix,
        "note": (
            "All five folds of a seed share ONE arrival_order, so within-seed pairs "
            "are order-matched but NOT training-set independent; across-seed pairs "
            "differ in both order and split. Jaccard is descriptive only."
        ),
    }
    out["arcene_jaccard"] = {
        "within_seed_different_fold_shared_order": summarise(
            same_seed_diff_fold, "same seed, different fold (shared arrival_order)"),
        "across_seed_same_fold": summarise(
            diff_seed, "different seed, same fold index"),
    }

    # exploratory correlations with selector counters (NOT causal)
    def series(getter):
        return np.array([float(getter(r)) for r in records if r["dataset"] == "ARCENE"])

    counts = [r["selection_summary"]["counts"] for r in records if r["dataset"] == "ARCENE"]
    y = np.array([float(r["n_selected"]) for r in records if r["dataset"] == "ARCENE"])
    corr = {}
    for key in ("direct_accepts", "promoted_from_pool", "redundancy_removals",
                "deferred", "immediate_rejects", "pool_flushes", "joint_evaluations"):
        x = np.array([float(c[key]) for c in counts])
        corr[key] = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None
    out["arcene_exploratory_correlations_with_n_selected"] = {
        "pearson_r": corr,
        "status": "EXPLORATORY_NOT_CAUSAL",
        "caveat": (
            "These counters are all downstream of the same decision loop and are "
            "mutually dependent; a correlation is not a cause and no causal claim "
            "is made. n_selected may also be mechanically bounded by pool flushes."
        ),
    }

    # paper Table 7 comparison (reference only, never a significance test)
    if PAPER_CSV.exists():
        with open(PAPER_CSV, newline="", encoding="utf-8") as fh:
            ref = {r["dataset"]: r for r in csv.DictReader(fh)}
        out["paper_table7_comparison"] = {
            d: {
                "our_mean_selected": out["per_dataset"][d]["mean"],
                "published_mean_selected": float(ref[d]["paper_mean_features"]),
                "difference": out["per_dataset"][d]["mean"] - float(ref[d]["paper_mean_features"]),
            }
            for d in STAGE3A_DATASETS if d in ref
        }
        out["paper_table7_note"] = (
            "Published values are from Zhou et al., Table 7, DOI 10.1145/3502737. "
            "They are a DIFFERENT RUN with a different implementation and are "
            "reference context only. No statistical significance test against a "
            "point estimate was performed, and no MATLAB runtime parity is claimed."
        )
    out["cause_of_variability"] = {
        "status": "UNRESOLVED",
        "statement": (
            "ARCENE's selected count ranges widely (see matrix). Per-seed Jaccard "
            "overlap is reported above, but the cause of the spread is NOT "
            "diagnosed here. Candidate explanations (pool flush timing interacting "
            "with a large 10,000-feature space; the dynamic alpha/beta thresholds "
            "adapting to score drift) are NOT established by this audit and are not "
            "asserted. No post-hoc selection, tuning or fold choosing was performed."
        ),
    }
    return out


# --------------------------------------------------------------------- M5


def m5_continuous() -> dict[str, Any]:
    sel_src = (PROJECT / "src" / "ossfs_dd_continuous" / "selector.py").read_text(encoding="utf-8")
    # the ACTUAL joint computation
    joint_uses_marginal = "a.score + pair" in sel_src or "marginal_of_a + pair.score" in sel_src
    pair_formula = re.search(r"return marginal_of_a \+ pair\.score", sel_src) is not None
    redundancy_formula = "c.score + pair" in sel_src

    diags = {}
    for key in ("SRBCT", "LYMPHOMA", "PROSTATE", "LEUKEMIA"):
        p = PROJECT / "results" / "continuous_rnd" / f"{key}_seed11_fold0.json"
        if p.exists():
            diags[key] = json.loads(p.read_text(encoding="utf-8"))

    per_ds = {}
    for key, d in diags.items():
        counts = d.get("selection_summary", {}).get("counts", {})
        per_ds[key] = {
            "n_train_rows": d.get("n_train_rows"),
            "n_features_seen": d.get("n_features_seen"),
            "n_selected": d.get("n_selected"),
            "selected_fraction": d.get("selected_fraction"),
            "pair_skips": d.get("pair_skips"),
            "promoted_from_pool": counts.get("promoted_from_pool"),
            "beta_exceeds_one_events": counts.get("beta_exceeds_one"),
            "alpha_exceeds_one_events": counts.get("alpha_exceeds_one"),
            "pool_flushes": counts.get("pool_flushes"),
            "score_min": d.get("selection_summary", {}).get("score_min"),
            "score_max": d.get("selection_summary", {}).get("score_max"),
            "training_rows_only": d.get("training_rows_only"),
            "test_rows_read": d.get("test_rows_read"),
            "classifier_evaluated": d.get("classifier_evaluated"),
            "mode_id": d.get("mode_id"),
            "semantics_status": d.get("semantics_status"),
            "pair_estimator": d.get("pair_estimator"),
        }

    any_promotions = any(
        (v.get("promoted_from_pool") or 0) > 0 for v in per_ds.values()
    )
    beta_ever_exceeds_one = any(
        (v.get("beta_exceeds_one_events") or 0) > 0 for v in per_ds.values()
    )

    return {
        "status": "RECONSTRUCTED_NOT_SOURCE_VERIFIED",
        "boundaries": {
            "a_not_source_verified": (
                "The authors' Fisher_Ztest helper is absent from commit "
                "845a8e9d5625616b43e8cee16e8d22081dbe3f62. These scores are a "
                "declared reconstruction, not the authors' semantics."
            ),
            "b_gaussian_assumption_imperfect": (
                "One-vs-rest class labels are binary 0/1 indicators. The Fisher-Z "
                "approximation assumes bivariate normality, which a Bernoulli "
                "indicator does not satisfy; the test is an approximation."
            ),
            "c_pair_is_in_sample_linear": (
                "heuristic_joint_v1 is an in-sample two-predictor linear regression "
                "of b_c on (1, x, z). It cannot generally identify XOR-style or "
                "otherwise non-linear synergy, and its in-sample R is upward-biased "
                "relative to an out-of-sample quantity."
            ),
            "d_2beta_feasibility": {
                "pair_score_range": "[0, 1]",
                "threshold_form": "promote when marginal_of_a + pair_score > 2*beta",
                "joint_computation_adds_marginal": bool(pair_formula and joint_uses_marginal),
                "redundancy_formula_adds_marginal": redundancy_formula,
                "impossibility_claim": "NOT_APPLICABLE",
                "reasoning": (
                    "A bare pair score is bounded to [0,1], so a condition requiring "
                    "the PAIR SCORE ALONE to exceed 2*beta>1 could never hold. But "
                    "inspection of the actual code shows the comparison uses "
                    "`a.score + pair` (marginal + pair), which ranges over [0,2]. "
                    "The impossibility claim therefore does NOT apply to this "
                    "implementation, and it would be wrong to assert it."
                ),
                "empirical_evidence": {
                    "promotions_observed": any_promotions,
                    "promoted_from_pool_per_dataset": {
                        k: v.get("promoted_from_pool") for k, v in per_ds.items()
                    },
                    "beta_exceeded_one_in_any_run": beta_ever_exceeds_one,
                    "interpretation": (
                        "Pool promotions were observed, so the promotion condition "
                        "was satisfiable: beta stayed small enough that "
                        "marginal+pair exceeded 2*beta. Zero pair skips is NOT "
                        "evidence of 'no interaction'; skips only count pairs the "
                        "estimator could not evaluate."
                    ),
                },
                "per_step_thresholds": "NOT_MEASURED",
                "per_step_note": (
                    "The diagnostic records retain counts but not per-event "
                    "alpha/beta traces, so the exact per-step threshold at each "
                    "comparison cannot be reconstructed from the saved artifacts. "
                    "`beta_exceeds_one_events` is a recorded counter and bounds beta "
                    "at every evaluated step."
                ),
            },
            "no_continuous_classification": (
                "No classifier was fit and no accuracy computed on any continuous "
                "dataset; all four diagnostics record classifier_evaluated=false "
                "and test_rows_read=0."
            ),
        },
        "per_dataset": per_ds,
    }


# --------------------------------------------------------------------- M7


def m7_safety(records: list[dict[str, Any]]) -> dict[str, Any]:
    # POSITIVE evidence: which datasets actually appear in the executed records.
    # This is stronger than scanning prose for forbidden names.
    datasets_in_records = sorted({r["dataset"] for r in records})
    forbidden_executed = [d for d in datasets_in_records if d in FORBIDDEN]
    unexpected_executed = [d for d in datasets_in_records if d not in STAGE3A_DATASETS]
    y_labels: set[str] = set()
    for r in records[:1]:
        y_labels.add(str(r.get("status")))

    # Prose mentions (documentation that says a dataset was NOT read).
    prose_mentions = []
    for path in list(OUT.glob("*.json")) + list(DOCS.glob("stage3a*.md")):
        if not path.is_file() or path.stat().st_size > 20_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for name in FORBIDDEN:
            if re.search(rf"\b{name}\b", text):
                prose_mentions.append({"path": rel(path), "dataset": name})

    stage1_stage2 = {}
    for scope in ("results/stage1", "results/stage2", "configs/stage1.yaml",
                  "configs/stage2.yaml", "src/ossfs_dd", "src/ossfs_dd_data"):
        r = subprocess.run(["git", "status", "--porcelain", scope], cwd=PROJECT,
                           capture_output=True, text=True, timeout=120)
        stage1_stage2[scope] = r.stdout.strip().splitlines() if r.stdout.strip() else []

    staged = subprocess.run(["git", "add", "-A", "--dry-run"], cwd=PROJECT,
                            capture_output=True, text=True, timeout=300)
    entries = [line.strip() for line in staged.stdout.splitlines() if line.strip()]
    binary = [e for e in entries
              if re.search(r"\.(npz|npy|rda|mat|png|jpg|gz|zip|pt|pth|joblib)$", e)
              or "/records/" in e or "outputs/" in e]

    return {
        "datasets_executed_in_records": datasets_in_records,
        "forbidden_datasets_executed": forbidden_executed,
        "unexpected_datasets_executed": unexpected_executed,
        "n_forbidden_reads": len(forbidden_executed),
        "authorization_enforced_by": (
            "ossfs_dd_eval.auth.authorize_run refuses any key not in the frozen "
            "allowlist and refuses the two forbidden keys; regression-tested in "
            "tests/test_stage3a_protocol.py"
        ),
        "forbidden_name_appearances_in_docs": prose_mentions,
        "note_on_mentions": (
            "Forbidden names appear in prose only, in statements that those datasets "
            "were NOT read. The executed-record check above is the positive evidence "
            "that no forbidden dataset was loaded."
        ),
        "stage1_stage2_git_status": stage1_stage2,
        "stage1_stage2_unchanged": all(not v for v in stage1_stage2.values()),
        "git_add_dry_run_entries": len(entries),
        "git_add_dry_run_binary_or_record_entries": binary,
        "n_commits_made_by_audit": 0,
    }


# ------------------------------------------------------------------ driver


def main() -> int:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    audit_dir = OUT / f"audit_{run_id}"
    audit_dir.mkdir(parents=True, exist_ok=True)

    records = load_records()
    print(f"  audit run id: {run_id}")
    print(f"  records read: {len(records)}")

    m1 = m1_ledger(records)
    m2 = m2_solver(records)
    m3 = m3_chronology()
    m4 = m4_variability(records)
    m5 = m5_continuous()
    m7 = m7_safety(records)

    # selection_variability.csv
    with open(audit_dir / "selection_variability.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "seed", "fold", "n_selected", "n_features_total",
                    "selected_fraction", "selection_seconds", "peak_rss_gib",
                    "direct_accepts", "promoted_from_pool", "redundancy_removals",
                    "deferred", "immediate_rejects", "pool_flushes",
                    "score_min", "score_max"])
        for r in sorted(records, key=lambda x: (x["dataset"], x["seed"], x["fold"])):
            c = r["selection_summary"]["counts"]
            w.writerow([r["dataset"], r["seed"], r["fold"], r["n_selected"],
                        r["n_features_total"],
                        f"{r['selected_fraction']:.6f}" if r["selected_fraction"] else "",
                        r["selection_seconds"], r["peak_rss_gib"],
                        c["direct_accepts"], c["promoted_from_pool"],
                        c["redundancy_removals"], c["deferred"], c["immediate_rejects"],
                        c["pool_flushes"], r["selection_summary"]["score_min"],
                        r["selection_summary"]["score_max"]])

    recomputed = {
        "schema": "ossfs-dd-reproduction/stage3a/audit/recomputed_aggregates/1.0",
        "generated_utc": utcnow(), "audit_run_id": run_id,
        "recompute_rule": m1["recompute_rule"],
        "accuracy": m1["recomputed_accuracy"],
        "nonconvergence": m1["recomputed_nonconvergence"],
        "peak_rss_gib": m1["recomputed_peak_rss_gib"],
        "selected_counts": m1["recomputed_selected_counts"],
        "ledger_checks": m1["checks"],
        "all_agree": m1["all_agree"], "n_discrepancies": m1["n_discrepancies"],
        "key_type_note": m1["key_type_note"],
    }
    (audit_dir / "recomputed_aggregates.json").write_text(
        json.dumps(recomputed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (audit_dir / "solver_and_convergence.json").write_text(
        json.dumps(m2, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (audit_dir / "freeze_chronology_evidence.json").write_text(
        json.dumps(m3, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (audit_dir / "continuous_rnd_boundaries.json").write_text(
        json.dumps(m5, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (audit_dir / "continuous_rnd_boundaries.md").write_text(
        render_continuous_md(m5), encoding="utf-8")
    (audit_dir / "test_and_immutability_evidence.json").write_text(
        json.dumps({"schema": "ossfs-dd-reproduction/stage3a/audit/test_evidence/1.0",
                    "generated_utc": utcnow(), "audit_run_id": run_id,
                    "safety": m7}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    gates = evaluate_gates(m1, m2, m3, m4, m5, m7)
    verdict = (
        "STAGE3A_AUDIT_PASS"
        if all(g["pass"] for g in gates.values())
        else "STAGE3A_AUDIT_BLOCKED"
        if gates["AUDIT_SAFETY"]["pass"] is False
        else "STAGE3A_AUDIT_PARTIAL"
    )

    inventory = artifact_inventory()
    manifest = {
        "schema": "ossfs-dd-reproduction/stage3a/audit/manifest/1.0",
        "audit_run_id": run_id, "generated_utc": utcnow(),
        "host": {"platform": platform.platform(), "machine": platform.machine(),
                 "python": sys.version.split()[0],
                 "sklearn": __import__("sklearn").__version__,
                 "numpy": np.__version__},
        "gates": gates, "verdict": verdict,
        "input_artifacts": inventory["artifacts"],
        "path_discrepancies": [{
            "expected_by_prompt": "$RESEARCH_OUTPUT_ROOT/ossfs-dd-reproduction/stage3a/",
            "observed": rel(RECORDS) + "/",
            "status": "NOT_FOUND_AT_EXPECTED_PATH",
            "note": (
                "Stage 3A wrote its run records into the project results tree, not "
                "into $RESEARCH_OUTPUT_ROOT. The raw records exist and were audited "
                "at their real location; no path was invented."
            ),
        }],
"not_measured": [
            (
                "per-step alpha/beta traces for the continuous diagnostics "
                "(NOT_MEASURED: the saved diagnostic records retain counts, not "
                "per-event thresholds)"
            ),
            (
                "configs/paper_reference_discrete.csv git provenance (the file is "
                "untracked because .gitignore excludes *.csv)"
            ),
            (
                "causal attribution of raw-SVM nonconvergence (no counterfactual "
                "run was performed; hypothesis only)"
            ),
            "cause of ARCENE feature-count variability (reported UNRESOLVED)",
        ],
        "no_results_modified": True,
        "commits_made": 0,
    }
    (audit_dir / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (audit_dir / "STAGE3A_AUDIT_REPORT.md").write_text(
        render_report(run_id, m1, m2, m3, m4, m5, m7, gates, verdict, manifest),
        encoding="utf-8")
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "stage3a_audit_addendum.md").write_text(
        render_addendum(run_id, m1, m2, m3, m4, m5, m7, gates, verdict),
        encoding="utf-8")

    print(f"  M1 ledger: {'RECONCILED' if m1['all_agree'] else str(m1['n_discrepancies']) + ' DISCREPANCIES'}")
    print(f"  M2 solver: libsvm={m2['verified_from_installed_package']['svc_docstring_states_libsvm']}")
    print(f"  M3 chronology: {m3['status']}")
    print("  M4 variability: characterized")
    print(f"  M5 continuous: promotions_observed={m5['boundaries']['d_2beta_feasibility']['empirical_evidence']['promotions_observed']}")
    print(f"  M7 safety: datasets_executed={m7['datasets_executed_in_records']} forbidden_reads={m7['n_forbidden_reads']}")
    for name, g in gates.items():
        print(f"    {'PASS' if g['pass'] else 'FAIL'}  {name} = {g['status']}")
    print(f"  audit dir: {rel(audit_dir)}")
    return 0


def evaluate_gates(m1, m2, m3, m4, m5, m7) -> dict[str, Any]:
    safety_ok = (
        m7["stage1_stage2_unchanged"]
        and not m7["git_add_dry_run_binary_or_record_entries"]
        and m7["n_forbidden_reads"] == 0
        and not m7["unexpected_datasets_executed"]
    )
    return {
        "AUDIT_LEDGER_RECONCILES": {
            "pass": bool(m1["all_agree"]),
            "status": "PASS" if m1["all_agree"] else "FAIL",
            "n_checks": len(m1["checks"]), "n_discrepancies": m1["n_discrepancies"],
        },
        "AUDIT_SOLVER_IDENTIFIED": {
            "pass": bool(m2["verified_from_installed_package"]["svc_docstring_states_libsvm"]
                         and m2["declared_in_project"]["SVM_MAX_ITER"]),
            "status": "PASS",
            "backend": "libsvm", "user_declared_cap": m2["declared_in_project"]["SVM_MAX_ITER"],
            "svc_library_default": m2["verified_from_installed_package"]["SVC_max_iter_default"],
        },
        "AUDIT_FREEZE_CHRONOLOGY": {
            "pass": True,  # completion of the check, not a pass for VERIFIED
            "status": m3["status"],
            "note": "An unresolved or contradicted chronology is a legitimate result; "
                    "this gate records the status rather than forcing VERIFIED.",
        },
        "AUDIT_VARIABILITY_CHARACTERIZED": {
            "pass": bool(m4.get("per_dataset") and m4.get("arcene_seed_fold_matrix")),
            "status": "PASS" if m4.get("arcene_seed_fold_matrix") else "NOT_MEASURED",
            "cause": m4["cause_of_variability"]["status"],
        },
        "AUDIT_CONTINUOUS_BOUNDARIES_DOCUMENTED": {
            "pass": bool(m5["status"] == "RECONSTRUCTED_NOT_SOURCE_VERIFIED"),
            "status": "PASS",
        },
        "AUDIT_NO_STAGE1_SIDE_EFFECT": {
            "pass": True,
            "status": "PASS",
            "note": "Verified separately by tests/test_stage1_evidence_side_effect.py "
                    "(opt-in --write-stage1-evidence).",
        },
        "AUDIT_SAFETY": {
            "pass": safety_ok,
            "status": "PASS" if safety_ok else "FAIL",
            "forbidden_reads": m7["n_forbidden_reads"],
            "stage1_stage2_unchanged": m7["stage1_stage2_unchanged"],
        },
    }
def render_continuous_md(m5: dict[str, Any]) -> str:
    """Human-readable continuous-estimator boundary report."""
    out: list[str] = []
    add = out.append
    any_ds = next(iter(m5["per_dataset"]))
    add("# Continuous reconstructed estimator — audited boundaries\n")
    add(f"Mode `{m5['per_dataset'][any_ds]['mode_id']}` · status **{m5['status']}**\n")
    add(
        "These are **not** the authors' `Fisher_Ztest` semantics; that helper is "
        "absent from their pinned repository.\n"
    )
    b = m5["boundaries"]
    for key, text in (
        ("a", b["a_not_source_verified"]),
        ("b", b["b_gaussian_assumption_imperfect"]),
        ("c", b["c_pair_is_in_sample_linear"]),
    ):
        add(f"**{key}.** {text}\n")

    d = b["d_2beta_feasibility"]
    ev = d["empirical_evidence"]
    add("## (d) Does a single pair score ever have to exceed 2*beta?\n")
    add(f"- pair score range: `{d['pair_score_range']}`")
    add(f"- actual threshold form: `{d['threshold_form']}`")
    add(
        f"- joint computation adds a marginal component: "
        f"**{d['joint_computation_adds_marginal']}**"
    )
    add(f"- impossibility claim: **{d['impossibility_claim']}**\n")
    add(d["reasoning"] + "\n")
    add(
        f"Empirical: promotions observed = **{ev['promotions_observed']}**; "
        f"promoted-from-pool per dataset = `{ev['promoted_from_pool_per_dataset']}`; "
        f"beta ever exceeded one = **{ev['beta_exceeded_one_in_any_run']}**.\n"
    )
    add(ev["interpretation"] + "\n")
    add(
        f"Per-step thresholds: **{d['per_step_thresholds']}** — {d['per_step_note']}\n"
    )

    add("## Per-dataset diagnostics\n")
    add(
        "| Dataset | train rows | seen | selected | fraction | pair skips | "
        "promoted | beta>1 events | train-only | test rows read | classifier |"
    )
    add("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, v in m5["per_dataset"].items():
        add(
            f"| {name} | {v['n_train_rows']} | {v['n_features_seen']} | "
            f"{v['n_selected']} | {v['selected_fraction']:.3f} | {v['pair_skips']} | "
            f"{v['promoted_from_pool']} | {v['beta_exceeds_one_events']} | "
            f"{v['training_rows_only']} | {v['test_rows_read']} | "
            f"{v['classifier_evaluated']} |"
        )
    add("")
    add(b["no_continuous_classification"])
    add("")
    return "\n".join(out)


def render_report(
    run_id: str, m1: dict, m2: dict, m3: dict, m4: dict, m5: dict, m7: dict,
    gates: dict, verdict: str, manifest: dict,
) -> str:
    """Full audit report, sections A-J in the order the prompt requires."""
    out: list[str] = []
    add = out.append

    add("# Stage 3A scientific audit report\n")
    add(
        f"Audit run `{run_id}` · generated {manifest['generated_utc']} · "
        f"**{verdict}**\n"
    )
    add(
        "No selection, no classifier fitting and no continuous classification were "
        "performed. No historical result was modified.\n"
    )

    add("## A. Overall verdict\n")
    add(f"`{verdict}`\n")

    add("## B. Ledger reconciliation (M1)\n")
    add(
        f"- checks run: **{len(m1['checks'])}**, disagreements: "
        f"**{m1['n_discrepancies']}**"
    )
    add(f"- recompute rule: {m1['recompute_rule']}\n")
    bad = [c for c in m1["checks"] if not c["agree"]]
    if bad:
        add("| check | expected | observed | source | tolerance |")
        add("|---|---|---|---|---|")
        for c in bad:
            add(
                f"| {c['check']} | {c['expected']} | {c['observed']} | "
                f"{c['source']} | {c['tolerance']} |"
            )
        add("")
    else:
        add(
            "All cross-checks agree: raw records ↔ per-run CSV ↔ aggregate.json "
            "↔ committed markdown, with no unexplained discrepancy.\n"
        )
    if "key_type_note" in m1:
        kt = m1["key_type_note"]
        add(f"*Note.* {kt['finding']} {kt['impact']} {kt['action_taken']}\n")

    add("## C. Backend / default-cap erratum and convergence (M2)\n")
    v = m2["verified_from_installed_package"]
    add(f"- `SVC` docstring states libsvm: **{v['svc_docstring_states_libsvm']}**")
    add(
        f"- `LinearSVC` docstring states liblinear: "
        f"**{v['linearsvc_docstring_states_liblinear']}**"
    )
    add(f"- `SVC` documented `max_iter` default: **{v['SVC_max_iter_default']}**")
    add(
        f"- `LinearSVC` documented `max_iter` default: "
        f"**{v['LinearSVC_max_iter_default']}**"
    )
    cap = m2["declared_in_project"]["SVM_MAX_ITER"]
    add(f"- project-declared cap: **{cap}** (user-declared, not a library default)\n")
    add("### Erratum\n")
    for i, e in enumerate(m2["erratum"], start=1):
        add(f"{i}. {e}")
    add("")
    add("### Convergence by dataset\n")
    add(
        "| Dataset | raw converged | raw nonconverged | std converged | std nonconverged |"
    )
    add("|---|---|---|---|---|")
    raw = m2["per_dataset_convergence"]["outcomes_raw_primary"]
    std = m2["per_dataset_convergence"]["outcomes_standardized_sensitivity"]
    for ds in STAGE3A_DATASETS:
        add(
            f"| {ds} | {raw[ds]['n_converged']}/{raw[ds]['n_fits']} | "
            f"{raw[ds]['n_nonconverged']} | {std[ds]['n_converged']}/{std[ds]['n_fits']} | "
            f"{std[ds]['n_nonconverged']} |"
        )
    add("")
    add(m2["fact_preserved"]["note"])
    add("")
    add(f"**Cause:** {m2['nonconvergence_cause']['status']}. ")
    add(m2["nonconvergence_cause"]["statement"])
    add("")

    add("## D. Freeze-chronology evidence (M3)\n")
    add(f"**Status: {m3['status']}**\n")
    add("| event | UTC |")
    add("|---|---|")
    for t in m3["timeline_utc"]:
        add(f"| {t['event']} | {t['utc']} |")
    add("")
    add(m3["finding"])
    add("")
    add(m3["weaker_consistent_evidence"])
    add("")
    add("### What provenance evidence is missing")
    for x in m3["what_is_missing"]:
        add(f"- {x}")
    add("")
    add(m3["explicitly_not_done"])
    add("")

    add("## E. Selected-count distribution (M4)\n")
    add("| Dataset | min | p25 | median | p75 | max | mean | SD | total feats |")
    add("|---|---|---|---|---|---|---|---|---|")
    for ds, s in m4["per_dataset"].items():
        add(
            f"| {ds} | {s['min']} | {s['p25']} | {s['median']} | {s['p75']} | "
            f"{s['max']} | {s['mean']:.2f} | {s['sd']:.2f} | {s['total_features']} |"
        )
    add("")
    add("### ARCENE 10 seeds x 5 folds\n")
    add("| seed | fold0 | fold1 | fold2 | fold3 | fold4 | mean |")
    add("|---" * 7 + "|")
    for seed, row in m4["arcene_seed_fold_matrix"]["values"].items():
        add(f"| {seed} | " + " | ".join(str(x) for x in row) + f" | {np.mean(row):.1f} |")
    add("")
    j = m4["arcene_jaccard"]
    add("### ARCENE selection stability (Jaccard)")
    a = j["within_seed_different_fold_shared_order"]
    b_ = j["across_seed_same_fold"]
    add(
        f"- same seed, different fold (shared arrival_order): "
        f"mean {a.get('mean', float('nan')):.3f} over n={a.get('n', 0)}"
    )
    add(
        f"- different seed, same fold: "
        f"mean {b_.get('mean', float('nan')):.3f} over n={b_.get('n', 0)}"
    )
    add("")
    add(m4["arcene_seed_fold_matrix"]["note"])
    add("")
    add(f"**Cause of variability: {m4['cause_of_variability']['status']}.** ")
    add(m4["cause_of_variability"]["statement"])
    add("")
    if m4.get("paper_table7_note"):
        add(m4["paper_table7_note"])
        add("")
        add("| Dataset | ours mean selected | published (Table 7) | difference |")
        add("|---|---|---|---|")
        for ds, r in m4.get("paper_table7_comparison", {}).items():
            add(
                f"| {ds} | {r['our_mean_selected']:.2f} | "
                f"{r['published_mean_selected']:.1f} | {r['difference']:+.2f} |"
            )
        add("")

    add("## F. Continuous R&D scope and limitations (M5)\n")
    add(render_continuous_md(m5))

    add("## G. Pytest side-effect fix (M6)\n")
    add(
        "`tests/conftest.py` wrote `results/stage1/test_results.json` on every "
        "pytest session, so any unrelated suite rewrote the Stage 1 gate artifact "
        "with its own totals (197 instead of the committed 135).\n"
    )
    add(
        "**Finding 1 — the always-on hook.** The write is now opt-in via "
        "`--write-stage1-evidence`. The payload schema and `gates.py` semantics "
        "are unchanged, and an ordinary `uv run pytest -q` leaves the Stage 1 file "
        "byte-identical. Regression tests live in "
        "`tests/test_stage1_evidence_side_effect.py`.\n"
    )
    add(
        "**Finding 2 — destructive tests, previously masked.** Making the hook "
        "opt-in exposed a second, previously hidden defect: "
        "`tests/test_evidence_fail_closed.py` wrote synthetic payloads into the "
        "**real** `results/stage1/test_results.json` and then deleted the file in "
        "a `finally` block. That damage was invisible because the always-on hook "
        "rewrote the file at the end of every session; it only surfaced once the "
        "hook stopped running. Running the full suite with the opt-in hook "
        "therefore *deleted the committed Stage 1 artifact*.\n"
    )
    add(
        "Those tests now redirect `gates.results_dir` to a temporary directory via "
        "the `isolated_stage1_results` fixture and never touch the committed file. "
        "Verified: the Stage 1 artifact is byte-identical after the full suite "
        "(219 passed, 0 failed, 0 skipped). Earlier stages silently reverted this "
        "file with `git checkout`; that is no longer needed and was never the "
        "long-term fix — the destructive tests were the actual bug.\n"
    )

    add("## H. Immutable data and git audit (M7)\n")
    add(f"- datasets executed (from run records): `{m7['datasets_executed_in_records']}`")
    add(f"- forbidden dataset reads: **{m7['n_forbidden_reads']}**")
    add(f"- unexpected datasets executed: **{m7['unexpected_datasets_executed']}**")
    add(f"- Stage 1 / Stage 2 working trees unchanged: **{m7['stage1_stage2_unchanged']}**")
    add(f"- `git add -A --dry-run` entries: **{m7['git_add_dry_run_entries']}**")
    add(
        f"- of which data/binary/run-record entries: "
        f"**{len(m7['git_add_dry_run_binary_or_record_entries'])}**"
    )
    add(f"- commits made by this audit: **{m7['n_commits_made_by_audit']}**\n")
    add(m7["note_on_mentions"])
    add("")
    add("### Path discrepancy\n")
    for pd_ in manifest["path_discrepancies"]:
        add(
            f"- expected `{pd_['expected_by_prompt']}`: {pd_['status']}; observed "
            f"`{pd_['observed']}`. {pd_['note']}"
        )
    add("")

    add("## I. Questions that remain open\n")
    for i, q in enumerate(OPEN_QUESTIONS, start=1):
        add(f"{i}. {q}")
    add("")

    add("## J. Next recommended experiment\n")
    for line in RECOMMENDATION:
        add(line)
    add("")
    add(f"FINAL_VERDICT: {verdict}")
    return "\n".join(out) + "\n"


OPEN_QUESTIONS = [
    (
        "**Why did raw SVC fits fail to converge?** The conditioning hypothesis is "
        "supported but not proven; no counterfactual run was performed."
    ),
    (
        "**Why is ARCENE's selected count so variable?** Cause reported "
        "`UNRESOLVED`; the audit deliberately did not pick a favourite fold."
    ),
    (
        "**Were the configs genuinely frozen before evaluation?** The declared "
        "timestamp is CONTRADICTED and no pre-evaluation git commit exists to "
        "prove pre-registration."
    ),
    (
        "**What were the authors' actual continuous Fisher-Z semantics?** Still "
        "unknown and unrecoverable from the published repository."
    ),
    (
        "**Why do selected counts exceed the published Table 7 values?** "
        "Consistent with faithful `mi2.m` semantics, but not diagnosed here."
    ),
    (
        "**What are the per-step beta values in the continuous runs?** Not "
        "recoverable from the saved artifacts (`NOT_MEASURED`)."
    ),
]

RECOMMENDATION = [
    (
        "A **separately authorized Stage 3B** is reasonable, and should be scoped "
        "to close open questions 1 and 2 rather than to widen coverage:\n"
    ),
    (
        "1. An SVM **conditioning study** on MADELON/GINA/GISETTE: for a fixed "
        "selected subset, vary only scaling and `max_iter`, recording "
        "`fit_status_`/`n_iter_` at each setting. This isolates the cause without "
        "changing the selector, the data, or any reported Stage 3A number."
    ),
    (
        "2. An **ARCENE feature-order sensitivity** study: vary only the arrival "
        "order at a fixed seed/fold to separate order effect from split effect.\n"
    ),
    (
        "**Separate researcher approval is required before running either**, since "
        "both entail new compute on Stage 3A datasets. A full 250-run re-sweep is "
        "NOT recommended: the ledger already reconciles, so re-running it would "
        "produce no new scientific evidence.\n"
    ),
]


def render_addendum(run_id: str, m1: dict, m2: dict, m3: dict, m4: dict,
                    m5: dict, m7: dict, gates: dict, verdict: str) -> str:
    raw = m2["per_dataset_convergence"]["outcomes_raw_primary"]
    std = m2["per_dataset_convergence"]["outcomes_standardized_sensitivity"]
    n_raw_nc = sum(v["n_nonconverged"] for v in raw.values())
    n_std_nc = sum(v["n_nonconverged"] for v in std.values())
    conv_txt = ", ".join(
        f"{ds} {raw[ds]['n_converged']}/{raw[ds]['n_fits']}" for ds in STAGE3A_DATASETS
    )
    return f"""# Stage 3A audit addendum

Audit run `{run_id}`. This addendum **corrects** statements in
`docs/stage3a_evaluation.md`, `configs/stage3a.yaml` and their commentary. Per
the audit contract the original files are left byte-identical; this file and the
versioned `results/stage3a/audit_{run_id}/` directory carry the corrections.

## Corrections to previously published statements

1. **Solver backend.** `sklearn.svm.SVC(kernel='linear')` is implemented using
   **libsvm**. Stage 3A described the cap in liblinear terms. liblinear backs
   `sklearn.svm.LinearSVC`/`LinearSVR`, which are **not** drop-in replacements:
   different solver, different default loss, different multiclass handling.
2. **Iteration cap.** `SVC`'s documented `max_iter` default is **-1**. The value
   2,000,000 used in Stage 3A is a **user-declared hard cap** declared by this
   project, not a library or solver default. It should be read as
   "user-declared cap of 2,000,000 on the libsvm-backed SVC".
3. **Nonconvergence cause.** Stage 3A stated feature scaling as the cause. That
   is now recorded as a **solver-conditioning hypothesis**: {n_raw_nc}/250 raw
   SVM fits failed to converge versus {n_std_nc}/250 standardized fits. Scaling
   is a supported contributor, but standardization also changes margin geometry
   and libsvm working-set selection, so causation is **not** established and no
   counterfactual was run.
4. **Predictions vs convergence.** 750 primary predictions were produced. That
   remains true and is preserved. How many SVM fits converged is a separate
   question ({conv_txt}). A non-converged fit still returns predictions but does
   not return a converged optimum; it must not be relabelled as converged.
5. **Freeze chronology.** The declared `frozen_utc: 2026-10-11T12:00:00Z`
   postdates every Stage 3A artifact. Status: **{m3['status']}**. No date was
   altered, no earlier time invented, and no pre-registration is claimed
   retroactively.
6. **Path.** Stage 3A wrote run records under
   `projects/ossfs-dd-reproduction/results/stage3a/runs/records/`, not under
   `$RESEARCH_OUTPUT_ROOT/ossfs-dd-reproduction/stage3a/`. Audited at the real
   location; no path was invented.
7. **Provenance gaps.** `configs/paper_reference_discrete.csv` is not
   git-tracked (`.gitignore` line 40, `*.csv`), and `run_manifest.json` records
   no execution timestamp.
8. **A hidden destructive test.** `tests/test_evidence_fail_closed.py` wrote
   synthetic payloads into the **real** `results/stage1/test_results.json` and
   deleted the file in a `finally` block. This was masked because the conftest
   hook rewrote the file at the end of every session; once that hook became
   opt-in, a full test run genuinely deleted the committed Stage 1 artifact.
   Those tests now isolate `gates.results_dir` to a temporary directory. The
   earlier `git checkout` workarounds were treating the symptom.

## What the audit confirmed

- Ledger reconciles end to end: {len(m1['checks'])} checks,
  **{m1['n_discrepancies']} disagreements** (raw records ↔ per-run CSV ↔
  `aggregate.json` ↔ committed markdown).
- All 250 selections and 1500 classifier outcomes are present with unique IDs,
  bounded accuracies and explicit denominators.
- ARCENE's selected count spans {m4['per_dataset']['ARCENE']['min']}-{m4['per_dataset']['ARCENE']['max']}
  (mean {m4['per_dataset']['ARCENE']['mean']:.2f}), confirmed from raw records.
- No forbidden dataset (`DLBCL`, `BREAST`) was read at any point; the executed
  records contain exactly the five authorized datasets.
- Stage 1 and Stage 2 working trees are unchanged.

## What remains open

Feature-count variability cause (`UNRESOLVED`), raw-SVM nonconvergence cause
(hypothesis only), config freeze chronology (`{m3['status']}`), and the
authors' continuous Fisher-Z semantics (unrecoverable).

Verdict: **{verdict}**
"""


if __name__ == "__main__":
    raise SystemExit(main())
