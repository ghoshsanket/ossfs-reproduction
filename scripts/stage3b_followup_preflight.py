#!/usr/bin/env python
"""Stage 3B follow-up preflight: verify sources, then FREEZE the protocol.

Runs BEFORE any real-data fit or selection. It verifies the authorized
datasets, the 45 Stage 3A subsets, the 50 Stage 3B ARCENE raw records and all
planned identifiers, then writes the frozen config with a **runtime** UTC
stamp, a write-once preflight manifest and the PRE_FLIGHT_WRITTEN journal entry.

This freeze is in-process and chronologically ordered. It is NOT an
independently registered or attested preregistration, and it does not repair or
relabel the Stage 3A contradicted chronology.

Usage:
    uv run python scripts/stage3b_followup_preflight.py [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ossfs_dd_stage3b.protocol import SubsetID  # noqa: E402
from ossfs_dd_stage3b.provenance import (  # noqa: E402
    BlockedInput,
    load_order,
    load_split,
    load_stage3a_record,
    sha256_file,
)
from ossfs_dd_stage3b_followup import EXP_A_DATASETS, EXP_B_DATASET  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
OUT_ROOT = (Path(os.environ["RESEARCH_OUTPUT_ROOT"]) / "ossfs-dd-reproduction"
            / "stage3b_followup")
STAGE3B_RUN = ("20261011T153549Z")

A_SEEDS = (11, 17, 23)
A_FOLDS = (0, 1, 2, 3, 4)
A_C_VALUES = (0.1, 1.0, 10.0)
A_TOL_VALUES = (0.001, 0.0001)
A_MAX_ITER = 2_000_000

B_SPLIT_SEED = 11
B_ORDER_SEEDS = (11, 17, 23, 31, 47, 59, 71, 83, 97, 109)

PLANNED_FITS = 3 * 3 * 5 * 3 * 2           # 270
PLANNED_SELECTIONS = 5 * 10                # 50
PLANNED_SUBSET_CLASSIFIERS = 50 * 3        # 150
PLANNED_BASELINES = 5 * 3                   # 15


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def source_fingerprint() -> str:
    parts = []
    for d in ("src/ossfs_dd", "src/ossfs_dd_data", "src/ossfs_dd_eval",
              "src/ossfs_dd_stage3b", "src/ossfs_dd_stage3b_followup"):
        base = PROJECT / d
        if not base.exists():
            continue
        for f in sorted(base.rglob("*.py")):
            parts.append(f"{f.relative_to(PROJECT).as_posix()}:{sha256_file(f)}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def write_once(path: Path, payload: dict) -> None:
    if path.exists():
        raise SystemExit(f"REFUSING to overwrite write-once artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def append_journal(path: Path, event: str, run_id: str, cfg_sha: str, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"event": event, "utc": utcnow(), "run_id": run_id, "config_sha256": cfg_sha}
    row.update(extra)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def build_config(frozen_utc: str) -> dict:
    return {
        "schema_version": 1,
        "stage": "3b_followup",
        "version": "v1",
        "frozen_utc": frozen_utc,
        "freeze_statement": (
            "Frozen and hashed before any real-data fit or selection. This is an "
            "in-process, chronologically ordered freeze. It is NOT an independently "
            "registered or attested preregistration."
        ),
        "supersedes_nothing": (
            "This is a NEW separately versioned follow-up. It does not replace or "
            "alter Stage 3B, Stage 3A, or the Stage 2 partial gate."
        ),
        "authorization": {
            "experiment_a_datasets": list(EXP_A_DATASETS),
            "experiment_b_datasets": [EXP_B_DATASET],
            "forbidden": ["DLBCL", "BREAST"],
            "continuous_excluded": ["SRBCT", "LYMPHOMA", "PROSTATE", "LEUKEMIA"],
            "required_manifest_status": "EXACT_VERIFIED",
        },
        "diagnostic_repairs": {
            "D1": {
                "file": "src/ossfs_dd/selector.py",
                "symbol": "Selector._redundancy_pass",
                "nature": "accounting_only",
                "description": (
                    "Report only candidate removals that were actually applied. "
                    "Queued-but-unapplied removals are no longer reported or counted."
                ),
                "must_not_change": ["selected feature ids", "arrival positions",
                                    "deletion order", "tie-breaking", "promotions"],
            },
            "D2": {
                "file": "src/ossfs_dd_stage3b/trace.py",
                "symbol": "checkpoint_rows",
                "nature": "accounting_only",
                "description": (
                    "Prefix checkpoints are chronological snapshots. The final pool "
                    "flush is counted only in the terminal summary, never in an "
                    "arrival checkpoint."
                ),
            },
        },
        "counter_semantics": {
            "direct_accepts": "entered the direct-accept path, even if removed immediately",
            "promoted_from_pool": "actual candidate insertions from a pool flush",
            "redundancy_removals": "only applied, successful candidate deletions",
            "n_selected": "candidate-list length (not a de-duplicated set size)",
            "warmup": "mi2 SKIPS the first two arrivals; they are not insertions",
            "conflict_policy": (
                "If these definitions conflict with the live selector, disclose it; "
                "never force an accounting equation by changing feature selection."
            ),
        },
        "experiment_a": {
            "purpose": "SVM C/tol sensitivity on fixed Stage 3A subsets",
            "datasets": list(EXP_A_DATASETS),
            "seeds": list(A_SEEDS),
            "folds": list(A_FOLDS),
            "subset_source": "results/stage3a/runs/records/ (raw selected_feature_ids)",
            "scale_variant": "standardized",
            "scaler_fit_rows": "selected_training_rows_only",
            "estimator": "sklearn.svm.SVC",
            "kernel": "linear",
            "C_values": list(A_C_VALUES),
            "tol_values": list(A_TOL_VALUES),
            "max_iter": A_MAX_ITER,
            "probability": False,
            "linear_svc_substitution_forbidden": True,
            "status_types": ["CONVERGED", "NONCONVERGED", "TIMEOUT", "FIT_ERROR",
                             "BLOCKED_INPUT"],
            "planned_fits": PLANNED_FITS,
            "control_cell": {
                "C": 1.0, "tol": 0.001, "max_iter": A_MAX_ITER,
                "compare_to": f"stage3b {STAGE3B_RUN} standardized highest-cap fits",
                "tolerance": 1e-10,
                "on_version_mismatch": "NOT_DIRECTLY_COMPARABLE",
            },
        },
        "hypotheses_a": {
            "A_H1": {
                "name": "C sensitivity",
                "datasets": list(EXP_A_DATASETS),
                "compare": "paired converged-fit mean accuracy at C=0.1 vs C=10, tol=1e-3",
                "threshold": 0.02,
                "min_datasets_meeting": 1,
                "type": "descriptive_effect_not_significance",
            },
            "A_H2": {
                "name": "tolerance sensitivity",
                "datasets": list(EXP_A_DATASETS),
                "compare": "median n_iter_max at C=1, tol 1e-3 vs 1e-4",
                "threshold_relative": 0.10,
                "min_datasets_meeting": 2,
                "zero_denominator": "UNDEFINED",
            },
            "A_H3": {
                "name": "feasible convergence",
                "statement": "standardized converges in >= 95% of 270 planned cells",
                "threshold": 0.95,
                "denominator": PLANNED_FITS,
            },
        },
        "experiment_b": {
            "purpose": "ARCENE decision trajectory + downstream accuracy",
            "dataset": EXP_B_DATASET,
            "split_seed": B_SPLIT_SEED,
            "folds": list(range(5)),
            "order_seeds": list(B_ORDER_SEEDS),
            "order_seed_selects_permutation_only": True,
            "baseline_records": (
                f"outputs/ossfs-dd-reproduction/stage3b/{STAGE3B_RUN}/arcene/runs/"
            ),
            "planned_selections": PLANNED_SELECTIONS,
            "planned_subset_classifiers": PLANNED_SUBSET_CLASSIFIERS,
            "planned_baselines": PLANNED_BASELINES,
            "classifiers": {
                "knn": "KNeighborsClassifier(n_neighbors=5, weights='uniform', "
                       "metric='minkowski', p=2) on raw selected values",
                "cart": "DecisionTreeClassifier(criterion='gini', splitter='best', "
                        "random_state=11) on raw selected values",
                "svm": "sklearn.svm.SVC(kernel='linear', C=1.0, tol=0.001, "
                       "max_iter=2000000) raw selected values",
            },
            "cart_seed_fixed_across_orders": 11,
            "baseline_model": "same three classifiers on all 10000 original features",
            "baseline_role": "contextual comparator, not a tuned reference model",
            "illustrative_pair": {"fold": 0, "order_seeds": [11, 17]},
            "divergence_gap_threshold": 100,
            "divergence_bin_size": 100,
            "jaccard_pairs_expected": 225,
            "threshold_tolerance": 1e-10,
        },
        "hypotheses_b": {
            "B_H1": {
                "name": "ledger correctness",
                "operational_gate": True,
                "criterion": "candidate-count conservation and selected-ID agreement "
                             "hold in 50/50 ARCENE runs",
            },
            "B_H2": {
                "name": "downstream order effect",
                "primary_metric": "CART",
                "criterion": "within-fold CART accuracy range across 10 orders >= 0.05",
                "threshold": 0.05,
                "min_folds_meeting": 3,
                "type": "within_fold_sensitivity_not_independent_sample_test",
            },
            "B_H3": {
                "name": "mechanism",
                "preregistered_mechanism": None,
                "default_outcome": "MECHANISM_UNRESOLVED",
                "note": "Correlation with arithmetic components is not causation.",
            },
        },
        "resources": {
            "cpu_only": True,
            "memory_limit_gib": 32,
            "max_fit_workers": 2,
            "selection_workers": 1,
            "blas_threads_per_worker": 1,
            "per_fit_timeout_minutes": 10,
            "experiment_a_max_hours": 3,
            "experiment_b_max_hours": 4,
            "total_max_hours": 8,
        },
        "operational_gates": [
            "F1_AUTHORIZATION_AND_SOURCE", "F2_FAIL_BEFORE_PASS_AFTER",
            "F3_SELECTOR_INVARIANCE", "F4_TRUE_LEDGER", "F5_C_TRIAL_COMPLETE",
            "F6_ARCENE_CLASSIFICATION_COMPLETE", "F7_NO_LEAKAGE",
            "F8_CHRONOLOGY_AND_HASH", "F9_REPRODUCIBILITY", "F10_CODE_AND_IMMUTABILITY",
        ],
        "hypotheses_are_not_operational_gates": True,
        "forbidden_actions": [
            "changing k1/k2/pool_size/mode/entry_file",
            "changing feature-selection thresholds to improve results",
            "selecting best C from test accuracy",
            "any work on continuous datasets or DLBCL/BREAST",
            "any retroactive edit of Stage 1/2/3A/3B artifacts",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    print(f"  run_id: {run_id}")

    # ---- verify authorized data -------------------------------------
    import json as _json

    from ossfs_dd_data.workspace import manifest_path, processed_dir, resolve

    data_checks = {}
    for ds in list(EXP_A_DATASETS) + [EXP_B_DATASET]:
        man = _json.loads(manifest_path(ds).read_text(encoding="utf-8"))
        schema = _json.loads((processed_dir(ds) / "schema.json").read_text(encoding="utf-8"))
        digest = sha256_file(resolve(schema["X"]["path"]))
        ok = (man["stage1_verdict"]["status"] == "EXACT_VERIFIED"
              and digest == schema["X"]["sha256"])
        data_checks[ds] = {"status": man["stage1_verdict"]["status"],
                           "sha256_matches": digest == schema["X"]["sha256"], "ok": ok}
        if not ok:
            raise BlockedInput(f"{ds}: manifest status or matrix hash failed verification")

    for ds in EXP_A_DATASETS:
        for seed in A_SEEDS:
            for fold in A_FOLDS:
                load_split(ds, seed, fold)
    for oseed in B_ORDER_SEEDS:
        load_order(EXP_B_DATASET, oseed)

    # ---- verify 45 Stage 3A subsets ---------------------------------
    subsets = []
    for ds in EXP_A_DATASETS:
        for seed in A_SEEDS:
            for fold in A_FOLDS:
                rec = load_stage3a_record(SubsetID(ds, seed, fold))
                subsets.append({"key": f"{ds}_seed{seed}_fold{fold}",
                                "n_selected": rec["n_selected"]})
    print(f"  Stage 3A subsets verified: {len(subsets)}/45")

    # ---- verify 50 Stage 3B ARCENE raw records -----------------------
    base_dir = (OUT_ROOT.parent / "stage3b" / STAGE3B_RUN / "arcene" / "runs")
    arc_records = []
    for fold in range(5):
        for oseed in B_ORDER_SEEDS:
            p = base_dir / f"ARCENE_split{B_SPLIT_SEED}_fold{fold}_order{oseed}.json"
            if not p.exists():
                raise BlockedInput(
                    f"Stage 3B baseline record NOT_AVAILABLE: {p}. Ordered selected "
                    "IDs must not be inferred from count summaries."
                )
            r = _json.loads(p.read_text(encoding="utf-8"))
            if r["split_seed"] != B_SPLIT_SEED or r["fold"] != fold or r["order_seed"] != oseed:
                raise BlockedInput(f"{p.name}: identity mismatch")
            arc_records.append({"path": str(p), "sha256": sha256_file(p),
                                "n_selected": r["n_selected"]})
    print(f"  Stage 3B ARCENE raw records verified: {len(arc_records)}/50")

    # ---- enumerate planned IDs --------------------------------------
    fit_ids = [f"{s['key']}__C{c}__tol{t}" for s in subsets
               for c in A_C_VALUES for t in A_TOL_VALUES]
    sel_ids = [f"ARCENE_split{B_SPLIT_SEED}_fold{f}_order{o}"
               for f in range(5) for o in B_ORDER_SEEDS]
    clf_ids = [f"{s}::{m}" for s in sel_ids for m in ("knn", "cart", "svm")]
    base_ids = [f"ARCENE_all_features_fold{f}::{m}" for f in range(5)
                for m in ("knn", "cart", "svm")]
    for name, lst, want in (("fits", fit_ids, PLANNED_FITS),
                            ("selections", sel_ids, PLANNED_SELECTIONS),
                            ("classifiers", clf_ids, PLANNED_SUBSET_CLASSIFIERS),
                            ("baselines", base_ids, PLANNED_BASELINES)):
        if len(lst) != want or len(set(lst)) != want:
            raise SystemExit(f"enumeration drift in {name}: {len(lst)}/{len(set(lst))} != {want}")
    print(f"  planned: {len(fit_ids)} fits, {len(sel_ids)} selections, "
          f"{len(clf_ids)} classifiers, {len(base_ids)} baselines")

    if args.dry_run:
        print("  --dry-run: nothing written, no real-data evaluation performed")
        return 0

    # ---- freeze ------------------------------------------------------
    frozen_utc = utcnow()
    config = build_config(frozen_utc)
    cfg_path = PROJECT / "configs" / "stage3b_followup_v1.yaml"
    if cfg_path.exists():
        raise SystemExit(f"REFUSING to overwrite frozen config {cfg_path}; start a new run id")
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                        encoding="utf-8")
    cfg_sha = sha256_file(cfg_path)
    print(f"  frozen config sha256={cfg_sha[:16]} at {frozen_utc}")

    out_dir = OUT_ROOT / run_id
    journal = out_dir / "logs" / "event_journal.jsonl"
    append_journal(journal, "PRE_FLIGHT_WRITTEN", run_id, cfg_sha,
                   {"note": "before any real-data fit or selection"})

    import numpy
    import sklearn

    preflight = {
        "schema": "ossfs-dd-reproduction/stage3b_followup/preflight/1.0",
        "run_id": run_id,
        "utc": frozen_utc,
        "config_sha256": cfg_sha,
        "source_fingerprint": source_fingerprint(),
        "versions": {"python": sys.version.split()[0], "numpy": numpy.__version__,
                     "scikit_learn": sklearn.__version__},
        "data_checks": data_checks,
        "stage3a_subsets": subsets,
        "stage3b_arcene_records": arc_records,
        "planned": {"fits": len(fit_ids), "selections": len(sel_ids),
                    "subset_classifiers": len(clf_ids), "baselines": len(base_ids)},
        "planned_run_ids": {"fits": fit_ids, "selections": sel_ids,
                            "classifiers": clf_ids, "baselines": base_ids},
        "stage3a_code_hash_limitation": (
            "Stage 3A's per-run code_hash is non-authoritative (11 distinct values "
            "observed because it was recomputed per task from a moving source tree). "
            "Validation uses protocol_digest, subset identity and the current source "
            "fingerprint instead."
        ),
        "freeze_limitation": (
            "In-process, chronologically ordered freeze. Not an independently "
            "registered or attested preregistration. The Stage 3A contradicted "
            "chronology is preserved unchanged."
        ),
    }
    write_once(PROJECT / "results" / "stage3b_followup" / run_id / "preflight.json", preflight)
    print(f"  preflight written (write-once); journal: {journal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
