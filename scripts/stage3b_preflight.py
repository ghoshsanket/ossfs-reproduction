#!/usr/bin/env python
"""Stage 3B preflight: validate inputs, FREEZE the protocol, write the journal.

This MUST run before any real Stage 3B fit or ARCENE selection. It

1. verifies Stage 1 hashes/splits/orders, Stage 3A records and authorisation;
2. enumerates the 75 subsets / 450 fit IDs / 50 ARCENE run IDs (dry run only);
3. writes the frozen ``configs/stage3b.yaml`` **with a runtime UTC stamp**;
4. writes a write-once ``stage3b_preflight.json`` capturing every digest;
5. appends ``PRE_FLIGHT_WRITTEN`` to the event journal.

The freeze is in-process, hashed and chronologically ordered. It is NOT an
independently attested preregistration, and it does not rewrite Stage 3A's
contradicted timestamp.

Usage:
    uv run python scripts/stage3b_preflight.py
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

from ossfs_dd_stage3b import EXP_A_DATASETS, EXP_B_DATASET  # noqa: E402
from ossfs_dd_stage3b.protocol import (  # noqa: E402  # noqa: E402
    A_PLANNED_FITS,
    B_PLANNED_RUNS,
    MAX_ITER_CAPS,
    SCALE_VARIANTS,
    all_subsets,
    enumerate_run_ids,
    protocol_digest,
    validate_protocol,
)
from ossfs_dd_stage3b.provenance import (  # noqa: E402
    BlockedInput,
    load_order,
    load_split,
    load_stage3a_record,
    selection_code_fingerprint,
    sha256_file,
    stage3a_manifest,
    subset_fingerprint,
    verify_stage3a_digests,
)

PROJECT = Path(__file__).resolve().parents[1]
OUT_ROOT = Path(os.environ["RESEARCH_OUTPUT_ROOT"]) / "ossfs-dd-reproduction" / "stage3b"
SOURCE_DIRS = (
    "src/ossfs_dd_stage3b", "src/ossfs_dd", "src/ossfs_dd_data",
    "src/ossfs_dd_eval", "scripts",
)


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def code_hash() -> str:
    parts = []
    for d in SOURCE_DIRS:
        base = PROJECT / d
        if not base.exists():
            continue
        for f in sorted(base.rglob("*.py")):
            parts.append(f"{f.relative_to(PROJECT).as_posix()}:{sha256_file(f)}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def write_json_once(path: Path, payload: dict) -> None:
    """Write-once: refuse to overwrite an existing artifact."""
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


def append_journal(path: Path, event: str, run_id: str, config_sha: str, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"event": event, "utc": utcnow(), "run_id": run_id, "config_sha256": config_sha}
    row.update(extra)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def build_config(frozen_utc: str) -> dict:
    """The preregistered Stage 3B protocol. Only the timestamp is runtime-derived."""
    return {
        "schema_version": 1,
        "stage": "3b",
        "frozen_utc": frozen_utc,
        "freeze_statement": (
            "Written and hashed before the first Stage 3B fit or ARCENE selection. "
            "This is an in-process, hashed, chronologically ordered freeze, NOT an "
            "independently attested preregistration."
        ),
        "authorization": {
            "experiment_a_datasets": list(EXP_A_DATASETS),
            "experiment_b_datasets": [EXP_B_DATASET],
            "forbidden": ["DLBCL", "BREAST"],
            "continuous_excluded": ["SRBCT", "LYMPHOMA", "PROSTATE", "LEUKEMIA"],
            "required_manifest_status": "EXACT_VERIFIED",
        },
        "experiment_a": {
            "purpose": "SVM conditioning on already-selected Stage 3A subsets",
            "datasets": list(EXP_A_DATASETS),
            "seeds": [11, 17, 23],
            "folds": [0, 1, 2, 3, 4],
            "subsets_source": "projects/ossfs-dd-reproduction/results/stage3a/runs/records",
            "feature_selection_performed": False,
            "scale_variants": list(SCALE_VARIANTS),
            "max_iter_caps": list(MAX_ITER_CAPS),
            "max_iter_minus_one_allowed": False,
            "planned_subsets": 75,
            "fits_per_subset": 6,
            "planned_fits": A_PLANNED_FITS,
            "a_h1_datasets": ["MADELON", "GINA", "GISETTE"],
            "a_h1_top_cap": max(MAX_ITER_CAPS),
            "a_h1_min_delta": 0.50,
            "a_h1_min_std_rate": 0.80,
            "a_h2_low_cap": min(MAX_ITER_CAPS),
            "a_h2_high_cap": max(MAX_ITER_CAPS),
            "a_h2_comparisons": 10,
            "control_check": "cap=2000000 + matching preprocessing must reproduce Stage 3A accuracy within 1e-10",
            "determinism_repeats": [
                {"dataset": "MADELON", "scale_variant": "raw", "max_iter": 200000},
                {"dataset": "GISETTE", "scale_variant": "standardized", "max_iter": 200000},
            ],
        },
        "experiment_b": {
            "purpose": "ARCENE feature-arrival-order sensitivity, CV split held fixed",
            "dataset": EXP_B_DATASET,
            "split_seed": 11,
            "split_source": "Stage 1 ARCENE seed_11.json (FIXED for all 50 runs)",
            "order_seeds": [11, 17, 23, 31, 47, 59, 71, 83, 97, 109],
            "folds": [0, 1, 2, 3, 4],
            "crossed_design": "5 folds x 10 orders; order_seed selects ONLY the permutation",
            "order_seed_must_not_select_split": True,
            "anchor_order_seed": 11,
            "planned_runs": B_PLANNED_RUNS,
            "b_h1_range_threshold": 500,
            "b_h1_min_folds": 3,
            "b_h2_mechanism_preregistered": False,
            "illustrative_traces": {"fold": 0, "order_seeds": [11, 17]},
            "jaccard_pairs_expected": 5 * 45,
            "jaccard_empty_convention": "empty/empty=1.0; empty/nonempty=0.0",
        },
        "estimator": {
            "class": "sklearn.svm.SVC",
            "params": {"kernel": "linear", "C": 1.0},
            "max_iter_swept": list(MAX_ITER_CAPS),
            "other_params": "installed-version defaults, recorded verbatim per fit",
            "linear_svc_substitution_forbidden": True,
        },
        "scaler": {
            "standardized": "sklearn.preprocessing.StandardScaler fitted on SELECTED TRAINING ROWS ONLY",
            "raw": "identity",
            "fitted_on_test_rows": False,
        },
        "selector": {
            "mode": "matlab_compatible_v1",
            "entry_file": "mi2",
            "k1": 1.0,
            "k2": 2.0,
            "pool_size": 100,
            "modified": False,
        },
        "status_semantics": {
            "CONVERGED": "fit completed, solver did not hit its cap",
            "NONCONVERGED": "predictions exist but the cap was hit or early termination warned",
            "TIMEOUT": "worker killed at the wall-clock limit; accuracy null",
            "FIT_ERROR": "fit raised; accuracy null",
            "BLOCKED_INPUT": "a required Stage 3A input was missing or inconsistent",
            "missing_is_never_zero": True,
            "nonconverged_accuracy_is_incomplete_fit_diagnostic": True,
        },
        "resources": {
            "cpu_only": True,
            "memory_limit_gib": 32,
            "experiment_a_max_wallclock_hours": 16,
            "experiment_b_max_wallclock_hours": 2,
            "per_fit_timeout_minutes": 10,
            "experiment_a_workers": 2,
            "experiment_b_workers": 1,
            "blas_threads_per_worker": 1,
        },
        "gates": [
            "B_PREFLIGHT_CHRONOLOGY", "B_AUTHORIZATION", "B_A_FIXED_SUBSETS",
            "B_A_GRID_COMPLETE", "B_B_CROSSED_DESIGN", "B_B_ANCHOR_MATCH",
            "B_TRACE_ARITHMETIC", "B_SVM_CONVERGENCE_HONEST", "B_REPRODUCIBILITY",
            "B_CODE_AND_SAFETY",
        ],
        "hypotheses_are_not_operational_gates": True,
    }


def tree_hashes() -> dict[str, str]:
    """Hash the Stage 1/2/3A trees that Stage 3B must leave byte-identical."""
    out = {}
    for scope in ("configs/stage1.yaml", "configs/stage2.yaml", "configs/stage3a.yaml",
                  "src/ossfs_dd", "src/ossfs_dd_data", "src/ossfs_dd_eval",
                  "results/stage1", "results/stage2", "results/stage3a/aggregate.json",
                  "results/stage3a/per_run_outcomes.csv",
                  "results/stage3a/report_manifest.json"):
        p = PROJECT / scope
        if p.is_file():
            out[scope] = sha256_file(p)
        elif p.is_dir():
            h = hashlib.sha256()
            for f in sorted(p.rglob("*")):
                if f.is_file() and ".venv" not in f.parts:
                    h.update(str(f.relative_to(PROJECT)).encode())
                    h.update(sha256_file(f).encode())
            out[scope + "/"] = h.hexdigest()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="enumerate only; write no files and touch no real data")
    args = ap.parse_args()

    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    print(f"  run_id: {run_id}")

    # ---- 1. Stage 1 data, splits, orders -------------------------------
    import json as _json

    from ossfs_dd_data.workspace import manifest_path, processed_dir, resolve

    data_checks = {}
    for ds in list(EXP_A_DATASETS):
        man = _json.loads(manifest_path(ds).read_text(encoding="utf-8"))
        schema = _json.loads((processed_dir(ds) / "schema.json").read_text(encoding="utf-8"))
        digest = sha256_file(resolve(schema["X"]["path"]))
        data_checks[ds] = {
            "manifest_status": man["stage1_verdict"]["status"],
            "x_sha256_matches_schema": digest == schema["X"]["sha256"],
            "n_samples": schema["n_samples"], "n_features": schema["n_features"],
        }
        if data_checks[ds]["manifest_status"] != "EXACT_VERIFIED":
            raise BlockedInput(f"{ds}: manifest status is not EXACT_VERIFIED")
        if not data_checks[ds]["x_sha256_matches_schema"]:
            raise BlockedInput(f"{ds}: stored matrix SHA-256 mismatch")

    split_order_checks = {}
    for ds in EXP_A_DATASETS:
        for seed in (11, 17, 23):
            for fold in range(5):
                load_split(ds, seed, fold)
    for order_seed in (11, 17, 23, 31, 47, 59, 71, 83, 97, 109):
        load_order(EXP_B_DATASET, order_seed)
    split_order_checks["all_loads_ok"] = True
    split_order_checks["arcene_orders_verified_permutations"] = 10

    # ---- 2. Stage 3A records -------------------------------------------
    manifest = stage3a_manifest()
    digest_check = verify_stage3a_digests(manifest)
    if not digest_check["protocol_digest_ok"]:
        raise BlockedInput("Stage 3A protocol_digest mismatch across records")

    subsets = all_subsets()
    subset_checks = []
    for s in subsets:
        rec = load_stage3a_record(s)
        subset_checks.append({
            "key": s.key, "n_selected": rec["n_selected"],
            "ids_sha256": subset_fingerprint(s.dataset, rec["selected_feature_ids"])["ids_sha256"],
        })
    n_unique_subsets = len({c["ids_sha256"] for c in subset_checks})
    print(f"  Stage 3A subsets verified: {len(subset_checks)}/75 "
          f"(unique id-sequences: {n_unique_subsets})")

    ids = enumerate_run_ids()
    # Fail closed on any enumeration drift rather than asserting, which ruff
    # rightly forbids in shipped scripts.
    expected = {
        "experiment_a_subset_ids": (75, 75),
        "experiment_a_fit_ids": (A_PLANNED_FITS, A_PLANNED_FITS),
        "experiment_b_run_ids": (B_PLANNED_RUNS, B_PLANNED_RUNS),
    }
    for key, (want_n, want_unique) in expected.items():
        got_n, got_unique = len(ids[key]), len(set(ids[key]))
        if (got_n, got_unique) != (want_n, want_unique):
            raise SystemExit(
                f"enumeration drift in {key}: n={got_n} unique={got_unique}, "
                f"expected n={want_n} unique={want_unique}"
            )
    print(f"  dry-run enumeration: 75 subsets / {A_PLANNED_FITS} fits / "
          f"{B_PLANNED_RUNS} ARCENE runs (all unique)")

    if args.dry_run:
        print("  --dry-run: no files written, no real data touched")
        return 0

    # ---- 3. freeze config with a RUNTIME timestamp ----------------------
    frozen_utc = utcnow()
    config = build_config(frozen_utc)
    problems = validate_protocol(config)
    if problems:
        raise SystemExit(f"frozen config is internally inconsistent: {problems}")
    cfg_path = PROJECT / "configs" / "stage3b.yaml"
    if cfg_path.exists():
        raise SystemExit(
            f"REFUSING to overwrite a frozen config: {cfg_path}. "
            "Start a new versioned Stage 3B run id instead."
        )
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                        encoding="utf-8")
    cfg_sha = sha256_file(cfg_path)
    proto_digest = protocol_digest(config)
    print(f"  frozen config: {cfg_path.name} sha256={cfg_sha[:16]} at {frozen_utc}")

    run_dir = PROJECT / "results" / "stage3b" / run_id
    out_dir = OUT_ROOT / run_id
    journal = out_dir / "logs" / "event_journal.jsonl"
    append_journal(journal, "PRE_FLIGHT_WRITTEN", run_id, cfg_sha,
                   {"note": "config hashed before any Stage 3B fit or selection"})

    import numpy
    import sklearn

    preflight = {
        "schema": "ossfs-dd-reproduction/stage3b/preflight/1.0",
        "run_id": run_id,
        "utc": frozen_utc,
        "config_path": f"projects/ossfs-dd-reproduction/{cfg_path.relative_to(PROJECT)}",
        "config_sha256": cfg_sha,
        "protocol_digest": proto_digest,
        "code_hash": code_hash(),
        "selection_code_fingerprint": selection_code_fingerprint(),
        "versions": {
            "python": sys.version.split()[0], "numpy": numpy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "scope": {
            "experiment_a_datasets": list(EXP_A_DATASETS),
            "experiment_b_datasets": [EXP_B_DATASET],
            "forbidden": ["DLBCL", "BREAST"],
            "continuous_excluded": ["SRBCT", "LYMPHOMA", "PROSTATE", "LEUKEMIA"],
        },
        "stage3a": {
            "manifest": digest_check,
            "n_subsets_verified": len(subset_checks),
            "subset_checks": subset_checks,
        },
        "stage1_data_checks": data_checks,
        "split_and_order_checks": split_order_checks,
        "planned_run_ids": ids,
        "tests_run_before_execution": "synthetic-only protocol/provenance/trace unit tests",
        "safety_checks": {
            "forbidden_datasets_read": 0,
            "downloads": 0,
            "feature_selections_performed_in_A": 0,
        },
        "data_not_mutated_baseline": tree_hashes(),
        "freeze_limitation": (
            "In-process, hashed, chronologically ordered freeze. No independent "
            "timestamp attestation exists. Stage 3A's contradictory freeze "
            "timestamp is preserved unchanged and is NOT retroactively repaired."
        ),
    }
    write_json_once(run_dir / "stage3b_preflight.json", preflight)
    print(f"  preflight: {run_dir.relative_to(PROJECT)}/stage3b_preflight.json (write-once)")
    print(f"  journal:   {journal}")
    print(f"  output:    {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
