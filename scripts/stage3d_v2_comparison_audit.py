#!/usr/bin/env python
"""Stage 3D V2 comparison — independent fail-closed auditor with ten gates.

Read-only against raw records. Writes a versioned audit report.

Usage:
    uv run python scripts/stage3d_v2_comparison_audit.py --run-id RUN_ID
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PROJECT = Path(__file__).resolve().parents[1]
OUT_ROOT = Path(os.environ.get("RESEARCH_OUTPUT_ROOT", "")) / "ossfs-dd-reproduction" / "stage3d_v2"
V1_ROOT = Path(os.environ.get("RESEARCH_OUTPUT_ROOT", "")) / "ossfs-dd-reproduction" / "stage3d" / "20261102T042840Z"


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jaccard(a, b) -> float:
    A, B = set(a), set(b)
    return len(A & B) / len(A | B) if A | B else 1.0


# Exact V2 -> V1 arm mapping, exported for tests. A naive
# ``.replace("_V2", "")`` would also mangle ``D2_V2_pilot``.
ARM_MAP = {"D2_V2": "D2", "D2_V2_pilot": "D2", "C2_V2": "C2", "C4_V2": "C4"}


def expected_v1_keys_from_v2(expected_sel_keys: set[str]) -> tuple[set[str], set[str]]:
    """Derive exact required V1 selection and classifier keys from V2 selection keys."""
    exp_sel: set[str] = set()
    for v2k in expected_sel_keys:
        head, arm = v2k.rsplit("__", 1)
        v1_arm = ARM_MAP.get(arm)
        if v1_arm is None:
            continue
        exp_sel.add(f"{head}__{v1_arm}")
    exp_clf = {f"{k}__{c}" for k in exp_sel for c in ("knn", "svm", "cart")}
    return exp_sel, exp_clf


def evaluate_gates(evidence: dict) -> tuple[dict, str]:
    """Pure gate evaluation from an evidence dict. Used by the auditor and tests.

    Every gate is driven by explicit, mutable evidence inputs so mutation tests can
    demonstrate real failures without rewriting returned gate objects.
    """
    sel_keys = set(evidence.get("sel_keys", set()))
    expected_sel = set(evidence.get("expected_sel", set()))
    clf_keys = set(evidence.get("clf_keys", set()))
    expected_clf = set(evidence.get("expected_clf", set()))
    sel_count = int(evidence.get("sel_count", len(sel_keys)))
    clf_count = int(evidence.get("clf_count", len(clf_keys)))
    sel_dupes = sel_count - len(sel_keys)
    clf_dupes = clf_count - len(clf_keys)

    e1 = {"outcome": "PASS" if (sel_keys == expected_sel and clf_keys == expected_clf
                                 and sel_dupes == 0 and clf_dupes == 0
                                 and sel_count == len(expected_sel)
                                 and clf_count == len(expected_clf)) else "FAIL",
          "measured": (f"V2 selections={sel_count}/{len(expected_sel)} "
                       f"equality={sel_keys == expected_sel} dupes={sel_dupes}; "
                       f"V2 classifiers={clf_count}/{len(expected_clf)} "
                       f"equality={clf_keys == expected_clf} dupes={clf_dupes}")}

    cfg_ok = bool(evidence.get("cfg_ok", False))
    e2 = {"outcome": "PASS" if cfg_ok else "FAIL",
          "measured": str(evidence.get("e2_measured", f"cfg_ok={cfg_ok}"))}

    exp_v1_sel, exp_v1_clf = expected_v1_keys_from_v2(expected_sel)
    v1_sel_keys = set(evidence.get("v1_batch_sel_keys", set()))
    v1_clf_keys = set(evidence.get("v1_batch_clf_keys", set()))
    v1_sel_count = int(evidence.get("v1_sel_count", len(v1_sel_keys)))
    v1_clf_count = int(evidence.get("v1_clf_count", len(v1_clf_keys)))
    v1_sel_dupes = v1_sel_count != len(v1_sel_keys)
    v1_clf_dupes = v1_clf_count != len(v1_clf_keys)
    missing_v1_sel = sorted(exp_v1_sel - v1_sel_keys)
    missing_v1_clf = sorted(exp_v1_clf - v1_clf_keys)
    e3 = {"outcome": "PASS" if (v1_sel_keys == exp_v1_sel and v1_clf_keys == exp_v1_clf
                                 and not v1_sel_dupes and not v1_clf_dupes) else "FAIL",
          "measured": (f"V1 sel {len(v1_sel_keys)}/{len(exp_v1_sel)} "
                       f"missing={len(missing_v1_sel)} dupes={v1_sel_dupes}; "
                       f"V1 clf {len(v1_clf_keys)}/{len(exp_v1_clf)} "
                       f"missing={len(missing_v1_clf)} dupes={v1_clf_dupes}")}

    map_failures = list(evidence.get("map_failures", []))
    e4 = {"outcome": "PASS" if len(map_failures) == 0 else "FAIL",
          "measured": f"map_failures={len(map_failures)}"}

    all_finalized = bool(evidence.get("all_finalized", False))
    e5 = {"outcome": "PASS" if all_finalized else "FAIL",
          "measured": f"all_finalized={all_finalized}"}

    oracle_ok = bool(evidence.get("oracle_ok", False))
    e6 = {"outcome": "PASS" if oracle_ok else "FAIL",
          "measured": str(evidence.get("e6_measured", f"oracle_ok={oracle_ok}"))}

    v2_clf = list(evidence.get("v2_clf", []))
    valid_statuses = {"COMPLETED", "EMPTY_SELECTION", "FIT_ERROR"}
    statuses_ok = bool(v2_clf) and all(
        isinstance(r, dict) and r.get("status") in valid_statuses for r in v2_clf
    )
    e7 = {"outcome": "PASS" if (clf_keys == expected_clf and clf_count == len(expected_clf)
                                 and len(v2_clf) == len(expected_clf) and statuses_ok) else "FAIL",
          "measured": (f"classifiers={len(v2_clf)}/{len(expected_clf)} "
                       f"equality={clf_keys == expected_clf} dupes={clf_dupes} "
                       f"statuses_ok={statuses_ok}")}

    v2_j = evidence.get("v2_j", None)
    v1_j = evidence.get("v1_j", None)
    paired_cart = list(evidence.get("paired_cart", []))
    missing_pairs = list(evidence.get("missing_pairs", []))
    v1_cart_dupes = list(evidence.get("v1_cart_dupes", []))
    expected_cart_pairs = int(evidence.get("expected_cart_pairs", len(exp_v1_sel)))
    finite_acc = bool(paired_cart) and all(
        isinstance(p.get("v2_accuracy"), (int, float)) and math.isfinite(float(p["v2_accuracy"]))
        and isinstance(p.get("v1_accuracy"), (int, float)) and math.isfinite(float(p["v1_accuracy"]))
        for p in paired_cart
    )
    arcene_complete = bool(evidence.get("arcene_complete", False))
    fit_statuses = list(evidence.get("fit_statuses", [r.get("status") for r in v2_clf if isinstance(r, dict)]))
    no_fit_errors = "FIT_ERROR" not in fit_statuses
    all_fits_completed = bool(evidence.get("all_fits_completed", no_fit_errors))
    e8 = {"outcome": "PASS" if (v2_j is not None and v1_j is not None
                                 and len(paired_cart) == expected_cart_pairs
                                 and len(missing_pairs) == 0 and len(v1_cart_dupes) == 0
                                 and finite_acc and arcene_complete
                                 and all_fits_completed and no_fit_errors) else "FAIL",
          "measured": (f"H1_delta={(v2_j - v1_j) if (v2_j is not None and v1_j is not None) else None} "
                       f"cart_pairs={len(paired_cart)}/{expected_cart_pairs} "
                       f"missing={len(missing_pairs)} dupes={len(v1_cart_dupes)} "
                       f"finite={finite_acc} arcene_complete={arcene_complete} "
                       f"fit_errors={fit_statuses.count('FIT_ERROR')}")}

    pytest_exit = evidence.get("pytest_exit", None)
    ruff_exit = evidence.get("ruff_exit", None)
    preflight_pass = evidence.get("preflight_pass", False)
    e9 = {"outcome": "PASS" if (pytest_exit == 0 and ruff_exit == 0 and preflight_pass is True) else "FAIL",
          "measured": (f"pytest_exit={pytest_exit} ruff_exit={ruff_exit} "
                       f"preflight_pass={preflight_pass}")}

    all_ok = all(g["outcome"] == "PASS" for g in (e1, e2, e3, e4, e5, e6, e7, e8, e9))
    e10 = {"outcome": "PASS" if all_ok else "FAIL", "measured": f"all_ok={all_ok}"}
    gates = {f"E{i}": g for i, g in enumerate(
        [e1, e2, e3, e4, e5, e6, e7, e8, e9, e10], 1)}
    verdict = "COMPARISON_EVIDENCE_PASS" if all_ok else "COMPARISON_EVIDENCE_PARTIAL"
    return gates, verdict


# Exact V2 -> V1 arm mapping, exported for tests. A naive
# ``.replace("_V2", "")`` would also mangle ``D2_V2_pilot``.
ARM_MAP = {"D2_V2": "D2", "D2_V2_pilot": "D2", "C2_V2": "C2", "C4_V2": "C4"}


def split_selection_key(record_id: str):
    """(dataset, arm, seed, fold) from a V2 selection key, or None."""
    m = re.search(r"^(.+)_seed(\d+)_fold(\d+)__(.+)$", record_id)
    if not m:
        return None
    return m.group(1), m.group(4), int(m.group(2)), int(m.group(3))


def split_classifier_key(record_id: str):
    """(dataset, arm, seed, fold, classifier) from a classifier key, or None."""
    if "__" not in record_id:
        return None
    sel_part, clf = record_id.rsplit("__", 1)
    if "__" not in sel_part:
        return None
    head, arm = sel_part.rsplit("__", 1)
    m = re.search(r"_seed(\d+)_fold(\d+)$", head)
    if not m:
        return None
    return head[:m.start()], arm, int(m.group(1)), int(m.group(2)), clf


def load_frozen_protocol(path: Path) -> dict:
    """Load the frozen YAML exactly once; raise a clear error, never a raw traceback."""
    import yaml as _y
    if not path.exists():
        raise RuntimeError(f"FROZEN_PROTOCOL_MISSING: {path}")
    try:
        result = _y.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - re-raise with a useful diagnosis
        raise RuntimeError(f"FROZEN_PROTOCOL_UNREADABLE: {type(exc).__name__}: {exc}") from exc
    if not isinstance(result, dict):
        raise RuntimeError(
            f"FROZEN_PROTOCOL_INVALID: expected a YAML mapping, got {type(result).__name__}"
        )
    return result


def build_paired_cart(v2_clf, v1_clf) -> tuple[list[dict], list[dict], list[str]]:
    """Exact one-to-one CART join on (dataset, seed, fold, mapped_arm).

    Pairs every V2 CART record (any status) with the corresponding V1 CART record.
    Missing or invalid counterparts are reported; no cross-arm pairing is allowed.
    """
    index: dict[tuple, dict] = {}
    dupes: list[str] = []
    for c1 in v1_clf:
        if c1.get("classifier") != "cart":
            continue
        parsed = split_classifier_key(c1.get("record_id", ""))
        if parsed is None:
            continue
        ds, arm, seed, fold, _ = parsed
        k = (ds, seed, fold, arm)
        if k in index:
            dupes.append(c1.get("record_id", str(k)))
        else:
            index[k] = c1
    paired, missing = [], []
    for c2 in v2_clf:
        if c2.get("classifier") != "cart":
            continue
        parsed = split_classifier_key(c2.get("record_id", ""))
        if parsed is None:
            missing.append((c2.get("record_id"), "unparseable key"))
            continue
        ds, arm, seed, fold, _ = parsed
        v1_arm = ARM_MAP.get(arm)
        if v1_arm is None:
            missing.append((c2.get("record_id"), "unmapped arm"))
            continue
        c1 = index.get((ds, seed, fold, v1_arm))
        if c1 is None:
            missing.append((c2.get("record_id"), f"no V1 {v1_arm} pair"))
            continue
        paired.append({
            "dataset": ds, "seed": seed, "fold": fold,
            "v2_arm": arm, "v1_arm": v1_arm, "classifier": "cart",
            "v2_status": c2.get("status"),
            "v1_status": c1.get("status"),
            "v2_accuracy": c2.get("accuracy"),
            "v1_accuracy": c1.get("accuracy"),
            "signed_delta_v2_minus_v1": (
                c2["accuracy"] - c1["accuracy"]
                if isinstance(c2.get("accuracy"), (int, float)) and isinstance(c1.get("accuracy"), (int, float))
                else None
            ),
        })
    return paired, missing, dupes


def verify_closure_hashes(run_dir: Path, frozen: dict, project_root: Path) -> tuple[bool, str, dict]:
    """Recompute closure file hashes from disk and verify config source hash.

    closure_hashes.json entries are treated as paths relative to project_root.
    The original config hash must be verified against a grounded retained source
    object (SOURCE_CONFIG.yaml or equivalent) rather than the post-stamped
    FROZEN_PROTOCOL.yaml bytes.
    """
    closure_path = run_dir / "closure_hashes.json"
    closure: dict = {}
    if closure_path.exists():
        try:
            closure = json.loads(closure_path.read_text(encoding="utf-8")) or {}
        except Exception:
            closure = {}
    if not isinstance(closure, dict) or not closure:
        return False, "closure_hashes missing or empty", {"closure_entries": 0}

    mismatches: list[str] = []
    missing_files: list[str] = []
    checked = 0
    for rel, expected in sorted(closure.items()):
        if not isinstance(rel, str) or not isinstance(expected, str):
            mismatches.append(f"{rel}: bad entry type")
            continue
        p = (project_root / rel).resolve()
        # keep verification within project root for safety/clarity
        try:
            p.relative_to(project_root.resolve())
        except Exception:
            mismatches.append(f"{rel}: outside project root")
            continue
        if not p.exists() or not p.is_file():
            missing_files.append(rel)
            continue
        actual = sha256_file(p)
        checked += 1
        if actual != expected:
            mismatches.append(f"{rel}: recorded={expected[:12]} actual={actual[:12]}")

    cfg_sha = frozen.get("config_sha256")
    src_cfg = run_dir / "SOURCE_CONFIG.yaml"
    cfg_ok = False
    cfg_note = "missing SOURCE_CONFIG.yaml"
    if src_cfg.exists() and isinstance(cfg_sha, str):
        actual_cfg = sha256_file(src_cfg)
        cfg_ok = (actual_cfg == cfg_sha)
        cfg_note = f"source_config_hash_match={cfg_ok}"
    elif isinstance(cfg_sha, str):
        cfg_note = "SOURCE_CONFIG.yaml missing (cannot ground original config hash)"

    ok = (not mismatches and not missing_files and checked == len(closure) and cfg_ok)
    measured = (f"closure_entries={len(closure)} checked={checked} "
                f"missing={len(missing_files)} mismatches={len(mismatches)} "
                f"config_sha256={'set' if cfg_sha else 'MISSING'} {cfg_note}")
    details = {
        "closure_entries": len(closure), "checked": checked,
        "missing_files": missing_files[:20], "mismatches": mismatches[:20],
        "config_sha256_ok": cfg_ok,
    }
    return ok, measured, details


def arcene_coverage_complete(records: list[dict], arm: str, expected_seeds: set[int], expected_folds: set[int]) -> bool:
    """Exact within-fold seed/order coverage for ARCENE arm records."""
    got: dict[int, set[int]] = {f: set() for f in expected_folds}
    for r in records:
        if r.get("dataset") != "ARCENE" or r.get("arm") != arm:
            continue
        f = r.get("fold")
        s = r.get("seed")
        if f in got and s in expected_seeds:
            got[f].add(int(s))
    return all(got[f] == expected_seeds for f in expected_folds)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    run_id = args.run_id
    run_dir = OUT_ROOT / f"comparison_{run_id}"
    if not run_dir.exists():
        raise SystemExit(f"run not found: {run_dir}")
    audit_dir = PROJECT / "results" / "stage3d_v2" / f"audit_{run_id}"
    audit_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((run_dir / "comparison_key_manifest.json").read_text(encoding="utf-8"))
    expected_sel = set(manifest["selection_keys"])
    expected_clf = set(manifest["classifier_keys"])

    # ---- load V2 records -------------------------------------------
    sel_dir = run_dir / "selection_records"
    clf_dir = run_dir / "classification_records"
    v2_sel = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(sel_dir.glob("*.json"))] if sel_dir.exists() else []
    v2_clf = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(clf_dir.glob("*.json"))] if clf_dir.exists() else []
    v2_sel_keys = {r.get("key") for r in v2_sel if isinstance(r, dict)}
    v2_clf_keys = {r.get("record_id") for r in v2_clf if isinstance(r, dict)}

    # ---- load V1 comparator (D2/C2/C4 batch arms only) -------------
    v1_raw = V1_ROOT / "raw_records"
    v1_clf_dir = V1_ROOT / "classifiers"
    v1_sel = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(v1_raw.glob("*.json"))] if v1_raw.exists() else []
    v1_clf = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(v1_clf_dir.glob("*.json"))] if v1_clf_dir.exists() else []
    v1_sel_by_key: dict[str, dict] = {}
    v1_sel_dupes: list[str] = []
    for r in v1_sel:
        if not isinstance(r, dict):
            continue
        arm = r.get("arm")
        if arm not in ("D2", "C2", "C4"):
            continue
        k = f"{r.get('dataset')}_seed{r.get('seed')}_fold{r.get('fold')}__{arm}"
        if k in v1_sel_by_key:
            v1_sel_dupes.append(k)
        else:
            v1_sel_by_key[k] = r
    v1_batch_sel_keys = set(v1_sel_by_key.keys())
    v1_batch_clf_keys: set[str] = set()
    v1_clf_dupes: list[str] = []
    seen_v1_clf: set[str] = set()
    for c in v1_clf:
        if not isinstance(c, dict):
            continue
        rid = c.get("record_id", "")
        parsed = split_classifier_key(rid)
        if parsed is None:
            continue
        _, arm, _, _, _ = parsed
        if arm not in ("D2", "C2", "C4"):
            continue
        if rid in seen_v1_clf:
            v1_clf_dupes.append(rid)
        else:
            seen_v1_clf.add(rid)
            v1_batch_clf_keys.add(rid)

    # ---- E2: config freeze (real hash verification) -----------------
    frozen: dict = {}
    frozen_load_error = None
    frozen_path = run_dir / "FROZEN_PROTOCOL.yaml"
    if not frozen_path.exists():
        frozen_load_error = "FROZEN_PROTOCOL.yaml missing"
    else:
        try:
            frozen = load_frozen_protocol(frozen_path)
        except Exception as exc:  # noqa: BLE001
            frozen = {}
            frozen_load_error = str(exc)
    cfg_ok, e2_measured, e2_details = (False, f"load_error={frozen_load_error}", {})
    if frozen_load_error is None:
        cfg_ok, e2_measured, e2_details = verify_closure_hashes(run_dir, frozen, PROJECT)

    # ---- E4: mapping (feature IDs in range) --------------------------
    from ossfs_dd_data.workspace import processed_dir
    true_p: dict[str, int] = {}
    map_failures: list[list] = []
    try:
        for ds in {r.get("dataset") for r in v2_sel if isinstance(r, dict)}:
            if ds is None:
                continue
            schema = json.loads((processed_dir(ds) / "schema.json").read_text(encoding="utf-8"))
            true_p[ds] = int(schema["n_features"])
        for r in v2_sel:
            if not isinstance(r, dict):
                map_failures.append(["<non-dict>", "record not a dict"])
                continue
            ds = r.get("dataset")
            p = true_p.get(ds)
            ids = r.get("selected_feature_ids") or []
            if p is None:
                map_failures.append([r.get("key"), "unknown dataset dimensionality"])
                continue
            if r.get("n_features_total") != p:
                map_failures.append([r.get("key"), f"n_features_total {r.get('n_features_total')} != {p}"])
            if not isinstance(ids, list):
                map_failures.append([r.get("key"), "selected_feature_ids not a list"])
                continue
            if any((not isinstance(i, int)) or i < 0 or i >= p for i in ids):
                map_failures.append([r.get("key"), f"ID out of [0,{p})"])
            if len(set(ids)) != len(ids):
                map_failures.append([r.get("key"), "duplicate selected IDs"])
    except Exception as exc:  # noqa: BLE001 - fail closed
        map_failures.append(["<exception>", f"{type(exc).__name__}: {exc}"])

    # ---- E5: V2 lifecycle (finalized) --------------------------------
    all_finalized = bool(v2_sel) and all(
        isinstance(r, dict) and r.get("finalized") is True for r in v2_sel
    )

    # ---- E6: Fisher oracle ------------------------------------------
    from scipy import stats as _st
    from ossfs_dd_stage3d.standard_fisherz import marginal_fisherz as _mfz
    rng6 = np.random.default_rng(20261118)
    oracle_cases = []
    worst_Z = worst_p = 0.0
    for _ in range(60):
        n = int(rng6.integers(6, 250))
        x = rng6.normal(size=n)
        k = int(rng6.integers(2, 4))
        y = rng6.choice([f"c{i}" for i in range(k)], size=n)
        m = _mfz(x, y, use_evidence=True)
        b = (y == m.winning_class).astype(float)
        if np.ptp(b) == 0 or np.ptp(x) == 0:
            continue
        r = float(np.corrcoef(x, b)[0, 1])
        Z = float(np.arctanh(np.clip(r, -(1 - 1e-12), 1 - 1e-12)) * np.sqrt(n - 3))
        pval = float(2 * _st.norm.sf(abs(Z)))
        worst_Z = max(worst_Z, abs(abs(m.Z) - abs(Z)))
        worst_p = max(worst_p, abs(m.p_raw - pval))
        oracle_cases.append({"n": n, "dZ": abs(abs(m.Z) - abs(Z)), "dp": abs(m.p_raw - pval)})
    oracle_ok = (len(oracle_cases) >= 50 and worst_Z <= 1e-8 and worst_p <= 1e-10)
    e6_measured = (f"cases={len(oracle_cases)} max_dZ={worst_Z:.3e} "
                   f"max_dp={worst_p:.3e} tol_Z=1e-8 tol_p=1e-10")

    # ---- E8: Jaccard + paired joins ----------------------------------
    expected_arc_seeds = set(range(11, 21))  # placeholder, replaced below if possible
    # derive ARCENE seeds from expected_sel keys
    arc_seeds_v2 = set()
    arc_folds_v2 = set()
    for k in expected_sel:
        p = split_selection_key(k)
        if p and p[0] == "ARCENE" and p[1] == "D2_V2":
            arc_seeds_v2.add(p[2]); arc_folds_v2.add(p[3])
    if not arc_seeds_v2:
        arc_seeds_v2 = {11, 17, 23, 31, 47, 59, 71, 83, 97, 109}
        arc_folds_v2 = {0, 1, 2, 3, 4}

    arc_v2 = [r for r in v2_sel if isinstance(r, dict) and r.get("dataset") == "ARCENE" and r.get("arm") == "D2_V2"]
    arc_v1 = [r for r in v1_sel if isinstance(r, dict) and r.get("dataset") == "ARCENE" and r.get("arm") == "D2"]
    v2_means, v1_means = [], []
    for fold in sorted(arc_folds_v2):
        fv2 = [r for r in arc_v2 if r.get("fold") == fold]
        fv1 = [r for r in arc_v1 if r.get("fold") == fold]
        if len(fv2) >= 2:
            js2 = [jaccard(a.get("selected_feature_ids", []), b.get("selected_feature_ids", []))
                   for a, b in combinations(sorted(fv2, key=lambda x: x.get("seed", 0)), 2)]
            v2_means.append(float(np.mean(js2)) if js2 else float("nan"))
        if len(fv1) >= 2:
            js1 = [jaccard(a.get("selected_feature_ids", []), b.get("selected_feature_ids", []))
                   for a, b in combinations(sorted(fv1, key=lambda x: x.get("seed", 0)), 2)]
            v1_means.append(float(np.mean(js1)) if js1 else float("nan"))
    v2_j = float(np.mean(v2_means)) if v2_means and all(np.isfinite(v2_means)) else None
    v1_j = float(np.mean(v1_means)) if v1_means and all(np.isfinite(v1_means)) else None
    delta_j = (v2_j - v1_j) if (v2_j is not None and v1_j is not None) else None

    arcene_complete = (arcene_coverage_complete(arc_v2, "D2_V2", arc_seeds_v2, arc_folds_v2)
                       and arcene_coverage_complete(arc_v1, "D2", arc_seeds_v2, arc_folds_v2))

    paired_cart, missing_pairs, v1_cart_dupes = build_paired_cart(v2_clf, v1_clf)
    cart_deltas = [p["signed_delta_v2_minus_v1"] for p in paired_cart
                   if isinstance(p.get("signed_delta_v2_minus_v1"), (int, float))]
    cart_mean = float(np.mean(cart_deltas)) if cart_deltas else None
    exp_v1_sel_keys, exp_v1_clf_keys = expected_v1_keys_from_v2(expected_sel)
    expected_cart_pairs = len(exp_v1_sel_keys)

    fit_statuses = [r.get("status") for r in v2_clf if isinstance(r, dict)]
    no_fit_errors = "FIT_ERROR" not in fit_statuses
    all_fits_completed = bool(fit_statuses) and all(s in ("COMPLETED", "EMPTY_SELECTION") for s in fit_statuses)

    h1 = {"outcome": ("SUPPORTED" if delta_j is not None and delta_j >= 0.10 else
                      "NOT_SUPPORTED" if delta_j is not None else "NOT_MEASURED"),
          "v2_mean_jaccard": v2_j, "v1_mean_jaccard": v1_j, "delta": delta_j,
          "threshold": 0.10, "arcene_complete": arcene_complete,
          "note": "five per-fold aggregates over 10 common seeds; not 225 independent samples"}
    h2 = {"outcome": ("SUPPORTED" if cart_mean is not None and cart_mean >= -0.03
                      and len(paired_cart) == expected_cart_pairs and no_fit_errors else
                      "NOT_SUPPORTED" if cart_mean is not None and len(paired_cart) == expected_cart_pairs
                      and no_fit_errors else "NOT_MEASURED"),
          "n_paired": len(cart_deltas), "n_missing_pairs": len(missing_pairs),
          "n_duplicate_v1_keys": len(v1_cart_dupes),
          "expected_cart_pairs": expected_cart_pairs,
          "mean_delta": cart_mean, "threshold": -0.03,
          "note": "descriptive tolerance, not a formal noninferiority test"}

    # ---- E9: user-run tests + lint + preservation --------------------
    e9_parts: dict[str, object] = {}
    for name in ("pytest", "ruff"):
        log = run_dir / f"{name}_preflight.log"
        ext = run_dir / f"{name}_preflight.exit"
        if not log.exists() or not ext.exists():
            e9_parts[name] = "MISSING"
            continue
        try:
            rc = int(ext.read_text(encoding="utf-8").strip())
        except ValueError:
            e9_parts[name] = "BAD_EXIT_FILE"
            continue
        e9_parts[name] = rc
    pp = run_dir / "PREFLIGHT_PASS.json"
    pp_ok = False
    if pp.exists():
        try:
            pp_ok = json.loads(pp.read_text(encoding="utf-8")).get("status") == "PASS"
        except Exception:
            pp_ok = False

    # ---- build evidence and evaluate gates ---------------------------
    evidence = {
        "sel_keys": v2_sel_keys, "expected_sel": expected_sel,
        "clf_keys": v2_clf_keys, "expected_clf": expected_clf,
        "sel_count": len(v2_sel), "clf_count": len(v2_clf),
        "cfg_ok": cfg_ok, "e2_measured": e2_measured,
        "v1_batch_sel_keys": v1_batch_sel_keys, "v1_batch_clf_keys": v1_batch_clf_keys,
        "v1_sel_count": len([r for r in v1_sel if isinstance(r, dict) and r.get("arm") in ("D2","C2","C4")]),
        "v1_clf_count": len([r for r in v1_clf if isinstance(r, dict) and split_classifier_key(r.get("record_id","")) and split_classifier_key(r.get("record_id",""))[1] in ("D2","C2","C4")]),
        "map_failures": map_failures, "all_finalized": all_finalized,
        "oracle_ok": oracle_ok, "e6_measured": e6_measured,
        "v2_clf": v2_clf,
        "v2_j": v2_j, "v1_j": v1_j,
        "paired_cart": paired_cart, "missing_pairs": missing_pairs, "v1_cart_dupes": v1_cart_dupes,
        "expected_cart_pairs": expected_cart_pairs,
        "arcene_complete": arcene_complete,
        "fit_statuses": fit_statuses, "all_fits_completed": all_fits_completed,
        "pytest_exit": e9_parts.get("pytest"), "ruff_exit": e9_parts.get("ruff"),
        "preflight_pass": pp_ok,
    }
    gates, verdict = evaluate_gates(evidence)

    # ---- write deliverables (always, even on failure) ---------------
    (audit_dir / "paired_join_ledger.csv").write_text(
        "dataset,seed,fold,v2_arm,v1_arm,classifier,v2_status,v1_status,v2_accuracy,v1_accuracy,signed_delta_v2_minus_v1\n" +
        "\n".join(
            f"{p.get('dataset')},{p.get('seed')},{p.get('fold')},{p.get('v2_arm')},{p.get('v1_arm')},cart,"
            f"{p.get('v2_status')},{p.get('v1_status')},"
            f"{'' if p.get('v2_accuracy') is None else p.get('v2_accuracy')},"
            f"{'' if p.get('v1_accuracy') is None else p.get('v1_accuracy')},"
            f"{'' if p.get('signed_delta_v2_minus_v1') is None else p.get('signed_delta_v2_minus_v1')}"
            for p in paired_cart
        ) + "\n",
        encoding="utf-8")
    (audit_dir / "paired_join_missing.json").write_text(json.dumps(
        {"n_missing": len(missing_pairs), "missing": missing_pairs[:100],
         "n_duplicate_v1_keys": len(v1_cart_dupes), "duplicate_v1_keys": v1_cart_dupes[:100],
         "missing_v1_sel_keys": sorted(set(exp_v1_sel_keys) - v1_batch_sel_keys)[:100],
         "missing_v1_clf_keys": sorted(set(exp_v1_clf_keys) - v1_batch_clf_keys)[:100],
         "duplicate_v1_sel_records": v1_sel_dupes[:100],
         "duplicate_v1_clf_records": v1_clf_dupes[:100]},
        indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (audit_dir / "e2_hash_verification.json").write_text(
        json.dumps({"ok": cfg_ok, "measured": e2_measured, "details": e2_details},
                   indent=2, sort_keys=True) + "\n", encoding="utf-8")

    (audit_dir / "independent_recompute.json").write_text(
        json.dumps({"H1": h1, "H2": h2}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (audit_dir / "gate_results.json").write_text(
        json.dumps(gates, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    has_clf = clf_dir.exists() and any(clf_dir.glob("*.json"))
    (audit_dir / "FINAL_VERDICT.json").write_text(
        json.dumps({"verdict": verdict, "utc": utcnow(), "run_id": run_id,
                    "gates": {k: v["outcome"] for k, v in gates.items()},
                    "h1": h1["outcome"], "h2": h2["outcome"],
                    "v2_performance_not_measured": not has_clf,
                    "v2_performance_note": ("Classifier outcomes exist; see report for actual metrics."
                                           if has_clf else "No classifier outcomes yet."),
                    "v2_performance_status": ("V2_PERFORMANCE_MEASURED" if has_clf
                                             else "V2_PERFORMANCE_NOT_MEASURED")},
                   indent=2, sort_keys=True) + "\n", encoding="utf-8")

    arc_rows = []
    for fold in range(5):
        v2m = v2_means[fold] if fold < len(v2_means) else None
        v1m = v1_means[fold] if fold < len(v1_means) else None
        d = (v2m - v1m) if (v2m is not None and v1m is not None) else None
        arc_rows.append(f"{fold},{'' if v2m is None else v2m},{'' if v1m is None else v1m},{'' if d is None else d}")
    (audit_dir / "arcene_stability_within_folds.csv").write_text(
        "fold,v2_mean_jaccard,v1_mean_jaccard,delta\n" + "\n".join(arc_rows) + "\n",
        encoding="utf-8")
    (audit_dir / "STAGE3D_V2_COMPARISON_AUDIT_REPORT.md").write_text(
        f"# Stage 3D V2 comparison audit\nRun `{run_id}` · {utcnow()} · **{verdict}**\n\n"
        f"## Gates\n" + "\n".join(f"- {k}: {v['outcome']} ({v['measured']})" for k, v in sorted(gates.items())) +
        f"\n\n## Hypotheses\n- H1: {h1['outcome']} (Δ={delta_j})\n"
        f"- H2: {h2['outcome']} (n={h2['n_paired']}, Δ={h2['mean_delta']})\n\n"
        f"FINAL_VERDICT: {verdict}\n", encoding="utf-8")

    print(f"  audit dir: {audit_dir}")
    print(f"  verdict: {verdict}")
    print(f"  H1: {h1['outcome']} (Δ={delta_j})")
    print(f"  H2: {h2['outcome']} (n={h2['n_paired']}, Δ={h2['mean_delta']})")
    return 0 if verdict == "COMPARISON_EVIDENCE_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


if __name__ == "__main__":
    raise SystemExit(main())
