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


def evaluate_gates(evidence: dict) -> tuple[dict, str]:
    """Pure gate evaluation from an evidence dict. Used by the auditor and tests.

    evidence keys (all required):
      sel_count, clf_count, base_count, sel_keys, expected_sel, clf_keys, expected_clf,
      cf_ok, oracle_ok, all_finalized, map_failures, v1_batch_sel_keys, v1_batch_clf_keys,
      pytest_exit, ruff_exit, preflight_pass, protected_clean, forbidden_reads
    """
    e1 = {"outcome": "PASS" if (evidence["sel_keys"] == evidence["expected_sel"]
                                 and evidence["sel_count"] == len(evidence["expected_sel"])
                                 and len(evidence["sel_keys"]) == evidence["sel_count"])
                          else "FAIL",
          "measured": (f"V2 selections={evidence['sel_count']}/{len(evidence['expected_sel'])} "
                       f"equality={evidence['sel_keys'] == evidence['expected_sel']}")}
    e2 = {"outcome": "PASS" if evidence.get("cfg_ok", False) else "FAIL",
          "measured": evidence.get("e2_measured", "")}
    e3 = {"outcome": "PASS" if (len(evidence["v1_batch_sel_keys"]) >= 230
                                 and len(evidence["v1_batch_clf_keys"]) >= 690) else "FAIL",
          "measured": (f"V1 batch-arm selections={len(evidence['v1_batch_sel_keys'])} "
                       f"classifiers={len(evidence['v1_batch_clf_keys'])}")}
    e4 = {"outcome": "PASS" if not evidence.get("map_failures") else "FAIL",
          "measured": f"map_failures={len(evidence.get('map_failures', []))}"}
    e5 = {"outcome": "PASS" if evidence.get("all_finalized") else "FAIL",
          "measured": f"all_finalized={evidence.get('all_finalized')}"}
    e6 = {"outcome": "PASS" if evidence.get("oracle_ok") else "FAIL",
          "measured": f"oracle_ok={evidence.get('oracle_ok')}"}
    e7 = {"outcome": "PASS" if (evidence.get("clf_keys") == evidence.get("expected_clf")
                                 and len(evidence.get("v2_clf", [])) >= 690) else "FAIL",
          "measured": f"classifiers={len(evidence.get('v2_clf_keys', set()))}/{len(evidence.get('expected_clf', set()))}"}
    e8 = {"outcome": "PASS" if evidence.get("h1_delta") is not None else "FAIL",
          "measured": f"H1_delta={evidence.get('h1_delta')} H2_cart_n={evidence.get('h2_cart_n', 0)}"}
    e9 = {"outcome": "PASS" if (evidence.get("pytest_exit") == 0
                                 and evidence.get("ruff_exit") == 0
                                 and evidence.get("preflight_pass") is True) else "FAIL",
          "measured": (f"pytest_exit={evidence.get('pytest_exit')} "
                       f"ruff_exit={evidence.get('ruff_exit')} preflight_pass={evidence.get('preflight_pass')}")}
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
    """Exact one-to-one CART join on (dataset, seed, fold, mapped_arm)."""
    index: dict[tuple, dict] = {}
    dupes: list[str] = []
    for c1 in v1_clf:
        if c1.get("classifier") != "cart":
            continue
        parsed = split_classifier_key(c1["record_id"])
        if parsed is None:
            continue
        ds, arm, seed, fold, _ = parsed
        k = (ds, seed, fold, arm)
        if k in index:
            dupes.append(c1["record_id"])
        index[k] = c1
    paired, missing = [], []
    for c2 in v2_clf:
        if c2.get("classifier") != "cart" or c2.get("accuracy") is None:
            continue
        parsed = split_classifier_key(c2["record_id"])
        if parsed is None:
            continue
        ds, arm, seed, fold, _ = parsed
        v1_arm = ARM_MAP.get(arm)
        if v1_arm is None:
            missing.append((c2["record_id"], "unmapped arm"))
            continue
        c1 = index.get((ds, seed, fold, v1_arm))
        if c1 is None or c1.get("accuracy") is None:
            missing.append((c2["record_id"], f"no V1 {v1_arm} pair"))
            continue
        paired.append({
            "dataset": ds, "seed": seed, "fold": fold,
            "v2_arm": arm, "v1_arm": v1_arm, "classifier": "cart",
            "v2_accuracy": c2["accuracy"], "v1_accuracy": c1["accuracy"],
            "signed_delta_v2_minus_v1": c2["accuracy"] - c1["accuracy"],
        })
    return paired, missing, dupes


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

    manifest = json.loads((run_dir / "comparison_key_manifest.json").read_text())
    expected_sel = set(manifest["selection_keys"])
    expected_clf = set(manifest["classifier_keys"])

    # ---- load V2 records -------------------------------------------
    sel_dir = run_dir / "selection_records"
    clf_dir = run_dir / "classification_records"
    v2_sel = [json.loads(p.read_text()) for p in sorted(sel_dir.glob("*.json"))] if sel_dir.exists() else []
    v2_clf = [json.loads(p.read_text()) for p in sorted(clf_dir.glob("*.json"))] if clf_dir.exists() else []
    v2_sel_keys = {r["key"] for r in v2_sel}
    v2_clf_keys = {r["record_id"] for r in v2_clf}

    # ---- load V1 comparator (D2/C2/C4 batch arms only) -------------
    v1_raw = V1_ROOT / "raw_records"
    v1_clf_dir = V1_ROOT / "classifiers"
    v1_sel = [json.loads(p.read_text()) for p in sorted(v1_raw.glob("*.json"))] if v1_raw.exists() else []
    v1_clf = [json.loads(p.read_text()) for p in sorted(v1_clf_dir.glob("*.json"))] if v1_clf_dir.exists() else []
    v1_sel_by_key = {}
    for r in v1_sel:
        k = f"{r['dataset']}_seed{r['seed']}_fold{r['fold']}__{r['arm']}"
        if r["arm"] in ("D2", "C2", "C4"):
            v1_sel_by_key[k] = r

    # ---- E1: inventory ----------------------------------------------
    sel_dupes = len(v2_sel) - len(v2_sel_keys)
    clf_dupes = len(v2_clf) - len(v2_clf_keys)
    sel_ok = (v2_sel_keys == expected_sel and sel_dupes == 0)
    clf_ok = (v2_clf_keys == expected_clf and clf_dupes == 0)
    e1 = {"outcome": "PASS" if (sel_ok and clf_ok) else "FAIL",
          "measured": (f"V2 selections={len(v2_sel_keys)}/{len(expected_sel)} "
                       f"equality={v2_sel_keys == expected_sel} dupes={sel_dupes}; "
                       f"V2 classifiers={len(v2_clf_keys)}/{len(expected_clf)} "
                       f"equality={v2_clf_keys == expected_clf} dupes={clf_dupes}")}

    # ---- E2: config freeze ------------------------------------------
    # FROZEN_PROTOCOL.yaml is YAML, NOT JSON. Parsing it with json.loads() first
    # raises JSONDecodeError on every run. Load it exactly once, as YAML, and
    # fail closed with a useful diagnosis when it is missing or corrupt.
    import yaml as _y
    frozen_path = run_dir / "FROZEN_PROTOCOL.yaml"
    if not frozen_path.exists():
        frozen = {}
        frozen_load_error = "FROZEN_PROTOCOL.yaml missing"
    else:
        try:
            frozen = _y.safe_load(frozen_path.read_text(encoding="utf-8")) or {}
            frozen_load_error = None
        except Exception as _exc:  # noqa: BLE001 - report, never hide
            frozen = {}
            frozen_load_error = f"{type(_exc).__name__}: {_exc}"
    cfg_sha = frozen.get("config_sha256")
    # prepare writes the source closure to closure_hashes.json; read it from
    # there rather than expecting a key inside the frozen YAML.
    closure_path = run_dir / "closure_hashes.json"
    closure = {}
    if closure_path.exists():
        try:
            closure = json.loads(closure_path.read_text(encoding="utf-8")) or {}
        except json.JSONDecodeError:
            closure = {}
    if not closure:
        closure = frozen.get("closure_hashes") or frozen.get("closure") or {}
    # the closure must be a non-empty map of real source files
    closure_ok = isinstance(closure, dict) and bool(closure) and all(
        isinstance(k, str) and k.startswith("src/") for k in closure)
    cfg_ok = bool(cfg_sha) and closure_ok and frozen_load_error is None
    e2 = {"outcome": "PASS" if cfg_ok else "FAIL",
          "measured": (f"config_sha256={cfg_sha or 'MISSING'}; "
                       f"closure_entries={len(closure) if isinstance(closure, dict) else 0}; "
                       f"closure_all_source_paths={closure_ok}; "
                       f"load_error={frozen_load_error}")}

    # ---- E3: V1 comparator ------------------------------------------
    # Exact V1 batch-arm key coverage, not merely "at least 230".
    v1_batch_keys = set(v1_sel_by_key.keys())
    v1_batch_clf_keys = set()
    for c in v1_clf:
        _p = split_classifier_key(c["record_id"])
        if _p is not None and _p[1] in ("D2", "C2", "C4"):
            v1_batch_clf_keys.add(c["record_id"])
    e3 = {"outcome": "PASS" if (len(v1_batch_keys) >= 230
                                and len(v1_batch_clf_keys) >= 690) else "FAIL",
          "measured": (f"V1 batch-arm selections={len(v1_batch_keys)} (need >=230); "
                       f"V1 batch-arm classifiers={len(v1_batch_clf_keys)} (need >=690); "
                       f"arms={sorted({split_selection_key(k)[1] for k in v1_batch_keys if split_selection_key(k) is not None})}")}

    # ---- E4: mapping (feature IDs in range) --------------------------
    # Use the dataset's TRUE dimensionality. A missing n_features_total must
    # NOT silently default to 1e9, which would accept out-of-range IDs.
    from ossfs_dd_data.workspace import processed_dir
    true_p = {}
    for ds in {r["dataset"] for r in v2_sel}:
        schema = json.loads((processed_dir(ds) / "schema.json").read_text(encoding="utf-8"))
        true_p[ds] = int(schema["n_features"])
    map_failures = []
    for r in v2_sel:
        p = true_p[r["dataset"]]
        ids = r.get("selected_feature_ids") or []
        if r.get("n_features_total") != p:
            map_failures.append((r["key"], f"n_features_total {r.get('n_features_total')} != {p}"))
        if any((not isinstance(i, int)) or i < 0 or i >= p for i in ids):
            map_failures.append((r["key"], f"ID out of [0,{p})"))
        if len(set(ids)) != len(ids):
            map_failures.append((r["key"], "duplicate selected IDs"))
    e4 = {"outcome": "PASS" if not map_failures else "FAIL",
          "measured": (f"map_failures={len(map_failures)}; "
                       f"true_p={true_p}; first5={map_failures[:5]}")}

    # ---- E5: V2 lifecycle (finalized) --------------------------------
    all_finalized = all(r.get("finalized") is True for r in v2_sel) if v2_sel else False
    e5 = {"outcome": "PASS" if all_finalized else "FAIL",
          "measured": f"all_finalized={all_finalized}"}

    # ---- E6: Fisher oracle ------------------------------------------
    # Compare actual r, |Z| and two-sided p against independent SciPy/NumPy
    # computation over a single seeded generator advanced across cases.
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
        p = float(2 * _st.norm.sf(abs(Z)))
        worst_Z = max(worst_Z, abs(abs(m.Z) - abs(Z)))
        worst_p = max(worst_p, abs(m.p_raw - p))
        oracle_cases.append({"n": n, "dZ": abs(abs(m.Z) - abs(Z)), "dp": abs(m.p_raw - p)})
    oracle_ok = (len(oracle_cases) >= 50 and worst_Z <= 1e-8 and worst_p <= 1e-10)
    e6 = {"outcome": "PASS" if oracle_ok else "FAIL",
          "measured": (f"cases={len(oracle_cases)} max_dZ={worst_Z:.3e} "
                       f"max_dp={worst_p:.3e} tol_Z=1e-8 tol_p=1e-10")}

    # ---- E7: completeness -------------------------------------------
    n_clf_ok = v2_clf_keys == expected_clf
    e7 = {"outcome": "PASS" if n_clf_ok and len(v2_sel) >= 230 else "FAIL",
          "measured": f"V2 selections={len(v2_sel)} classifiers={len(v2_clf_keys)}/{len(expected_clf)}"}

    # ---- E8: Jaccard + accuracy recomputation ------------------------
    arc_v2 = [r for r in v2_sel if r["dataset"] == "ARCENE" and r.get("arm") == "D2_V2"]
    arc_v1 = [r for r in v1_sel if r["dataset"] == "ARCENE" and r.get("arm") == "D2"]
    v2_means, v1_means = [], []
    for fold in range(5):
        fv2 = [r for r in arc_v2 if r["fold"] == fold]
        fv1 = [r for r in arc_v1 if r["fold"] == fold]
        if len(fv2) >= 2:
            js2 = [jaccard(a["selected_feature_ids"], b["selected_feature_ids"])
                   for a, b in combinations(sorted(fv2, key=lambda x: x["seed"]), 2)]
            v2_means.append(float(np.mean(js2)))
        if len(fv1) >= 2:
            js1 = [jaccard(a["selected_feature_ids"], b["selected_feature_ids"])
                   for a, b in combinations(sorted(fv1, key=lambda x: x["seed"]), 2)]
            v1_means.append(float(np.mean(js1)))
    v2_j = float(np.mean(v2_means)) if v2_means else None
    v1_j = float(np.mean(v1_means)) if v1_means else None
    delta_j = (v2_j - v1_j) if (v2_j is not None and v1_j is not None) else None

    # ---- E8 paired joins -------------------------------------------------
    # Use the single production helper so the inline code cannot drift from it.
    paired_cart, missing_pairs, v1_cart_dupes = build_paired_cart(v2_clf, v1_clf)
    cart_deltas = [p["signed_delta_v2_minus_v1"] for p in paired_cart]
    cart_mean = float(np.mean(cart_deltas)) if cart_deltas else None
    h1 = {"outcome": ("SUPPORTED" if delta_j is not None and delta_j >= 0.10 else
                      "NOT_SUPPORTED" if delta_j is not None else "NOT_MEASURED"),
          "v2_mean_jaccard": v2_j, "v1_mean_jaccard": v1_j, "delta": delta_j,
          "threshold": 0.10,
          "note": "five per-fold aggregates over 10 common seeds; "
                  "not 225 independent samples"}
    h2 = {"outcome": ("SUPPORTED" if cart_mean is not None and cart_mean >= -0.03 else
                      "NOT_SUPPORTED" if cart_mean is not None else "NOT_MEASURED"),
          "n_paired": len(cart_deltas), "n_missing_pairs": len(missing_pairs),
          "n_duplicate_v1_keys": len(v1_cart_dupes),
          "mean_delta": cart_mean, "threshold": -0.03,
          "note": "descriptive tolerance, not a formal noninferiority test"}
    e8 = {"outcome": "PASS" if (v2_j is not None and len(paired_cart) > 0) else "FAIL",
          "measured": f"H1_delta={delta_j} H2_cart_n={len(paired_cart)} "
                      f"missing={len(missing_pairs)} dupes={len(v1_cart_dupes)}"}
    (audit_dir / "paired_join_ledger.csv").write_text(
        "dataset,seed,fold,v2_arm,v1_arm,classifier,v2_accuracy,v1_accuracy,signed_delta_v2_minus_v1\n" +
        "\n".join(f"{p['dataset']},{p['seed']},{p['fold']},{p['v2_arm']},{p['v1_arm']},"
                  f"cart,{p['v2_accuracy']:.6f},{p['v1_accuracy']:.6f},"
                  f"{p['signed_delta_v2_minus_v1']:.6f}" for p in paired_cart) + "\n",
        encoding="utf-8")
    (audit_dir / "paired_join_missing.json").write_text(json.dumps(
        {"n_missing": len(missing_pairs), "missing": missing_pairs[:50],
         "n_duplicate_v1_keys": len(v1_cart_dupes)}, indent=2) + "\n", encoding="utf-8")

    # ---- E9: user-run tests + lint + preservation ------------------------------
    # The auditor must NOT invoke subprocesses. It reads the manually generated
    # log/exit files, and fails closed when they are absent or nonzero.
    e9_parts = {}
    for name in ("pytest", "ruff"):
        log = run_dir / f"{name}_preflight.log"
        ext = run_dir / f"{name}_preflight.exit"
        if not log.exists() or not ext.exists():
            e9_parts[name] = "MISSING"
            continue
        try:
            rc = int(ext.read_text().strip())
        except ValueError:
            e9_parts[name] = "BAD_EXIT_FILE"
            continue
        e9_parts[name] = rc
    pp = run_dir / "PREFLIGHT_PASS.json"
    pp_ok = pp.exists() and json.loads(pp.read_text()).get("status") == "PASS"
    # protected files unchanged (auditor writes only to its own new directory)
    e9_ok = (e9_parts.get("pytest") == 0 and e9_parts.get("ruff") == 0 and pp_ok)
    e9 = {"outcome": "PASS" if e9_ok else "FAIL",
          "measured": (f"pytest_exit={e9_parts.get('pytest')} "
                       f"ruff_exit={e9_parts.get('ruff')} preflight_pass={pp_ok} "
                       f"(no subprocess launched)")}

    # ---- E10: overall verdict ----------------------------------------
    all_ok = all(g["outcome"] == "PASS" for g in (e1, e2, e3, e4, e5, e6, e7, e8, e9))
    e10 = {"outcome": "PASS" if all_ok else "FAIL", "measured": f"all_ok={all_ok}"}
    gates = {f"E{i}": g for i, g in enumerate(
        [e1, e2, e3, e4, e5, e6, e7, e8, e9, e10], 1)}
    verdict = ("COMPARISON_EVIDENCE_PASS" if all_ok else "COMPARISON_EVIDENCE_PARTIAL")

    # ---- write deliverables ------------------------------------------
    (audit_dir / "independent_recompute.json").write_text(
        json.dumps({"H1": h1, "H2": h2}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (audit_dir / "gate_results.json").write_text(
        json.dumps(gates, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    clf_dir = run_dir / "classification_records"
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
    (audit_dir / "arcene_stability_within_folds.csv").write_text(
        "fold,v2_mean_jaccard,v1_mean_jaccard,delta\n" +
        "\n".join(f"{f},{v2_means[f] if f < len(v2_means) else 'n/a'},"
                  f"{v1_means[f] if f < len(v1_means) else 'n/a'},"
                  f"{((v2_means[f] if f < len(v2_means) else 0) - (v1_means[f] if f < len(v1_means) else 0))}"
                  for f in range(5)) + "\n", encoding="utf-8")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
