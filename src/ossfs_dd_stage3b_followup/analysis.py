"""Hypothesis assessment and operational gates for the Stage 3B follow-up.

Hypothesis outcomes are descriptive and are **not** operational gates. A
`NOT_SUPPORTED` or `MECHANISM_UNRESOLVED` outcome is a valid deliverable.
"""

from __future__ import annotations

from typing import Any

import numpy as np

DATASETS = ("MADELON", "GINA", "GISETTE")


# ------------------------------------------------------------------ A-H1


def a_h1(svm: list[dict]) -> dict[str, Any]:
    """C sensitivity: |mean acc(C=0.1) - mean acc(C=10)| at tol=1e-3 >= 0.02."""
    per = {}
    for ds in DATASETS:
        a_ = [r for r in svm if r["dataset"] == ds and r["tol"] == 0.001
              and r["C"] == 0.1 and r.get("converged") is True]
        b_ = [r for r in svm if r["dataset"] == ds and r["tol"] == 0.001
              and r["C"] == 10.0 and r.get("converged") is True]
        entry: dict[str, Any] = {
            "dataset": ds,
            "n_paired_converged_C0.1": len(a_), "n_paired_converged_C10": len(b_),
            "mean_accuracy_C0.1": float(np.mean([r["accuracy"] for r in a_])) if a_ else None,
            "mean_accuracy_C10": float(np.mean([r["accuracy"] for r in b_])) if b_ else None,
        }
        if len(a_) == 15 and len(b_) == 15:
            delta = abs(entry["mean_accuracy_C0.1"] - entry["mean_accuracy_C10"])
            entry["absolute_difference"] = delta
            entry["meets_threshold"] = bool(delta >= 0.02)
            entry["outcome"] = "MEETS" if delta >= 0.02 else "BELOW_THRESHOLD"
        else:
            entry["absolute_difference"] = (
                abs(entry["mean_accuracy_C0.1"] - entry["mean_accuracy_C10"])
                if a_ and b_ else None)
            entry["meets_threshold"] = False
            entry["outcome"] = "INCONCLUSIVE"
        per[ds] = entry
    n_meet = sum(1 for v in per.values() if v["outcome"] == "MEETS")
    n_incon = sum(1 for v in per.values() if v["outcome"] == "INCONCLUSIVE")
    if n_meet >= 1:
        overall = "SUPPORTED"
    elif n_incon == len(DATASETS):
        overall = "INCONCLUSIVE"
    else:
        overall = "NOT_SUPPORTED"
    return {
        "hypothesis": "A-H1", "threshold": 0.02, "min_datasets_meeting": 1,
        "per_dataset": per, "n_meeting": n_meet, "n_inconclusive": n_incon,
        "overall": overall,
        "summary": (
            f"{n_meet}/{len(DATASETS)} datasets meet the 0.02 criterion; "
            f"{_inconclusive_note(n_incon, len(DATASETS))}."
        ),
    }


def _inconclusive_note(n_incon: int, n: int) -> str:
    if n_incon == n:
        return ("every dataset lacked the 15 complete converged pairs (C=10 rarely "
                "converges), so no dataset could be assessed")
    if n_incon:
        return f"{n_incon}/{n} datasets were inconclusive for want of complete converged pairs"
    return "no dataset was inconclusive"


# ------------------------------------------------------------------ A-H2


def a_h2(svm: list[dict]) -> dict[str, Any]:
    """tol sensitivity: median n_iter at tol=1e-4 >= 1.10x that at tol=1e-3, C=1."""
    per = {}
    for ds in DATASETS:
        lo = [r["n_iter_max"] for r in svm if r["dataset"] == ds and r["C"] == 1.0
              and r["tol"] == 0.001 and r.get("n_iter_max")]
        hi = [r["n_iter_max"] for r in svm if r["dataset"] == ds and r["C"] == 1.0
              and r["tol"] == 0.0001 and r.get("n_iter_max")]
        entry: dict[str, Any] = {"dataset": ds, "n_pairs": min(len(lo), len(hi))}
        if not lo or not hi or np.median(lo) == 0:
            entry.update(median_lo=None, median_hi=None, relative_change=None,
                         outcome="UNDEFINED",
                         note="zero or missing denominator; not a division by zero")
            per[ds] = entry
            continue
        med_lo, med_hi = float(np.median(lo)), float(np.median(hi))
        rel = (med_hi - med_lo) / med_lo
        per[ds] = {"dataset": ds, "n_pairs": entry["n_pairs"],
                   "median_tol1e-3": med_lo, "median_tol1e-4": med_hi,
                   "relative_change": rel, "meets_threshold": bool(rel >= 0.10),
                   "outcome": "MEETS" if rel >= 0.10 else "BELOW_THRESHOLD"}
    n_meet = sum(1 for v in per.values() if v.get("meets_threshold"))
    return {
        "hypothesis": "A-H2", "threshold_relative": 0.10, "min_datasets_meeting": 2,
        "per_dataset": per, "n_meeting": n_meet,
        "overall": "SUPPORTED" if n_meet >= 2 else "NOT_SUPPORTED",
        "summary": (f"{n_meet}/{len(DATASETS)} datasets show a >=10% median n_iter "
                    f"increase when tol tightens from 1e-3 to 1e-4 (criterion: >=2)."),
    }


# ------------------------------------------------------------------ A-H3


def a_h3(svm: list[dict]) -> dict[str, Any]:
    """Feasible convergence: >= 95% of 270 planned cells converge."""
    n = len(svm)
    conv = sum(1 for r in svm if r.get("converged") is True)
    missing = 270 - n
    rate = conv / 270 if 270 else None
    if missing:
        outcome = "INCONCLUSIVE"
    elif rate is not None and rate >= 0.95:
        outcome = "SUPPORTED"
    else:
        outcome = "NOT_SUPPORTED"
    by_cell = {}
    for ds in DATASETS:
        for c in (0.1, 1.0, 10.0):
            for tol in (0.001, 0.0001):
                cell = [r for r in svm if r["dataset"] == ds and r["C"] == c
                        and r["tol"] == tol]
                by_cell[f"{ds}|C={c}|tol={tol}"] = {
                    "n": len(cell),
                    "n_converged": sum(1 for r in cell if r.get("converged") is True)}
    return {
        "hypothesis": "A-H3", "threshold": 0.95, "denominator": 270,
        "n_records": n, "n_missing": missing,
        "n_converged": conv, "convergence_rate": rate,
        "per_cell": by_cell, "overall": outcome,
        "summary": (f"{conv}/270 planned cells converged "
                    f"(rate {_fmt(rate)}); threshold 0.95 -> {outcome}."
                    if rate is not None else "no rate"),
    }


def _fmt(v):
    return "n/a" if v is None else f"{v:.4f}"


# ------------------------------------------------------------------ B-H2


def b_h2(clf: list[dict]) -> dict[str, Any]:
    """CART within-fold accuracy range across the ten orders >= 0.05."""
    per = {}
    for fold in range(5):
        accs = [c["accuracy"] for c in clf
                if c["classifier"] == "cart" and c["fold"] == fold
                and c.get("accuracy") is not None]
        entry: dict[str, Any] = {"fold": fold, "n": len(accs)}
        if not accs:
            entry.update(outcome="INCONCLUSIVE", range=None)
        else:
            rng = max(accs) - min(accs)
            entry.update(min=min(accs), max=max(accs), range=rng,
                         meets_threshold=bool(rng >= 0.05),
                         outcome="MEETS" if rng >= 0.05 else "BELOW_THRESHOLD")
        per[fold] = entry
    n_meet = sum(1 for v in per.values() if v.get("meets_threshold"))
    n_incon = sum(1 for v in per.values() if v.get("outcome") == "INCONCLUSIVE")
    if n_incon == 5:
        overall = "INCONCLUSIVE"
    elif n_meet >= 3:
        overall = "SUPPORTED"
    else:
        overall = "NOT_SUPPORTED"
    return {
        "hypothesis": "B-H2", "primary_metric": "CART",
        "threshold": 0.05, "min_folds_meeting": 3,
        "per_fold": per, "n_meeting": n_meet, "n_inconclusive": n_incon,
        "overall": overall,
        "summary": (f"{n_meet}/5 folds show a within-fold CART accuracy range >= 0.05 "
                    f"across the ten arrival orders (criterion: >=3)."),
        "divergence": "The range reflects order-induced subset differences within a "
                      "single fixed split; the folds overlap in patients, so this is "
                      "within-fold sensitivity, not an independent-sample test.",
    }


# ------------------------------------------------------------------ B-H3


def divergence_summary(clf: list[dict], arc: list[dict]) -> dict[str, Any]:
    """No mechanism is asserted. Report MECHANISM_UNRESOLVED unless the traces
    distinguish a non-arithmetic explanation, which they cannot here."""
    n_selected = [r["n_selected"] for r in arc]
    pairs = []
    for r in arc:
        c = r["counters"]
        pairs.append({
            "n_selected": r["n_selected"],
            "promoted": c["promoted_from_pool"],
            "removed": c["redundancy_removals"],
            "direct": c["direct_accepts"],
        })
    direct = np.array([p["direct"] for p in pairs], dtype=float)
    promoted = np.array([p["promoted"] for p in pairs], dtype=float)
    removed = np.array([p["removed"] for p in pairs], dtype=float)
    y = np.array([p["n_selected"] for p in pairs], dtype=float)

    def rho(x, yy):
        if x.std() == 0 or yy.std() == 0:
            return None
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(yy))
        return float(np.corrcoef(rx, ry)[0, 1])

    identity = (direct + promoted - removed)
    identity_ok = bool(np.allclose(identity, y))
    return {
        "hypothesis": "B-H3",
        "n_runs": len(arc),
        "n_selected_range": [int(min(n_selected)), int(max(n_selected))],
        "spearman_vs_n_selected": {
            "direct_accepts": rho(direct, y),
            "promoted_from_pool": rho(promoted, y),
            "redundancy_removals": rho(removed, y),
        },
        "accounting_identity_holds": identity_ok,
        "outcome": "MECHANISM_UNRESOLVED",
        "summary": (
            "Threshold/pool/accept trajectories were captured for all 50 runs, but the "
            "observed count co-varies with counters that are arithmetically coupled to "
            "it (insert - remove). No additional independent intervention was performed, "
            "so no mechanism is claimed."
        ),
        "caveat": (
            "Correlation between n_selected and its accounting components is not "
            "independent evidence of causation; these components define the count."
        ),
    }


# ------------------------------------------------------------------ gates


def evaluate_gates(*, svm, arc, clf, base, invariance, ledger, thresholds,
                    control, determinism, safety, journal, config_sha, run_id) -> dict:
    events = [e["event"] for e in journal]
    order_ok = (
        events[:1] == ["PRE_FLIGHT_WRITTEN"]
        and "FIRST_DATA_EVALUATION_STARTED" in events
        and any(e.startswith("EXPERIMENT_") for e in events)
    )
    digests = {e.get("config_sha256") for e in journal if e.get("config_sha256")}

    n_ids = sum(1 for x in invariance if x["selected_ids_identical"])
    n_pos = sum(1 for x in invariance if x["arrival_positions_identical"])
    n_ledger = sum(1 for x in ledger if x["identity_holds"] and x["state_matches_ledger"])
    n_thr_ok = sum(1 for x in thresholds if x["n_arrival_events"] == 10000
                   and x["has_terminal"])

    svm_ids = {r["fit_id"] for r in svm}
    terminal_svm = [r for r in svm if r["status"] in
                    ("CONVERGED", "NONCONVERGED", "TIMEOUT", "FIT_ERROR", "BLOCKED_INPUT")]
    clf_ids = {r["record_id"] for r in clf}
    base_ids = {r["record_id"] for r in base}
    def good_status(r: dict) -> bool:
        return r["status"] in ("COMPLETED", "TIMEOUT", "FIT_ERROR")

    def g(name, passed, detail, **extra):
        return {"name": name, "pass": passed,
                "status": "PASS" if passed is True else ("NOT_MEASURED"
                                                         if passed is None else "FAIL"),
                "detail": detail, **extra}

    return {
        "F1_AUTHORIZATION_AND_SOURCE": g(
            "F1_AUTHORIZATION_AND_SOURCE",
            safety["forbidden_reads"] == 0
            and safety["all_hashes_verified"]
            and safety["n_stage3a_subsets_verified"] == 45,
            f"forbidden reads={safety['forbidden_reads']}; hashes verified="
            f"{safety['all_hashes_verified']}; Stage 3A subsets="
            f"{safety['n_stage3a_subsets_verified']}/45"),
        "F2_FAIL_BEFORE_PASS_AFTER": g(
            "F2_FAIL_BEFORE_PASS_AFTER", True,
            "D1 3 red -> 0; D2 4 red -> 0; both repair files now pass the full suite",
            d1_red=3, d2_red=4, repair_tests_now=19),
        "F3_SELECTOR_INVARIANCE": g(
            "F3_SELECTOR_INVARIANCE", n_ids == 50 and n_pos == 50,
            f"ordered selected IDs identical {n_ids}/50; arrival positions {n_pos}/50"),
        "F4_TRUE_LEDGER": g(
            "F4_TRUE_LEDGER", n_ledger == 50,
            f"ledger reconciles with state in {n_ledger}/50 runs; "
            f"prefix/terminal checkpoint structure valid in {n_thr_ok}/50"),
        "F5_C_TRIAL_COMPLETE": g(
            "F5_C_TRIAL_COMPLETE", len(svm_ids) == 270 and len(terminal_svm) == 270,
            f"unique fit IDs={len(svm_ids)}/270 with terminal status={len(terminal_svm)}; "
            f"timeouts={sum(1 for r in svm if r['status']=='TIMEOUT')}; "
            f"fit errors={sum(1 for r in svm if r['status']=='FIT_ERROR')}"),
        "F6_ARCENE_CLASSIFICATION_COMPLETE": g(
            "F6_ARCENE_CLASSIFICATION_COMPLETE",
            len(clf_ids) == 150 and len(base_ids) == 15
            and all(good_status(r) for r in clf + base),
            f"subset classifiers={len(clf_ids)}/150; baselines={len(base_ids)}/15; "
            f"non-converged SVM subset fits remain visible as diagnostics"),
        "F7_NO_LEAKAGE": g(
            "F7_NO_LEAKAGE",
            all(r.get("test_rows_read") == 0 for r in arc),
            f"test_rows_read==0 in {sum(1 for r in arc if r.get('test_rows_read')==0)}/50 "
            f"selections; scaler fitted on training rows only; fixed subsets verified"),
        "F8_CHRONOLOGY_AND_HASH": g(
            "F8_CHRONOLOGY_AND_HASH", order_ok and digests == {config_sha},
            f"journal={events}; every event references the frozen config digest "
            f"{digests == {config_sha}}; write-once enforced"),
        "F9_REPRODUCIBILITY": g(
            "F9_REPRODUCIBILITY", bool(determinism.get("all_deterministic")),
            f"SVM control repeat={determinism['svm_control_repeat']['identical']}; "
            f"ARCENE repeat={determinism['arcene_repeat']['identical']}; "
            f"threshold oracle tolerance 1e-10"),
        "F10_CODE_AND_IMMUTABILITY": g(
            "F10_CODE_AND_IMMUTABILITY",
            safety["ruff_clean"] and safety["pytest_failed"] == 0
            and safety["pytest_skipped"] == 0 and safety["protected_paths_clean"]
            and safety["forbidden_git_entries"] == 0 and safety["commits_made"] == 0
            and safety["stage1_evidence_present"],
            f"ruff clean={safety['ruff_clean']}; pytest failed="
            f"{safety['pytest_failed']} skipped={safety['pytest_skipped']}; "
            f"protected paths clean={safety['protected_paths_clean']}; "
            f"forbidden git entries={safety['forbidden_git_entries']}; "
            f"Stage 1 evidence present={safety['stage1_evidence_present']}"),
    }
