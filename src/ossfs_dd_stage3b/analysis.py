"""Prespecified aggregate analyses and hypothesis outcomes.

Hypothesis outcomes are reported as SUPPORTED / REFUTED / INCONCLUSIVE and are
explicitly **not** operational pass gates. Denominators are never silently
reduced: a cell only gets a convergence rate when all 15 of its fits reached a
terminal observed status.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .protocol import (
    A_FOLDS,
    B_H1_MIN_FOLDS,
    B_H1_RANGE_THRESHOLD,
    EXP_A_DATASETS,
    MAX_ITER_CAPS,
    SCALE_VARIANTS,
)

TERMINAL = ("CONVERGED", "NONCONVERGED", "TIMEOUT", "FIT_ERROR", "BLOCKED_INPUT")
A_H1_DATASETS = ("MADELON", "GINA", "GISETTE")


def convergence_rate(n_converged: int, n_terminal: int, n_planned: int = 15) -> float | None:
    """Rate only when every planned fit reached a terminal observed status."""
    if n_terminal != n_planned:
        return None
    return n_converged / n_planned


def summarize_cells(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """dataset x scale_variant x max_iter aggregates."""
    out = []
    for ds in EXP_A_DATASETS:
        for sv in SCALE_VARIANTS:
            for cap in MAX_ITER_CAPS:
                cell = [r for r in records
                        if r["dataset"] == ds and r["scale_variant"] == sv
                        and r["max_iter"] == cap]
                conv = [r for r in cell if r["status"] == "CONVERGED"]
                nonconv = [r for r in cell if r["status"] == "NONCONVERGED"]
                accs_conv = [r["accuracy"] for r in conv if r.get("accuracy") is not None]
                accs_non = [r["accuracy"] for r in nonconv if r.get("accuracy") is not None]
                iters = [r["n_iter_max"] for r in cell if r.get("n_iter_max") is not None]
                secs = [r["seconds"] for r in cell if r.get("seconds") is not None]
                out.append({
                    "dataset": ds, "scale_variant": sv, "max_iter": cap,
                    "n_planned": 15, "n_records": len(cell),
                    "n_converged": len(conv), "n_nonconverged": len(nonconv),
                    "n_timed_out": sum(1 for r in cell if r["status"] == "TIMEOUT"),
                    "n_fit_error": sum(1 for r in cell if r["status"] == "FIT_ERROR"),
                    "n_blocked_input": sum(1 for r in cell if r["status"] == "BLOCKED_INPUT"),
                    "n_terminal": sum(1 for r in cell if r["status"] in TERMINAL),
                    "convergence_rate": convergence_rate(len(conv), len(cell)),
                    "accuracy_mean_converged": float(np.mean(accs_conv)) if accs_conv else None,
                    "n_accuracy_converged": len(accs_conv),
                    "accuracy_mean_nonconverged_diagnostic": float(np.mean(accs_non)) if accs_non else None,
                    "n_accuracy_nonconverged": len(accs_non),
                    "n_iter_max_median": float(np.median(iters)) if iters else None,
                    "n_iter_max_min": int(min(iters)) if iters else None,
                    "n_iter_max_max": int(max(iters)) if iters else None,
                    "fit_seconds_mean": float(np.mean(secs)) if secs else None,
                    "fit_seconds_median": float(np.median(secs)) if secs else None,
                })
    return out


def a_h1_outcome(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Standardized convergence must exceed raw by >=0.50 AND be >=0.80."""
    cells = summarize_cells(records)
    top_cap = max(MAX_ITER_CAPS)
    per_dataset: dict[str, Any] = {}
    for ds in A_H1_DATASETS:
        c = {(x["scale_variant"], x["max_iter"]): x for x in cells
             if x["dataset"] == ds}
        raw = c.get(("raw", top_cap))
        std = c.get(("standardized", top_cap))
        detail: dict[str, Any] = {
            "dataset": ds, "max_iter": top_cap,
            "raw_converged": raw["n_converged"] if raw else None,
            "raw_terminal": raw["n_terminal"] if raw else None,
            "std_converged": std["n_converged"] if std else None,
            "std_terminal": std["n_terminal"] if std else None,
        }
        if not raw or not std or raw["convergence_rate"] is None \
                or std["convergence_rate"] is None:
            detail.update(
                outcome="INCONCLUSIVE_MISSING_FITS",
                delta=None,
                reason="one or both cells lacks a complete 15-fit terminal denominator",
            )
        else:
            delta = std["convergence_rate"] - raw["convergence_rate"]
            detail["delta"] = delta
            detail["criterion_delta_ge_0_50"] = delta >= 0.50
            detail["criterion_std_ge_0_80"] = std["convergence_rate"] >= 0.80
            ok = delta >= 0.50 and std["convergence_rate"] >= 0.80
            detail["outcome"] = "SUPPORTED" if ok else "REFUTED"
        per_dataset[ds] = detail
    outcomes = [v["outcome"] for v in per_dataset.values()]
    if all(o == "SUPPORTED" for o in outcomes):
        overall = "SUPPORTED"
    elif any(o == "INCONCLUSIVE_MISSING_FITS" for o in outcomes):
        overall = "INCONCLUSIVE"
    else:
        overall = "REFUTED"
    return {
        "hypothesis": "A-H1",
        "statement": (
            f"At max_iter={top_cap}, standardized convergence exceeds raw by >=0.50 "
            f"AND standardized >=0.80, per dataset."
        ),
        "scope_note": (
            "Association under a controlled intervention, not proof of physical cause."
        ),
        "per_dataset": per_dataset, "overall": overall,
    }


def a_h2_outcome(records: list[dict[str, Any]]) -> dict[str, Any]:
    """10 prespecified comparisons: convergence at 2e6 >= at 20k, per scale."""
    caps = sorted(MAX_ITER_CAPS)
    low, high = min(caps), max(caps)
    comparisons = []
    for ds in EXP_A_DATASETS:
        for sv in SCALE_VARIANTS:
            # pair every subset by its (split_seed, fold) key across the two caps
            by_subset: dict[tuple, dict] = {}
            for r in [x for x in records if x["dataset"] == ds and x["scale_variant"] == sv]:
                by_subset.setdefault((r["split_seed"], r["fold"]), {})[r["max_iter"]] = r
            pairs: list[dict[str, Any]] = []
            for (seed, fold), d in sorted(by_subset.items()):
                lo, hi = d.get(low), d.get(high)
                if not lo or not hi or lo["status"] not in TERMINAL or hi["status"] not in TERMINAL:
                    pairs.append({"split_seed": seed, "fold": fold, "status": "MISSING_OR_CENSORED",
                                  "low": lo["status"] if lo else None,
                                  "high": hi["status"] if hi else None})
                    continue
                pairs.append({
                    "split_seed": seed, "fold": fold,
                    "status": "PAIRED",
                    "low_status": lo["status"], "high_status": hi["status"],
                    "low_converged": lo["converged"], "high_converged": hi["converged"],
                    "not_lower": bool(hi["converged"]) >= bool(lo["converged"]),
                    "accuracy_low": lo.get("accuracy"), "accuracy_high": hi.get("accuracy"),
                })
            n_paired = sum(1 for p in pairs if p["status"] == "PAIRED")
            n_ok = sum(1 for p in pairs if p["status"] == "PAIRED" and p["not_lower"])
            violations = [p for p in pairs if p["status"] == "PAIRED" and not p["not_lower"]]
            comparisons.append({
                "dataset": ds, "scale_variant": sv, "low_cap": low, "high_cap": high,
                "n_pairs": n_paired, "n_pairs_not_lower": n_ok,
                "n_violations": len(violations), "violations": violations[:5],
                "outcome": ("INCONCLUSIVE" if n_paired < 15
                            else ("SUPPORTED" if n_ok == n_paired else "REFUTED")),
            })
    outcomes = [c["outcome"] for c in comparisons]
    overall = ("INCONCLUSIVE" if "INCONCLUSIVE" in outcomes
               else "SUPPORTED" if all(o == "SUPPORTED" for o in outcomes) else "REFUTED")
    return {
        "hypothesis": "A-H2",
        "statement": (
            f"With scaling held fixed, convergence at max_iter={high} is no lower "
            f"than at max_iter={low} (10 prespecified comparisons)."
        ),
        "n_prespecified_comparisons": len(comparisons),
        "comparisons": comparisons, "overall": overall,
    }


def jaccard(a: set, b: set) -> float:
    """Fixed convention: empty/empty = 1.0; empty/nonempty = 0.0."""
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def b_h1_outcome(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Within-fold range of n_selected across 10 orders, >=500 on >=3 folds."""
    per_fold = {}
    n_folds_meeting = 0
    incomplete = []
    for fold in A_FOLDS:
        cell = [r for r in records if r["fold"] == fold and r["dataset"] == "ARCENE"]
        if len(cell) < 10:
            incomplete.append(fold)
            per_fold[fold] = {"n_runs": len(cell), "outcome": "INCONCLUSIVE_MISSING_RUNS",
                              "range": None}
            continue
        counts = np.array([r["n_selected"] for r in cell], dtype=float)
        rng = int(counts.max() - counts.min())
        meets = rng >= B_H1_RANGE_THRESHOLD
        n_folds_meeting += int(meets)
        per_fold[fold] = {"n_runs": len(cell), "min": int(counts.min()),
                          "max": int(counts.max()), "range": rng,
                          "meets_threshold": bool(meets)}
    if incomplete:
        overall = "INCONCLUSIVE_MISSING_RUNS"
    elif n_folds_meeting >= B_H1_MIN_FOLDS:
        overall = "SUPPORTED"
    else:
        overall = "REFUTED"
    return {
        "hypothesis": "B-H1",
        "statement": (
            f"Within-fold range of n_selected across the 10 orders reaches "
            f">= {B_H1_RANGE_THRESHOLD} on at least {B_H1_MIN_FOLDS} of 5 folds."
        ),
        "criterion_note": "Preregistered EFFECT criterion, not an operational success gate.",
        "per_fold": per_fold, "n_folds_meeting_threshold": n_folds_meeting,
        "overall": overall,
    }


def b_h2_outcome(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Diagnostic co-variation between count and counters WITHIN each fixed fold."""
    per_fold = {}
    for fold in A_FOLDS:
        cell = [r for r in records if r["fold"] == fold and r["dataset"] == "ARCENE"]
        if len(cell) < 3:
            per_fold[fold] = {"n_runs": len(cell), "status": "NOT_MEASURED"}
            continue
        y = np.array([r["n_selected"] for r in cell], dtype=float)
        stats = {}
        for key in ("redundancy_removals", "promoted_from_pool", "direct_accepts",
                    "peak_candidates", "pool_flushes", "immediate_rejects"):
            x = np.array([r["counters"][key] for r in cell], dtype=float)
            if x.std() == 0 or y.std() == 0:
                stats[key] = None
                continue
            rho = float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])
            stats[key] = rho
        per_fold[fold] = {"n_runs": len(cell), "spearman_rank_rho": stats}
    resolved = any(v for f, d in per_fold.items()
                   for k, v in (d.get("spearman_rank_rho") or {}).items()
                   if k is not None and abs(v) >= 0.7)
    return {
        "hypothesis": "B-H2",
        "statement": (
            "Investigate whether order-driven count differences co-occur with "
            "redundancy_removals / promoted_from_pool / direct accepts / threshold "
            "trajectories. No mechanism is preregistered."
        ),
        "per_fold": per_fold,
        "status": "ASSOCIATION_OBSERVED_NOT_ISOLATED" if resolved else "MECHANISM_UNRESOLVED",
        "caution": (
            "Counters are arithmetically coupled (a selected feature is accepted, "
            "possibly promoted or removed), so even |rho|=1 does not establish a "
            "causal mechanism."
        ),
    }
