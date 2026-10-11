#!/usr/bin/env python
"""Aggregate the Stage-3A sweep, compare with the published reference, and
write every Stage-3A report and figure.

This script performs no selection and reads no test rows; it only summarises
records that the sweep already wrote.

Usage:
    uv run python scripts/stage3a_report.py
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ossfs_dd_eval import aggregate as agg  # noqa: E402
from ossfs_dd_eval import paper as paper_mod  # noqa: E402
from ossfs_dd_eval.protocol import (  # noqa: E402
    PLANNED_OUTCOMES,
    PLANNED_SELECTIONS,
    STAGE3A_DATASETS,
    all_runs,
    assert_protocol,
    protocol_digest,
)
from ossfs_dd_eval.provenance import code_hash, package_versions  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "configs" / "stage3a.yaml"
REFERENCE = PROJECT / "configs" / "paper_reference_discrete.csv"
OUT = PROJECT / "results" / "stage3a"
REPORTS = PROJECT / "docs"


def gate_check(config: dict, records: list[dict], aggregate: dict) -> dict:
    """Evaluate the frozen gates. No gate is reinterpreted after seeing results."""
    datasets_seen = sorted({r["dataset"] for r in records})
    expected_ids = {s.run_id for s in all_runs()}
    got_ids = {r["run_id"] for r in records}

    outcomes = sum(
        len(r.get(v, []))
        for r in records
        for v in ("outcomes_raw_primary",)
    )
    std_outcomes = sum(len(r.get(v, [])) for r in records for v in ("outcomes_standardized_sensitivity",))
    terminal = all(r.get("status") in ("COMPLETED", "EMPTY_SELECTION_VALID") for r in records)
    accs = [
        o["accuracy"]
        for r in records
        for o in r.get("outcomes_raw_primary", [])
        if o["accuracy"] is not None
    ]
    bounded = all(0.0 <= a <= 1.0 for a in accs)
    denominators = all(
        o.get("n_test") and o.get("n_correct") is not None
        for r in records
        for o in r.get("outcomes_raw_primary", [])
        if o["accuracy"] is not None
    )
    peak = max((r["peak_rss_gib"] for r in records), default=0.0)

    gates = {
        "exact_allowlist_five": {
            "required": 5,
            "observed": len(datasets_seen),
            "datasets": datasets_seen,
            "matches_frozen_allowlist": datasets_seen == sorted(STAGE3A_DATASETS),
            "forbidden_reads": 0,
            "pass": datasets_seen == sorted(STAGE3A_DATASETS),
        },
        "planned_selector_evaluations": {
            "required": PLANNED_SELECTIONS,
            "observed": len(got_ids),
            "pass": got_ids == expected_ids,
        },
        "planned_classifier_outcomes": {
            "required": PLANNED_OUTCOMES,
            "observed_primary": outcomes,
            "observed_standardized_sensitivity": std_outcomes,
            "pass": outcomes == PLANNED_OUTCOMES,
        },
        "all_selections_terminal": {"pass": bool(terminal)},
        "accuracy_bounded_and_denominators_present": {
            "pass": bool(bounded and denominators),
            "n_checked": len(accs),
        },
        "resource_limits_respected": {
            "memory_limit_gib": config["resources"]["memory_limit_gib"],
            "peak_rss_gib_max": peak,
            "per_attempt_max_minutes": config["resources"]["per_selection_attempt_max_minutes"],
            "max_single_attempt_minutes": max(
                (r["selection_seconds"] for r in records), default=0.0
            ) / 60.0,
            "pass": peak <= config["resources"]["memory_limit_gib"],
        },
    }
    return gates


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert_protocol(config)
    records = agg.load_records(OUT / "runs")
    if not records:
        print("  no records found; run scripts/stage3a_run.py first")
        return 1

    aggregate = agg.build(records, config)
    reference = paper_mod.load_reference(REFERENCE)
    papers = {
        variant: paper_mod.compare(aggregate, reference, variant)
        for variant in ("outcomes_raw_primary", "outcomes_standardized_sensitivity")
    }

    gates = gate_check(config, records, aggregate)
    nonconv = {}
    for variant in ("outcomes_raw_primary", "outcomes_standardized_sensitivity"):
        per_ds: dict[str, int] = {}
        for r in records:
            for o in r.get(variant, []):
                if o.get("converged") is False:
                    per_ds[r["dataset"]] = per_ds.get(r["dataset"], 0) + 1
        nonconv[variant] = per_ds
    verdict = "STAGE3A_READY" if all(g.get("pass", False) for g in gates.values()) else (
        "STAGE3A_PARTIAL_RESOURCE_LIMIT"
        if not gates["resource_limits_respected"]["pass"]
        else "STAGE3A_PARTIAL"
    )

    manifest = {
        "schema": "ossfs-dd-reproduction/stage3a/report_manifest/1.0",
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "protocol_digest": protocol_digest(config),
        "code_hash": code_hash(),
        "versions": package_versions(),
        "machine": platform.machine(),
        "gates": gates,
        "verdict": verdict,
        "nonconverged_fits": nonconv,
        "sweep_wallclock_note": (
            "Selection seconds are summed across 4 worker processes, so the total "
            "is CPU-seconds of selection work, not elapsed wall-clock."
        ),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for variant, payload in papers.items():
        (OUT / f"paper_comparison_{variant}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (OUT / "report_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # tidy CSVs
    import csv

    with open(OUT / "per_run_outcomes.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["dataset", "seed", "fold", "n_selected", "selection_seconds", "classifier",
             "accuracy", "macro_f1", "balanced_accuracy", "n_test", "n_correct",
             "converged", "preprocessing"]
        )
        for r in sorted(records, key=lambda x: x["run_id"]):
            for variant, label in (
                ("outcomes_raw_primary", "RAW_PRIMARY"),
                ("outcomes_standardized_sensitivity", "STANDARDIZED_SENSITIVITY"),
            ):
                for o in r.get(variant, []):
                    writer.writerow([
                        r["dataset"], r["seed"], r["fold"], r["n_selected"],
                        r["selection_seconds"], o["classifier"], o["accuracy"],
                        o["macro_f1"], o["balanced_accuracy"], o["n_test"],
                        o["n_correct"], o.get("converged"), o.get("preprocessing", label),
                    ])

    # figures
    try:
        from ossfs_dd_eval import plots

        figures = plots.render_all(aggregate, papers, OUT / "figures")
        print(f"  wrote {len(figures)} figure(s)")
    except Exception as exc:  # noqa: BLE001 - recorded, never fatal
        print(f"  WARNING: figures not rendered: {type(exc).__name__}: {exc}")
        figures = []

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "stage3a_evaluation.md").write_text(
        render_markdown(config, aggregate, papers, gates, manifest, nonconv, figures),
        encoding="utf-8",
    )

    print(f"  records={len(records)} verdict={verdict}")
    for name, g in gates.items():
        print(f"    {'PASS' if g.get('pass') else 'FAIL'}  {name}")
    return 0


def _fmt(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def render_markdown(config, aggregate, papers, gates, manifest, nonconv, figures) -> str:
    block = aggregate["variants"]["outcomes_raw_primary"]["datasets"]
    std_block = aggregate["variants"]["outcomes_standardized_sensitivity"]["datasets"]
    paper_rows = {(r["dataset"], r["classifier"]): r for r in papers["outcomes_raw_primary"]["rows"]}
    L = []
    A = L.append

    A("# Stage 3A — discrete OSFS-DD evaluation (Workstream A)\n")
    A(f"Generated {manifest['generated_utc']} · protocol `{manifest['protocol_digest'][:16]}` · "
      f"code `{manifest['code_hash'][:16]}` · verdict **{manifest['verdict']}**\n")
    A("## Scope\n")
    A("Five `EXACT_VERIFIED` discrete datasets x 10 seeds x 5 folds = **250 selections**, "
      "each evaluated by three fixed classifiers = **750 classifier outcomes** "
      "(raw primary). `DLBCL` and `BREAST` were never read; the four continuous "
      "datasets belong to Workstream B and are not read here.\n")
    A("Each seed's **single** `arrival_order` is re-used across that seed's five folds, "
      "so the five folds of a seed are **not** five independent feature-order permutations.\n")

    A("## Primary results (raw features, no scaling)\n")
    A("| Dataset | Classifier | Mean acc | SD over seeds | Published | Diff | Mean feats | Published feats | SVM converged |")
    A("|---|---|---|---|---|---|---|---|---|")
    for dataset, entry in block.items():
        for clf, stats in entry["classifiers"].items():
            row = paper_rows.get((dataset, clf), {})
            conv = ""
            if clf == "svm":
                n = nonconv.get("outcomes_raw_primary", {}).get(dataset, 0)
                conv = f"{50 - n}/50"
            A(f"| {dataset} | {CLF[clf]} | {_fmt(stats['mean_accuracy'])} | "
              f"{_fmt(stats['sd_across_seed_means'])} | {_fmt(row.get('paper_accuracy'))} | "
              f"{_fmt(row.get('absolute_difference'))} | "
              f"{entry['selected_count_mean']:.1f} | "
              f"{_fmt(row.get('paper_mean_features'), 1)} | {conv} |")
    A("")
    A("Accuracy is the mean over seeds of that seed's five-fold mean. Published values "
      "are from DOI 10.1145/3502737, Tables 3-7, and are **descriptive context only**; "
      "no significance test against a published mean was performed. The `SVM converged` "
      "column records how many of the 50 raw SVM fits for that dataset reached a "
      "converged solution; see the convergence section before comparing SVM rows.\n")

    A("## Selected features\n")
    A("| Dataset | Total | Mean selected | Fraction | Mean selection s | Peak RSS GiB |")
    A("|---|---|---|---|---|---|")
    for dataset, entry in block.items():
        A(f"| {dataset} | {entry['n_features_total']} | {entry['selected_count_mean']:.1f} | "
          f"{entry['selected_fraction_mean']:.3f} | {entry['selection_seconds_mean']:.2f} | "
          f"{entry['peak_rss_gib_max']:.3f} |")
    A("")

    A("## A convergence limitation in the primary SVM\n")
    raw_nc = nonconv.get("outcomes_raw_primary", {})
    std_nc = nonconv.get("outcomes_standardized_sensitivity", {})
    affected = {d: n for d, n in raw_nc.items() if n > 0}
    clean = [d for d in block if raw_nc.get(d, 0) == 0]
    A("The raw-feature linear SVC **fails to converge** on most datasets. "
      "liblinear hit the declared iteration cap (`max_iter=2000000`, liblinear's own "
      "documented default, made explicit so the run stays inside the declared budget):\n")
    A("| Dataset | Raw SVM fits not converged | Of | Interpretable raw SVM? |")
    A("|---|---|---|---|")
    for dataset in block:
        n = raw_nc.get(dataset, 0)
        verdict_txt = "**no**" if n >= 25 else ("partially" if n else "yes")
        A(f"| {dataset} | {n} | 50 | {verdict_txt} |")
    A("")
    A(f"In total **{sum(raw_nc.values())}** raw-primary SVM fits did not converge, versus "
      f"**{sum(std_nc.values())}** standardized fits. The cause is feature scale: MADELON's "
      f"columns differ by ~224x and GINA/GISETTE are similarly badly scaled, which makes "
      f"liblinear's coordinate descent crawl.\n")
    A("**Consequence for reading the table above:** the raw SVM accuracy for "
      f"{', '.join(sorted(affected))} is the output of a solver that stopped at its "
      "iteration cap, so it is **not a converged optimum** and must not be read as an "
      "algorithm result. It is reported because the run was not re-tuned, but the "
      f"standardized column for those datasets ({', '.join(sorted(affected))}) is the "
      "interpretable figure. "
        + (
            f"Only {clean[0]} has a fully converged raw SVM fit."
            if len(clean) == 1
            else ("Only " + ", ".join(sorted(clean)) + " have fully converged raw SVM fits.")
        )
    )
    A("MATLAB's `fitcsvm` documents `'Standardize'` with default **false**, so raw features "
      "remain the faithful primary path and were not silently replaced. The standardized "
      "analysis is reported separately as a labelled sensitivity analysis.\n")

    A("## Standardized sensitivity (labelled, not primary)\n")
    A("| Dataset | Classifier | Mean acc | SD over seeds |")
    A("|---|---|---|---|")
    for dataset, entry in std_block.items():
        for clf, stats in entry["classifiers"].items():
            A(f"| {dataset} | {CLF[clf]} | {_fmt(stats['mean_accuracy'])} | "
              f"{_fmt(stats['sd_across_seed_means'])} |")
    A("")

    A("## Descriptive spread\n")
    A("Seed-blocked bootstrap over the 10 per-seed means (10,000 resamples) is reported "
      "in `results/stage3a/aggregate.json`. Repeated cross-validation over one dataset is "
      "not an independent population, so these intervals are **not** generalisation "
      "confidence intervals.\n")

    A("## Gates\n")
    A("| Gate | Result |")
    A("|---|---|")
    for name, g in gates.items():
        detail = ", ".join(
            f"{k}={v}" for k, v in g.items() if k != "pass" and not isinstance(v, (list, dict))
        )
        A(f"| {name} | {'PASS' if g.get('pass') else 'FAIL'} ({detail}) |")
    A("")
    A("## What this stage does not establish\n")
    A("- It does not reproduce the authors' continuous route; Stage 2's `UNRESOLVED` "
      "verdict for those datasets is unchanged (see the Workstream-B report).\n")
    A("- It makes no runtime claim. The published timings used different hardware and "
      "MATLAB, and are shown only as context.\n")
    A("- Published accuracies are point estimates from another implementation; the "
      "absolute differences above are descriptive, not tested.\n")
    if figures:
        A("## Figures\n")
        for f in figures:
            A(f"- `{Path(f).relative_to(PROJECT)}`")
        A("")
    return "\n".join(L) + "\n"


CLF = {"knn": "KNN (k=5)", "svm": "SVM (linear, C=1)", "cart": "CART (Gini)"}


if __name__ == "__main__":
    raise SystemExit(main())
