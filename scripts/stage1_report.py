"""Stage 1d — produce every committed report.

Every headline number in every artefact produced by this script is computed by
reading the saved manifests and gate results from disk. Nothing is typed from
memory, and nothing is overwritten destructively: a rerun writes a
timestamped copy alongside the current report rather than erasing the ledger.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ossfs_dd_data.fetch import configure_logging
from ossfs_dd_data.gates import evaluate_gates
from ossfs_dd_data.registry import load_registry, load_stage1_config
from ossfs_dd_data.workspace import (
    data_root,
    manifest_path,
    rel_to_research,
    results_dir,
    utc_now,
    write_json,
)

LOG = logging.getLogger("ossfs_dd_data.stage1.report")

CSV_HEADER = [
    "id", "dataset", "paper_n", "paper_p", "paper_c",
    "actual_n", "actual_p", "actual_c",
    "shape_match", "class_match",
    "source_tier", "provenance_tier", "provenance_confidence",
    "source_url", "raw_archive", "sha256", "raw_bytes",
    "source_split_composition", "class_counts",
    "nonfinite_total", "constant_columns", "duplicate_column_groups",
    "duplicate_row_groups", "zero_only_rows",
    "split_plan_count", "feature_order_plan_count",
    "status", "mismatch_reason",
]


def load_manifests(registry) -> OrderedDict[str, dict]:
    out: OrderedDict[str, dict] = OrderedDict()
    for spec in registry.datasets:
        p = manifest_path(spec.key)
        if p.exists():
            out[spec.key] = json.loads(p.read_text(encoding="utf-8"))
    return out


def row_for(spec, m: dict | None) -> dict:
    if m is None:
        return {
            "id": spec.id, "dataset": spec.key,
            "paper_n": spec.n_samples, "paper_p": spec.n_features, "paper_c": spec.n_classes,
            "actual_n": "not measured", "actual_p": "not measured", "actual_c": "not measured",
            "shape_match": "unknown", "class_match": "unknown",
            "source_tier": spec.provenance_tier, "provenance_tier": spec.provenance_tier,
            "provenance_confidence": spec.provenance_confidence,
            "source_url": spec.sources[0].url, "raw_archive": "", "sha256": "", "raw_bytes": "",
            "source_split_composition": "not measured", "class_counts": "",
            "nonfinite_total": "not measured", "constant_columns": "not measured",
            "duplicate_column_groups": "not measured", "duplicate_row_groups": "not measured",
            "zero_only_rows": "not measured",
            "split_plan_count": 0, "feature_order_plan_count": 0,
            "status": "NOT_ATTEMPTED", "mismatch_reason": "no manifest on disk",
        }

    obs = m.get("observed", {})
    audit = m.get("data_quality_audit", {})
    dl = m.get("source", {}).get("download", {})
    n_ok = obs.get("n_samples") == spec.n_samples
    p_ok = obs.get("n_features") == spec.n_features
    c_ok = obs.get("n_classes") == spec.n_classes
    return {
        "id": spec.id, "dataset": spec.key,
        "paper_n": spec.n_samples, "paper_p": spec.n_features, "paper_c": spec.n_classes,
        "actual_n": obs.get("n_samples", "not measured"),
        "actual_p": obs.get("n_features", "not measured"),
        "actual_c": obs.get("n_classes", "not measured"),
        "shape_match": "yes" if (n_ok and p_ok) else ("no" if obs.get("n_samples") else "unknown"),
        "class_match": "yes" if c_ok else ("no" if obs.get("n_classes") else "unknown"),
        "source_tier": m.get("source", {}).get("tier", ""),
        "provenance_tier": m.get("provenance_assessment", {}).get("tier", ""),
        "provenance_confidence": m.get("provenance_assessment", {}).get("confidence", ""),
        "source_url": m.get("source", {}).get("url", ""),
        "raw_archive": dl.get("raw_archive", "") or "",
        "sha256": dl.get("sha256") or "",
        "raw_bytes": dl.get("bytes") if dl.get("bytes") is not None else "",
        "source_split_composition": json.dumps(obs.get("source_split_counts", {})),
        "class_counts": json.dumps(obs.get("class_counts", {})),
        "nonfinite_total": (audit.get("nonfinite") or {}).get("total", "not measured"),
        "constant_columns": audit.get("constant_columns", "not measured"),
        "duplicate_column_groups": audit.get("duplicate_column_groups", "not measured"),
        "duplicate_row_groups": audit.get("duplicate_row_groups", "not measured"),
        "zero_only_rows": audit.get("zero_only_rows", "not measured"),
        "split_plan_count": obs.get("split_plan_count", 0),
        "feature_order_plan_count": obs.get("feature_order_plan_count", 0),
        "status": m.get("stage1_verdict", {}).get("status", "NOT_ATTEMPTED"),
        "mismatch_reason": m.get("stage1_verdict", {}).get("reason", ""),
    }


def write_status_csv(rows: list[dict], path: Path) -> Path:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def write_source_ledger(registry, manifests: dict, path: Path) -> Path:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "dataset", "chosen_tier", "url", "format", "raw_archive",
            "sha256", "bytes", "download_ok", "reused_existing",
            "provenance_tier", "provenance_confidence", "license",
            "rejected_alternatives", "notes",
        ])
        for spec in registry.datasets:
            src = spec.sources[0]
            m = manifests.get(spec.key)
            dl = (m or {}).get("source", {}).get("download", {}) or {}
            rejected = "; ".join(
                f"{a.get('what')} -> {a.get('why_rejected_at_stage1', '')}".strip()
                for a in spec.alternatives_rejected
            )
            w.writerow([
                spec.key, src.tier, src.url, src.fmt,
                dl.get("raw_archive", ""), dl.get("sha256", ""), dl.get("bytes", ""),
                dl.get("ok", "not measured"), dl.get("reused_existing", "not measured"),
                spec.provenance_tier, spec.provenance_confidence, spec.license,
                rejected, (src.notes or spec.provenance_note).replace("\n", " ")[:400],
            ])
    return path


def dir_stats(d: Path, pattern: str = "*") -> tuple[int, int]:
    files = [p for p in d.rglob(pattern) if p.is_file()] if d.exists() else []
    return len(files), sum(p.stat().st_size for p in files)


def build_markdown(registry, manifests: dict, rows: list[dict], gates: dict, cfg: dict) -> str:
    L: list[str] = []
    A = L.append

    A("# Stage 1 Report — Dataset Acquisition, Provenance and Validation")
    A("")
    A("**Project:** `ossfs-dd-reproduction`  ")
    A(f"**Generated (UTC):** {utc_now()}  ")
    A("**Stage:** 1 of 3 — **DATA ONLY**  ")
    A("**Paper:** Zhou, Zhao, Yan & Wu, *Online Scalable Streaming Feature Selection via "
      "Dynamic Decision*, ACM TKDD 16(5) art. 87 (2022), DOI 10.1145/3502737")
    A("")
    A("> No feature selection, no classifier, no accuracy measurement was performed. "
      "Stage 1 prepares and audits data only.")
    A("")

    # ---------------- verdict ----------------
    A("## Final verdict")
    A("")
    A(f"```\nSTAGE_1_VERDICT: {gates['verdict']}\n```")
    A("")
    A(f"Failed hard gates: {', '.join(gates['failed_gates']) if gates['failed_gates'] else 'none'}")
    A("")

    # ---------------- gate table ----------------
    A("## Preregistered gates")
    A("")
    A("Thresholds come from `configs/stage1.yaml` and were fixed before any download. "
      "None was lowered to force a pass.")
    A("")
    A("| Gate | Threshold | Observed | Result | Detail |")
    A("|---|---|---|---|---|")
    for g in gates["gates"]:
        A(f"| `{g['name']}` | {g['threshold']} | {g['observed']} | "
          f"{'**PASS**' if g['pass'] else '**FAIL**'} | {g['detail']} |")
    A("")

    # ---------------- dataset table ----------------
    A("## Dataset status: paper Table 1 vs observed")
    A("")
    A("`features` counts predictor columns only; the class column is never counted as a feature.")
    A("")
    A("| # | Dataset | paper n×p×c | observed n×p×c | source tier | SHA-256 (first 16) | source splits | status |")
    A("|---:|---|---|---|---|---|---|---|")
    for spec, r in zip(registry.datasets, rows, strict=False):
        sha = (r["sha256"] or "")[:16] or "—"
        obs = f"{r['actual_n']}×{r['actual_p']}×{r['actual_c']}"
        paper = f"{spec.n_samples}×{spec.n_features}×{spec.n_classes}"
        A(f"| {spec.id} | {spec.key} | {paper} | {obs} | {r['source_tier']} | `{sha}` | "
          f"`{r['source_split_composition']}` | **{r['status']}** |")
    A("")

    # ---------------- per-dataset detail ----------------
    A("## Per-dataset detail, provenance and data quality")
    A("")
    for spec, r in zip(registry.datasets, rows, strict=False):
        m = manifests.get(spec.key)
        A(f"### {spec.id}. {spec.key} — `{r['status']}`")
        A("")
        A(f"- **Expected (Table 1):** {spec.n_samples} samples × {spec.n_features} features × "
          f"{spec.n_classes} classes; paper feature type `{spec.paper_feature_type}`")
        A(f"- **Observed:** n={r['actual_n']}, p={r['actual_p']}, c={r['actual_c']}")
        A(f"- **Upstream origin:** <{r['source_url']}> (tier `{r['source_tier']}`)")
        A(f"- **Raw archive:** `{r['raw_archive'] or '—'}` — SHA-256 `{r['sha256'] or '—'}`, "
          f"{r['raw_bytes'] or '—'} bytes")
        A(f"- **Provenance:** tier `{spec.provenance_tier}`, confidence `{spec.provenance_confidence}`")
        A(f"- **Citation:** {spec.citation}")
        A(f"- **Licence:** {spec.license}")
        if r["class_counts"]:
            A(f"- **Class counts:** `{r['class_counts']}`")
        if r["source_split_composition"] not in ("not measured", "{}"):
            A(f"- **Source split composition:** `{r['source_split_composition']}`")
        if m:
            dq = m.get("data_quality_audit", {})
            nf = (dq.get("nonfinite") or {})
            if nf:
                A(f"- **Non-finite values:** total={nf.get('total')} "
                  f"(NaN={nf.get('nan')}, +Inf={nf.get('posinf')}, -Inf={nf.get('neginf')})")
            A(f"- **Constant columns:** {dq.get('constant_columns', 'not measured')}; "
              f"**duplicate column groups:** {dq.get('duplicate_column_groups', 'not measured')}; "
              f"**duplicate row groups:** {dq.get('duplicate_row_groups', 'not measured')}; "
              f"**zero-only rows:** {dq.get('zero_only_rows', 'not measured')}")
            ir = dq.get("value_range", {})
            sp = dq.get("sparsity", {})
            A(f"- **Value range:** [{ir.get('min')}, {ir.get('max')}] "
              f"(all values integral: {dq.get('value_integrality', {}).get('all_values_integral')}); "
              f"stored as `{m.get('observed', {}).get('storage')}` "
              f"with dtype `{m.get('observed', {}).get('storage_dtype')}`, "
              f"density {sp.get('density')}, zero columns {sp.get('zero_columns')}")
            extra = m.get("observed", {}).get("parsing_extra") or {}
            checks = extra.get("source_checksum_verification") or {}
            if checks:
                A("- **Independent source checksum verification** (NIPS `.param` published `Check_sum`):")
                for split, c in checks.items():
                    A(f"  - `{split}`: published `{c['source_check_sum']}`, observed "
                      f"`{c['observed_value_sum']}` — **{'MATCH' if c['match'] else 'MISMATCH'}**; "
                      f"class balance published {c['source_pos_neg']} vs observed {c['observed_pos_neg']}")
            A(f"- **Preexisting preprocessing in the source:** "
              f"{m.get('observed', {}).get('preexisting_preprocessing_in_source')}")
            A(f"- **Transformations applied by this pipeline:** "
              f"{m.get('observed', {}).get('transformations_applied_by_this_pipeline') or 'none'}")
        A(f"- **Status reason:** {r['mismatch_reason'] or '—'}")
        if spec.alternatives_rejected:
            A("- **Alternatives considered and rejected:**")
            for a in spec.alternatives_rejected:
                A(f"  - *{a.get('what')}* — {a.get('why_rejected_at_stage1')}")
        A("")

    # ---------------- preparation summary ----------------
    raw_n, raw_b = dir_stats(data_root() / "raw")
    proc_n, proc_b = dir_stats(data_root() / "processed")
    split_n, split_b = dir_stats(data_root() / "splits")
    order_n, order_b = dir_stats(data_root() / "stream_orders")
    A("## Preparation summary")
    A("")
    A("| Artifact tree | Files | Bytes |")
    A("|---|---:|---:|")
    A(f"| `dataset/ossfs-dd-reproduction/raw/` | {raw_n} | {raw_b} |")
    A(f"| `dataset/ossfs-dd-reproduction/processed/` | {proc_n} | {proc_b} |")
    A(f"| `dataset/ossfs-dd-reproduction/splits/` | {split_n} | {split_b} |")
    A(f"| `dataset/ossfs-dd-reproduction/stream_orders/` | {order_n} | {order_b} |")
    A("")
    total_folds = sum(int(r["split_plan_count"]) * int(cfg["evaluation_plan"]["n_splits"]) for r in rows)
    total_orders = sum(int(r["feature_order_plan_count"]) for r in rows)
    A(f"- CV test folds defined: **{total_folds}** "
      f"(target {cfg['gates']['cv_test_folds_defined']['required']})")
    A(f"- Feature-order permutations defined: **{total_orders}** "
      f"(target {cfg['gates']['feature_order_permutations_defined']['required']})")
    A(f"- Seeds: `{cfg['evaluation_plan']['seeds']}`, "
      f"`n_splits={cfg['evaluation_plan']['n_splits']}`")
    A("")

    # ---------------- unresolved ----------------
    A("## Unresolved items and next actions")
    A("")
    unresolved = [r for r in rows if r["status"] != "EXACT_VERIFIED"]
    if not unresolved:
        A("No unresolved dataset items.")
    else:
        A("| Dataset | Status | Reason | Minimum corrective action |")
        A("|---|---|---|---|")
        for r in unresolved:
            A(f"| {r['dataset']} | {r['status']} | {r['mismatch_reason']} | "
              f"{_action_for(r)} |")
    A("")
    A("## Reproduction assumptions (explicit)")
    A("")
    A("- The paper describes 5-fold CV with 10 repetitions of randomized feature arrival "
      "orders, but **does not disclose its splits or seeds**. The plans in "
      "`dataset/ossfs-dd-reproduction/splits/` and `stream_orders/` use fixed preregistered "
      f"seeds `{cfg['evaluation_plan']['seeds']}`. They are a reproduction assumption, "
      "not a recovery of the original experiment.")
    A("- Original NIPS-2003 train/valid membership is preserved per row, but is not used "
      "as an evaluation holdout: the paper's totals are combined labeled counts.")
    A("- No preprocessing of any kind is applied in Stage 1. Any scaling, imputation or "
      "discretisation must be fitted on training rows only, separately within each fold, "
      "in a later stage.")
    A("")
    A("## Next stage")
    A("")
    if gates["verdict"] == "STAGE1_READY":
        A("`APPROVE STAGE 2` — all hard gates pass. Stage 2 still requires the researcher's "
          "review before OSSFS-DD is implemented.")
    else:
        A("`DO NOT START STAGE 2` — at least one hard gate fails. Gates must not be "
          "weakened to permit Stage 2. See the unresolved-items table above.")
    A("")
    return "\n".join(L)


def _without_timestamp(md: str) -> str:
    """Drop the generation-timestamp line so a no-op rerun is not mistaken for a
    changed scientific result."""
    return "\n".join(
        line for line in md.splitlines()
        if not line.startswith("**Generated (UTC):**")
    )


def _action_for(r: dict) -> str:
    st = r["status"]
    if st == "SHAPE_MATCH_PROVENANCE_UNVERIFIED":
        return ("Confirm with the authors, or locate an original Kent Ridge / Shipp et al. "
                "release with a published checksum, to substantiate the exact variant. "
                "No code change can close this gap.")
    if st == "DOWNLOAD_FAILED":
        return "Re-run `stage1_fetch.py` when the source is reachable; check the download log for the exact HTTP response."
    if st == "PARSE_FAILED":
        return "Inspect the preserved raw bytes and the parse error in `stage1_prepare.log`; the raw file is intact and may be re-parsed without re-downloading."
    if st == "DATA_QUALITY_BLOCKED":
        return "Escalate to the researcher for an explicit, train-fold-only preprocessing decision. Stage 1 must not impute."
    return "Investigate the recorded reason."


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 1: write all committed reports")
    ap.parse_args(argv)

    logfile = configure_logging("stage1_report.log")
    LOG.info("stage1_report starting; log also at %s", rel_to_research(logfile))

    cfg = load_stage1_config()
    registry = load_registry()
    manifests = load_manifests(registry)
    LOG.info("loaded %d/%d manifests from disk", len(manifests), len(registry.datasets))

    rs = results_dir()
    if not (rs / "dataset_status.json").exists():
        LOG.error("no dataset_status.json — run stage1_verify.py first")
        return 2

    # The report is the FINAL verdict authority: it re-evaluates every gate now
    # that the test suite (and therefore the git audit) has run.
    plan = cfg["evaluation_plan"]
    gates = evaluate_gates(cfg, len(registry.datasets), int(plan["n_splits"]), list(plan["seeds"]))
    write_json(rs / "readiness_summary.json", gates)
    LOG.info("final verdict: %s", gates["verdict"])

    rows = [row_for(spec, manifests.get(spec.key)) for spec in registry.datasets]

    csv_path = write_status_csv(rows, rs / "dataset_status.csv")
    write_json(rs / "dataset_status.json",
               {"generated_utc": utc_now(),
                "note": "Every count here is computed from the saved per-dataset manifests.",
                "datasets": rows})
    ledger_path = write_source_ledger(registry, manifests, rs / "source_registry_ledger.csv")
    md = build_markdown(registry, manifests, rows, gates, cfg)
    md_path = rs / "STAGE1_REPORT.md"
    previous = md_path.read_text(encoding="utf-8") if md_path.exists() else None
    md_path.write_text(md, encoding="utf-8")
    # A rerun must never overwrite a previous *scientific* result silently. The
    # comparison deliberately ignores the generation timestamp, so a no-op rerun
    # leaves the ledger alone while a genuine change to inputs or findings keeps
    # the earlier report verbatim under a stamped name.
    if previous is not None and _without_timestamp(previous) != _without_timestamp(md):
        stamped = rs / f"STAGE1_REPORT_{utc_now().replace(':', '').replace('-', '')}.md"
        i = 1
        while stamped.exists():
            stamped = rs / f"STAGE1_REPORT_{utc_now().replace(':', '').replace('-', '')}_{i}.md"
            i += 1
        stamped.write_text(previous, encoding="utf-8")
        LOG.info("scientific content changed; previous report retained at %s", rel_to_research(stamped))
    elif previous is not None:
        LOG.info("rerun produced identical findings; previous report not duplicated")

    LOG.info("wrote: %s", rel_to_research(csv_path))
    LOG.info("wrote: %s", rel_to_research(ledger_path))
    LOG.info("wrote: %s", rel_to_research(md_path))
    LOG.info("verdict: %s", gates["verdict"])
    for g in gates["gates"]:
        LOG.info("  gate %-48s %s", g["name"], "PASS" if g["pass"] else f"FAIL ({g['detail']})")
    if gates["verdict"] != "STAGE1_READY":
        LOG.error(
            "STAGE 1 verdict is %s. Reports are written; the exit code is non-zero "
            "because a hard gate failed. Gates must not be lowered to proceed.",
            gates["verdict"],
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
