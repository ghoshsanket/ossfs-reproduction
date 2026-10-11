#!/usr/bin/env python
"""Write the Workstream-B (continuous reconstruction) report.

Reads the four fold-0 train-only diagnostics and summarises what the
reconstruction does and, more importantly, what it does not establish.

Usage:
    uv run python scripts/continuous_rnd_report.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ossfs_dd_continuous import ALLOWLIST, MODE_ID, SEMANTICS_STATUS  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "configs" / "continuous_rnd_v1.yaml"
RESULTS = PROJECT / "results" / "continuous_rnd"
REPORTS = PROJECT / "docs"


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    # Fail closed if the frozen config and the package disagree about scope.
    frozen_allow = tuple(config["authorization"]["allowlist"])
    frozen_forbidden = tuple(config["authorization"]["forbidden"])
    if frozen_allow != ALLOWLIST:
        raise SystemExit(f"frozen allowlist {frozen_allow} != package allowlist {ALLOWLIST}")
    if not set(frozen_forbidden) >= {"DLBCL", "BREAST"}:
        raise SystemExit(f"frozen forbidden set {frozen_forbidden} must include DLBCL and BREAST")
    records = []
    for key in ALLOWLIST:
        path = RESULTS / f"{key}_seed11_fold0.json"
        if path.exists():
            records.append(json.loads(path.read_text(encoding="utf-8")))
    if not records:
        print("  no Workstream-B diagnostics found; run scripts/continuous_rnd_run.py first")
        return 1

    completed = [r for r in records if r.get("status") == "DIAGNOSTIC_COMPLETED"]
    train_only = all(r.get("training_rows_only") and r.get("test_rows_read") == 0 for r in records)
    no_clf = all(r.get("classifier_evaluated") is False for r in records)

    verdict = (
        "CONTINUOUS_RND_DIAGNOSTIC_READY"
        if len(completed) == len(ALLOWLIST) and train_only and no_clf
        else "CONTINUOUS_RND_PARTIAL"
    )

    L = []
    A = L.append
    A("# Workstream B — reconstructed continuous Fisher-Z estimator\n")
    A(f"Generated {datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')} · "
      f"mode `{MODE_ID}` · semantics **{SEMANTICS_STATUS}** · verdict **{verdict}**\n")

    A("## Status: this is research, not reproduction\n")
    A("The authors' `OSFS_DD_z2.m` calls `Fisher_Ztest`, which **does not exist** in their "
      "pinned repository (commit `845a8e9d5625616b43e8cee16e8d22081dbe3f62`). Its output, "
      "units, sign handling and pairwise semantics are unknown. Everything below is our "
      "declared reconstruction from `configs/continuous_rnd_v1.yaml`.\n")
    A("**Stage 2's verdict is unchanged.** The `gaussian_fisher_z` route remains "
      "`UNRESOLVED` and fail-closed for these same four datasets. Workstream B does not "
      "unlock, replace or reinterpret it.\n")

    A("## What is source-verified vs reconstructed\n")
    A("The *decision structure* is taken from the pinned `z2.m` and is verifiable "
      "independently of the missing helper:\n")
    A("| Behaviour | Source | Reproduced |")
    A("|---|---|---|")
    A("| Warm-up **accepts** the first two features (unlike `mi2.m`, which skips them) | `z2.m:41-46` | yes |")
    A("| NaN relevance score is mapped to 0 | `z2.m:35-37` | yes |")
    A("| A pair is skipped when the helper reports `CI==1` or a NaN cmi | `z2.m:124`, `z2.m:152` | yes |")
    A("| Inner-loop break on pool promotion, outer loop continues | `z2.m:129-131` | yes |")
    A("")
    A("The *scores* are ours and are not the authors' `Fisher_Ztest`:\n")
    A("- **marginal** — one-vs-rest Pearson correlation, Bonferroni-corrected, `tanh`-bounded to `[0,1]`;\n")
    A("- **pair** — an in-sample multiple correlation of `b_c ~ 1 + x + z`. This is a "
      "**heuristic** (`heuristic_joint_v1`): a two-predictor Fisher conditional-independence "
      "theorem does not transfer to an in-sample R, so the formula is declared heuristic and "
      "labelled as such everywhere it is used.\n")

    A("## Diagnostics (seed 11, fold 0, training rows only)\n")
    A("| Dataset | Train rows | Features seen | Selected | Fraction | Pair skips | Seconds |")
    A("|---|---|---|---|---|---|---|")
    for r in sorted(records, key=lambda x: x["dataset"]):
        if r.get("status") != "DIAGNOSTIC_COMPLETED":
            A(f"| {r['dataset']} | — | — | — | — | — | FAILED |")
            continue
        A(f"| {r['dataset']} | {r['n_train_rows']} | {r['n_features_seen']} | "
          f"{r['n_selected']} | {r['selected_fraction']:.3f} | {r['pair_skips']} | "
          f"{r['seconds']:.1f} |")
    A("")
    A(f"Leakage contract: every record has `training_rows_only=true`, "
      f"`test_rows_read=0` and `classifier_evaluated=false` "
      f"(verified: {train_only and no_clf}). No accuracy was computed for any continuous "
      f"dataset, because no source-verified continuous selector exists to evaluate.\n")

    A("## Stated limitations\n")
    A("- One-hot class indicators are not Gaussian, so the Fisher-Z approximation is not exact.\n")
    A("- The estimator captures **linear** dependence only; XOR-style synergy is invisible to it.\n")
    A("- With K=2 the one-vs-rest contrasts are reflected duplicates. The declared K=2 "
      "Bonferroni correction is **retained** rather than silently changed.\n")
    A("- One seed and one fold per dataset: this is a smoke test of the estimator's "
      "behaviour, not a performance measurement.\n")
    A("- Selection fractions (~8-9%) reflect the reconstructed score, not the authors' "
      "algorithm. They must not be read as a reproduction of their continuous results.\n")

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "continuous_rnd_workstream.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    summary = {
        "schema": "ossfs-dd-reproduction/continuous_rnd/summary/1.0",
        "mode_id": MODE_ID,
        "semantics_status": SEMANTICS_STATUS,
        "verdict": verdict,
        "datasets": [r["dataset"] for r in completed],
        "training_rows_only_verified": train_only,
        "classifier_evaluated": not no_clf,
        "stage2_verdict_unchanged": "STAGE2_PARTIAL | DISCRETE_CORRECTNESS_PASS",
        "results": {r["dataset"]: {k: r.get(k) for k in
                 ("status", "n_train_rows", "n_features_seen", "n_selected",
                  "selected_fraction", "pair_skips", "seconds")}
                 for r in records},
    }
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"  verdict={verdict} datasets={len(completed)}/{len(ALLOWLIST)} "
          f"train_only={train_only} classifier_evaluated={not no_clf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
