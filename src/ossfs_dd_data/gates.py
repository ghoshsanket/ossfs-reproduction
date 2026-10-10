"""Stage 1 gate evaluation — the single authority for the stage verdict.

Gates are read from ``configs/stage1.yaml``. They are never lowered, widened or
reinterpreted to force a pass, and a quantity that was not measured is reported
as ``not measured`` — never as ``0``.

Gate-dependent evidence sources:
  * per-dataset manifests (written by stage1_verify)
  * plan file counts on disk (written by stage1_prepare)
  * ``results/stage1/test_results.json`` (written by the pytest session hook)

A gate whose evidence is missing is NOT counted as passing.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .registry import STATUS_EXACT_VERIFIED, STATUS_SHAPE_MATCH_PROVENANCE_UNVERIFIED
from .workspace import (
    data_root,
    results_dir,
    splits_dir,
    utc_now,
)

LOG = logging.getLogger("ossfs_dd_data.gates")


def _gate(name: str, passed: bool, threshold: Any, observed: Any, detail: str) -> dict[str, Any]:
    return {
        "name": name,
        "pass": bool(passed),
        "threshold": threshold,
        "observed": observed,
        "detail": detail,
    }


def _count_plan_files() -> tuple[int, int]:
    splits = sum(1 for d in data_root().glob("splits/*/*.json"))
    orders = sum(1 for d in data_root().glob("stream_orders/*/*.json"))
    return splits, orders


def load_test_results() -> dict[str, Any] | None:
    p = results_dir() / "test_results.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_manifests() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in sorted(data_root().glob("manifests/*.json")):
        if p.name.endswith(".raw_inventory.json"):
            continue
        try:
            out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # A corrupt manifest must be visible, not silently dropped: the gate
            # would otherwise be evaluated against a smaller dataset set.
            LOG.error("could not read manifest %s: %s", p.name, exc)
            continue
    return out


def evaluate_gates(cfg: dict[str, Any], n_datasets: int, n_splits: int, seeds: list[int]) -> dict[str, Any]:
    g = cfg["gates"]
    manifests = load_manifests()
    statuses = {k: m.get("stage1_verdict", {}).get("status", "NOT_ATTEMPTED")
                for k, m in manifests.items()}
    n_plan_splits, n_plan_orders = _count_plan_files()
    tests = load_test_results()

    attempted = sum(1 for s in statuses.values() if s != "NOT_ATTEMPTED")
    exact = sum(1 for s in statuses.values() if s == STATUS_EXACT_VERIFIED)
    shape_unproven = sum(
        1 for s in statuses.values() if s == STATUS_SHAPE_MATCH_PROVENANCE_UNVERIFIED
    )
    dimensionally_exact = exact + shape_unproven
    blocked = sorted(
        k for k, s in statuses.items()
        if s not in {STATUS_EXACT_VERIFIED, STATUS_SHAPE_MATCH_PROVENANCE_UNVERIFIED}
    )

    gates: list[dict[str, Any]] = []

    gates.append(_gate(
        "inventory_attempted",
        attempted >= g["inventory_attempted"]["required"],
        g["inventory_attempted"]["required"],
        attempted,
        f"{attempted}/{n_datasets} registry datasets attempted",
    ))

    gates.append(_gate(
        "exact_dimension_and_label_match",
        dimensionally_exact >= g["exact_dimension_and_label_match"]["required"],
        g["exact_dimension_and_label_match"]["required"],
        dimensionally_exact,
        f"{dimensionally_exact}/{n_datasets} match Table 1 (n, p, classes) with aligned, "
        f"reversible labels; {shape_unproven} of those have unproven lineage",
    ))

    gates.append(_gate(
        "exact_verified_provenance",
        exact >= g["exact_verified_provenance"]["required"],
        g["exact_verified_provenance"]["required"],
        exact,
        f"{exact}/{n_datasets} EXACT_VERIFIED; "
        f"{shape_unproven} shape-matched with lineage recorded as unproven"
        + (f"; blocked: {blocked}" if blocked else ""),
    ))

    integrity_bad = [
        k for k, m in manifests.items()
        if m.get("data_quality_audit", {}).get("status") not in {"not_run", None}
        and m.get("stage1_verdict", {}).get("status") == "DATA_QUALITY_BLOCKED"
    ]
    gates.append(_gate(
        "sample_feature_label_corruption",
        not integrity_bad,
        g["unresolved_corrupt_rows_or_columns"]["max_allowed"],
        len(integrity_bad),
        "round-trip, label-alignment and plan-structure checks passed for every dataset"
        if not integrity_bad else f"datasets blocked on integrity: {integrity_bad}",
    ))

    nonfinite_total = sum(
        int((m.get("data_quality_audit", {}).get("nonfinite") or {}).get("total", 0) or 0)
        for m in manifests.values()
    )
    nonfinite_in_ready = sum(
        int((m.get("data_quality_audit", {}).get("nonfinite") or {}).get("total", 0) or 0)
        for k, m in manifests.items()
        if m.get("stage1_verdict", {}).get("status") in
        {STATUS_EXACT_VERIFIED, STATUS_SHAPE_MATCH_PROVENANCE_UNVERIFIED}
    )
    gates.append(_gate(
        "unresolved_nonfinite_or_missing_in_ready_data",
        nonfinite_in_ready <= g["unresolved_nonfinite_or_missing_in_ready_data"]["max_allowed"],
        g["unresolved_nonfinite_or_missing_in_ready_data"]["max_allowed"],
        nonfinite_in_ready,
        f"{nonfinite_in_ready} non-finite/missing values across ready datasets "
        f"({nonfinite_total} across all audited datasets); Stage 1 never imputes, so any "
        "unresolved value would mark its dataset DATA_QUALITY_BLOCKED instead",
    ))

    test_rows_used = 0
    for m in manifests.values():
        extra = m.get("observed", {}).get("parsing_extra") or {}
        test_rows_used += len(extra.get("source_test_rows_parsed", []) or [])
    gates.append(_gate(
        "unlabeled_challenge_test_examples_used",
        test_rows_used <= g["unlabeled_challenge_test_examples_used"]["max_allowed"],
        g["unlabeled_challenge_test_examples_used"]["max_allowed"],
        test_rows_used,
        "parsers refuse a 'test' split outright; asserted by test_no_unlabeled_test_data",
    ))

    required_folds = n_datasets * len(seeds) * n_splits
    observed_folds = sum(
        json.loads((splits_dir(k) / f"seed_{s}.json").read_text(encoding="utf-8"))["n_splits"]
        for k in manifests
        for s in seeds
        if (splits_dir(k) / f"seed_{s}.json").exists()
    ) if manifests else 0
    gates.append(_gate(
        "cv_test_folds_defined",
        observed_folds >= g["cv_test_folds_defined"]["required"],
        g["cv_test_folds_defined"]["required"],
        observed_folds,
        f"{n_plan_splits} split-plan files covering {observed_folds} test folds "
        f"({n_datasets} datasets x {len(seeds)} seeds x {n_splits} folds = {required_folds})",
    ))

    gates.append(_gate(
        "feature_order_permutations_defined",
        n_plan_orders >= g["feature_order_permutations_defined"]["required"],
        g["feature_order_permutations_defined"]["required"],
        n_plan_orders,
        f"{n_plan_orders} feature-order permutations "
        f"({n_datasets} datasets x {len(seeds)} seeds = {n_datasets * len(seeds)})",
    ))

    # ---- gates whose evidence is the test suite --------------------------
    # Every one of these is FAIL-CLOSED. A value of ``null`` or a non-"ok"
    # audit status means the evidence was not produced, which is reported as
    # "not measured" and does NOT pass. v1 read a bare integer that defaulted
    # to 0 whenever git was unavailable, which converted missing evidence into
    # a green gate.
    if tests is None:
        # No evidence file at all: every test-dependent gate reports
        # "not measured" and does NOT pass.
        for name in (
            "stream_leakage_tests_passing",
            "data_or_result_binaries_in_git",
            "required_tests_passing",
        ):
            spec = g[name]
            # these gates express their bar differently (a fraction vs a count)
            threshold = spec.get("required_fraction", spec.get("max_allowed"))
            gates.append(_gate(
                name, False, threshold, "not measured",
                "results/stage1/test_results.json is absent; run `uv run pytest` before judging",
            ))
    else:
        git_status = tests.get("git_audit_status", "missing")
        git_violations = tests.get("git_data_violations")

        leaked = tests.get("stream_leakage_failures")
        leaked_ok = isinstance(leaked, list) and len(leaked) == 0
        gates.append(_gate(
            "stream_leakage_tests_passing",
            bool(leaked_ok and tests.get("failed", 1) == 0),
            g["stream_leakage_tests_passing"]["required_fraction"],
            f"{tests.get('passed', 0)} passed, {tests.get('failed', 0)} failed",
            f"stream-leakage test failures: {len(leaked) if isinstance(leaked, list) else 'not measured'}"
            f" of {len(tests.get('stream_leakage_tests') or [])} leakage tests",
        ))

        git_measured = git_status == "ok" and git_violations is not None
        gates.append(_gate(
            "data_or_result_binaries_in_git",
            bool(git_measured and git_violations == 0),
            g["data_or_result_binaries_in_git"]["max_allowed"],
            git_violations if git_violations is not None else "not measured",
            f"git audit status={git_status!r}: {tests.get('git_audit_detail', 'no detail')}"
            + ("" if git_measured else "  <-- evidence not measured; this gate cannot pass"),
        ))

        missing = tests.get("missing_required_tests") or []
        skipped_req = tests.get("skipped_required_tests") or []
        config_error = tests.get("config_error")
        gates.append(_gate(
            "required_tests_passing",
            bool(
                tests.get("failed", 1) == 0
                and not missing
                and not skipped_req
                and config_error is None
            ),
            g["required_tests_passing"]["required_fraction"],
            f"{tests.get('passed', 0)} passed, {tests.get('failed', 0)} failed, "
            f"{tests.get('skipped', 0)} skipped",
            (
                "every required named test ran and passed"
                if not missing and not skipped_req and not config_error
                else f"missing={missing}; skipped_required={skipped_req}; config_error={config_error}"
            ),
        ))

    failed = [x["name"] for x in gates if not x["pass"]]
    if not failed:
        verdict = "STAGE1_READY"
    elif attempted == 0 or not manifests:
        verdict = "STAGE1_BLOCKED"
    else:
        verdict = "STAGE1_PARTIAL"

    return {
        "schema": "ossfs-dd-reproduction/stage1/readiness/1.1",
        "generated_utc": utc_now(),
        "verdict": verdict,
        "gates": gates,
        "failed_gates": failed,
        "status_counts": _counts(statuses),
        "datasets_total": n_datasets,
        "test_evidence_present": tests is not None,
        "note": (
            "Thresholds come from configs/stage1.yaml and were fixed before any download. "
            "No gate was lowered to force a pass. A gate whose evidence is missing is "
            "reported as 'not measured' and does not pass."
        ),
    }


def _counts(statuses: dict[str, str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in statuses.values():
        out[s] = out.get(s, 0) + 1
    return dict(sorted(out.items()))
