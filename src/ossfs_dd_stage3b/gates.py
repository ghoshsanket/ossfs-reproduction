"""Operational gates for Stage 3B.

Gates are execution-quality checks. A refuted scientific hypothesis is a
successful experiment; a gate failure means the execution or its provenance
was incomplete. Hypothesis outcomes are reported alongside but never used to
gate the verdict.
"""

from __future__ import annotations

from typing import Any

from .protocol import A_PLANNED_FITS, B_PLANNED_RUNS

PASS, FAIL, NOT_MEASURED = "PASS", "FAIL", "NOT_MEASURED"


def _g(name: str, passed: bool | None, detail: str, **extra: Any) -> dict[str, Any]:
    return {
        "name": name,
        "status": NOT_MEASURED if passed is None else (PASS if passed else FAIL),
        "pass": passed,
        "detail": detail,
        **extra,
    }


def evaluate_gates(*, preflight: dict[str, Any], journal: list[dict[str, Any]],
                   svm_records: list[dict[str, Any]],
                   arcene_records: list[dict[str, Any]],
                   trace_checks: dict[str, Any],
                   anchor_checks: list[dict[str, Any]],
                   reproducibility: dict[str, Any],
                   safety: dict[str, Any],
                   config_digest: str,
                   run_id: str) -> dict[str, Any]:
    from .analysis import TERMINAL

    events = [e["event"] for e in journal]
    order_ok = (
        events[:1] == ["PRE_FLIGHT_WRITTEN"]
        and "FIRST_DATA_EVALUATION_STARTED" in events
        and "LAST_EVALUATION_FINISHED" in events
        and events.index("PRE_FLIGHT_WRITTEN") < events.index("FIRST_DATA_EVALUATION_STARTED")
        < events.index("LAST_EVALUATION_FINISHED")
    )
    digest_refs = {e.get("config_sha256") for e in journal if e.get("config_sha256")}

    svm_ids = {r["fit_id"] for r in svm_records}
    svm_planned = A_PLANNED_FITS
    svm_terminal = [r for r in svm_records if r["status"] in TERMINAL]
    arc_ids = {r["run_id"] for r in arcene_records}

    n_timed_out = sum(1 for r in svm_records if r["status"] == "TIMEOUT")
    n_fit_error = sum(1 for r in svm_records if r["status"] == "FIT_ERROR")
    n_blocked = sum(1 for r in svm_records if r["status"] == "BLOCKED_INPUT")

    gates = {
        "B_PREFLIGHT_CHRONOLOGY": _g(
            "B_PREFLIGHT_CHRONOLOGY", bool(order_ok and digest_refs == {config_digest}),
            f"journal={events}; config digest referenced by every event={digest_refs == {config_digest}}",
            run_id=run_id, config_sha256=config_digest,
            freeze_type="in-process, hashed, chronologically ordered freeze",
            independent_attestation=False,
        ),
        "B_AUTHORIZATION": _g(
            "B_AUTHORIZATION",
            safety.get("forbidden_reads") == 0 and not safety.get("unexpected_datasets")
            and safety.get("all_hashes_verified") is True,
            f"forbidden reads={safety.get('forbidden_reads')}; "
            f"datasets A={safety.get('experiment_a_datasets')}; "
            f"B={safety.get('experiment_b_datasets')}; "
            f"hashes verified={safety.get('all_hashes_verified')}",
        ),
        "B_A_FIXED_SUBSETS": _g(
            "B_A_FIXED_SUBSETS",
            safety.get("n_stage3a_subsets_verified") == 75
            and safety.get("n_selections_performed_in_A") == 0,
            f"Stage 3A subsets verified={safety.get('n_stage3a_subsets_verified')}/75; "
            f"feature selections performed in A={safety.get('n_selections_performed_in_A')}",
        ),
        "B_A_GRID_COMPLETE": _g(
            "B_A_GRID_COMPLETE",
            len(svm_ids) == svm_planned and len(svm_terminal) == svm_planned,
            f"unique fit IDs={len(svm_ids)}/{svm_planned}; terminal records={len(svm_terminal)}; "
            f"timeouts={n_timed_out}; fit errors={n_fit_error}; blocked inputs={n_blocked}",
            resource_limited=(n_timed_out > 0),
        ),
        "B_B_CROSSED_DESIGN": _g(
            "B_B_CROSSED_DESIGN",
            len(arc_ids) == B_PLANNED_RUNS
            and all(r["split_seed"] == 11 for r in arcene_records)
            and len({r["order_seed"] for r in arcene_records}) == 10
            and len({r["fold"] for r in arcene_records}) == 5,
            f"unique run IDs={len(arc_ids)}/{B_PLANNED_RUNS}; "
            f"split seeds={sorted({r['split_seed'] for r in arcene_records})}; "
            f"orders={len({r['order_seed'] for r in arcene_records})}; "
            f"folds={len({r['fold'] for r in arcene_records})}",
        ),
        "B_B_ANCHOR_MATCH": _g(
            "B_B_ANCHOR_MATCH",
            len(anchor_checks) == 5 and all(c["match"] for c in anchor_checks),
            "; ".join(f"fold{c['fold']}={'MATCH' if c['match'] else 'MISMATCH'}"
                      for c in anchor_checks) or "no anchor comparisons",
            n_matched=sum(1 for c in anchor_checks if c["match"]),
        ),
        "B_TRACE_ARITHMETIC": _g(
            "B_TRACE_ARITHMETIC",
            trace_checks.get("all_ok") is True,
            f"traces checked={trace_checks.get('n_traces')}; "
            f"checkpoint alpha/beta mismatches={trace_checks.get('n_alpha_beta_mismatch')}; "
            f"event-order mismatches={trace_checks.get('n_event_order_mismatch')}",
        ),
        "B_SVM_CONVERGENCE_HONEST": _g(
            "B_SVM_CONVERGENCE_HONEST",
            all(
                ("status" in r and "converged" in r and "accuracy" in r
                 and (r.get("n_iter_vector") is not None or r.get("n_iter_max") is not None
                      or r["status"] in ("TIMEOUT", "FIT_ERROR", "BLOCKED_INPUT")))
                for r in svm_records
            ),
            f"all {len(svm_records)} fit records carry status/converged/accuracy and an "
            f"n_iter vector (or a non-fit terminal status)",
        ),
        "B_REPRODUCIBILITY": _g(
            "B_REPRODUCIBILITY",
            bool(reproducibility.get("all_deterministic")),
            f"SVM repeats={reproducibility.get('svm_repeats')}; "
            f"ARCENE repeat={reproducibility.get('arcene_repeat')}",
            details=reproducibility,
        ),
        "B_CODE_AND_SAFETY": _g(
            "B_CODE_AND_SAFETY",
            safety.get("stage123a_trees_unchanged") is True
            and safety.get("ruff_clean") is True
            and safety.get("pytest_failed") == 0
            and safety.get("pytest_skipped") == 0
            and safety.get("forbidden_git_entries") == 0
            and safety.get("commits_made") == 0,
            f"Stage1/2/3A trees unchanged={safety.get('stage123a_trees_unchanged')}; "
            f"ruff clean={safety.get('ruff_clean')}; "
            f"pytest failed={safety.get('pytest_failed')} skipped={safety.get('pytest_skipped')}; "
            f"forbidden git entries={safety.get('forbidden_git_entries')}; "
            f"commits={safety.get('commits_made')}",
        ),
    }
    failed = [k for k, v in gates.items() if v["pass"] is not True]
    return {
        "gates": gates,
        "n_gates": len(gates),
        "n_passed": sum(1 for v in gates.values() if v["pass"] is True),
        "failed_gates": failed,
        "resource_limited": bool(gates["B_A_GRID_COMPLETE"].get("resource_limited")),
        "any_fit_timeout_or_error": bool(n_timed_out or n_fit_error or n_blocked),
    }
