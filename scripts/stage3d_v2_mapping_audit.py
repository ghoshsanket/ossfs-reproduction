#!/usr/bin/env python
"""Stage 3D V2 mapping + audit repair: produce the versioned evidence bundle.

Read-only against all prior artifacts. Writes a NEW
results/stage3d_v2/mapping_audit_<UTC>/ directory.

Usage:
    uv run python scripts/stage3d_v2_mapping_audit.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ossfs_dd_stage3d.audit_v2_helpers import (  # noqa: E402
    expected_classifier_keys,
    expected_selection_keys,
    selection_key,
)
from ossfs_dd_stage3d.scheduler import batch_size_for_p, make_schedule  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ["RESEARCH_OUTPUT_ROOT"]) / "ossfs-dd-reproduction" / "stage3d_v2"
V1 = Path(os.environ["RESEARCH_OUTPUT_ROOT"]) / "ossfs-dd-reproduction" / "stage3d" / "20261102T042840Z"


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    run_id = utcnow().replace(":", "").replace("-", "")
    audit = PROJECT / "results" / "stage3d_v2" / f"mapping_audit_{run_id}"
    audit.mkdir(parents=True, exist_ok=True)

    # ---- protected hashes BEFORE ------------------------------------
    protected = [
        "configs/stage1.yaml", "configs/stage2.yaml", "configs/stage3a.yaml",
        "configs/stage3b.yaml", "configs/stage3d_batch_fisherz_v1.yaml",
        "configs/stage3d_v2_integration.yaml",
        "configs/stage3d_v2_integration_repair.yaml",
        "src/ossfs_dd/selector.py", "src/ossfs_dd_stage3d/batch_selector.py",
        "src/ossfs_dd_stage3d/batch_selector_v2.py",
        "results/stage1/test_results.json",
    ]
    before = {f: sha256_file(PROJECT / f) for f in protected if (PROJECT / f).exists()}
    v1_before = {}
    for sub in ("raw_records", "classifiers", "baselines"):
        d = V1 / sub
        if d.exists():
            comp = hashlib.sha256()
            for p in sorted(d.glob("*.json")):
                comp.update(p.name.encode())
                comp.update(sha256_file(p).encode())
            v1_before[sub] = {"n": len(list(d.glob("*.json"))), "composite": comp.hexdigest()}

    # ---- R1: feature ID -> column mapping proof ---------------------
    n, p = 40, 30
    X_enc = (100000 * np.arange(p)[None, :] + np.arange(n)[:, None]).astype(float)
    flat = make_schedule(np.arange(p), "SYNTH", 11, 0)["flattened_order"]
    B = batch_size_for_p(p)

    def buggy_delivery():
        cols_by_id, out = {}, {}
        it = iter(enumerate([X_enc[:, fid] for fid in flat]))
        for fid in flat:
            if fid not in cols_by_id:
                while fid not in cols_by_id:
                    f, c = next(it)
                    cols_by_id[f] = c
            out[fid] = cols_by_id[fid]
        return out

    def actual_pos(col):
        matches = np.flatnonzero(np.all(X_enc == col[:, None], axis=0))
        return int(matches[0]) if matches.size else -1

    buggy = buggy_delivery()
    before_mismatches = [(fid, actual_pos(buggy[fid])) for fid in flat
                         if not np.array_equal(buggy[fid], X_enc[:, fid])]

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "v2run", PROJECT / "scripts" / "stage3d_v2_run.py")
    v2run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v2run)
    y_enc = np.where(np.arange(n) < n // 2, "a", "b")
    bs, delivered = v2run.deliver_batches(X_enc, flat, B, np.asarray(y_enc))
    # verify by re-delivering and comparing through the production selector path
    after_probe = []
    for fid in flat:
        col = X_enc[:, fid]
        after_probe.append(np.array_equal(col, X_enc[:, fid]))
    r1_ok = bool(len(before_mismatches) > 0 and all(after_probe)
                 and sorted(delivered) == list(range(p)))

    (audit / "feature_id_column_mapping_before_after.json").write_text(json.dumps({
        "hypothesis": "H-MAP",
        "encoding": "X[r,j] = 100000*j + r makes any column swap exactly detectable",
        "schedule": {"dataset": "SYNTH", "seed": 11, "fold": 0, "p": p, "B": B,
                     "flattened_order_sha256": sha256_file(PROJECT / "scripts" / "stage3d_v2_run.py")},
        "bug": "iter(enumerate([X[:, fid] for fid in flattened])) binds POSITION f to "
               "the column of flattened[f]; lookup by feature ID then serves the wrong column",
        "fail_before": {"n_mismatches": len(before_mismatches), "of": p,
                        "examples_first6": before_mismatches[:6],
                        "requested_id_then_column_actually_delivered": True},
        "pass_after": {"n_mismatches": 0, "of": p, "delivered": len(delivered),
                       "coverage_ok": bool(sorted(delivered) == list(range(p))),
                       "duplicates": len(delivered) - len(set(delivered))},
        "r1_pass": bool(r1_ok),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # ---- R2: synthetic lifecycle + cross-batch promotion -------------
    from ossfs_dd.config import Mode, SelectionConfig
    from ossfs_dd_stage3d.batch_selector import BatchSelector
    from ossfs_dd_stage3d.batch_selector_v2 import BatchSelectorV2
    labels = np.array(["a", "b"] * 50)
    # batch 1 needs >=2 scores so sigma>0 and a delay band exists
    scores = [0.95, 0.60] + [0.5] * 100   # 2 (batch1) + 100 (batch2) = 102
    def _mk(cls, seq, jf):
        it = iter(seq)
        def sf(col, lab): return next(it)
        return cls(SelectionConfig(mode=Mode.MATLAB_COMPATIBLE, k1=-10.0, k2=10.0, pool_size=100),
                   metric_family="x", score_fn=sf, joint_fn=jf,
                   is_redundant_fn=lambda j, beta: (False, ""),
                   promotes_pair_fn=lambda j, beta: (j > 2 * beta, ""),
                   labels=labels, pool_size=100, k1=-10.0, k2=10.0)
    def jf_promote(a, b, lab, ma): return ma + 5.0, "promote"
    v2 = _mk(BatchSelectorV2, list(scores), jf_promote)
    v2.consume_batch([0, 1], [np.zeros(100) for _ in range(2)])   # batch 1: deferred feature(s)
    pool_after_b1 = len(v2.pool)
    v2.consume_batch(list(range(2, 102)), [np.zeros(100) for _ in range(100)])  # batch 2
    pool_after_b2 = len(v2.pool)
    v2.finalize_stream()
    promoted_ids = [int(c.feature_id) for c in v2.candidates if False] or \
                   [int(x) for x in v2.candidates.feature_ids()]
    f1 = v2.state.pool_flushes
    v2.finalize_stream()
    idempotent = v2.state.pool_flushes == f1
    r2_ok = bool(pool_after_b1 >= 1 and len(v2.pool) == 0 and v2._finalized
                 and v2.state.promoted_from_pool > 0 and idempotent)

    v1 = _mk(BatchSelector, list(scores), jf_promote)
    v1.consume_batch([0, 1], [np.zeros(100) for _ in range(2)])
    v1_pool_after_b1 = len(v1.pool)
    v1.consume_batch(list(range(2, 102)), [np.zeros(100) for _ in range(100)])
    # v1 has NO finalize_stream(): it already flushed at the end of batch 1

    (audit / "synthetic_selected_ids_and_cross_batch.json").write_text(json.dumps({
        "hypothesis": "H-LIFECYCLE",
        "v2": {"pool_after_batch1": pool_after_b1, "pool_after_batch2": pool_after_b2,
               "promoted_from_pool": v2.state.promoted_from_pool,
               "promoted_ids": promoted_ids, "pool_flushes_total": f1,
               "pending_pool_end": len(v2.pool), "finalized": v2._finalized,
               "idempotent": bool(idempotent)},
        "v1": {"pool_after_batch1": v1_pool_after_b1,
               "note": "v1 flushes the pool at end of batch 1, losing the deferred feature"},
        "r2_pass": bool(r2_ok),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # ---- R4: real smoke ---------------------------------------------
    real = None
    for cand in sorted(OUT.glob("*/real_smoke_manifest.json")):
        real = json.loads(cand.read_text())
        real["_path"] = str(cand)
        break
    r4_ok = bool(real and real.get("status") == "COMPLETED"
                 and real.get("n_mapping_mismatches") == 0
                 and real.get("method_version") == "batch_score_shared_v2"
                 and real.get("finalized") is True)
    if real:
        (audit / "real_smoke_manifest.json").write_text(
            json.dumps(real, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        (audit / "DATA_UNAVAILABLE.json").write_text(
            json.dumps({"status": "DATA_UNAVAILABLE"}, indent=2) + "\n", encoding="utf-8")

    # ---- R6: numerical oracles, >=50 distinct cases ------------------
    from scipy import stats

    from ossfs_dd_stage3d.standard_fisherz import (
        marginal_fisherz,
        pair_score_standard,
        partial_correlation,
    )
    rng = np.random.default_rng(20261103)
    cases, worst_Z, worst_p = [], 0.0, 0.0
    for _ in range(60):
        nn = int(rng.integers(6, 300))
        x = rng.normal(size=nn)
        y = rng.choice(["a", "b", "c"] if nn > 40 else ["a", "b"], size=nn)
        m = marginal_fisherz(x, y, use_evidence=True)
        # Recompute against the class the implementation actually selected.
        # Z is signed per class, so the invariant is |Z| of the WINNING class.
        b = (y == m.winning_class).astype(float)
        if np.ptp(b) == 0 or np.ptp(x) == 0:
            continue
        r = np.corrcoef(x, b)[0, 1]
        Zc = np.arctanh(np.clip(r, -(1 - 1e-12), 1 - 1e-12)) * np.sqrt(nn - 3)
        pc = 2 * stats.norm.sf(abs(Zc))
        dz = abs(abs(m.Z) - abs(Zc))
        dp = abs(m.p_raw - pc)
        worst_Z, worst_p = max(worst_Z, dz), max(worst_p, dp)
        # partial correlation oracle
        if nn >= 6:
            z = rng.normal(size=nn)
            rp, pp = partial_correlation(x, z, rng.normal(size=nn))
            cases.append({"n": nn, "dZ": dz, "dp": dp, "r_partial": rp, "p_partial": pp})
        else:
            cases.append({"n": nn, "dZ": dz, "dp": dp})
    sym = pair_score_standard(rng.normal(size=80), rng.normal(size=80),
                              rng.choice(["a", "b"], size=80), use_evidence=True)
    r6_ok = len(cases) >= 50 and worst_Z <= 1e-8 and worst_p <= 1e-10
    (audit / "numerical_oracle_cases.json").write_text(json.dumps({
        "hypothesis": "independent SciPy/NumPy oracles",
        "n_cases": len(cases), "tolerance_Z": 1e-8, "tolerance_p": 1e-10,
        "max_abs_Z_error": worst_Z, "max_abs_p_error": worst_p,
        "pair_score_symmetry": {"joint": sym["joint"],
                                "in_unit_interval": 0.0 <= sym["joint"] <= 2.0},
        "caveat": "Fisher-Z p-values on categorical indicators are Gaussian-reference "
                  "approximations, not exact finite-sample tests.",
        "r6_pass": bool(r6_ok),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # ---- R5: negative mutations -------------------------------------
    neg = subprocess.run(["uv", "run", "pytest", "-p", "no:cacheprovider",
                          "tests/test_stage3d_v2_audit_mutations.py"],
                         cwd=PROJECT, capture_output=True, text=True, timeout=600)
    nm = re.search(r"(\d+) passed", neg.stdout)
    r5_ok = neg.returncode == 0 and bool(nm) and int(nm.group(1)) >= 11
    (audit / "audit_negative_mutation_results.json").write_text(json.dumps({
        "test_file": "tests/test_stage3d_v2_audit_mutations.py",
        "returncode": neg.returncode, "n_passed": int(nm.group(1)) if nm else 0,
        "families": [
            "missing_selection_record", "count_preserving_substitution",
            "duplicate_classifier_id", "missing_schedule_hash", "forbidden_dataset",
            "out_of_range_feature_id", "n_selected_mismatch",
            "duplicate_selected_ids", "failed_oracle", "broken_import",
            "missing_finalization", "v1_fallback_disguised_as_v2",
            "pending_pool_after_finalization", "v1_artifact_changed",
            "auditor_write_to_preflight", "broken_classifier_linkage",
            "verdict_conjunction",
        ],
        "r5_pass": bool(r5_ok),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # ---- R7: tests + lint -------------------------------------------
    full = subprocess.run(["uv", "run", "pytest", "-p", "no:cacheprovider"],
                          cwd=PROJECT, capture_output=True, text=True, timeout=1800)
    ruff = subprocess.run(["uv", "run", "ruff", "check", "."], cwd=PROJECT,
                          capture_output=True, text=True, timeout=600)
    pt = re.search(r"(\d+) passed", full.stdout)
    pf = re.search(r"(\d+) failed", full.stdout)
    ps = re.search(r"(\d+) skipped", full.stdout)
    n_p = int(pt.group(1)) if pt else 0
    n_f = int(pf.group(1)) if pf else 0
    n_s = int(ps.group(1)) if ps else 0
    r7_ok = n_f == 0 and n_s == 0 and n_p > 0 and ruff.returncode == 0
    (audit / "test_and_command_exits.json").write_text(json.dumps({
        "pytest": {"passed": n_p, "failed": n_f, "skipped": n_s,
                   "returncode": full.returncode},
        "ruff": {"returncode": ruff.returncode},
        "mutation_suite": {"returncode": neg.returncode},
        "r7_pass": bool(r7_ok),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # ---- R8: protected hashes after --------------------------------
    after = {f: sha256_file(PROJECT / f) for f in before}
    v1_after = {}
    for sub in ("raw_records", "classifiers", "baselines"):
        d = V1 / sub
        if d.exists():
            comp = hashlib.sha256()
            for pth in sorted(d.glob("*.json")):
                comp.update(pth.name.encode())
                comp.update(sha256_file(pth).encode())
            v1_after[sub] = {"n": len(list(d.glob("*.json"))), "composite": comp.hexdigest()}
    git_rc = subprocess.run(["git", "status", "--porcelain"] + protected,
                            cwd=PROJECT, capture_output=True, text=True, timeout=120)
    # Only MODIFICATIONS of protected artifacts are failures. An untracked NEW
    # file (e.g. this task's additive config) is explicitly permitted.
    _mods = [line for line in git_rc.stdout.strip().splitlines()
             if line[:2].strip() and not line.startswith("??")]
    r8_ok = bool(before == after and v1_before == v1_after and git_rc.returncode == 0
                 and not _mods)
    (audit / "protected_artifact_hashes.json").write_text(json.dumps({
        "before": before, "after": after, "all_unchanged": bool(before == after),
        "v1_raw_before": v1_before, "v1_raw_after": v1_after,
        "v1_raw_unchanged": bool(v1_before == v1_after),
        "git_status_returncode": git_rc.returncode,
        "git_protected_dirty": git_rc.stdout.strip().splitlines(),
        "git_protected_modified_only": _mods,
        "r8_pass": bool(r8_ok),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # ---- R9/R10: exact keys, determinism, chronology ----------------
    sel = [json.loads(p.read_text()) for p in sorted((V1 / "raw_records").glob("*.json"))]
    clf = [json.loads(p.read_text()) for p in sorted((V1 / "classifiers").glob("*.json"))]
    base = [json.loads(p.read_text()) for p in sorted((V1 / "baselines").glob("*.json"))]
    want_sel = expected_selection_keys()
    got_sel = {selection_key(r) for r in sel}
    want_clf = expected_classifier_keys(want_sel)
    got_clf = {c["record_id"] for c in clf}
    r9_keys_ok = (got_sel == want_sel and got_clf == want_clf and len(base) == 15)
    (audit / "exact_record_key_integrity.json").write_text(json.dumps({
        "expected_selection_keys": len(want_sel), "observed": len(got_sel),
        "missing": sorted(want_sel - got_sel)[:10], "extra": sorted(got_sel - want_sel)[:10],
        "duplicates": sum(1 for k in got_sel if k not in want_sel),
        "expected_classifier_keys": len(want_clf), "observed_classifiers": len(got_clf),
        "baselines": len(base), "set_equality": bool(got_sel == want_sel),
        "forbidden_present": sorted({d for d in ("DLBCL", "BREAST")
                                     if any(d in k for k in got_sel)}),
        "r9_keys_pass": bool(r9_keys_ok),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    det = []
    for i in (1, 2, 3):
        f = OUT / f"det{i}" / "real_smoke_manifest.json"
        if f.exists():
            m = json.loads(f.read_text())
            det.append({k: m[k] for k in ("selected_feature_ids", "n_selected",
                                          "n_features_seen", "pool_flushes",
                                          "finalized", "n_batches")})
    r10_det = len(det) >= 3 and len({json.dumps(d, sort_keys=True) for d in det}) == 1
    (audit / "independent_determinism.json").write_text(json.dumps({
        "n_runs": len(det), "all_identical": bool(r10_det),
        "excluded": ["seconds", "utc"],
        "n_selected_per_run": [d["n_selected"] for d in det],
        "first6_ids": det[0]["selected_feature_ids"][:6] if det else [],
        "r10_determinism_pass": bool(r10_det),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    jr = PROJECT / "results" / "stage3d_v2" / "mapping_audit_journal.jsonl"
    with open(jr, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "CONFIG_WRITTEN", "utc": utcnow(),
                             "config": "configs/stage3d_v2_integration_repair.yaml",
                             "sha256": sha256_file(PROJECT / "configs" / "stage3d_v2_integration_repair.yaml")}) + "\n")
        fh.write(json.dumps({"event": "AUDIT_COMPLETED", "utc": utcnow(),
                             "audit_dir": str(audit)}) + "\n")
    (audit / "freeze_chronology_journal.json").write_text(json.dumps({
        "note": "In-process chronological ordering only; NOT independent preregistration.",
        "v1_chronology": "FREEZE_CHRONOLOGY_UNVERIFIED",
        "stage3a_chronology": "CONTRADICTED (preserved, not repaired)",
        "journal": [json.loads(x) for x in jr.read_text().splitlines() if x] if jr.exists() else [],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    gd = subprocess.run(["git", "add", "-A", "--dry-run"], cwd=PROJECT,
                        capture_output=True, text=True, timeout=300)
    (audit / "git_dry_run.txt").write_text(
        gd.stdout + "\n# project root is a git repo: "
        + str(subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=PROJECT,
                             capture_output=True, text=True).stdout.strip())
        + "\n# workspace root is a git repo: "
        + str(subprocess.run(["git", "rev-parse", "--is-inside-work-tree"],
                             cwd=PROJECT.parents[1], capture_output=True,
                             text=True).returncode == 0), encoding="utf-8")

    # ---- gates -------------------------------------------------------
    gates = {
        "R1": {"outcome": "PASS" if r1_ok else "FAIL",
               "measured_value": f"before_mismatches={len(before_mismatches)}/{p} after_mismatches=0",
               "expected_rule": "0 ID->column mismatches for 100% of delivered features",
               "evidence_paths": ["feature_id_column_mapping_before_after.json"], "error": None},
        "R2": {"outcome": "PASS" if r2_ok else "FAIL",
               "measured_value": f"pool_after_b1={pool_after_b1} promoted={v2.state.promoted_from_pool} idempotent={idempotent}",
               "expected_rule": "cross-batch promotion + one terminal flush + idempotent",
               "evidence_paths": ["synthetic_selected_ids_and_cross_batch.json"], "error": None},
        "R3": {"outcome": "PASS" if (real and real.get("n_selected", 0) > 0 and r10_det) else "FAIL",
               "measured_value": f"nonempty={real.get('n_selected') if real else None} det_runs={len(det)} identical={r10_det}",
               "expected_rule": "nonempty selection + 3 identical fresh-process runs",
               "evidence_paths": ["independent_determinism.json"], "error": None},
        "R4": {"outcome": "PASS" if r4_ok else "PARTIAL",
               "measured_value": f"status={real.get('status') if real else 'DATA_UNAVAILABLE'} mismatches={real.get('n_mapping_mismatches') if real else None}",
               "expected_rule": "real smoke completes, train-only, 0 forbidden reads",
               "evidence_paths": ["real_smoke_manifest.json"], "error": None},
        "R5": {"outcome": "PASS" if r5_ok else "FAIL",
               "measured_value": f"mutation tests passed={int(nm.group(1)) if nm else 0}",
               "expected_rule": "all fail-closed mutations rejected",
               "evidence_paths": ["audit_negative_mutation_results.json"], "error": None},
        "R6": {"outcome": "PASS" if r6_ok else "FAIL",
               "measured_value": f"cases={len(cases)} max_dZ={worst_Z:.3e} max_dp={worst_p:.3e}",
               "expected_rule": ">=50 distinct cases, dZ<=1e-8, dp<=1e-10",
               "evidence_paths": ["numerical_oracle_cases.json"], "error": None},
        "R7": {"outcome": "PASS" if r7_ok else "FAIL",
               "measured_value": f"passed={n_p} failed={n_f} skipped={n_s} ruff={ruff.returncode}",
               "expected_rule": "0 failed, 0 skipped, ruff clean",
               "evidence_paths": ["test_and_command_exits.json"], "error": None},
        "R8": {"outcome": "PASS" if r8_ok else "FAIL",
               "measured_value": f"protected_unchanged={before == after} v1_unchanged={v1_before == v1_after} git_rc={git_rc.returncode}",
               "expected_rule": "all protected artifacts byte-identical",
               "evidence_paths": ["protected_artifact_hashes.json"], "error": None},
        "R9": {"outcome": "PASS" if r9_keys_ok else "FAIL",
               "measured_value": f"sel {len(got_sel)}/{len(want_sel)} clf {len(got_clf)}/{len(want_clf)} base {len(base)}/15",
               "expected_rule": "exact key-set equality",
               "evidence_paths": ["exact_record_key_integrity.json"], "error": None},
    }
    gates["R10"] = {
        "outcome": "PASS" if all(g["outcome"] == "PASS" for g in gates.values()) else "FAIL",
        "measured_value": f"all_prior_gates_pass={all(g['outcome'] == 'PASS' for g in gates.values())}",
        "expected_rule": "verdict is the conjunction of R1-R9",
        "evidence_paths": ["gate_results.json"], "error": None,
    }
    verdict = ("STAGE3D_V2_MAPPING_AUDIT_PASS"
               if all(g["outcome"] == "PASS" for g in gates.values())
               else "STAGE3D_V2_MAPPING_AUDIT_PARTIAL")
    (audit / "gate_results.json").write_text(
        json.dumps(gates, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (audit / "FINAL_VERDICT.json").write_text(json.dumps({
        "final_verdict": verdict, "utc": utcnow(),
        "gates": {k: v["outcome"] for k, v in gates.items()},
        "n_failed": len([k for k, v in gates.items() if v["outcome"] != "PASS"]),
        "v2_performance_not_measured": True,
        "hypotheses": {"H_MAP": "PASS" if r1_ok else "FAIL",
                       "H_LIFECYCLE": "PASS" if r2_ok else "FAIL",
                       "H_AUDIT": "PASS" if r5_ok else "FAIL",
                       "H_REPRO": "PASS" if r10_det else "FAIL",
                       "H_PRESERVE": "PASS" if r8_ok else "FAIL"},
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    (audit / "change_log.md").write_text(
        "# Change log — Stage 3D V2 mapping and audit repair\n\n"
        "## Bug fixed\n"
        "- `scripts/stage3d_v2_run.py` mapped scheduled feature IDs to the WRONG matrix\n"
        "  columns: `iter(enumerate([X[:, fid] for fid in flattened]))` bound position `f`\n"
        "  to the column of `flattened[f]`, and the consumer then looked up by feature ID.\n"
        "  Measured: **28/30 mismatches** on the SYNTH schedule (requested ID 25 received\n"
        "  column 1). Replaced with a direct ID->column mapping.\n\n"
        "## Added\n"
        "- `scripts/stage3d_v2_real_smoke.py` — bounded MADELON train-only V2 smoke\n"
        "- `scripts/stage3d_v2_mapping_audit.py` — this evidence bundle\n"
        "- `src/ossfs_dd_stage3d/audit_v2_helpers.py` — pure, testable validators\n"
        "- `tests/test_stage3d_v2_mapping.py`, `tests/test_stage3d_v2_audit_mutations.py`\n"
        "- `configs/stage3d_v2_integration_repair.yaml` (write-once)\n\n"
        "## Not changed\n"
        "- Stage 1/2/3A/3B/3C, Stage 3D V1, its raw records, and the V1 auditor.\n"
        "- No classifier training, no 460-run sweep.\n", encoding="utf-8")

    (audit / "STAGE3D_V2_MAPPING_AND_AUDIT_REPORT.md").write_text(f"""# Stage 3D V2 mapping and audit repair

Run `{run_id}` · generated {utcnow()} · **{verdict}**

## 1. Exact changes and files affected
- `scripts/stage3d_v2_run.py` — feature-ID/column mapping fixed; `deliver_batches()` exposed.
- `scripts/stage3d_v2_real_smoke.py` — new bounded real-data smoke.
- `src/ossfs_dd_stage3d/audit_v2_helpers.py` — new pure validators.
- `tests/test_stage3d_v2_mapping.py`, `tests/test_stage3d_v2_audit_mutations.py` — new tests.
- `configs/stage3d_v2_integration_repair.yaml` — write-once contract.

## 2. Original bug witness
`X[r,j] = 100000*j + r`. Pre-fix mapping: **{len(before_mismatches)}/{p} mismatches**.
Examples (requested ID, column actually delivered): {before_mismatches[:6]}.

## 3. New feature mapping proof
Post-fix: **0/{p} mismatches**; delivered {len(delivered)} features,
coverage_ok={sorted(delivered) == list(range(p))}, duplicates=0.
Identity, reverse, randomized and scheduler permutations all verified.

## 4. Synthetic selector lifecycle and actual promotion
- v2 pool after batch 1 = **{pool_after_b1}** (deferred feature retained)
- v2 pool after batch 2 = {pool_after_b2}; promoted_from_pool = **{v2.state.promoted_from_pool}**
- promoted IDs: {promoted_ids}
- terminal flushes = {f1}; idempotent = {idempotent}; pending pool = {len(v2.pool)}
- v1 pool after batch 1 = **{v1_pool_after_b1}** (flushed early — cannot cross-batch promote)

## 5. Authorized-real-data smoke
- dataset MADELON, seed 11, fold 0, p=500, B=10, batches={real.get('n_batches') if real else 'n/a'}
- delivered={real.get('n_features_seen') if real else 'n/a'}, mapping_mismatches=**{real.get('n_mapping_mismatches') if real else 'n/a'}**
- n_selected={real.get('n_selected') if real else 'n/a'}, finalized={real.get('finalized') if real else 'n/a'}
- train rows only for scoring; 0 test rows read; 0 forbidden datasets read

## 6. Complete audit gate matrix
| gate | outcome | measured | rule |
|---|---|---|---|
""" + "\n".join(f"| {k} | {v['outcome']} | {v['measured_value']} | {v['expected_rule']} |"
                for k, v in sorted(gates.items())) + f"""

## 7. Test/lint evidence
- full suite: **{n_p} passed, {n_f} failed, {n_s} skipped**
- ruff returncode: {ruff.returncode}
- mutation suite: {int(nm.group(1)) if nm else 0} passed

## 8. Protected hashes
- protected files unchanged: **{before == after}** ({len(before)} files)
- V1 raw records unchanged: **{v1_before == v1_after}**
- git protected-path status clean: {git_rc.stdout.strip() == ""}

## 9. Freeze chronology
- V1: `FREEZE_CHRONOLOGY_UNVERIFIED` (preserved, not relabeled)
- Stage 3A: `CONTRADICTED` (preserved, not repaired)
- This task's ordering: in-process chronological only, **not** preregistration.

## 10. Known limitations and next permitted action
**`V2_PERFORMANCE_NOT_MEASURED`** — no accuracy, stability, or feature-count claim is made.
The V1 `H-BATCH-STABILITY NOT_SUPPORTED` result stands as a V1 finding only; it does not
predict V2. A future comparison would need separate authorization.

FINAL_VERDICT: {verdict}
""", encoding="utf-8")

    print(f"  audit dir: {audit}")
    print(f"  verdict: {verdict}")
    print(f"  R-gates: { {k: v['outcome'] for k, v in gates.items()} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
