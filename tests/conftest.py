"""Shared pytest fixtures.

A synthetic fixture is used to exercise the contracts (parse, stream, split).
It is a TEST FIXTURE ONLY and is never written into the dataset tree, never
placed in the registry, and never presented as a paper dataset.

The session-finish hook records pass/fail counts, the presence of every required
named test, and the git ignore-audit outcome into
``results/stage1/test_results.json`` so the Stage 1 gate evaluator has real
evidence rather than an assumption.

That write is **opt-in** (``--write-stage1-evidence``). It used to happen on
every pytest session, so running an unrelated suite (for example a Stage 2/3
regression suite) silently rewrote the Stage 1 gate artifact with that run's
totals -- 197 instead of the committed 135 -- which made the Stage 1 evidence
depend on what else happened to be in the tree. Stage 1 evidence is now written
only when it is explicitly asked for; an ordinary ``uv run pytest -q`` leaves
``results/stage1/test_results.json`` byte-identical. The payload schema and the
gate semantics in ``ossfs_dd_data.gates`` are unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ossfs_dd_data.registry import load_registry, load_stage1_config
from ossfs_dd_data.workspace import data_root, project_root, results_dir

WRITE_EVIDENCE_FLAG = "--write-stage1-evidence"

STREAM_LEAKAGE_TESTS = {
    "test_stream_contains_only_training_rows",
    "test_no_target_column_in_features",
    "test_no_unlabeled_test_data",
    "test_cv_train_test_disjoint",
}


def pytest_addoption(parser) -> None:
    """Register the opt-in Stage 1 evidence switch."""
    parser.addoption(
        WRITE_EVIDENCE_FLAG,
        action="store_true",
        default=False,
        help=(
            "Write results/stage1/test_results.json (the Stage 1 gate evidence). "
            "Off by default so unrelated test runs never mutate Stage 1 artifacts."
        ),
    )


def stage1_evidence_requested(config) -> bool:
    """True only when the caller explicitly asked for Stage 1 evidence."""
    try:
        return bool(config.getoption(WRITE_EVIDENCE_FLAG))
    except (ValueError, AttributeError):  # pragma: no cover - defensive
        return False


@pytest.fixture(scope="session")
def registry():
    return load_registry()


@pytest.fixture(scope="session")
def stage1_cfg():
    return load_stage1_config()


@pytest.fixture(scope="session")
def synthetic_fixture(tmp_path_factory):
    """Small synthetic matrix used only to exercise the contracts."""
    rng = np.random.default_rng(20220101)
    n, p = 40, 12
    X = rng.normal(size=(n, p))
    y = np.array(["alpha", "beta"] * (n // 2), dtype=object)
    train_idx = np.array([i for i in range(n) if i % 4 != 0], dtype=np.int64)
    order = np.random.Generator(np.random.PCG64(7)).permutation(p)
    d = tmp_path_factory.mktemp("synthetic_fixture")
    (d / "README.txt").write_text(
        "Synthetic TEST FIXTURE for contract tests. Not a paper dataset.\n", encoding="utf-8"
    )
    return {"X": X, "y": y, "train_idx": train_idx, "order": order, "dir": d, "n": n, "p": p}


def pytest_sessionfinish(session, exitstatus):
    """Persist test evidence for the Stage 1 gate evaluator.

    Opt-in only: an ordinary pytest run must not mutate Stage 1 artifacts.
    """
    if not stage1_evidence_requested(session.config):
        return
    config_error: str | None = None
    try:
        cfg = load_stage1_config()
        required = set(cfg.get("required_named_tests") or [])
        if not required:
            config_error = "configs/stage1.yaml declared no required_named_tests"
    except Exception as exc:  # noqa: BLE001 - evidence must be reported, not assumed
        # Failing over to an empty required-set would make `missing_required_tests`
        # vacuously empty and turn the gate green. Record the failure instead.
        required = set()
        config_error = f"{type(exc).__name__}: {exc}"

    reporter = None
    pm = getattr(session.config, "pluginmanager", None)
    if pm is not None:
        reporter = pm.getplugin("terminalreporter")

    stats = getattr(reporter, "stats", {}) if reporter is not None else {}
    passed = len(stats.get("passed", []))
    failed = len(stats.get("failed", [])) + len(stats.get("error", []))
    names = [
        rep.nodeid.split("::")[-1].split("[")[0]
        for bucket in stats.values()
        for rep in bucket
    ]
    distinct = set(names)
    missing = sorted(required - distinct)
    # A required test that was SKIPPED has not produced evidence. Treating a skip
    # as a pass would let an inaccessible dataset source turn a hard gate green.
    skipped_names = {
        rep.nodeid.split("::")[-1].split("[")[0] for rep in stats.get("skipped", [])
    }
    skipped_required = sorted(required & skipped_names)
    effective_passed = {
        rep.nodeid.split("::")[-1].split("[")[0] for rep in stats.get("passed", [])
    }
    skipped_required = [n for n in skipped_required if n not in effective_passed]
    leakage_failures = sorted(
        n for n in STREAM_LEAKAGE_TESTS if n in distinct and n not in effective_passed
    )

    git_audit = _git_data_violations()

    payload = {
        "schema": "ossfs-dd-reproduction/stage1/testresults/2.0",
        "exitstatus": int(exitstatus),
        # Sanitised: a committed artefact must not carry private absolute paths.
        "interpreter_is_project_venv": str(Path(sys.executable).resolve())
        == str((project_root() / ".venv" / "bin" / "python").resolve()),
        "python_version": sys.version.split()[0],
        "passed": passed,
        "failed": failed,
        "skipped": len(stats.get("skipped", [])),
        "total": passed + failed + len(stats.get("skipped", [])),
        "required_named_tests": sorted(required),
        "missing_required_tests": missing,
        "skipped_required_tests": skipped_required,
        "observed_test_names": sorted(distinct),
        "stream_leakage_tests": sorted(STREAM_LEAKAGE_TESTS),
        "stream_leakage_failures": leakage_failures,
        "git_audit_status": git_audit["status"],
        "git_data_violations": git_audit["violations"],
        "git_audit_detail": git_audit["detail"],
        "config_error": config_error,
        "evidence_complete": bool(
            not missing
            and not skipped_required
            and git_audit["status"] == "ok"
            and not leakage_failures
            and config_error is None
        ),
        "fail_closed_note": (
            "git_data_violations is null when the git audit could not run. A gate "
            "evaluated against null evidence must report 'not measured' and must not pass."
        ),
    }
    try:
        results_dir().mkdir(parents=True, exist_ok=True)
        (results_dir() / "test_results.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as exc:  # pragma: no cover - results tree unwritable
        print(f"WARNING: could not persist test_results.json: {exc}", file=sys.stderr)


def _git_data_violations() -> dict[str, Any]:
    """Audit whether git would stage data or binaries.

    FAIL-CLOSED. v1 returned ``0`` violations whenever git was missing, the
    project was not a repository, or a subprocess raised, which silently turned
    absent evidence into a passing gate. The result now carries an explicit
    status:

      ``ok``       the audit ran; ``violations`` is a measured integer
      ``unavailable``  git is not a repository here -> ``violations`` is null
      ``error``     git raised or returned unusable output -> ``violations`` null

    ``null`` means *not measured*, and a gate evaluated against it must not pass.
    """
    import subprocess

    root = project_root()
    try:
        inside = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"status": "error", "violations": None,
                "detail": f"could not execute git: {type(exc).__name__}: {exc}"}

    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return {
            "status": "unavailable",
            "violations": None,
            "detail": "project is not inside a git work tree; the ignore rules could not be audited",
        }

    try:
        dry = subprocess.run(
            ["git", "-C", str(root), "add", "-A", "--dry-run", "--verbose"],
            capture_output=True, text=True, timeout=180, check=False,
        )
        ignored = subprocess.run(
            ["git", "-C", str(root), "status", "--ignored", "--porcelain"],
            capture_output=True, text=True, timeout=180, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"status": "error", "violations": None,
                "detail": f"git subprocess failed: {type(exc).__name__}: {exc}"}

    if dry.returncode != 0 or not dry.stdout:
        return {
            "status": "error",
            "violations": None,
            "detail": f"git add --dry-run failed (rc={dry.returncode}): {dry.stderr.strip()[:200]}",
        }
    if ignored.returncode != 0:
        return {
            "status": "error",
            "violations": None,
            "detail": f"git status --ignored failed (rc={ignored.returncode})",
        }

    violations: list[str] = []
    for line in dry.stdout.splitlines():
        if not line.startswith("add "):
            continue
        path = line.split("'")[1] if "'" in line else line.split()[-1]
        low = path.lower()
        if low.startswith(("dataset/", "data/", "outputs/", "runs/")) or low.endswith(
            (".npy", ".npz", ".zip", ".arff", ".pt", ".pth", ".safetensors")
        ):
            violations.append(path)
    if data_root().exists() and "!!" not in ignored.stdout:
        violations.append("dataset tree present but nothing reported as ignored")

    return {
        "status": "ok",
        "violations": len(violations),
        "detail": sorted(violations) if violations else "no data or binary path would be staged",
    }
