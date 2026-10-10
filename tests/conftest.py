"""Shared pytest fixtures.

A synthetic fixture is used to exercise the contracts (parse, stream, split).
It is a TEST FIXTURE ONLY and is never written into the dataset tree, never
placed in the registry, and never presented as a paper dataset.

The session-finish hook records pass/fail counts, the presence of every required
named test, and the git ignore-audit outcome into
``results/stage1/test_results.json`` so the Stage 1 gate evaluator has real
evidence rather than an assumption.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ossfs_dd_data.registry import load_registry, load_stage1_config
from ossfs_dd_data.workspace import data_root, project_root, results_dir

STREAM_LEAKAGE_TESTS = {
    "test_stream_contains_only_training_rows",
    "test_no_target_column_in_features",
    "test_no_unlabeled_test_data",
    "test_cv_train_test_disjoint",
}


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
    """Persist test evidence for the Stage 1 gate evaluator."""
    try:
        cfg = load_stage1_config()
        required = set(cfg.get("required_named_tests") or [])
    except Exception:
        required = set()

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
    passed_names = {
        rep.nodeid.split("::")[-1].split("[")[0] for rep in stats.get("passed", [])
    }
    distinct = set(names)
    missing = sorted(required - distinct)
    leakage_failures = sorted(
        n for n in STREAM_LEAKAGE_TESTS if n in distinct and n not in passed_names
    )

    payload = {
        "schema": "ossfs-dd-reproduction/stage1/testresults/1.0",
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
        "observed_test_names": sorted(distinct),
        "stream_leakage_tests": sorted(STREAM_LEAKAGE_TESTS),
        "stream_leakage_failures": leakage_failures,
        "git_data_violations": _git_data_violations(),
    }
    try:
        results_dir().mkdir(parents=True, exist_ok=True)
        (results_dir() / "test_results.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as exc:  # pragma: no cover - results tree unwritable
        print(f"WARNING: could not persist test_results.json: {exc}", file=sys.stderr)


_GIT_CACHE: dict[str, object] = {}


def _git_data_violations() -> int:
    """Count data/binary paths git would actually stage.

    Runs the real ``git add --dry-run`` so the answer reflects git's own
    matching rules rather than a reimplementation of them.
    """
    if "v" in _GIT_CACHE:
        return _GIT_CACHE["v"]  # type: ignore[return-value]
    import subprocess

    root = project_root()
    violations = 0
    try:
        inside = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        if inside.returncode != 0 or inside.stdout.strip() != "true":
            _GIT_CACHE["v"] = 0
            return 0
        dry = subprocess.run(
            ["git", "-C", str(root), "add", "-A", "--dry-run", "--verbose"],
            capture_output=True, text=True, timeout=120, check=False,
        )
        for line in dry.stdout.splitlines():
            if not line.startswith("add "):
                continue
            path = line.split("'", 2)[-1].rstrip("'") if "'" in line else line.split()[-1]
            p = Path(path)
            if p.suffix.lower() in {".npy", ".npz", ".zip", ".arff", ".csv.gz", ".pt", ".pth"} \
               or "/dataset/" in path or "/outputs/" in path:
                violations += 1
        ignored = subprocess.run(
            ["git", "-C", str(root), "status", "--ignored", "--porcelain"],
            capture_output=True, text=True, timeout=120, check=False,
        )
        if ignored.returncode == 0 and data_root().exists() and "!!" not in ignored.stdout:
            violations += 1
    except Exception:
        violations = 0
    _GIT_CACHE["v"] = violations
    return violations
