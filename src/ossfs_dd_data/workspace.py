"""Workspace paths and small shared helpers.

All paths are resolved from the environment variables exported by the
workspace's ``scripts/research-env.sh``. No user-specific absolute path is
hard-coded anywhere in this package.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class WorkspaceError(RuntimeError):
    """Raised when the workspace contract is not satisfied."""


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise WorkspaceError(
            f"{name} is not set. Source the workspace environment first:\n"
            "  export RESEARCH_PROJECT_NAME=ossfs-dd-reproduction\n"
            "  cd \"$RESEARCH_ROOT\" && source scripts/research-env.sh"
        )
    return value


def research_root() -> Path:
    return Path(_require("RESEARCH_ROOT"))


def project_root() -> Path:
    return Path(_require("RESEARCH_PROJECT_ROOT"))


def project_name() -> str:
    return _require("RESEARCH_PROJECT_NAME")


def data_root() -> Path:
    """``$RESEARCH_DATA_ROOT/<project>/`` — raw and processed datasets."""
    return Path(_require("RESEARCH_DATA_ROOT")) / project_name()


def output_root() -> Path:
    """``$RESEARCH_OUTPUT_ROOT/<project>/`` — run artifacts and logs."""
    return Path(_require("RESEARCH_OUTPUT_ROOT")) / project_name()


def stage_output_root(stage: str = "stage1") -> Path:
    return output_root() / stage


def log_dir(stage: str = "stage1") -> Path:
    d = stage_output_root(stage) / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_dir(stage: str = "stage1") -> Path:
    d = project_root() / "results" / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def config_dir() -> Path:
    return project_root() / "configs"


def raw_dir(dataset: str) -> Path:
    return data_root() / "raw" / dataset


def processed_dir(dataset: str) -> Path:
    return data_root() / "processed" / dataset


def splits_dir(dataset: str) -> Path:
    return data_root() / "splits" / dataset


def stream_orders_dir(dataset: str) -> Path:
    return data_root() / "stream_orders" / dataset


def manifest_path(dataset: str) -> Path:
    return data_root() / "manifests" / f"{dataset}.json"


def ensure_data_dirs(dataset: str) -> None:
    for d in (
        raw_dir(dataset),
        processed_dir(dataset),
        splits_dir(dataset),
        stream_orders_dir(dataset),
        data_root() / "manifests",
    ):
        d.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    """ISO-8601 UTC timestamp, second resolution, always suffixed with Z."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def free_bytes(path: Path) -> int:
    st = os.statvfs(path if path.exists() else path.parent)
    return st.f_bavail * st.f_frsize


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, payload: Any, *, indent: int = 2) -> Path:
    """Write JSON atomically. Never overwrite a different existing version
    silently: callers decide, this only guarantees the write itself is atomic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=indent, sort_keys=False, default=_json_default)
        fh.write("\n")
    os.replace(tmp, path)
    return path


def _json_default(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def rel_to_research(path: Path) -> str:
    """Path relative to RESEARCH_ROOT, for reports that must not leak private
    absolute paths into committed artefacts."""
    try:
        return str(Path(path).resolve().relative_to(research_root().resolve()))
    except ValueError:
        return str(path)


def resolve(stored: str | Path) -> Path:
    """Resolve a path recorded in a manifest.

    Manifest paths are stored relative to ``RESEARCH_ROOT`` so that no private
    absolute path leaks into a committed artefact. This turns one back into a
    usable absolute path regardless of the caller's working directory.
    """
    p = Path(stored)
    if p.is_absolute():
        return p
    return research_root() / p
