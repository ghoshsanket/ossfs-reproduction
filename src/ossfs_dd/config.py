"""Frozen Stage-2 configuration, mode enum, and typed configuration objects."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ossfs_dd_data.workspace import config_dir, project_name, project_root


class UnsupportedSemanticsError(RuntimeError):
    """Raised when a metric family has no source-verified definition.

    The Stage-2 prompt forbids substituting a plausible proxy statistic for
    one whose reference implementation could not be inspected. This error is
    how that refusal is made explicit rather than silent.
    """

    def __init__(self, metric_family: str, reason: str, source: str = "") -> None:
        self.metric_family = metric_family
        self.reason = reason
        self.source = source
        msg = (
            f"UNSUPPORTED_UNVERIFIED_SEMANTICS: metric_family={metric_family!r}. {reason}"
        )
        if source:
            msg += f" (reference: {source})"
        super().__init__(msg)


class Mode(enum.StrEnum):
    PAPER_SPEC = "paper_spec_v1"
    MATLAB_COMPATIBLE = "matlab_compatible_v1"


class MetricFamily(enum.StrEnum):
    DISCRETE_INFORMATION = "discrete_information"
    GAUSSIAN_FISHER_Z = "gaussian_fisher_z"


@dataclass(frozen=True)
class SelectionConfig:
    """Algorithm parameters. Defaults are the paper's Sec. 4.2 choices."""

    mode: Mode = Mode.MATLAB_COMPATIBLE
    k1: float = 1.0
    k2: float = 2.0
    pool_size: int = 100
    log_base: float = 2.0

    def __post_init__(self) -> None:
        if not self.k1 < self.k2:
            raise ValueError(
                f"k1 ({self.k1}) must be strictly less than k2 ({self.k2}); the "
                "three-way decision is undefined otherwise and this must not be relaxed"
            )
        if self.pool_size < 1:
            raise ValueError(f"pool_size must be >= 1, got {self.pool_size}")
        if self.log_base <= 1.0:
            raise ValueError(f"log_base must be > 1, got {self.log_base}")

    @property
    def requires_two_beta(self) -> float:
        return 2.0  # documented literal from mi2.m:117 and :138


@dataclass
class Stage2Plan:
    """The frozen `configs/stage2.yaml`, parsed."""

    allowlist: list[str]
    forbidden: list[str]
    required_manifest_status: str
    primary_mode: Mode
    routes: dict[str, dict[str, Any]]
    seed: int
    fold: int
    memory_limit_gib: int
    max_walltime_minutes: int
    gates: dict[str, Any]
    raw: dict[str, Any] = field(default_factory=dict)
    path: str = ""

    def route_for(self, paper_feature_type: str) -> dict[str, Any]:
        try:
            return self.routes[paper_feature_type]
        except KeyError as exc:
            raise KeyError(
                f"no metric route preregistered for paper feature type "
                f"{paper_feature_type!r}; registered routes: {sorted(self.routes)}"
            ) from exc


def stage2_config_path() -> Path:
    return config_dir() / "stage2.yaml"


def load_stage2_config(path: Path | None = None) -> Stage2Plan:
    p = path or stage2_config_path()
    if not p.exists():
        raise FileNotFoundError(
            f"missing frozen Stage-2 config at {p}. The config must exist and be frozen "
            "before any real-data smoke run."
        )
    doc = yaml.safe_load(p.read_text(encoding="utf-8"))

    auth = doc["authorization"]
    plan = doc["run_plan"]
    modes = doc["modes"]
    res = doc["resources"]
    routes = doc["metric_routing"]["routes"]

    # Route families by name so config changes are honoured, not hard-coded.
    family_by_name = {m.value: m for m in MetricFamily}
    normalised_routes: dict[str, dict[str, Any]] = {}
    for key, spec in routes.items():
        entry = dict(spec)
        entry["metric_family"] = family_by_name[entry["metric_family"]]
        normalised_routes[key] = entry

    return Stage2Plan(
        allowlist=list(auth["allowlist"]),
        forbidden=list(auth["forbidden"]),
        required_manifest_status=auth["require_manifest_status"],
        primary_mode=Mode(modes["primary"]),
        routes=normalised_routes,
        seed=int(plan["seeds"][0]),
        fold=int(plan["folds"][0]),
        memory_limit_gib=int(res["memory_limit_gib"]),
        max_walltime_minutes=int(res["max_walltime_per_smoke_minutes"]),
        gates=doc["gates"],
        raw=doc,
        path=str(p),
    )


def package_paths() -> dict[str, str]:
    return {"project": project_name(), "project_root": str(project_root())}


def dataset_paper_type(spec) -> str:
    """Route by the paper's Table 1 wording, never by numpy dtype."""
    return spec.paper_feature_type
