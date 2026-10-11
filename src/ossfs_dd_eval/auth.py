"""Fail-closed Stage-3A authorization, layered on Stage 1 and Stage 2.

Stage 2's ``authorize`` already refuses a forbidden or non-allowlisted key and
verifies the stored-matrix hash. Workstream A reuses it so the exact
authorization semantics cannot drift, then adds two Stage-3A-specific refusals:
the continuous Workstream-B datasets and any non-discrete metric family.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ossfs_dd.config import MetricFamily, Mode, Stage2Plan
from ossfs_dd.runner import authorize as stage2_authorize

STAGE2_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "stage2.yaml"


class Stage3AError(RuntimeError):
    """Raised when a Stage-3A request must be refused."""


def _stage2_routes() -> dict[str, Any]:
    """Reuse Stage-2's own Table-1 routing verbatim.

    Duplicating the route table here would let the two drift; reading Stage-2's
    config keeps Workstream A on exactly the routing Stage 2 verified. The
    ``metric_family`` strings are normalized to ``MetricFamily`` exactly as
    ``load_stage2_config`` does, so the Selector receives an enum.
    """
    routes = yaml.safe_load(STAGE2_CONFIG.read_text(encoding="utf-8"))["metric_routing"]["routes"]
    family_by_name = {m.value: m for m in MetricFamily}
    return {
        key: {**spec, "metric_family": family_by_name[spec["metric_family"]]}
        for key, spec in routes.items()
    }


def stage2_plan_for(key: str, config: dict[str, Any], *, seed: int, fold: int) -> Stage2Plan:
    """A Stage-2 plan pinned to the Stage-3A datasets and the given seed/fold."""
    allow = list(config["authorization"]["allowlist"])
    return Stage2Plan(
        allowlist=allow,
        forbidden=list(config["authorization"]["forbidden"]),
        required_manifest_status=config["authorization"]["require_manifest_status"],
        primary_mode=Mode.MATLAB_COMPATIBLE,
        routes=_stage2_routes(),
        seed=seed,
        fold=fold,
        memory_limit_gib=int(config["resources"]["memory_limit_gib"]),
        max_walltime_minutes=int(config["resources"]["per_selection_attempt_max_minutes"]),
        gates={},
    )


def authorize_run(key: str, config: dict[str, Any], *, seed: int, fold: int):
    """Authorize one Stage-3A run; returns the Stage-2 AuthorizedDataset."""
    if key in config["authorization"]["forbidden"]:
        raise Stage3AError(f"{key} is FORBIDDEN by researcher authorization")
    if key in config["authorization"]["continuous_datasets_excluded_from_A"]:
        raise Stage3AError(
            f"{key} is a continuous Workstream-B dataset and must not be read by Stage 3A"
        )
    if key not in config["authorization"]["allowlist"]:
        raise Stage3AError(f"{key} is not in the Stage-3A allowlist")

    plan = stage2_plan_for(key, config, seed=seed, fold=fold)
    auth = stage2_authorize(key, plan)
    family = getattr(auth.metric_family, "value", auth.metric_family)
    if family != MetricFamily.DISCRETE_INFORMATION.value:
        raise Stage3AError(f"{key}: Workstream A requires the discrete information family")
    return auth, plan
