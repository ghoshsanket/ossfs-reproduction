"""Fail-closed authorization for the continuous Workstream B.

Two refusals matter and are tested:

* ``DLBCL``/``BREAST`` are forbidden everywhere, as in Stage 1/2/3A.
* A Stage-3A discrete dataset is refused here, so Workstream B cannot quietly
  become a second, differently-preprocessed copy of Workstream A.
"""

from __future__ import annotations

from typing import Any

from ossfs_dd.config import Stage2Plan
from ossfs_dd.runner import authorize as stage2_authorize

from . import ALLOWLIST, FORBIDDEN

STAGE3A_DATASETS = ("ARCENE", "DEXTER", "MADELON", "GINA", "GISETTE")


class ContinuousAuthError(RuntimeError):
    """Raised when a Workstream-B request must be refused."""


def authorize_continuous(key: str, *, seed: int, fold: int) -> Any:
    if key in FORBIDDEN:
        raise ContinuousAuthError(f"{key} is FORBIDDEN by researcher authorization")
    if key in STAGE3A_DATASETS:
        raise ContinuousAuthError(
            f"{key} is a Stage-3A discrete dataset and must not be read by Workstream B"
        )
    if key not in ALLOWLIST:
        raise ContinuousAuthError(f"{key} is not in the continuous allowlist {ALLOWLIST}")

    plan = Stage2Plan(
        allowlist=list(ALLOWLIST),
        forbidden=list(FORBIDDEN),
        required_manifest_status="EXACT_VERIFIED",
        primary_mode=None,  # type: ignore[arg-type] - unused; Stage 2 stays fail-closed
        routes={
            "Real": {
                "metric_family": "gaussian_fisher_z",
                "source_reference": "RECONSTRUCTED_FISHER_Z_NOT_SOURCE_VERIFIED",
                "semantics_status": "RECONSTRUCTED",
            },
            "Integer": {
                "metric_family": "gaussian_fisher_z",
                "source_reference": "RECONSTRUCTED_FISHER_Z_NOT_SOURCE_VERIFIED",
                "semantics_status": "RECONSTRUCTED",
            },
        },
        seed=seed,
        fold=fold,
        memory_limit_gib=32,
        max_walltime_minutes=60,
        gates={},
    )
    return stage2_authorize(key, plan)
