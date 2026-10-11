"""Stage-1 integration: the restricted dataset gate and the stream adapter.

Two responsibilities, kept separate:

``authorize``
    Fail closed. Only the nine researcher-authorized datasets, each with a
    manifest status of ``EXACT_VERIFIED``, correct Table-1 shape and a stored
    matrix hash matching the schema. Anything else — including a forbidden key
    supplied explicitly by a user — is refused here.

``run_dataset``
    Build a Stage-1 ``FeatureStream`` from the persisted seed-11 fold-0 plan
    and the persisted seed-11 ``arrival_order``, then drive the selector over
    it. The adapter never sees a held-out row: the stream is constructed with
    training indices only and the selector has no access to the full matrix.
"""

from __future__ import annotations

import hashlib
import json
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from ossfs_dd_data.stream import FeatureStream
from ossfs_dd_data.workspace import (
    manifest_path,
    processed_dir,
    resolve,
    splits_dir,
    stream_orders_dir,
)

from .config import (
    MetricFamily,
    Mode,
    SelectionConfig,
    Stage2Plan,
    UnsupportedSemanticsError,
)
from .reference_semantics import mode_record
from .selector import Selector

TABLE1 = {
    "SRBCT": (63, 2308, 4), "LYMPHOMA": (62, 4026, 3), "PROSTATE": (102, 6033, 2),
    "LEUKEMIA": (72, 7129, 2), "ARCENE": (200, 10000, 2), "DEXTER": (600, 20000, 2),
    "BREAST": (97, 24481, 2), "MADELON": (2600, 500, 2), "GINA": (3468, 970, 2),
    "GISETTE": (7000, 5000, 2),
}


class AuthorizationError(RuntimeError):
    """Raised when a dataset is not permitted for Stage 2."""


@dataclass
class AuthorizedDataset:
    key: str
    n_samples: int
    n_features: int
    n_classes: int
    paper_feature_type: str
    manifest_status: str
    metric_family: MetricFamily
    source_reference: str
    semantics_status: str


def authorize(key: str, plan: Stage2Plan) -> AuthorizedDataset:
    """Fail-closed authorization for one dataset key."""
    if key in plan.forbidden:
        raise AuthorizationError(
            f"dataset {key!r} is FORBIDDEN in Stage 2 by researcher authorization; "
            f"permitted: {plan.allowlist}"
        )
    if key not in plan.allowlist:
        raise AuthorizationError(
            f"dataset {key!r} is not in the Stage-2 allowlist {plan.allowlist}"
        )

    mp = manifest_path(key)
    if not mp.exists():
        raise AuthorizationError(f"dataset {key!r}: no Stage-1 manifest at {mp}")
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    status = manifest.get("stage1_verdict", {}).get("status")
    if status != plan.required_manifest_status:
        raise AuthorizationError(
            f"dataset {key!r}: manifest status is {status!r}, Stage 2 requires "
            f"{plan.required_manifest_status!r}"
        )

    obs = manifest.get("observed", {})
    exp_n, exp_p, exp_c = TABLE1[key]
    got = (obs.get("n_samples"), obs.get("n_features"), obs.get("n_classes"))
    if got != (exp_n, exp_p, exp_c):
        raise AuthorizationError(
            f"dataset {key!r}: manifest shape {got} does not match Table 1 "
            f"{(exp_n, exp_p, exp_c)}"
        )

    schema_path = processed_dir(key) / "schema.json"
    if not schema_path.exists():
        raise AuthorizationError(f"dataset {key!r}: no processed schema at {schema_path}")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    xp = resolve(schema["X"]["path"])
    if not xp.exists():
        raise AuthorizationError(f"dataset {key!r}: stored matrix missing at {schema['X']['path']}")
    digest = _sha256(xp)
    if digest != schema["X"]["sha256"]:
        raise AuthorizationError(
            f"dataset {key!r}: stored matrix hash mismatch (schema {schema['X']['sha256'][:16]}, "
            f"disk {digest[:16]})"
        )

    paper_type = manifest["expected_from_paper_table1"]["paper_feature_type"]
    route = plan.route_for(paper_type)
    return AuthorizedDataset(
        key=key,
        n_samples=exp_n, n_features=exp_p, n_classes=exp_c,
        paper_feature_type=paper_type,
        manifest_status=status,
        metric_family=route["metric_family"],
        source_reference=route["source_reference"],
        semantics_status=route["semantics_status"],
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_dataset(key: str) -> tuple[Any, np.ndarray, dict[str, Any]]:
    """Load through ``resolve`` so the working directory is irrelevant."""
    schema = json.loads((processed_dir(key) / "schema.json").read_text(encoding="utf-8"))
    xrec = schema["X"]
    X = (
        sparse.load_npz(str(resolve(xrec["path"])))
        if xrec["format"] == "scipy_csr"
        else np.load(resolve(xrec["path"]), allow_pickle=False)
    )
    y = np.load(resolve(schema["y"]["path"]), allow_pickle=False).astype(str)
    return X, y, schema


def build_stream(key: str, plan: Stage2Plan):
    """Stage-1 ``FeatureStream`` for the authorized seed/fold."""
    X, y, schema = load_dataset(key)
    split = json.loads(
        (splits_dir(key) / f"seed_{plan.seed}.json").read_text(encoding="utf-8")
    )
    fold = split["folds"][plan.fold]
    train_idx = np.asarray(fold["train"], dtype=np.int64)
    rec = json.loads(
        (stream_orders_dir(key) / f"seed_{plan.seed}.json").read_text(encoding="utf-8")
    )
    order = np.asarray(rec["arrival_order"], dtype=np.int64)
    if rec.get("schema") != "ossfs-dd-reproduction/stage1/stream_order/2.0":
        raise AuthorizationError(
            f"{key}: stream-order plan schema is {rec.get('schema')!r}; Stage 2 requires 2.0"
        )
    return X, y, FeatureStream(X, y, train_idx, order), train_idx


def run_dataset(
    key: str,
    plan: Stage2Plan,
    *,
    mode: Mode | None = None,
) -> dict[str, Any]:
    """Run one authorized dataset end to end. Never raises for a blocked family;
    records the structured reason instead."""
    auth = authorize(key, plan)
    chosen_mode = mode or plan.primary_mode
    cfg = SelectionConfig(mode=chosen_mode)

    entry = "mi2" if auth.metric_family is MetricFamily.DISCRETE_INFORMATION else "z2"
    meta = mode_record(
        chosen_mode,
        metric_family=auth.metric_family.value,
        source_reference=auth.source_reference,
        joint_metric=(
            "SU(A;Y) + cmi(A,Y,B)"
            if chosen_mode is Mode.MATLAB_COMPATIBLE
            else "SU({A,B};Y)"
        ),
        entry_file=f"OSFS_DD_{entry}.m",
    )

    result: dict[str, Any] = {
        "dataset": key,
        "n_samples": auth.n_samples,
        "n_features": auth.n_features,
        "n_classes": auth.n_classes,
        "paper_feature_type": auth.paper_feature_type,
        "metric_family": auth.metric_family.value,
        "semantics_status": auth.semantics_status,
        "mode": chosen_mode.value,
        "seed": plan.seed,
        "fold": plan.fold,
    }

    if auth.semantics_status == "UNRESOLVED":
        result.update(
            status="UNSUPPORTED_UNVERIFIED_SEMANTICS",
            error=(
                f"dataset {key!r} is paper-type {auth.paper_feature_type!r}, routed to "
                f"{auth.source_reference}, whose helper Fisher_Ztest is absent from the pinned "
                "authors' repository. No proxy statistic is substituted."
            ),
            n_selected=None, seconds=None, peak_rss_gib=None,
        )
        return result

    X, y, stream, train_idx = build_stream(key, plan)
    labels = stream.train_labels
    result["n_train"] = int(train_idx.size)
    result["n_test_withheld"] = int(X.shape[0] - train_idx.size)

    t0 = time.perf_counter()
    try:
        selector = Selector(
            cfg,
            metric_family=auth.metric_family,
            labels=labels,
            entry_file=entry,
            mode_metadata=meta,
        )
        selection = selector.fit_stream(iter(stream))
        elapsed = time.perf_counter() - t0
    except UnsupportedSemanticsError as exc:
        result.update(status="UNSUPPORTED_UNVERIFIED_SEMANTICS", error=str(exc))
        return result
    except MemoryError as exc:
        result.update(status="RESOURCE_BUDGET_EXCEEDED", error=f"MemoryError: {exc}")
        return result

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    summary = selection.summary()
    result.update(
        status="COMPLETED",
        error=None,
        seconds=round(elapsed, 3),
        peak_rss_gib=round(peak_rss, 3),
        n_selected=len(selection.feature_indices),
        selected_fraction=(len(selection.feature_indices) / auth.n_features)
        if auth.n_features else None,
        selected_feature_ids=selection.feature_indices,
        selected_arrival_positions=selection.arrival_positions,
        score_min=selection.score_min,
        score_max=selection.score_max,
        counts=summary["counts"],
        warnings=selection.warnings,
        mode_metadata=meta,
    )
    if peak_rss > plan.memory_limit_gib:
        result["status"] = "RESOURCE_BUDGET_EXCEEDED"
        result["error"] = f"peak RSS {peak_rss:.1f} GiB exceeded {plan.memory_limit_gib} GiB"
    if elapsed > plan.max_walltime_minutes * 60:
        result["status"] = "RESOURCE_BUDGET_EXCEEDED"
        result["error"] = f"elapsed {elapsed:.1f}s exceeded {plan.max_walltime_minutes} minutes"
    return result
