"""Provenance manifests, artifact hashing, and environment snapshots.

A manifest is the audit record for one dataset. It is written under the
workspace ``dataset/`` tree (never committed) and a sanitised summary is
mirrored into the project's ``results/`` tree (committed, small, no private
absolute paths).
"""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn
from scipy import sparse

from .fetch import sha256_file
from .registry import DatasetSpec, SourceSpec
from .workspace import rel_to_research, utc_now, write_json

SCHEMA = "ossfs-dd-reproduction/stage1/manifest/1.0"


def library_versions() -> dict[str, str]:
    import yaml

    try:
        import pyreadr  # type: ignore

        pyreadr_v = getattr(pyreadr, "__version__", "installed")
    except Exception:
        pyreadr_v = "not_installed"
    return {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "pyyaml": yaml.__version__,
        "pyreadr": pyreadr_v,
    }


def environment_snapshot() -> dict[str, Any]:
    """Sanitised host/environment record.

    Contains no credentials and no user identity: no HOME, no USER, no hostname,
    no private absolute paths outside the workspace root.
    """
    def _cmd(*args: str) -> str | None:
        try:
            return subprocess.run(
                args, capture_output=True, text=True, timeout=15, check=False
            ).stdout.strip() or None
        except Exception:
            return None

    return {
        "captured_utc": utc_now(),
        "architecture": platform.machine(),
        "system": platform.system(),
        "kernel_release": platform.release(),
        "cpu_count": _cpu_count(),
        "python_version": sys.version.split()[0],
        "python_executable_is_in_workspace_runtime": _runtime_check(),
        "cuda_visible_devices": "unset_or_empty",
        "gpu_used_by_stage1": False,
        "libraries": library_versions(),
    }


def _cpu_count() -> int | None:
    import os

    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux
        return os.cpu_count()


def _runtime_check() -> bool:
    try:
        from .workspace import research_root

        return str(research_root()) in sys.executable
    except Exception:
        return False


def git_commit(project_dir: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(project_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def build_manifest(
    spec: DatasetSpec,
    source: SourceSpec,
    download: dict[str, Any],
    extraction: dict[str, Any],
    parsed: Any,
    artifacts: dict[str, Any],
    audit: dict[str, Any] | None = None,
    *,
    status: str,
    status_reason: str,
    stage: str = "stage1",
) -> dict[str, Any]:
    """Assemble the full provenance manifest for one dataset."""
    return {
        "schema": SCHEMA,
        "stage": stage,
        "generated_utc": utc_now(),
        "dataset": {
            "id": spec.id,
            "key": spec.key,
            "name": spec.name,
            "domain": spec.domain,
            "label_semantics": spec.label_semantics,
            "citation": spec.citation,
            "license": spec.license,
        },
        "expected_from_paper_table1": {
            "n_samples": spec.n_samples,
            "n_features": spec.n_features,
            "n_classes": spec.n_classes,
            "paper_feature_type": spec.paper_feature_type,
            "features_exclude_class_column": True,
            "note": "Immutable preregistered specification. Never edited to match a download.",
        },
        "source": {
            "tier": source.tier,
            "url": source.url,
            "format": source.fmt,
            "filename_in_archive": source.filename_in_archive,
            "expected_sha256": source.expected_sha256,
            "notes": source.notes,
            "download": download,
            "extraction": extraction,
            "raw_immutable": True,
        },
        "provenance_assessment": {
            "tier": spec.provenance_tier,
            "confidence": spec.provenance_confidence,
            "note": spec.provenance_note,
            "alternatives_considered_and_rejected": spec.alternatives_rejected,
            "paper_cited_collections": [],
        },
        "observed": {
            "n_samples": parsed.n_samples,
            "n_features": parsed.n_features,
            "n_classes": len(np.unique(parsed.y)),
            "orientation": parsed.orientation,
            "label_name": parsed.label_name,
            "label_position": parsed.label_position,
            "label_position_rule": "explicit registry/arff declaration, not last-column guess",
            "label_map_original_to_canonical": parsed.label_map,
            "class_counts": {
                str(k): int(v)
                for k, v in zip(*np.unique(parsed.y.astype(str), return_counts=True), strict=False)
            },
            "source_split_counts": _count(parsed.source_split),
            "source_split_rows": _count(parsed.source_row_index, key="split"),
            "storage": "scipy_csr" if sparse.issparse(parsed.X) else "numpy_dense",
            "storage_dtype": parsed.dtype,
            "feature_dtype": parsed.feature_dtype,
            "feature_names_available": bool(parsed.feature_names and parsed.feature_names[0] != ""),
            "feature_name_scheme": (
                "source_attribute_names"
                if parsed.feature_names and not parsed.feature_names[0].startswith("feature_")
                else "positional_identity_0_to_p_minus_1"
            ),
            "parsing_notes": parsed.notes,
            "parsing_extra": parsed.extra,
            "transformations_applied_by_this_pipeline": [],
            "preexisting_preprocessing_in_source": (
                "Binghamton '-std' release: values arrive standardised by the source "
                "collection. Documented as preexisting, NOT applied here."
                if spec.key in {"SRBCT", "LYMPHOMA", "PROSTATE", "LEUKEMIA"}
                else "none documented"
            ),
            "transient_dates": {
                "observation_window": "not documented — historical cross-sectional public benchmarks",
                "sample_observation_dates": "not documented",
                "source_release_version": "not published by the source",
                "download_timestamp_utc": download.get("finished_utc"),
            },
        },
        "artifacts": artifacts,
        "data_quality_audit": audit or {"status": "not_run"},
        "environment": environment_snapshot(),
        "stage1_verdict": {"status": status, "reason": status_reason},
    }


def _count(arr: np.ndarray, key: str | None = None) -> dict[str, int]:
    vals, counts = np.unique(arr.astype(str), return_counts=True)
    return {str(v): int(c) for v, c in zip(vals, counts, strict=False)}


def write_manifest(manifest: dict[str, Any], path: Path) -> Path:
    return write_json(path, manifest)


def artefact_record(path: Path, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "path": rel_to_research(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        **(extra or {}),
    }


def sanitised_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    """Small, path-safe projection of a manifest for the committed results tree."""
    obs = manifest.get("observed", {})
    src = manifest.get("source", {})
    return {
        "dataset": manifest["dataset"]["key"],
        "expected_n": manifest["expected_from_paper_table1"]["n_samples"],
        "expected_p": manifest["expected_from_paper_table1"]["n_features"],
        "expected_c": manifest["expected_from_paper_table1"]["n_classes"],
        "observed_n": obs.get("n_samples"),
        "observed_p": obs.get("n_features"),
        "observed_c": obs.get("n_classes"),
        "source_url": src.get("url"),
        "source_tier": src.get("tier"),
        "sha256": (src.get("download") or {}).get("sha256"),
        "bytes": (src.get("download") or {}).get("bytes"),
        "download_utc": (src.get("download") or {}).get("finished_utc"),
        "source_split_counts": obs.get("source_split_counts"),
        "class_counts": obs.get("class_counts"),
        "provenance_tier": manifest.get("provenance_assessment", {}).get("tier"),
        "provenance_confidence": manifest.get("provenance_assessment", {}).get("confidence"),
        "status": manifest.get("stage1_verdict", {}).get("status"),
        "status_reason": manifest.get("stage1_verdict", {}).get("reason"),
    }
