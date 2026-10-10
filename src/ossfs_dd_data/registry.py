"""Declarative dataset registry.

Table 1 expectations are read from ``configs/dataset_registry.yaml`` and are
treated as an immutable specification. Nothing in this module (or anywhere else
in Stage 1) may write back to the expectations to match what was downloaded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .workspace import config_dir, project_name, project_root

# Stage 1 status vocabulary. Each dataset gets exactly one of these.
STATUS_EXACT_VERIFIED = "EXACT_VERIFIED"
STATUS_SHAPE_MATCH_PROVENANCE_UNVERIFIED = "SHAPE_MATCH_PROVENANCE_UNVERIFIED"
STATUS_SHAPE_MISMATCH = "SHAPE_MISMATCH"
STATUS_LABEL_MISMATCH = "LABEL_MISMATCH"
STATUS_DATA_QUALITY_BLOCKED = "DATA_QUALITY_BLOCKED"
STATUS_DOWNLOAD_FAILED = "DOWNLOAD_FAILED"
STATUS_PARSE_FAILED = "PARSE_FAILED"
STATUS_LICENSE_BLOCKED = "LICENSE_BLOCKED"
STATUS_NOT_ATTEMPTED = "NOT_ATTEMPTED"

ALL_STATUSES = frozenset(
    {
        STATUS_EXACT_VERIFIED,
        STATUS_SHAPE_MATCH_PROVENANCE_UNVERIFIED,
        STATUS_SHAPE_MISMATCH,
        STATUS_LABEL_MISMATCH,
        STATUS_DATA_QUALITY_BLOCKED,
        STATUS_DOWNLOAD_FAILED,
        STATUS_PARSE_FAILED,
        STATUS_LICENSE_BLOCKED,
        STATUS_NOT_ATTEMPTED,
    }
)

# Provenance tiers, strongest first.
TIER_ORDER = ["paper_cited", "official", "academic_repository", "mirror"]


@dataclass(frozen=True)
class SourceSpec:
    """One acquisition candidate for a dataset."""

    url: str
    tier: str
    fmt: str
    filename_in_archive: str
    expected_sha256: str | None = None
    verified_at_runtime: bool = False
    members: dict[str, list[str]] = field(default_factory=dict)
    notes: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SourceSpec:
        return cls(
            url=d["url"],
            tier=d["tier"],
            fmt=d["format"],
            filename_in_archive=d["filename_in_archive"],
            expected_sha256=None
            if d.get("expected_sha256", "unknown") == "unknown"
            else d["expected_sha256"],
            verified_at_runtime=bool(d.get("verified_at_runtime", False)),
            members={k: list(v) for k, v in (d.get("members") or {}).items()},
            notes=d.get("notes"),
        )

    def member(self, role: str) -> str | None:
        """First archive member matching ``role``, or None."""
        cands = self.members.get(role) or []
        return cands[0] if cands else None


@dataclass(frozen=True)
class DatasetSpec:
    """Immutable expectation plus acquisition plan for one Table 1 dataset."""

    id: int
    key: str
    name: str
    n_samples: int
    n_features: int
    n_classes: int
    paper_feature_type: str
    domain: str
    label_semantics: str
    source_splits: dict[str, Any]
    provenance_tier: str
    provenance_confidence: str
    provenance_note: str
    alternatives_rejected: list[dict[str, str]]
    citation: str
    license: str
    sources: list[SourceSpec]

    @property
    def primary_source(self) -> SourceSpec:
        return self.sources[0]

    @property
    def total_attributes(self) -> int:
        """Attributes expected in the source, i.e. features + the class column."""
        return self.n_features + 1

    def combine_splits(self) -> list[str]:
        return list(self.source_splits.get("combine") or [])

    def excluded_splits(self) -> list[str]:
        return list(self.source_splits.get("exclude") or [])

    def expected_split_counts(self) -> dict[str, int]:
        return dict(self.source_splits.get("counts") or {})

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DatasetSpec:
        return cls(
            id=int(d["id"]),
            key=d["key"],
            name=d["name"],
            n_samples=int(d["n_samples"]),
            n_features=int(d["n_features"]),
            n_classes=int(d["n_classes"]),
            paper_feature_type=d["paper_feature_type"],
            domain=d["domain"],
            label_semantics=d["label_semantics"],
            source_splits=dict(d.get("source_splits") or {}),
            provenance_tier=d["provenance_tier"],
            provenance_confidence=d["provenance_confidence"],
            provenance_note=d["provenance_note"],
            alternatives_rejected=list(d.get("alternatives_rejected") or []),
            citation=d["citation"],
            license=d["license"],
            sources=[SourceSpec.from_dict(s) for s in d["sources"]],
        )


@dataclass(frozen=True)
class Registry:
    schema_version: int
    paper: dict[str, Any]
    datasets: list[DatasetSpec]
    path: Path

    def keys(self) -> list[str]:
        return [d.key for d in self.datasets]

    def by_key(self, key: str) -> DatasetSpec:
        k = key.upper()
        for d in self.datasets:
            if d.key.upper() == k:
                return d
        raise KeyError(f"unknown dataset {key!r}; known: {self.keys()}")

    def select(self, keys: list[str] | None) -> list[DatasetSpec]:
        if not keys:
            return list(self.datasets)
        return [self.by_key(k) for k in keys]


def registry_path() -> Path:
    return config_dir() / "dataset_registry.yaml"


def load_registry(path: Path | None = None) -> Registry:
    p = path or registry_path()
    with open(p, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    datasets = [DatasetSpec.from_dict(d) for d in doc["datasets"]]
    return Registry(
        schema_version=int(doc.get("schema_version", 1)),
        paper=dict(doc.get("paper") or {}),
        datasets=datasets,
        path=p,
    )


# --------------------------------------------------------------------------
# Stage 1 protocol config
# --------------------------------------------------------------------------

DEFAULTS_STAGE1: dict[str, Any] = {
    "evaluation_plan": {
        "n_splits": 5,
        "seeds": [11, 17, 23, 31, 47, 59, 71, 83, 97, 109],
        "splitter": "sklearn.model_selection.StratifiedKFold",
        "shuffle": True,
        "feature_order_bit_generator": "PCG64",
    },
    "download": {
        "max_attempts_per_url": 3,
        "connect_timeout_seconds": 30,
        "read_timeout_seconds": 300,
        "backoff_base_seconds": 2.0,
        "backoff_max_seconds": 30.0,
        "retry_on_http_status": [429, 500, 502, 503, 504],
        "do_not_retry_on_http_status": [401, 403, 404, 410],
        "max_concurrent_downloads": 3,
        "user_agent": "ossfs-dd-reproduction/stage1",
        "reuse_existing_raw_by_sha256": True,
        "min_free_bytes_required": 500_000_000,
    },
    "extraction": {
        "allow_symlinks": False,
        "allow_hardlinks": False,
        "allow_absolute_paths": False,
        "allow_parent_traversal": False,
        "max_total_uncompressed_bytes": 2_000_000_000,
        "max_compression_ratio": 200,
        "max_members": 10_000,
    },
    "parsing": {
        "float_dtype": "float64",
        "sparse_if_density_below": 0.30,
        "label_position": "explicit",
    },
    "data_quality": {
        "allow_imputation": False,
        "allow_scaling": False,
        "allow_feature_dropping": False,
        "allow_deduplication": False,
    },
    "gates": {
        "inventory_attempted": {"required": 11, "of": 11},
        "exact_dimension_and_label_match": {"required": 11, "of": 11},
        "exact_verified_provenance": {"required": 11, "of": 11},
        "unresolved_corrupt_rows_or_columns": {"max_allowed": 0},
        "unresolved_nonfinite_or_missing_in_ready_data": {"max_allowed": 0},
        "unlabeled_challenge_test_examples_used": {"max_allowed": 0},
        "cv_test_folds_defined": {"required": 550},
        "feature_order_permutations_defined": {"required": 110},
        "stream_leakage_tests_passing": {"required_fraction": 1.0},
        "data_or_result_binaries_in_git": {"max_allowed": 0},
        "required_tests_passing": {"required_fraction": 1.0},
    },
    "verdict_values": ["STAGE1_READY", "STAGE1_PARTIAL", "STAGE1_BLOCKED"],
    "required_named_tests": [],
}


def stage1_config_path() -> Path:
    return config_dir() / "stage1.yaml"


def load_stage1_config(path: Path | None = None) -> dict[str, Any]:
    """Load the preregistered Stage 1 config.

    Missing keys fall back to the module defaults so a partially specified
    config still yields a usable object; anything explicitly written in YAML
    always wins, because gates must come from the file and not from code.
    """
    p = path or stage1_config_path()
    if not p.exists():
        raise FileNotFoundError(
            f"missing preregistered Stage 1 config at {p}; gates must be defined "
            "in config before any download"
        )
    with open(p, encoding="utf-8") as fh:
        user = yaml.safe_load(fh) or {}
    cfg = dict(DEFAULTS_STAGE1)
    for key, val in user.items():
        if isinstance(val, dict) and isinstance(cfg.get(key), dict):
            merged = dict(cfg[key])
            merged.update(val)
            cfg[key] = merged
        else:
            cfg[key] = val
    cfg["_config_path"] = str(p)
    cfg["_project_root"] = str(project_root())
    cfg["_project_name"] = project_name()
    return cfg


__all__ = [
    "ALL_STATUSES",
    "STATUS_DATA_QUALITY_BLOCKED",
    "STATUS_DOWNLOAD_FAILED",
    "STATUS_EXACT_VERIFIED",
    "STATUS_LABEL_MISMATCH",
    "STATUS_LICENSE_BLOCKED",
    "STATUS_NOT_ATTEMPTED",
    "STATUS_PARSE_FAILED",
    "STATUS_SHAPE_MATCH_PROVENANCE_UNVERIFIED",
    "STATUS_SHAPE_MISMATCH",
    "TIER_ORDER",
    "DatasetSpec",
    "Registry",
    "SourceSpec",
    "load_registry",
    "load_stage1_config",
]
