"""One-time, deterministic, versioned migration of Stage 1 artifacts.

Two v1 artifact defects are corrected here without touching raw data and
without altering any predictor value, label, split membership or feature
arrival order:

  ``stream_order/1.0``
      ``inverse_position_of_feature`` stored ``str(arrival_position) ->
      feature_index``, i.e. a copy of ``arrival_order``. The true inverse is
      ``str(feature_index) -> arrival_position``.

  ``processed/1.0``
      ``sample_ids.npy`` was an object-dtype array that could only be read with
      ``allow_pickle=True``.

Both migrations are:

  * **explicit** — nothing is rewritten implicitly by a normal rerun; this
    module is invoked deliberately and records what it did;
  * **versioned** — the new record carries ``schema: .../2.0``;
  * **deterministic** — output depends only on the input bytes;
  * **backed up** — the original file is copied to ``<name>.v1backup`` before
    replacement, and is never deleted;
  * **verified** — the migration refuses to write unless it can prove the
    scientific content (``arrival_order``, split indices, sample ids) is
    unchanged.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .registry import DatasetSpec
from .splits import build_feature_orders, feature_order_record
from .workspace import processed_dir, rel_to_research, stream_orders_dir, utc_now, write_json

LOG = logging.getLogger("ossfs_dd_data.migrations")

STREAM_ORDER_SCHEMA = "ossfs-dd-reproduction/stage1/stream_order/2.0"
PROCESSED_SCHEMA = "ossfs-dd-reproduction/stage1/processed/2.0"
BACKUP_SUFFIX = ".v1backup"


class MigrationError(RuntimeError):
    """Raised when a migration cannot prove it is safe to write."""


# --------------------------------------------------------------------------
# stream arrival-order plans
# --------------------------------------------------------------------------


def _is_v2(record: dict[str, Any]) -> bool:
    return record.get("schema") == STREAM_ORDER_SCHEMA


def _migrate_one_order_file(path: Path) -> dict[str, Any]:
    """Rewrite a single plan file in place, after backing it up."""
    original = json.loads(path.read_text(encoding="utf-8"))
    order = original.get("arrival_order")
    n_features = original.get("n_features")
    if not isinstance(order, list) or not isinstance(n_features, int):
        raise MigrationError(f"{path.name}: missing arrival_order/n_features; refusing to migrate")

    if sorted(order) != list(range(n_features)):
        raise MigrationError(
            f"{path.name}: arrival_order is not a permutation of 0..{n_features - 1}; "
            "refusing to migrate a corrupt plan"
        )

    dataset = original.get("dataset", path.parent.name)
    seed = int(original["seed"])

    # Deterministic re-derivation from the same preregistered seed.
    expected_order = build_feature_orders(n_features, [seed])[seed]
    if expected_order.tolist() != order:
        raise MigrationError(
            f"{path.name}: re-deriving seed {seed} did not reproduce the stored arrival_order. "
            "The plan was not produced by the preregistered generator; refusing to migrate."
        )

    new = feature_order_record(dataset, n_features, seed, np.asarray(order, dtype=np.int64))

    # The scientific payload must be identical, by construction.
    if new["arrival_order"] != original["arrival_order"]:
        raise MigrationError(f"{path.name}: arrival_order changed during migration")
    if original.get("arrival_order") == list(new["inverse_position_of_feature"].values()) and len(
        new["inverse_position_of_feature"]
    ) == n_features:
        # a v1 file in disguise; the inverse would equal the forward order only if
        # the permutation is the identity, which is not a valid arrival order.
        raise MigrationError(f"{path.name}: unexpected identity permutation; refusing to migrate")

    backup = path.with_suffix(path.suffix + BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(path, backup)
    write_json(path, new)
    return {
        "file": rel_to_research(path),
        "backup": rel_to_research(backup),
        "arrival_order_unchanged": True,
        "seed": seed,
        "n_features": n_features,
    }


def migrate_stream_orders(specs: list[DatasetSpec], seeds: list[int]) -> dict[str, Any]:
    """Migrate every existing stream-order plan to schema 2.0."""
    migrated, skipped, absent = [], [], []
    for spec in specs:
        d = stream_orders_dir(spec.key)
        if not d.exists():
            absent.append(spec.key)
            continue
        for seed in seeds:
            p = d / f"seed_{seed}.json"
            if not p.exists():
                absent.append(f"{spec.key}/seed_{seed}")
                continue
            rec = json.loads(p.read_text(encoding="utf-8"))
            if _is_v2(rec):
                skipped.append(spec.key)
                continue
            migrated.append(_migrate_one_order_file(p))
    LOG.info(
        "stream-order migration: %d migrated, %d already v2, %d absent",
        len(migrated), len(skipped), len(absent),
    )
    return {
        "migration": "stream_order/1.0 -> 2.0",
        "change": (
            "inverse_position_of_feature now maps str(feature_index) -> arrival_position, "
            "the true mathematical inverse of arrival_order"
        ),
        "arrival_order_modified": False,
        "migrated_files": migrated,
        "already_current": sorted(set(skipped)),
        "absent": absent,
        "backup_suffix": BACKUP_SUFFIX,
        "completed_utc": utc_now(),
    }


# --------------------------------------------------------------------------
# sample_ids
# --------------------------------------------------------------------------


def migrate_sample_ids(specs: list[DatasetSpec]) -> dict[str, Any]:
    """Convert object-dtype ``sample_ids.npy`` to fixed-width Unicode.

    The migration is non-lossy: the string values are compared before and after,
    and the old file is retained under ``<name>.v1backup``. No predictor value,
    label or row ordering is touched.
    """
    migrated, skipped, absent, problems = [], [], [], []
    for spec in specs:
        p = processed_dir(spec.key) / "sample_ids.npy"
        if not p.exists():
            absent.append(spec.key)
            continue
        old = np.load(p, allow_pickle=True)
        if old.dtype.kind == "U":
            skipped.append(spec.key)
            continue
        if old.dtype.kind != "O":
            raise MigrationError(
                f"{spec.key}: unexpected sample_ids dtype {old.dtype}; refusing to migrate"
            )

        values = [str(v) for v in old.tolist()]
        if len(values) != spec.n_samples:
            raise MigrationError(
                f"{spec.key}: sample_ids has {len(values)} entries but the registry declares "
                f"{spec.n_samples}; refusing to migrate"
            )
        expected = [f"{spec.key}:{i}" for i in range(spec.n_samples)]
        if values != expected:
            raise MigrationError(
                f"{spec.key}: stored sample_ids are not the canonical '<KEY>:<row>' identifiers; "
                "refusing to migrate"
            )

        new = np.asarray(values, dtype=np.str_)  # fixed-width Unicode, no pickle
        backup = p.with_suffix(p.suffix + BACKUP_SUFFIX)
        if not backup.exists():
            shutil.copy2(p, backup)
        np.save(p, new, allow_pickle=False)

        # non-lossy proof
        reread = np.load(p, allow_pickle=False)
        if reread.dtype.kind != "U" or reread.tolist() != values:
            raise MigrationError(f"{spec.key}: sample_ids migration was lossy")
        migrated.append(
            {
                "file": rel_to_research(p),
                "backup": rel_to_research(backup),
                "old_dtype": str(old.dtype),
                "new_dtype": str(reread.dtype),
                "bytes_before": backup.stat().st_size,
                "bytes_after": p.stat().st_size,
                "values_unchanged": True,
            }
        )
    LOG.info(
        "sample_ids migration: %d migrated, %d already fixed-width, %d absent",
        len(migrated), len(skipped), len(absent),
    )
    if problems:
        raise MigrationError("; ".join(problems))
    return {
        "migration": "processed/1.0 sample_ids object-dtype -> fixed-width Unicode",
        "change": "sample_ids.npy no longer requires allow_pickle=True to read",
        "raw_data_modified": False,
        "predictor_or_label_values_modified": False,
        "migrated_files": migrated,
        "already_current": sorted(set(skipped)),
        "absent": absent,
        "backup_suffix": BACKUP_SUFFIX,
        "completed_utc": utc_now(),
    }
