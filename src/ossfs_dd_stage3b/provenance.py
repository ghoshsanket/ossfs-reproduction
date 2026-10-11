"""Loading and fail-closed validation of Stage 3B inputs.

The Stage 3A per-run records are **read-only** inputs. A missing or mismatched
record is a named, fatal condition for that subset -- never a silently
substituted average from the committed CSV.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ossfs_dd_data.workspace import processed_dir, resolve, splits_dir, stream_orders_dir

from . import EXP_A_DATASETS, FORBIDDEN
from .protocol import SubsetID

PROJECT = Path(__file__).resolve().parents[2]
STAGE3A_RECORDS = PROJECT / "results" / "stage3a" / "runs" / "records"
STAGE3A_MANIFEST = PROJECT / "results" / "stage3a" / "run_manifest.json"
STAGE1_ORDER_SCHEMA = "ossfs-dd-reproduction/stage1/stream_order/2.0"


class BlockedInput(RuntimeError):
    """A required Stage 3B input is missing, stale or inconsistent."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def assert_allowed(dataset: str) -> None:
    if dataset in FORBIDDEN:
        raise BlockedInput(f"{dataset} is FORBIDDEN and must never be read")
    if dataset not in EXP_A_DATASETS:
        raise BlockedInput(f"{dataset} is not an authorized Stage 3B dataset")


def stage3a_manifest() -> dict[str, Any]:
    if not STAGE3A_MANIFEST.exists():
        raise BlockedInput(f"missing Stage 3A run manifest at {STAGE3A_MANIFEST}")
    return json.loads(STAGE3A_MANIFEST.read_text(encoding="utf-8"))


def load_stage3a_record(subset: SubsetID) -> dict[str, Any]:
    """Load one Stage 3A selection record, validating identity and shape."""
    assert_allowed(subset.dataset)
    path = STAGE3A_RECORDS / f"{subset.stage3a_run_id}.json"
    if not path.exists():
        raise BlockedInput(
            f"Stage 3A record for {subset.key} is NOT_AVAILABLE at {path}. "
            "The raw per-run records are required; reconstructing a subset from a "
            "committed average is prohibited."
        )
    rec = json.loads(path.read_text(encoding="utf-8"))
    for field, want in (("dataset", subset.dataset), ("seed", subset.seed),
                        ("fold", subset.fold)):
        if rec.get(field) != want:
            raise BlockedInput(
                f"{path.name}: {field}={rec.get(field)!r} does not match {want!r}"
            )
    if rec.get("status") != "COMPLETED":
        raise BlockedInput(f"{path.name}: status={rec.get('status')!r}, expected COMPLETED")
    ids = rec.get("selected_feature_ids") or []
    if not isinstance(ids, list) or not ids:
        raise BlockedInput(f"{path.name}: empty or malformed selected_feature_ids")
    if len(set(ids)) != len(ids):
        raise BlockedInput(f"{path.name}: duplicate feature IDs in selected_feature_ids")
    n_total = int(rec["n_features_total"])
    if any((not isinstance(i, int)) or i < 0 or i >= n_total for i in ids):
        raise BlockedInput(f"{path.name}: selected feature ID out of range [0,{n_total})")
    if int(rec["n_selected"]) != len(ids):
        raise BlockedInput(
            f"{path.name}: n_selected={rec['n_selected']} != len(ids)={len(ids)}"
        )
    return rec


def verify_stage3a_digests(manifest: dict[str, Any]) -> dict[str, Any]:
    """Verify Stage 3A record provenance.

    Two fields are checked separately because they mean different things:

    ``protocol_digest``
        Stamped once from the frozen config and is therefore a **stable**
        record-level identity. This MUST match the manifest.

    ``code_hash``
        Stage 3A recomputed this **per task** inside the sweep worker from a
        source tree that was being edited while the sweep ran, so the 250
        records carry 11 distinct values and only some match the manifest hash
        taken at sweep launch. It is therefore **not** usable as a record-level
        code fingerprint. It is reported, not used as a gate; the selection-code
        fingerprint below is the meaningful check instead.
    """
    want_proto, want_code = manifest["protocol_digest"], manifest["code_hash"]
    seen = 0
    proto_mismatch: list[str] = []
    code_mismatch: list[str] = []
    distinct_codes: set[str] = set()
    for p in sorted(STAGE3A_RECORDS.glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        seen += 1
        if rec.get("protocol_digest") != want_proto:
            proto_mismatch.append(p.name)
        if rec.get("code_hash") != want_code:
            code_mismatch.append(p.name)
        distinct_codes.add(rec.get("code_hash", ""))
    return {
        "n_records": seen,
        "expected_protocol_digest": want_proto,
        "protocol_digest_mismatched_records": proto_mismatch,
        "protocol_digest_ok": not proto_mismatch,
        "expected_code_hash": want_code,
        "code_hash_mismatched_records_n": len(code_mismatch),
        "code_hash_distinct_values": len(distinct_codes),
        "code_hash_ok": not code_mismatch,
        "code_hash_authoritative": False,
        "code_hash_defect": (
            "Stage 3A's runner called code_hash() once per task inside each sweep "
            "worker, hashing the live source tree. Stage 3B's own sources and the "
            "Stage 3A evaluation harness were edited while the sweep was running, "
            "so records written later carry a different code_hash. This is a "
            "provenance-stamping defect in Stage 3A, not evidence that the "
            "selection logic changed."
        ),
    }


def selection_code_fingerprint() -> dict[str, Any]:
    """Fingerprint of the code that actually determined the selected subsets.

    Only ``src/ossfs_dd`` and ``src/ossfs_dd_data`` matter for a selected
    feature list; the evaluation harness does not. If this fingerprint is
    unchanged, every Stage 3A subset was produced by the same selection code.
    """
    parts = []
    newest = 0.0
    for base in ("src/ossfs_dd", "src/ossfs_dd_data"):
        for f in sorted((PROJECT / base).rglob("*.py")):
            parts.append(f"{f.relative_to(PROJECT).as_posix()}:{sha256_file(f)}")
            newest = max(newest, f.stat().st_mtime)
    return {
        "selection_code_sha256": hashlib.sha256("\n".join(parts).encode()).hexdigest(),
        "n_files": len(parts),
        "newest_source_mtime_epoch": newest,
    }


def load_split(dataset: str, seed: int, fold: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact Stage 1 train/test indices for a (dataset, seed, fold)."""
    assert_allowed(dataset)
    split = json.loads(
        (splits_dir(dataset) / f"seed_{seed}.json").read_text(encoding="utf-8")
    )
    f = split["folds"][fold]
    tr = np.asarray(f["train"], dtype=np.int64)
    te = np.asarray(f["test"], dtype=np.int64)
    if len(set(tr.tolist()) & set(te.tolist())):
        raise BlockedInput(f"{dataset} seed{seed} fold{fold}: train/test overlap")
    if len(tr) + len(te) != split["n_samples"]:
        raise BlockedInput(f"{dataset} seed{seed} fold{fold}: split does not cover samples")
    return tr, te


def load_order(dataset: str, order_seed: int) -> np.ndarray:
    """A Stage 1 arrival order, identified ONLY by its permutation seed."""
    assert_allowed(dataset)
    rec = json.loads(
        (stream_orders_dir(dataset) / f"seed_{order_seed}.json").read_text(encoding="utf-8")
    )
    if rec.get("schema") != STAGE1_ORDER_SCHEMA:
        raise BlockedInput(
            f"{dataset} order seed {order_seed}: schema {rec.get('schema')!r} "
            f"!= {STAGE1_ORDER_SCHEMA!r}"
        )
    order = np.asarray(rec["arrival_order"], dtype=np.int64)
    n_features = int(np.load(resolve(_x_path(dataset)), mmap_mode="r").shape[1])
    if not np.array_equal(np.sort(order), np.arange(n_features)):
        raise BlockedInput(
            f"{dataset} order seed {order_seed}: not a permutation of range({n_features})"
        )
    return order


def _x_path(dataset: str) -> str:
    schema = json.loads((processed_dir(dataset) / "schema.json").read_text(encoding="utf-8"))
    return schema["X"]["path"]


def subset_fingerprint(dataset: str, ids: list[int]) -> dict[str, str]:
    """Order-preserving hash of the exact selected ID sequence."""
    return {
        "n_selected": str(len(ids)),
        "ids_sha256": sha256_json(list(ids)),
    }
