"""Stage 1c — audit every prepared dataset and write its provenance manifest.

Reads only what earlier stages wrote. Writes manifests under the dataset tree
and a sanitised summary under results/. Exits non-zero when a hard gate fails,
but only after every report has been written.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from ossfs_dd_data.fetch import configure_logging
from ossfs_dd_data.gates import evaluate_gates
from ossfs_dd_data.provenance import build_manifest, write_manifest
from ossfs_dd_data.registry import (
    STATUS_DOWNLOAD_FAILED,
    STATUS_NOT_ATTEMPTED,
    load_registry,
    load_stage1_config,
)
from ossfs_dd_data.validate import audit_dataset
from ossfs_dd_data.workspace import (
    ensure_data_dirs,
    manifest_path,
    raw_dir,
    rel_to_research,
    resolve,
    results_dir,
    splits_dir,
    stream_orders_dir,
    utc_now,
    write_json,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_prepare import load_processed

LOG = logging.getLogger("ossfs_dd_data.stage1.verify")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 1: audit and assign statuses")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dataset", action="append", default=[])
    args = ap.parse_args(argv)

    logfile = configure_logging("stage1_verify.log")
    LOG.info("stage1_verify starting; log also at %s", rel_to_research(logfile))

    cfg = load_stage1_config()
    registry = load_registry()
    specs = registry.datasets if (args.all or not args.dataset) else registry.select(args.dataset)
    prepare_summary = _load_prepare_summary()

    results = results_dir()
    rows: list[dict] = []
    statuses: dict[str, str] = {}

    for spec in specs:
        ensure_data_dirs(spec.key)
        source = spec.primary_source
        inv_path = manifest_path(spec.key).parent / f"{spec.key}.raw_inventory.json"
        inventory = json.loads(inv_path.read_text(encoding="utf-8")) if inv_path.exists() else {"files": []}
        raw_files = inventory.get("files", [])
        raw_ok = any(f["path"].endswith(source.filename_in_archive) for f in raw_files) or bool(
            raw_dir(spec.key).exists() and any(raw_dir(spec.key).iterdir())
        )

        prep = next((d for d in prepare_summary if d["dataset"] == spec.key), None)

        if not raw_ok:
            status = STATUS_DOWNLOAD_FAILED
            reason = "no raw bytes present"
        elif prep is None:
            status = STATUS_NOT_ATTEMPTED
            reason = "no prepare record; stage1_prepare has not run for this dataset"
        else:
            status = prep.get("status", STATUS_NOT_ATTEMPTED)
            reason = prep.get("reason", "")

        parsed = None
        audit = None
        if raw_ok and prep is not None and status not in {"DOWNLOAD_FAILED", "PARSE_FAILED"}:
            try:
                X, y, meta = load_processed(spec)
                parsed = _AsParsed(X, y, meta)
                audit = audit_dataset(parsed, spec)
            except Exception as exc:  # noqa: BLE001
                status = "PARSE_FAILED"
                reason = f"reverification could not load the processed dataset: {type(exc).__name__}: {exc}"
                LOG.error("[%s] %s", spec.key, reason)

        manifest = build_manifest(
            spec,
            source,
            _download_record(spec, raw_files),
            _extraction_record(spec),
            parsed if parsed is not None else _Placeholder(spec),
            _artifact_record(spec),
            audit,
            status=status,
            status_reason=reason,
        )
        manifest["observed"]["split_plan_count"] = _count_files(splits_dir(spec.key), "*.json")
        manifest["observed"]["feature_order_plan_count"] = _count_files(stream_orders_dir(spec.key), "*.json")
        manifest["source"]["download"]["raw_file_inventory"] = raw_files
        write_manifest(manifest, manifest_path(spec.key))

        statuses[spec.key] = status
        rows.append(
            {
                "dataset": spec.key,
                "paper_n": spec.n_samples,
                "paper_p": spec.n_features,
                "paper_c": spec.n_classes,
                "actual_n": (audit or {}).get("n_samples"),
                "actual_p": (audit or {}).get("n_features"),
                "actual_c": (audit or {}).get("n_classes"),
                "source_url": source.url,
                "source_tier": source.tier,
                "sha256": (_download_record(spec, raw_files)).get("sha256"),
                "status": status,
                "reason": reason,
            }
        )
        LOG.info("[%s] %s — %s", spec.key, status, reason)

    write_json(results / "dataset_status.json", {"generated_utc": utc_now(), "datasets": rows})

    plan = cfg["evaluation_plan"]
    gates = evaluate_gates(cfg, len(registry.datasets), int(plan["n_splits"]), list(plan["seeds"]))
    write_json(results / "readiness_summary.json", gates)

    LOG.info("interim verdict: %s", gates["verdict"])
    for g in gates["gates"]:
        LOG.info("  gate %-48s %s", g["name"], "PASS" if g["pass"] else f"FAIL ({g['detail']})")

    data_only_failures = [
        g["name"] for g in gates["gates"]
        if not g["pass"] and g["name"] not in {
            "stream_leakage_tests_passing",
            "data_or_result_binaries_in_git",
            "required_tests_passing",
        }
    ]
    if data_only_failures:
        LOG.error(
            "data gates failed: %s. Run `uv run pytest` then "
            "`scripts/stage1_report.py`, which is the final verdict authority.",
            ", ".join(data_only_failures),
        )
        return 2
    LOG.info("all data gates pass; run pytest and then stage1_report.py for the final verdict")
    return 0


class _AsParsed:
    """Adapter exposing the ParsedDataset surface from reloaded artifacts."""

    def __init__(self, X, y, meta):
        self.X = X
        self.y = y
        self.meta = meta
        self.source_split = np.load(resolve(meta["source_split"]["path"]), allow_pickle=False).astype(str)
        self.source_row_index = np.load(resolve(meta["source_row_index"]["path"]), allow_pickle=False)
        self.label_map = meta.get("label_map_original_to_canonical", {})
        self.n_samples = int(X.shape[0])
        self.n_features = int(X.shape[1])
        self.orientation = "rows_are_samples"
        self.label_name = meta.get("label_name", "label")
        self.label_position = meta.get("label_position", "")
        self.dtype = str(X.dtype)
        self.sparse = not isinstance(X, np.ndarray)
        self.feature_dtype = meta.get("feature_dtype", "")
        self.feature_names = []
        self.notes = meta.get("notes", [])
        self.extra = meta.get("parsing_extra", {})


class _Placeholder:
    """Stand-in when a dataset never parsed, so the manifest still exists."""

    def __init__(self, spec):
        self.X = np.zeros((0, 0))
        self.y = np.array([], dtype=object)
        self.source_split = np.array([], dtype=object)
        self.source_row_index = np.array([], dtype=np.int64)
        self.label_map = {}
        self.n_samples = None
        self.n_features = None
        self.orientation = "not_applicable"
        self.label_name = None
        self.label_position = None
        self.dtype = None
        self.sparse = False
        self.feature_dtype = None
        self.feature_names = []
        self.notes = ["dataset was not parsed"]


def _download_record(spec, raw_files: list[dict]) -> dict:
    target = spec.primary_source.filename_in_archive
    match = next((f for f in raw_files if f["path"].endswith("/" + target)), None)
    if match is None:
        return {"ok": False, "error": "raw archive not found in inventory", "sha256": None, "bytes": None}
    return {
        "ok": True,
        "raw_archive": match["path"],
        "sha256": match["sha256"],
        "bytes": match["bytes"],
        "finished_utc": None,
        "url": spec.primary_source.url,
    }


def _extraction_record(spec) -> dict:
    d = raw_dir(spec.key) / "extracted"
    if d.exists():
        return {"ok": True, "extracted_root": rel_to_research(d)}
    return {"ok": False, "note": "no extraction directory (single-file source)"}


def _artifact_record(spec) -> dict:
    from ossfs_dd_data.workspace import processed_dir

    d = processed_dir(spec.key)
    if not d.exists():
        return {"note": "no processed artifacts"}
    return {
        "processed_root": rel_to_research(d),
        "files": sorted(p.name for p in d.iterdir() if p.is_file()),
    }


def _count_files(d: Path, pattern: str) -> int:
    if not d.exists():
        return 0
    return len(list(d.glob(pattern)))


def _load_prepare_summary() -> list[dict]:
    p = results_dir() / "prepare_summary.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8")).get("datasets", [])


if __name__ == "__main__":
    raise SystemExit(main())
