"""Stage 1a — acquire immutable raw bytes for every registry dataset.

Idempotent: an intact raw file already present is re-used by SHA-256 and never
re-fetched or overwritten. A partially downloaded file is safe to retry.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import requests

from ossfs_dd_data.fetch import (
    configure_logging,
    download_source,
    extract_source,
    raw_inventory,
    write_ledger,
)
from ossfs_dd_data.registry import load_registry, load_stage1_config
from ossfs_dd_data.workspace import ensure_data_dirs, rel_to_research, write_json

LOG = logging.getLogger("ossfs_dd_data.stage1.fetch")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 1: fetch raw dataset bytes")
    ap.add_argument("--all", action="store_true", help="fetch every registry dataset")
    ap.add_argument("--dataset", action="append", default=[], help="restrict to one key (repeatable)")
    args = ap.parse_args(argv)

    logfile = configure_logging("stage1_fetch.log")
    LOG.info("stage1_fetch starting; log also at %s", rel_to_research(logfile))

    cfg = load_stage1_config()
    registry = load_registry()
    specs = registry.select(args.dataset or None)
    if args.all or not args.dataset:
        specs = registry.datasets
    LOG.info("selected %d/%d datasets: %s", len(specs), len(registry.datasets), registry.keys())

    ledger: list[dict] = []
    failures = 0

    with requests.Session() as session:
        for spec in specs:
            ensure_data_dirs(spec.key)
            per_source = []
            ok_any = False
            for source in spec.sources:
                LOG.info("--- %s <- %s (%s) ---", spec.key, source.url, source.tier)
                result = download_source(spec, source, cfg, session=session)
                record = result.to_dict()
                record["tier"] = source.tier
                record["format"] = source.fmt
                if result.ok:
                    extraction = extract_source(spec, source, cfg, result)
                    record["extraction"] = extraction
                    ok_any = True
                else:
                    failures += 1
                    record["extraction"] = {"ok": False, "error": "download failed"}
                per_source.append(record)
                ledger.append(record)

            inventory = raw_inventory(spec.key)
            write_json(_manifest_dir() / f"{spec.key}.raw_inventory.json", inventory)
            LOG.info(
                "[%s] %s — %d raw file(s), %d source attempt(s)",
                spec.key,
                "OK" if ok_any else "DOWNLOAD_FAILED",
                len(inventory["files"]),
                len(per_source),
            )

    ledger_path = write_ledger("download_ledger.jsonl", ledger)
    LOG.info("download ledger: %s", rel_to_research(ledger_path))
    LOG.info("stage1_fetch complete: %d source attempt(s), %d failed", len(ledger), failures)
    return 0


def _manifest_dir() -> Path:
    from ossfs_dd_data.workspace import data_root

    d = data_root() / "manifests"
    d.mkdir(parents=True, exist_ok=True)
    return d


if __name__ == "__main__":
    raise SystemExit(main())
