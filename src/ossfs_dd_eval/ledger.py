"""Resumable, content-addressed run ledger.

Every completed (dataset, seed, fold) selection appends one immutable JSON
record named by its run ID. A resumed sweep reuses a record only when the run ID,
the frozen protocol digest and the recorded code hash all match; otherwise the
record is recomputed rather than trusted.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

SCHEMA = "ossfs-dd-reproduction/stage3a/record/1.0"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class Ledger:
    def __init__(self, root: Path, *, protocol_digest: str, code_hash: str) -> None:
        self.root = Path(root)
        self.records = self.root / "records"
        self.protocol_digest = protocol_digest
        self.code_hash = code_hash

    def _path(self, run_id: str) -> Path:
        return self.records / f"{run_id}.json"

    def load(self, run_id: str) -> dict[str, Any] | None:
        path = self._path(run_id)
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if record.get("schema") != SCHEMA:
            return None
        if record.get("protocol_digest") != self.protocol_digest:
            return None
        if record.get("code_hash") != self.code_hash:
            return None
        return record

    def save(self, record: dict[str, Any]) -> None:
        payload = dict(record)
        payload.setdefault("schema", SCHEMA)
        payload["protocol_digest"] = self.protocol_digest
        payload["code_hash"] = self.code_hash
        _atomic_write_json(self._path(payload["run_id"]), payload)

    def completed_ids(self) -> set[str]:
        return {p.stem for p in self.records.glob("*.json")}

    def count(self) -> int:
        return len(self.completed_ids())
