"""Environment, code and dataset digests recorded with every Stage-3A artifact.

Nothing here decides anything; it only fixes *what was run, with which code and
which data* so a record can be audited or a sweep safely resumed.
"""

from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRS = ("src/ossfs_dd", "src/ossfs_dd_data", "src/ossfs_dd_eval", "src/ossfs_dd_continuous")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def code_hash(dirs: tuple[str, ...] = SOURCE_DIRS) -> str:
    """SHA-256 over the sorted (relative path, content hash) of the sources."""
    parts: list[str] = []
    for d in dirs:
        base = PROJECT_ROOT / d
        if not base.exists():
            continue
        for f in sorted(base.rglob("*.py")):
            parts.append(f"{f.relative_to(PROJECT_ROOT).as_posix()}:{_sha256_file(f)}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def package_versions() -> dict[str, str]:
    import numpy
    import scipy
    import sklearn

    return {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def dataset_fingerprints(keys: tuple[str, ...]) -> dict[str, Any]:
    """Re-verify each authorized dataset's stored-matrix hash against its schema."""
    import json

    from ossfs_dd_data.workspace import manifest_path, processed_dir, resolve

    out: dict[str, Any] = {}
    for key in keys:
        try:
            schema = json.loads((processed_dir(key) / "schema.json").read_text(encoding="utf-8"))
            digest = _sha256_file(resolve(schema["X"]["path"]))
            manifest = json.loads(manifest_path(key).read_text(encoding="utf-8"))
            out[key] = {
                "sha256_recorded": schema["X"]["sha256"],
                "sha256_observed": digest,
                "match": digest == schema["X"]["sha256"],
                "stage1_status": manifest["stage1_verdict"]["status"],
            }
        except Exception as exc:  # noqa: BLE001 - recorded, never silently ignored
            out[key] = {"error": f"{type(exc).__name__}: {exc}"}
    return out
