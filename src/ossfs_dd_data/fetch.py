"""Streaming acquisition: download, hash, and safely extract.

Guarantees enforced here:
  * every byte written lands under the workspace TEMP area or the dataset raw
    directory, never under ``~``, ``/tmp`` or another user's tree;
  * a body is streamed to ``<name>.partial`` and atomically renamed only after it
    completes, so a partial file can never masquerade as a dataset;
  * an HTML error page, a 0-byte body, a GitHub directory listing, a truncated
    archive or a login wall is reported as a FAILURE, not as data;
  * raw files are immutable once present and are re-used by SHA-256;
  * archive extraction refuses path traversal, escaping links and bombs.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tarfile
import time
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from .registry import DatasetSpec, SourceSpec
from .workspace import free_bytes, log_dir, raw_dir, rel_to_research, utc_now

LOG = logging.getLogger("ossfs_dd_data.fetch")

ZIP_MAGICS = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
GZIP_MAGIC = b"\x1f\x8b"
BZIP_MAGIC = b"BZh"
XZ_MAGIC = b"\xfd7zXZ\x00"
# Compared against a lowercased head, so all entries must be lowercase.
HTML_HINTS = (b"<!doctype html", b"<html")
TEXT_HINTS = (
    b"sign in",
    b"log in",
    b"login required",
    b"access denied",
    b"403 forbidden",
    b"internal server error",
)

_CHUNK = 1024 * 256


@dataclass
class DownloadResult:
    """Everything observed about one URL attempt. Failures are preserved."""

    dataset: str
    url: str
    ok: bool
    attempts: list[dict[str, Any]] = field(default_factory=list)
    path: Path | None = None
    sha256: str | None = None
    bytes: int | None = None
    error: str | None = None
    started_utc: str = field(default_factory=utc_now)
    finished_utc: str | None = None
    reused_existing: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "url": self.url,
            "ok": self.ok,
            "attempts": self.attempts,
            "raw_path": rel_to_research(self.path) if self.path else None,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "error": self.error,
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "reused_existing": self.reused_existing,
        }


def sha256_file(path: Path, *, chunk: int = _CHUNK) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sniff(path: Path) -> str:
    with open(path, "rb") as fh:
        head = fh.read(2048)
    return head


def looks_like_archive(path: Path) -> str | None:
    head = sniff(path)
    if any(head.startswith(m) for m in ZIP_MAGICS):
        return "zip"
    if head.startswith(GZIP_MAGIC):
        return "gzip"
    if head.startswith(BZIP_MAGIC):
        return "bzip2"
    if head.startswith(XZ_MAGIC):
        return "xz"
    return None


def assert_not_error_page(path: Path) -> None:
    """Reject the three most common false-positive downloads."""
    size = path.stat().st_size
    if size == 0:
        raise ValueError("0-byte response: the server returned an empty body")
    head = sniff(path).lstrip()[:512].lower()
    if head.startswith(HTML_HINTS):
        raise ValueError(
            "HTML response body, not a dataset (likely an error page or a "
            "directory listing)"
        )
    low = head.decode("latin-1", "replace")
    for hint in TEXT_HINTS:
        if hint.decode() in low:
            raise ValueError(f"response body looks like an access/error page: {hint!r}")


def _temp_dir() -> Path:
    """Workspace TMPDIR, falling back to the project root only if unset.

    Never uses the system /tmp: the workspace contract requires TMPDIR to be
    isolated inside RESEARCH_ROOT.
    """
    tmp = os.environ.get("TMPDIR")
    d = Path(tmp) if tmp else None
    if d is None or not str(d).startswith(str(os.environ.get("RESEARCH_ROOT", "/nonexistent"))):
        raise RuntimeError(
            "TMPDIR is not set inside RESEARCH_ROOT; source scripts/research-env.sh "
            "before running Stage 1 (the workspace contract forbids writing to /tmp)"
        )
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------
# Safe extraction
# --------------------------------------------------------------------------


def _validate_member_name(name: str, cfg: dict[str, Any]) -> None:
    if name.startswith("/") and not cfg.get("allow_absolute_paths", False):
        raise ValueError(f"archive member has an absolute path: {name!r}")
    norm = os.path.normpath(name)
    if norm.startswith("..") or "/../" in f"/{norm}/":
        raise ValueError(f"archive member escapes the target directory: {name!r}")


def _check_bomb(total: int, largest: int, compressed: int, cfg: dict[str, Any]) -> None:
    if total > int(cfg.get("max_total_uncompressed_bytes", 2_000_000_000)):
        raise ValueError(
            f"archive expands to {total} bytes, above the configured cap "
            f"{cfg.get('max_total_uncompressed_bytes')}"
        )
    if compressed > 0:
        ratio = largest / compressed
        if ratio > float(cfg.get("max_compression_ratio", 200)):
            raise ValueError(
                f"archive compression ratio {ratio:.1f} exceeds the configured cap "
                f"{cfg.get('max_compression_ratio')} (possible file bomb)"
            )


def safe_extract_zip(archive: Path, dest: Path, cfg: dict[str, Any]) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        if len(infos) > int(cfg.get("max_members", 10_000)):
            raise ValueError(f"archive has {len(infos)} members, above the cap")
        total = 0
        largest = 0
        written: list[str] = []
        for info in infos:
            _validate_member_name(info.filename, cfg)
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000 and not cfg.get("allow_symlinks", False):
                raise ValueError(f"archive contains a symlink: {info.filename!r}")
            if info.is_dir():
                (dest / info.filename).mkdir(parents=True, exist_ok=True)
                continue
            target = (dest / info.filename).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise ValueError(f"archive member escapes target: {info.filename!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out, _CHUNK)
            total += info.file_size
            largest = max(largest, info.file_size)
            written.append(info.filename)
        _check_bomb(total, largest, archive.stat().st_size, cfg)
    return written


def safe_extract_tar(archive: Path, dest: Path, cfg: dict[str, Any]) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    total = 0
    largest = 0
    with tarfile.open(archive) as tf:
        members = tf.getmembers()
        if len(members) > int(cfg.get("max_members", 10_000)):
            raise ValueError(f"archive has {len(members)} members, above the cap")
        for m in members:
            _validate_member_name(m.name, cfg)
            if m.issym() or m.islnk():
                if not (cfg.get("allow_symlinks") or cfg.get("allow_hardlinks")):
                    raise ValueError(f"archive contains a link: {m.name!r}")
                raise ValueError(f"archive contains a link: {m.name!r}")
            if not (m.isfile() or m.isdir()):
                raise ValueError(f"archive contains an unexpected member type: {m.name!r}")
            if m.isfile():
                total += m.size
                largest = max(largest, m.size)
        _check_bomb(total, largest, archive.stat().st_size, cfg)
        for m in members:
            tf.extract(m, dest, filter="data")
            if m.isfile():
                written.append(m.name)
    return written


def safe_extract(archive: Path, dest: Path, cfg: dict[str, Any]) -> tuple[str, list[str]]:
    kind = looks_like_archive(archive)
    if kind == "zip":
        return "zip", safe_extract_zip(archive, dest, cfg)
    if kind in {"gzip", "bzip2", "xz"}:
        import bz2
        import gzip
        import lzma

        opener = {"gzip": gzip.open, "bzip2": bz2.open, "xz": lzma.open}[kind]
        with opener(archive, "rb") as src:  # type: ignore[operator]
            data = src.read(int(cfg.get("max_total_uncompressed_bytes", 2_000_000_000)) + 1)
        if len(data) > int(cfg.get("max_total_uncompressed_bytes", 2_000_000_000)):
            raise ValueError("single-file archive expands beyond the configured cap")
        dest.mkdir(parents=True, exist_ok=True)
        name = archive.name.split(".")[0]
        target = dest / name
        target.write_bytes(data)
        return kind, [name]
    raise ValueError(f"not a recognised archive format: {archive.name}")


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------


def _attempt_download(
    session: requests.Session,
    url: str,
    tmp_path: Path,
    cfg: dict[str, Any],
    idx: int,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "attempt": idx,
        "utc": utc_now(),
        "url": url,
        "t0": time.monotonic(),
    }
    t0 = time.monotonic()
    try:
        resp = session.get(
            url,
            stream=True,
            timeout=(
                float(cfg["connect_timeout_seconds"]),
                float(cfg["read_timeout_seconds"]),
            ),
            allow_redirects=True,
        )
        rec["http_status"] = resp.status_code
        rec["final_url"] = resp.url
        rec["redirects"] = [
            {"status": r.status_code, "url": r.url, "location": r.headers.get("Location")}
            for r in resp.history
        ]
        rec["content_type"] = resp.headers.get("Content-Type")
        rec["content_length_header"] = resp.headers.get("Content-Length")
        rec["etag"] = resp.headers.get("ETag")
        rec["last_modified"] = resp.headers.get("Last-Modified")

        if resp.status_code >= 400:
            rec["ok"] = False
            rec["error"] = f"HTTP {resp.status_code}"
            resp.close()
            return rec

        declared = resp.headers.get("Content-Length")
        if declared is not None and int(declared) > 0:
            min_free = int(cfg.get("min_free_bytes_required", 500_000_000))
            avail = free_bytes(tmp_path.parent)
            if avail < int(declared) + min_free:
                rec["ok"] = False
                rec["error"] = (
                    f"insufficient free space: need {declared} + {min_free} bytes "
                    f"headroom, {avail} available"
                )
                resp.close()
                return rec

        written = 0
        with open(tmp_path, "wb") as out:
            for block in resp.iter_content(chunk_size=_CHUNK):
                if block:
                    out.write(block)
                    written += len(block)
        resp.close()
        rec["bytes"] = written
        rec["ok"] = True
    except requests.RequestException as exc:
        rec["ok"] = False
        rec["error"] = f"{type(exc).__name__}: {exc}"
    rec["elapsed_seconds"] = round(time.monotonic() - t0, 3)
    return rec


def download_source(
    spec: DatasetSpec,
    source: SourceSpec,
    cfg: dict[str, Any],
    *,
    session: requests.Session | None = None,
) -> DownloadResult:
    """Download one source into the immutable raw directory."""
    dcfg = cfg["download"]
    dest_dir = raw_dir(spec.key)
    dest_dir.mkdir(parents=True, exist_ok=True)
    raw_path = dest_dir / source.filename_in_archive

    result = DownloadResult(dataset=spec.key, url=source.url, ok=False, path=raw_path)

    # ---- idempotence: never re-fetch or overwrite an intact raw file -------
    if raw_path.exists() and dcfg.get("reuse_existing_raw_by_sha256", True):
        try:
            assert_not_error_page(raw_path)
        except ValueError as exc:
            LOG.warning("[%s] existing raw file rejected (%s); quarantining", spec.key, exc)
            quarantine = raw_path.with_suffix(raw_path.suffix + ".rejected")
            if not quarantine.exists():
                os.replace(raw_path, quarantine)
            raw_path.unlink(missing_ok=True)
        else:
            result.ok = True
            result.sha256 = sha256_file(raw_path)
            result.bytes = raw_path.stat().st_size
            result.reused_existing = True
            result.finished_utc = utc_now()
            LOG.info("[%s] reusing existing raw file (%s bytes)", spec.key, result.bytes)
            return result

    tmp_dir = _temp_dir() / "ossfs_dd_stage1"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{spec.key}.{source.filename_in_archive}.partial"

    owns = session is None
    session = session or requests.Session()
    session.headers.update({"User-Agent": dcfg.get("user_agent", "ossfs-dd-reproduction/stage1")})
    retry_on = set(dcfg.get("retry_on_http_status", []))
    never = set(dcfg.get("do_not_retry_on_http_status", []))
    max_attempts = int(dcfg.get("max_attempts_per_url", 3))

    try:
        last_error = None
        for idx in range(1, max_attempts + 1):
            rec = _attempt_download(session, source.url, tmp_path, dcfg, idx)
            result.attempts.append(rec)
            LOG.info(
                "[%s] attempt %d: status=%s bytes=%s %.1fs %s",
                spec.key,
                idx,
                rec.get("http_status"),
                rec.get("bytes"),
                rec.get("elapsed_seconds", 0.0),
                rec.get("error") or "ok",
            )
            if rec.get("ok"):
                last_error = None
                break
            last_error = rec.get("error", "unknown error")
            status = rec.get("http_status")
            if status in never:
                LOG.warning("[%s] HTTP %s is not retryable; stopping", spec.key, status)
                break
            if status is not None and status not in retry_on:
                LOG.warning("[%s] HTTP %s is not in the retry set; stopping", spec.key, status)
                break
            if idx < max_attempts:
                delay = min(
                    float(dcfg.get("backoff_base_seconds", 2.0)) ** idx,
                    float(dcfg.get("backoff_max_seconds", 30.0)),
                )
                LOG.info("[%s] backing off %.1fs before retry", spec.key, delay)
                time.sleep(delay)

        if last_error is not None:
            tmp_path.unlink(missing_ok=True)
            result.error = last_error
            result.finished_utc = utc_now()
            LOG.error("[%s] DOWNLOAD_FAILED %s: %s", spec.key, source.url, last_error)
            return result

        # ---- body-level validation before the atomic rename -----------------
        try:
            assert_not_error_page(tmp_path)
        except ValueError as exc:
            tmp_path.unlink(missing_ok=True)
            result.error = str(exc)
            result.finished_utc = utc_now()
            LOG.error("[%s] REJECTED %s: %s", spec.key, source.url, exc)
            return result

        digest = sha256_file(tmp_path)
        if source.expected_sha256 and digest != source.expected_sha256:
            tmp_path.unlink(missing_ok=True)
            result.error = f"checksum mismatch: expected {source.expected_sha256}, got {digest}"
            result.finished_utc = utc_now()
            LOG.error("[%s] %s", spec.key, result.error)
            return result

        os.replace(tmp_path, raw_path)  # atomic: raw file appears only when complete
        result.ok = True
        result.sha256 = digest
        result.bytes = raw_path.stat().st_size
        result.finished_utc = utc_now()
        LOG.info("[%s] acquired %s (%s bytes, sha256 %s)", spec.key, raw_path.name, result.bytes, digest[:16])
        return result
    finally:
        if owns:
            session.close()


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def extract_source(
    spec: DatasetSpec,
    source: SourceSpec,
    cfg: dict[str, Any],
    download: DownloadResult,
) -> dict[str, Any]:
    """Extract a downloaded archive into ``raw/<key>/`` (never over raw bytes)."""
    record: dict[str, Any] = {
        "dataset": spec.key,
        "source_url": source.url,
        "raw_archive": rel_to_research(download.path) if download.path else None,
        "extracted_at_utc": utc_now(),
    }
    if download.path is None or not download.path.exists():
        record.update(ok=False, error="no raw archive to extract")
        return record

    kind = looks_like_archive(download.path)
    if kind is None:
        record.update(ok=True, format="single_file", extracted=[download.path.name], extracted_count=1)
        return record

    dest = raw_dir(spec.key) / "extracted"
    try:
        kind, written = safe_extract(download.path, dest, cfg["extraction"])
    except Exception as exc:
        record.update(ok=False, error=f"{type(exc).__name__}: {exc}")
        LOG.error("[%s] extraction failed: %s", spec.key, record["error"])
        return record

    record.update(
        ok=True,
        format=kind,
        extracted=[str((Path(dest) / w).relative_to(dest)) for w in written],
        extracted_count=len(written),
        extracted_root=rel_to_research(dest),
    )
    LOG.info("[%s] extracted %d members from %s", spec.key, len(written), kind)
    return record


def raw_inventory(dataset: str) -> dict[str, Any]:
    """Hash every immutable raw byte for a dataset, without modifying anything."""
    d = raw_dir(dataset)
    files: list[dict[str, Any]] = []
    if d.exists():
        for p in sorted(d.rglob("*")):
            if p.is_file():
                files.append(
                    {
                        "path": rel_to_research(p),
                        "bytes": p.stat().st_size,
                        "sha256": sha256_file(p),
                    }
                )
    return {"dataset": dataset, "root": rel_to_research(d), "files": files}


def configure_logging(logfile_name: str, *, verbose: bool = True) -> Path:
    """Log to stdout and to the workspace output tree. Never to /tmp."""
    path = log_dir() / logfile_name
    root = logging.getLogger("ossfs_dd_data")
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%dT%H:%M:%S%z")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    return path


def write_ledger(name: str, records: Iterable[dict[str, Any]]) -> Path:
    path = log_dir() / name
    import json

    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(rec, default=str) + "\n" for rec in records)
    return path
