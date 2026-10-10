"""Source-format adapters.

One adapter per format actually encountered in Stage 1:

  arff                 dense ARFF, target identified explicitly (never by
                       "it must be the last column" alone)
  csv_label_last       dense comma-separated matrix whose class column is
                       declared by the registry, not guessed
  uci_data_labels_pair NIPS-2003 `.data` + `.labels` pairs, concatenated along
                       ROWS after explicit label alignment

Design rules that hold for every adapter:
  * the class label is never emitted as a feature;
  * a transpose is performed only when the registry or the parser can *prove*
    the orientation, never on shape alone;
  * original within-source row indices are preserved per row;
  * the parser reports what it did, so the caller can record it verbatim.

No adapter imputes, scales, standardises, clips, rounds or deduplicates.
"""

from __future__ import annotations

import csv
import io
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

LOG = logging.getLogger("ossfs_dd_data.parsers")

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


class ParseError(RuntimeError):
    """Raised when a source cannot be parsed into an unambiguous dataset."""


@dataclass
class ParsedDataset:
    """Canonical in-memory representation produced by every adapter."""

    X: np.ndarray | sparse.csr_matrix
    y: np.ndarray                      # (n_samples,) original label strings
    feature_names: list[str]
    source_split: np.ndarray           # (n_samples,) 'train' | 'valid' | ...
    source_row_index: np.ndarray       # (n_samples,) row index within its source file
    label_name: str
    label_position: str
    dtype: str
    sparse: bool
    feature_dtype: str
    label_map: dict[str, int] = field(default_factory=dict)
    orientation: str = "rows_are_samples"
    notes: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _to_numeric_block(values: Sequence[Sequence[str]], dtype: str = "float64") -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=dtype)
    except ValueError as exc:
        raise ParseError(f"non-numeric value in the feature block: {exc}") from exc
    return arr


def _read_text(path: Path) -> str:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _maybe_sparse(arr: np.ndarray, density_threshold: float) -> np.ndarray | sparse.csr_matrix:
    if arr.size == 0:
        return arr
    nonzero = int(np.count_nonzero(arr))
    density = nonzero / arr.size
    if density < density_threshold:
        LOG.info("density %.4f below %.2f -> storing as CSR", density, density_threshold)
        return sparse.csr_matrix(arr)
    return arr


def build_label_map(labels: np.ndarray) -> dict[str, int]:
    """Explicit, reversible original-to-canonical label map.

    Sorted by the ORIGINAL label text so the mapping is deterministic and does
    not depend on row order.
    """
    uniq = sorted({str(v) for v in labels.tolist()})
    return {orig: i for i, orig in enumerate(uniq)}


# --------------------------------------------------------------------------
# ARFF
# --------------------------------------------------------------------------


def parse_arff(
    path: Path,
    *,
    label_column: int | None = None,
    label_name_hint: str | None = None,
    density_threshold: float = 0.30,
    dtype: str = "float64",
) -> ParsedDataset:
    """Parse a dense ARFF file.

    ``label_column`` is an explicit 0-based attribute index. When it is None the
    target is located from the attribute TYPE (nominal) and, failing that, by a
    declared name; the "last attribute is the class" rule is only used as a
    final fallback and is recorded in the manifest when it fires.
    """
    text = _read_text(path)
    lines = text.splitlines()

    attrs: list[tuple[str, str]] = []
    relation = ""
    data_start: int | None = None
    is_sparse = False
    for i, raw in enumerate(lines):
        line = raw.strip()
        low = line.lower()
        if not attrs and low.startswith("@relation"):
            relation = line.split(None, 1)[1].strip() if " " in line else ""
            continue
        if low.startswith("@attribute"):
            rest = line[len("@attribute") :].strip()
            # name may be quoted; type follows the last whitespace-delimited token
            if rest.startswith(("'", '"')):
                quote = rest[0]
                end = rest.index(quote, 1)
                name = rest[1:end]
                typ = rest[end + 1 :].strip()
            else:
                parts = rest.rsplit(None, 1)
                if len(parts) != 2:
                    raise ParseError(f"malformed @attribute line: {raw!r}")
                name, typ = parts[0], parts[1]
            typ = typ.strip().strip("{}").lower()
            attrs.append((name, typ))
            continue
        if low.startswith("@data"):
            is_sparse = is_sparse or "sparse" in low
            data_start = i + 1
            break

    if data_start is None:
        raise ParseError(f"{path.name}: no @data section found")
    if not attrs:
        raise ParseError(f"{path.name}: no @attribute declarations found")

    # ---- locate the target explicitly ------------------------------------
    how = None
    if label_column is not None:
        how = "registry_declared_index"
        if not 0 <= label_column < len(attrs):
            raise ParseError(
                f"{path.name}: declared label column {label_column} out of range "
                f"(0..{len(attrs) - 1})"
            )
    else:
        nominal = [i for i, (_, t) in enumerate(attrs) if t.startswith("nominal")]
        if label_name_hint:
            named = [
                i for i, (n, _) in enumerate(attrs) if n.strip().strip("'\"").lower() == label_name_hint.lower()
            ]
            if named:
                label_column = named[-1]
                how = "declared_name"
        if label_column is None and len(nominal) == 1:
            label_column = nominal[0]
            how = "sole_nominal_attribute"
        if label_column is None:
            label_column = len(attrs) - 1
            how = "last_attribute_fallback"

    label_attr = attrs[label_column]

    # ---- parse the data section ------------------------------------------
    n_attr = len(attrs)
    feature_cols = [i for i in range(n_attr) if i != label_column]

    dense_rows: list[list[str]] = []
    labels: list[str] = []
    for raw in lines[data_start:]:
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        if is_sparse:
            row, label = _parse_sparse_arff_row(line, n_attr, label_column)
        else:
            fields = next(csv.reader(io.StringIO(line)))
            if len(fields) != n_attr:
                raise ParseError(
                    f"{path.name}: row has {len(fields)} fields but {n_attr} "
                    f"attributes are declared"
                )
            row = fields
            label = fields[label_column]
        labels.append(label)
        dense_rows.append([row[i] for i in feature_cols])

    if not dense_rows:
        raise ParseError(f"{path.name}: @data section is empty")

    X = _to_numeric_block(dense_rows, dtype)
    X = _maybe_sparse(X, density_threshold)
    y = np.asarray(labels, dtype=object)

    feature_names = _disambiguate([attrs[i][0].strip().strip("'\"") for i in feature_cols])

    LOG.info(
        "%s: parsed %d x %d from ARFF relation %r, target %r located by %s",
        path.name,
        X.shape[0],
        X.shape[1],
        relation,
        label_attr[0],
        how,
    )
    return ParsedDataset(
        X=X,
        y=y,
        feature_names=feature_names,
        source_split=np.array(["original_labeled"] * X.shape[0], dtype=object),
        source_row_index=np.arange(X.shape[0], dtype=np.int64),
        label_name=label_attr[0],
        label_position=str(label_column),
        dtype=str(X.dtype),
        sparse=sparse.issparse(X),
        feature_dtype=str(dtype),
        label_map=build_label_map(y),
        notes=[
            f"arff_relation={relation}",
            f"target_located_by={how}",
            f"target_attribute_type={label_attr[1]}",
        ],
        extra={
            "n_declared_attributes": n_attr,
            "sparse_arff": is_sparse,
            "label_column_index": label_column,
        },
    )


def _parse_sparse_arff_row(line: str, n_attr: int, label_column: int) -> tuple[list[str], str]:
    body = line.strip().strip("{}")
    parts = [p.strip() for p in body.split(",")]
    default = parts[0]
    cells = [default] * n_attr
    for item in parts[1:]:
        if not item:
            continue
        idx_str, _, val = item.partition(" ")
        # ARFF sparse indices are 0-based
        idx = int(idx_str)
        cells[idx] = val.strip().strip("'\"")
    return cells, cells[label_column]


def _disambiguate(names: list[str]) -> list[str]:
    """Make duplicate names unique without discarding any column.

    Duplicates are suffixed with ``__dupN`` in order of appearance, so the
    mapping back to original source names stays reversible via the suffix.
    """
    seen: dict[str, int] = {}
    out: list[str] = []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}__dup{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    return out


# --------------------------------------------------------------------------
# dense CSV with an explicitly declared label column
# --------------------------------------------------------------------------


def parse_csv_label_last(
    path: Path,
    *,
    label_column: int = -1,
    expected_columns: int | None = None,
    density_threshold: float = 0.30,
    dtype: str = "float64",
    split_name: str = "original_labeled",
) -> ParsedDataset:
    """Parse a headerless dense CSV whose class column is declared, not guessed.

    ``label_column`` may be negative (counted from the end) or a non-negative
    absolute index. The class column is dropped from the feature block and never
    counted as a feature.
    """
    rows: list[list[str]] = []
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh)
        for rec in reader:
            if rec and any(c.strip() for c in rec):
                rows.append(rec)
    if not rows:
        raise ParseError(f"{path.name}: no data rows")

    width = len(rows[0])
    bad = [i for i, r in enumerate(rows) if len(r) != width]
    if bad:
        raise ParseError(
            f"{path.name}: ragged rows at positions {bad[:5]} (width {width} expected); "
            "refusing to guess a delimiter or pad rows"
        )

    col = label_column if label_column >= 0 else width + label_column
    if not 0 <= col < width:
        raise ParseError(f"{path.name}: label column {label_column} resolves to {col}, out of range")

    if expected_columns is not None and width != expected_columns:
        raise ParseError(
            f"{path.name}: {width} columns found but {expected_columns} expected "
            "(class column included)"
        )

    feature_idx = [i for i in range(width) if i != col]
    labels = [r[col] for r in rows]
    block = [[r[i] for i in feature_idx] for r in rows]

    X = _to_numeric_block(block, dtype)
    X = _maybe_sparse(X, density_threshold)
    y = np.asarray(labels, dtype=object)
    feature_names = [f"feature_{i}" for i in range(len(feature_idx))]

    LOG.info(
        "%s: parsed %d x %d from CSV, label column index %d (%d total columns)",
        path.name,
        X.shape[0],
        X.shape[1],
        col,
        width,
    )
    return ParsedDataset(
        X=X,
        y=y,
        feature_names=feature_names,
        source_split=np.array([split_name] * X.shape[0], dtype=object),
        source_row_index=np.arange(X.shape[0], dtype=np.int64),
        label_name=f"column_{col}",
        label_position=str(col),
        dtype=str(X.dtype),
        sparse=sparse.issparse(X),
        feature_dtype=str(dtype),
        label_map=build_label_map(y),
        notes=[f"label_column_index={col}", f"total_columns={width}", "headerless_csv"],
        extra={"label_column_index": col, "total_columns": width},
    )


# --------------------------------------------------------------------------
# UCI .data + .labels pairs (NIPS-2003 challenge layout)
# --------------------------------------------------------------------------


def _parse_uci_data_file(
    path: Path,
    n_columns: int | None,
    density_threshold: float,
    dtype: str,
    *,
    declared_sparse: bool = False,
) -> tuple[np.ndarray | sparse.csr_matrix, dict[str, Any]]:
    """Parse a NIPS-2003 `.data` file.

    Supports the dense whitespace layout and the ``index:value`` sparse layout.
    The sparse index base is DETECTED, not assumed, using the declared column
    count: if the largest observed index would fall outside 0..p-1 it must be
    1-based. An ambiguous file is resolved in favour of the original NIPS
    1-based convention and the ambiguity is reported rather than hidden.
    """
    n_lines = 0
    all_sparse = True
    min_index: int | None = None
    max_index: int | None = None
    dense_widths: set[int] = set()

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            tokens = line.split()
            if ":" in tokens[0]:
                for tok in tokens:
                    head = tok.split(":", 1)[0]
                    if head.lstrip("-").isdigit():
                        v = int(head)
                        min_index = v if min_index is None else min(min_index, v)
                        max_index = v if max_index is None else max(max_index, v)
            else:
                all_sparse = False
                dense_widths.add(len(tokens))

    if n_lines == 0:
        raise ParseError(f"{path.name}: empty data file")

    if all_sparse and declared_sparse:
        base, ambiguous = 1, False
        base_reason = (
            "the source's own .param file declares data_type='sparse-integer', which is the "
            "NIPS 2003 challenge format documented to use 1-based feature indices"
        )
    elif all_sparse:
        base, base_reason, ambiguous = _detect_index_base(
            path, min_index, max_index, n_columns, all_sparse=True
        )
    else:
        base, base_reason, ambiguous = 0, "file is dense; no sparse index base applies", False

    indptr = [0]
    indices: list[int] = []
    data: list[float] = []
    dense_rows: list[list[str]] = []

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            if all_sparse:
                for tok in tokens:
                    head, _, val = tok.partition(":")
                    indices.append(int(head) - base)
                    data.append(float(val))
                indptr.append(len(indices))
            else:
                dense_rows.append(tokens)

    if all_sparse:
        idx = np.asarray(indices, dtype=np.int64)
        val = np.asarray(data, dtype=dtype)
        width = n_columns or (int(idx.max()) + 1 if idx.size else 0)
        if idx.size and (idx.min() < 0 or idx.max() >= width):
            raise ParseError(
                f"{path.name}: sparse index out of range after the detected {base}-based "
                f"correction (min {int(idx.min())}, max {int(idx.max())}, width {width})"
            )
        if len(indptr) - 1 != n_lines:
            raise ParseError(f"{path.name}: sparse row count mismatch")
        X = sparse.csr_matrix(
            (val, idx, np.asarray(indptr, dtype=np.int64)), shape=(n_lines, width)
        )
        info = {
            "layout": "sparse_index_value",
            "index_base_detected": base,
            "index_base_convention": "1-based" if base == 1 else "0-based",
            "index_base_reason": base_reason,
            "index_base_ambiguous": ambiguous,
            "min_index_in_source": min_index,
            "max_index_in_source": max_index,
            "implicit_zero_columns": width,
            "columns_including_implicit_zeros": width,
            "declared_sparse_in_source_param": bool(declared_sparse),
        }
    else:
        widths = {len(r) for r in dense_rows}
        if len(widths) != 1:
            raise ParseError(f"{path.name}: ragged rows, widths observed: {sorted(widths)[:5]}")
        width = widths.pop()
        if n_columns is not None and width != n_columns:
            raise ParseError(
                f"{path.name}: {width} value columns but the registry declares {n_columns}. "
                "Transposing is not attempted: the NIPS .data format is row-major "
                "(one row per sample) by construction, so a width mismatch means a "
                "different variant, not a transposed copy of the right one."
            )
        X = _to_numeric_block(dense_rows, dtype)
        X = _maybe_sparse(X, density_threshold)
        info = {
            "layout": "dense_whitespace",
            "implicit_zero_columns": None,
            "columns_including_implicit_zeros": width,
        }

    LOG.info(
        "%s: parsed %d x %d (%s, index base %s — %s)",
        path.name,
        X.shape[0],
        X.shape[1],
        info["layout"],
        info.get("index_base_detected", "n/a"),
        info.get("index_base_reason", "n/a"),
    )
    return X, info


def _detect_index_base(
    path: Path,
    min_index: int | None,
    max_index: int | None,
    n_columns: int | None,
    *,
    all_sparse: bool,
) -> tuple[int, str, bool]:
    """Decide whether a sparse file uses 0-based or 1-based feature indices.

    The source's own ``.param`` declaration wins when it is available: the NIPS
    2003 challenge's ``sparse-integer`` format is documented as 1-based. Absent
    that, the declared column count is used: an index outside 0..p-1 forces a
    1-based reading. A genuinely ambiguous file is resolved in favour of the NIPS
    1-based convention and flagged as ambiguous rather than silently decided.
    """
    if not all_sparse:
        return 0, "file is dense; no sparse index base applies", False
    if min_index is None or max_index is None:
        raise ParseError(f"{path.name}: sparse file contained no index:value tokens")

    if n_columns:
        if max_index >= n_columns and min_index >= 1:
            return 1, (
                f"largest source index {max_index} is outside 0..{n_columns - 1}, so the "
                f"source must be 1-based"
            ), False
        if min_index == 0:
            return 0, (
                f"source contains a 0 index and every index fits 0..{n_columns - 1}, "
                "so the source is 0-based"
            ), False
    if min_index == 0:
        return 0, "source contains a 0 index, so the source is 0-based", False
    return 1, (
        f"indices {min_index}..{max_index} fit inside 0..{(n_columns or 0) - 1} under "
        "either convention, so the file alone is ambiguous; resolved in favour of the "
        "original NIPS 1-based convention and recorded as ambiguous"
    ), True


def parse_param_file(path: Path) -> dict[str, Any]:
    """Read a NIPS-2003 ``.param`` file.

    This file is shipped by the challenge authors and declares the data type,
    the feature count, and — crucially — a published value ``Check_sum`` for
    every split. Those checksums are an independent, source-provided integrity
    witness: if the sum of the values we parsed for a split equals the published
    checksum, the rows we read are the rows the source intended.
    """
    text = _read_text(path)
    data_type = None
    n_features = None
    splits: dict[str, dict[str, Any]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        low = line.lower()
        if low.startswith("data type:"):
            data_type = line.split(":", 1)[1].strip()
        elif low.startswith("number of features:"):
            n_features = int(float(line.split(":", 1)[1].strip()))
        else:
            parts = line.split()
            if len(parts) == 5:
                name = parts[0].lower()
                try:
                    pos, neg, tot, chk = (float(x) for x in parts[1:])
                except ValueError:
                    continue
                splits[name] = {
                    "pos": int(pos), "neg": int(neg), "total": int(tot), "check_sum": chk
                }
    return {
        "file": path.name,
        "data_type": data_type,
        "n_features": n_features,
        "splits": splits,
        "source": "NIPS 2003 challenge .param file shipped inside the UCI archive",
    }


def _read_labels(path: Path) -> np.ndarray:
    out: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if s:
                out.append(s)
    if not out:
        raise ParseError(f"{path.name}: empty labels file")
    return np.asarray(out, dtype=object)


def parse_uci_data_labels_pair(
    data_paths: dict[str, Path],
    label_paths: dict[str, Path],
    *,
    n_columns: int,
    density_threshold: float = 0.30,
    dtype: str = "float64",
    strict_order: Sequence[str] = ("train", "valid"),
    param_path: Path | None = None,
) -> ParsedDataset:
    """Combine labeled ``.data`` + ``.labels`` pairs along ROWS.

    ``strict_order`` fixes the concatenation order so the combined row order is
    reproducible. Unlabeled challenge test files are never passed in here; if one
    appears in the mapping this function refuses it.

    When the source ships its NIPS ``.param`` file, the declared data type drives
    the sparse index-base decision and each split's value sum is checked against
    the source's published ``Check_sum``.
    """
    if "test" in data_paths or "test" in label_paths:
        raise ParseError(
            "refusing to parse an unlabeled challenge 'test' split into the dataset; "
            "test records carry no labels and must never contribute samples"
        )

    param = parse_param_file(param_path) if param_path and param_path.exists() else None
    if param and param.get("data_type"):
        LOG.info(
            "%s: source .param declares data_type=%r, n_features=%r",
            param_path.name if param_path else "?",
            param["data_type"],
            param.get("n_features"),
        )

    blocks: list[np.ndarray | sparse.csr_matrix] = []
    labels: list[np.ndarray] = []
    splits: list[np.ndarray] = []
    row_index: list[np.ndarray] = []
    layout_notes: list[str] = []
    checksum_checks: dict[str, Any] = {}
    orientation = "rows_are_samples"

    for name in strict_order:
        if name not in data_paths:
            continue
        dpath, lpath = data_paths[name], label_paths.get(name)
        if lpath is None:
            raise ParseError(f"split {name!r} has a data file but no labels file")

        Xb, info = _parse_uci_data_file(
            dpath,
            n_columns,
            density_threshold,
            dtype,
            declared_sparse=str((param or {}).get("data_type") or "").strip() == "sparse-integer",
        )
        yb = _read_labels(lpath)

        if Xb.shape[0] != yb.shape[0]:
            raise ParseError(
                f"{name}: {Xb.shape[0]} data rows but {yb.shape[0]} labels — refusing to "
                "pad or truncate to force alignment"
            )

        # --- orientation proof -------------------------------------------
        # A .data file is by construction row-major (one row per sample). We do
        # NOT transpose on shape alone; we only ever reject a file whose width
        # contradicts the registry, which _parse_uci_data_file already enforces.
        if Xb.shape[1] != n_columns:
            raise ParseError(
                f"{name}: parsed {Xb.shape[1]} feature columns, registry declares "
                f"{n_columns}. Transposing is not attempted: the NIPS .data format "
                "is row-major by construction and a mismatch means a different variant."
            )

        # --- independent integrity witness from the source itself ---------
        if param and name in param.get("splits", {}):
            declared = float(param["splits"][name]["check_sum"])
            observed = float(Xb.sum())
            checksum_checks[name] = {
                "source_check_sum": declared,
                "observed_value_sum": observed,
                "match": abs(observed - declared) <= max(1.0, abs(declared) * 1e-9),
                "source_row_count": int(Xb.shape[0]),
                "source_pos_neg": {
                    "pos": param["splits"][name]["pos"],
                    "neg": param["splits"][name]["neg"],
                },
                "observed_pos_neg": {
                    "pos": int(np.sum(yb.astype(str) == "1")),
                    "neg": int(np.sum(yb.astype(str) == "-1")),
                },
            }

        layout_notes.append(f"{name}:{info['layout']}")
        if "index_base_detected" in info:
            layout_notes.append(
                f"{name}:index_base={info['index_base_convention']}"
                f"{'(ambiguous)' if info.get('index_base_ambiguous') else ''}"
            )
            if info.get("declared_sparse_in_source_param"):
                layout_notes.append(f"{name}:source_param_declares_sparse_integer")
        blocks.append(Xb)
        labels.append(yb)
        splits.append(np.array([name] * Xb.shape[0], dtype=object))
        row_index.append(np.arange(Xb.shape[0], dtype=np.int64))

    if not blocks:
        raise ParseError("no labeled splits were supplied")

    X = (
        sparse.vstack(blocks, format="csr")
        if any(sparse.issparse(b) for b in blocks)
        else np.vstack(blocks)
    )
    y = np.concatenate(labels)
    source_split = np.concatenate(splits)
    source_row_index = np.concatenate(row_index)

    LOG.info(
        "combined %s: %d x %d, splits %s",
        "+".join(str(s) for s in strict_order),
        X.shape[0],
        X.shape[1],
        {str(s): int(np.sum(source_split == s)) for s in set(source_split.tolist())},
    )
    return ParsedDataset(
        X=X,
        y=y,
        feature_names=[f"feature_{i}" for i in range(int(X.shape[1]))],
        source_split=source_split,
        source_row_index=source_row_index,
        label_name="label",
        label_position="separate_labels_file",
        dtype=str(X.dtype),
        sparse=sparse.issparse(X),
        feature_dtype=str(dtype),
        label_map=build_label_map(y),
        orientation=orientation,
        notes=layout_notes,
        extra={
            "split_order": list(strict_order),
            "split_row_counts": {
                str(s): int(np.sum(source_split == s)) for s in set(source_split.tolist())
            },
            "excluded_splits": ["test"],
            "source_param": param,
            "source_checksum_verification": checksum_checks,
        },
    )


# --------------------------------------------------------------------------
# RData (.rda / .RData) — optional, guarded
# --------------------------------------------------------------------------


def parse_rdata(
    path: Path, x_key: str = "x", y_key: str = "y", dtype: str = "float64"
) -> ParsedDataset:
    """Read an R ``.rda``/``.RData`` archive.

    Requires the optional ``rdata`` extra (``uv sync --extra rdata``). Stage 1's
    selected sources do not need this path; it exists so a documented R fallback
    can be parsed without bootstrapping a system R installation.
    """
    try:
        import pyreadr  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ParseError(
            "reading .rda requires the optional extra: run `uv sync --extra rdata`"
        ) from exc

    frames = pyreadr.read_r(str(path))  # type: ignore[attr-defined]
    if x_key not in frames:
        raise ParseError(
            f"{path.name}: object {x_key!r} not found; archive holds {sorted(frames)}"
        )
    frame = frames[x_key]
    if isinstance(frame, np.ndarray):
        X = np.asarray(frame, dtype=dtype)
    else:  # pandas DataFrame
        feature_names = [str(c) for c in frame.columns]
        if y_key in feature_names:
            feature_names.remove(y_key)
        X = frame[feature_names].to_numpy(dtype=dtype)
    y = frames.get(y_key)
    if y is None:
        raise ParseError(f"{path.name}: object {y_key!r} not found in the archive")
    yv = np.asarray(y).reshape(-1)
    if yv.shape[0] != X.shape[0]:
        raise ParseError(f"{path.name}: x has {X.shape[0]} rows but y has {yv.shape[0]}")
    return ParsedDataset(
        X=X,
        y=yv.astype(object),
        feature_names=feature_names if not isinstance(frame, np.ndarray) else [f"feature_{i}" for i in range(X.shape[1])],
        source_split=np.array(["original_labeled"] * X.shape[0], dtype=object),
        source_row_index=np.arange(X.shape[0], dtype=np.int64),
        label_name=y_key,
        label_position="separate_r_object",
        dtype=str(X.dtype),
        sparse=False,
        feature_dtype=str(dtype),
        label_map=build_label_map(yv.astype(object)),
        notes=["rdata"],
        extra={"objects": sorted(frames)},
    )


# --------------------------------------------------------------------------
# dispatcher
# --------------------------------------------------------------------------


def resolve_member(extracted_root: Path, candidates: list[str]) -> Path:
    """Locate an archive member, honouring alternative published layouts."""
    for cand in candidates:
        p = extracted_root / cand
        if p.exists():
            return p
    # case-insensitive fallback, for archives that differ only in case
    low = {str(p.relative_to(extracted_root)).lower(): p for p in extracted_root.rglob("*") if p.is_file()}
    for cand in candidates:
        hit = low.get(cand.lower())
        if hit is not None:
            return hit
    raise ParseError(
        f"none of the expected members {candidates} were found under {extracted_root}"
    )
