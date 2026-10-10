"""Dimensional and data-quality validation, plus status assignment.

Two hard rules:
  * the audit NEVER modifies feature data — no imputation, scaling, clipping,
    dropping, deduplication or coercion;
  * an unresolved NaN/Inf/missing value surfaces as ``DATA_QUALITY_BLOCKED``
    rather than being quietly repaired.

Matching ``(n, p, classes)`` is necessary but NOT sufficient for
``EXACT_VERIFIED``: the provenance tier recorded at fetch time must also be
substantiated. A lineage that is plausible but unproven yields
``SHAPE_MATCH_PROVENANCE_UNVERIFIED``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import sparse

from .parsers import ParsedDataset
from .registry import (
    STATUS_DATA_QUALITY_BLOCKED,
    STATUS_EXACT_VERIFIED,
    STATUS_LABEL_MISMATCH,
    STATUS_SHAPE_MATCH_PROVENANCE_UNVERIFIED,
    STATUS_SHAPE_MISMATCH,
    DatasetSpec,
)

LOG = logging.getLogger("ossfs_dd_data.validate")

# A tier of `paper_cited` or `official` is treated as substantiated lineage.
# `academic_repository` is dimensionally verified but its lineage cannot be
# proven to be the paper's variant, because the paper does not say which release
# it consumed.
SUBSTANTIATED_TIERS = {"paper_cited", "official"}


def audit_dataset(parsed: ParsedDataset, spec: DatasetSpec) -> dict[str, Any]:
    """Read-only audit. Modifies nothing.

    Sparse inputs are never densified globally: non-finite values are checked on
    the stored values, constant columns on the CSC structure, and duplicate rows
    on CSR row bytes.
    """
    X = parsed.X
    is_sparse = sparse.issparse(X)
    n_samples, n_features = int(X.shape[0]), int(X.shape[1])

    # ---- non-finite audit ------------------------------------------------
    if is_sparse:
        stored = X.data
        n_nan = int(np.isnan(stored).sum())
        n_posinf = int(np.isposinf(stored).sum())
        n_neginf = int(np.isneginf(stored).sum())
        n_nonfinite = n_nan + n_posinf + n_neginf
        n_stored = int(stored.size)
        n_explicit_zeros = int((stored == 0).sum())
    else:
        arr = np.asarray(X)
        n_nan = int(np.isnan(arr).sum())
        n_posinf = int(np.isposinf(arr).sum())
        n_neginf = int(np.isneginf(arr).sum())
        n_nonfinite = n_nan + n_posinf + n_neginf
        n_stored = int(arr.size)
        n_explicit_zeros = int((arr == 0).sum())

    # ---- constant columns, without a pairwise matrix --------------------
    if is_sparse:
        csc = X.tocsc()
        counts = np.diff(csc.indptr)
        all_zero_cols = counts == 0
        full_cols = counts == n_samples
        constant_mask = np.zeros(n_features, dtype=bool)
        constant_mask |= all_zero_cols
        # a fully-populated column is constant iff every stored value is equal
        for j in np.flatnonzero(full_cols):
            s, e = csc.indptr[j], csc.indptr[j + 1]
            constant_mask[j] = bool(np.all(csc.data[s:e] == csc.data[s]))
        constant_columns = int(constant_mask.sum())
        zero_columns = int(all_zero_cols.sum())
        col_min = col_max = None
    else:
        arr = np.asarray(X)
        col_min = arr.min(axis=0)
        col_max = arr.max(axis=0)
        constant_columns = int(np.sum(col_min == col_max))
        zero_columns = int(np.sum(col_max == 0))

    duplicate_columns = _count_duplicate_columns(X)
    duplicate_rows = _count_duplicate_rows(X)

    if is_sparse:
        zero_rows = int(np.sum(np.diff(X.tocsr().indptr) == 0))
    else:
        zero_rows = int(np.sum(np.all(arr == 0, axis=1)))

    uniq_labels = np.unique(parsed.y.astype(str))
    label_map = parsed.label_map
    class_counts = {
        str(k): int(v) for k, v in zip(*np.unique(parsed.y.astype(str), return_counts=True), strict=False)
    }

    sample_ids = [f"{spec.key}:{i}" for i in range(n_samples)]
    duplicate_sample_ids = int(len(sample_ids) - len(set(sample_ids)))

    value_range = _value_range(X, is_sparse)

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_classes": int(uniq_labels.size),
        "class_counts": class_counts,
        "label_map_original_to_canonical": label_map,
        "label_map_is_reversible": sorted(label_map.keys()) == sorted(
            {str(v) for v in parsed.y.tolist()}
        ),
        "n_labels_equals_rows": int(parsed.y.shape[0]) == n_samples,
        "nonfinite": {
            "total": n_nonfinite,
            "nan": n_nan,
            "posinf": n_posinf,
            "neginf": n_neginf,
            "checked": "stored (non-zero) values" if is_sparse else "all values",
            "note": (
                "Implied zeros cannot be non-finite. Stored values are checked exactly."
                if is_sparse
                else ""
            ),
        },
        "constant_columns": constant_columns,
        "duplicate_column_groups": duplicate_columns,
        "duplicate_row_groups": duplicate_rows,
        "zero_only_rows": zero_rows,
        "duplicate_sample_ids": duplicate_sample_ids,
        "sparsity": {
            "storage": "scipy_csr" if is_sparse else "numpy_dense",
            "stored_values": n_stored,
            "explicit_zeros": n_explicit_zeros,
            "density": float(n_stored / (n_samples * n_features)) if n_samples * n_features else 0.0,
            "zero_columns": zero_columns,
            "densified_globally": False,
        },
        "value_range": value_range,
        "value_integrality": _integrality(X, is_sparse),
        "modifications_applied": [],
        "modification_policy": "Stage 1 audits only; it never imputes, scales, drops or deduplicates.",
        "flags": _flags(n_nonfinite, constant_columns, duplicate_columns, duplicate_rows, zero_rows),
    }


def _value_range(X: Any, is_sparse: bool) -> dict[str, float | None]:
    if is_sparse:
        d = X.data
        if d.size == 0:
            return {"min": 0.0, "max": 0.0, "all_zero": True}
        return {"min": float(np.min(d)), "max": float(np.max(d)), "all_zero": False}
    arr = np.asarray(X)
    if arr.size == 0:
        return {"min": None, "max": None}
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"min": None, "max": None}
    return {"min": float(finite.min()), "max": float(finite.max())}


def _integrality(X: Any, is_sparse: bool) -> dict[str, Any]:
    d = X.data if is_sparse else np.asarray(X).ravel()
    d = d[np.isfinite(d)] if d.size else d
    all_int = bool(np.all(d == np.floor(d))) if d.size else None
    return {
        "all_values_integral": all_int,
        "checked": "stored (non-zero) values" if is_sparse else "all values",
        "note": (
            "The paper's 'Integer'/'Real' wording is a publication-level description. "
            "The parsed dtype is reported separately and is never coerced to match it."
        ),
    }


def _flags(n_nonfinite: int, const: int, dupcols: int, duprows: int, zerorows: int) -> list[str]:
    out: list[str] = []
    if n_nonfinite:
        out.append(f"nonfinite_values={n_nonfinite}")
    if const:
        out.append(f"constant_columns={const}")
    if dupcols:
        out.append(f"duplicate_column_groups={dupcols}")
    if duprows:
        out.append(f"duplicate_row_groups={duprows}")
    if zerorows:
        out.append(f"zero_only_rows={zerorows}")
    return out


def _count_duplicate_columns(X: Any) -> int:
    """Exact duplicate columns via hashing — never a pairwise correlation matrix."""
    if sparse.issparse(X):
        csc = X.tocsc()
        groups: dict[bytes, int] = {}
        for j in range(csc.shape[1]):
            s, e = csc.indptr[j], csc.indptr[j + 1]
            rows = csc.indices[s:e].tobytes()
            vals = csc.data[s:e].tobytes()
            key = rows + b"|" + vals
            groups[key] = groups.get(key, 0) + 1
    else:
        arr = np.asarray(X)
        # column-wise bytes view without materialising a transposed copy
        groups = {}
        for j in range(arr.shape[1]):
            key = arr[:, j].tobytes()
            groups[key] = groups.get(key, 0) + 1
    return int(sum(v - 1 for v in groups.values() if v > 1))


def _count_duplicate_rows(X: Any) -> int:
    """Exact duplicate rows via hashing — never a pairwise distance matrix.

    Sparse input is hashed from its CSR row bytes so no global densification is
    needed.
    """
    if X.shape[0] < 2:
        return 0
    if sparse.issparse(X):
        csr = X.tocsr()
        indptr, indices, data = csr.indptr, csr.indices, csr.data
        keys = []
        for i in range(csr.shape[0]):
            s, e = indptr[i], indptr[i + 1]
            keys.append(indices[s:e].tobytes() + b"|" + data[s:e].tobytes())
    else:
        view = np.ascontiguousarray(np.asarray(X))
        keys = [view[i].tobytes() for i in range(view.shape[0])]
    counts: dict[bytes, int] = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    return int(sum(v - 1 for v in counts.values() if v > 1))


def assign_status(
    spec: DatasetSpec,
    parsed: ParsedDataset | None,
    audit: dict[str, Any] | None,
    *,
    download_ok: bool,
    parse_ok: bool,
    provenance_tier: str,
) -> tuple[str, str]:
    """Assign exactly one Stage 1 status, with a human-readable reason."""
    if not download_ok:
        return "DOWNLOAD_FAILED", "no raw bytes were acquired from any candidate source"
    if parsed is None or not parse_ok:
        return "PARSE_FAILED", "raw bytes were acquired but could not be parsed into an unambiguous dataset"

    if audit is None:
        return "PARSE_FAILED", "the dataset parsed but produced no audit; refusing to assign a status"
    n_ok = parsed.n_samples == spec.n_samples
    p_ok = parsed.n_features == spec.n_features
    c_ok = int(audit["n_classes"]) == spec.n_classes

    if not (n_ok and p_ok and c_ok):
        diffs = []
        if not n_ok:
            diffs.append(f"samples {parsed.n_samples} != {spec.n_samples}")
        if not p_ok:
            diffs.append(f"features {parsed.n_features} != {spec.n_features}")
        if not c_ok:
            diffs.append(f"classes {audit['n_classes']} != {spec.n_classes}")
        return STATUS_SHAPE_MISMATCH, "Table 1 mismatch: " + "; ".join(diffs)

    if not audit["n_labels_equals_rows"] or not audit["label_map_is_reversible"]:
        return STATUS_LABEL_MISMATCH, "label vector does not align with rows, or the label map is not reversible"

    if audit["nonfinite"]["total"] > 0:
        return (
            STATUS_DATA_QUALITY_BLOCKED,
            (
                f"{audit['nonfinite']['total']} non-finite or missing feature values; "
                "Stage 1 does not impute, so canonical use is blocked pending a "
                "train-fold-only preprocessing decision"
            ),
        )

    if provenance_tier in SUBSTANTIATED_TIERS:
        return (
            STATUS_EXACT_VERIFIED,
            (
                f"dimensions ({spec.n_samples}, {spec.n_features}, {spec.n_classes}) match "
                f"Table 1, labels align and are reversible, no non-finite values, and "
                f"lineage is substantiated by a {provenance_tier} source"
            ),
        )
    return (
        STATUS_SHAPE_MATCH_PROVENANCE_UNVERIFIED,
        (
            f"dimensions ({spec.n_samples}, {spec.n_features}, {spec.n_classes}) and labels "
            f"match Table 1, but the variant came from an {provenance_tier} record and the "
            "paper does not state which release it consumed, so exact scientific "
            "equivalence is not proven"
        ),
    )
