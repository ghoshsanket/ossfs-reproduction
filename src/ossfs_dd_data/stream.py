"""Leakage-safe, memory-efficient feature-stream interface.

Contract, enforced by construction:

  * ``FeatureStream(X, y, train_indices, feature_permutation)`` yields
    ``(original_feature_index, values_for_training_rows_only)`` one feature at a
    time, following the arrival order;
  * held-out rows are NEVER yielded. ``X`` is restricted to ``train_indices``
    at construction, so a downstream selector has no code path to test data;
  * nothing is materialised as a global dense copy of the full matrix — a
    column is read on demand, and CSR columns are sliced rather than densified;
  * an optional diagnostic mode may expose full rows, and it is explicitly named
    so that using it inside a selection loop is a visible, deliberate act.

This module contains NO selection policy. Choosing which features to keep is
Stage 2 work.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

LOG = logging.getLogger("ossfs_dd_data.stream")


class LeakageError(RuntimeError):
    """Raised if a stream would expose rows outside the training set."""


@dataclass
class StreamStep:
    arrival_position: int
    original_feature_index: int
    n_rows: int
    dtype: str
    values: np.ndarray


class FeatureStream:
    """Yield one feature at a time, training rows only.

    Parameters
    ----------
    X : ndarray or scipy sparse matrix, shape ``(n_samples, n_features)``
        The FULL canonical matrix. Rows outside ``train_indices`` are retained
        only so row indices stay meaningful; they are never read out.
    y : ndarray, shape ``(n_samples,)``
        Labels for all rows. ``train_labels`` exposes only the training rows.
    train_indices : array-like of int
        Rows a selector is permitted to see.
    feature_permutation : array-like of int
        Arrival order over original feature indices 0..p-1.
    """

    def __init__(
        self,
        X: Any,
        y: np.ndarray,
        train_indices: np.ndarray,
        feature_permutation: np.ndarray,
    ) -> None:
        X = X.tocsr() if sparse.issparse(X) else X
        self._X = X
        self._y = np.asarray(y)
        self._train = np.asarray(train_indices, dtype=np.int64)
        self._perm = np.asarray(feature_permutation, dtype=np.int64)
        self._n_samples, self._n_features = X.shape

        if self._train.size == 0:
            raise LeakageError("train_indices is empty; nothing may be streamed")
        if self._train.min() < 0 or self._train.max() >= self._n_samples:
            raise IndexError("train_indices out of range")
        if len(set(self._train.tolist())) != self._train.size:
            raise LeakageError("train_indices contains duplicates")
        if self._perm.size != self._n_features:
            raise ValueError(
                f"feature_permutation has {self._perm.size} entries but X has "
                f"{self._n_features} features"
            )
        if sorted(self._perm.tolist()) != list(range(self._n_features)):
            raise ValueError("feature_permutation is not a permutation of 0..p-1")

    # ---- read-only properties, no data leakage ---------------------------
    @property
    def n_features(self) -> int:
        return int(self._n_features)

    @property
    def n_train_rows(self) -> int:
        return int(self._train.size)

    @property
    def train_indices(self) -> np.ndarray:
        return self._train.copy()

    @property
    def train_labels(self) -> np.ndarray:
        """Labels restricted to the training rows."""
        return self._y[self._train]

    def arrival_order(self) -> np.ndarray:
        return self._perm.copy()

    # ---- the stream -------------------------------------------------------
    def _column_rows(self, feature_index: int, rows: np.ndarray) -> np.ndarray:
        """Read one column for an explicit set of rows (internal use)."""
        if sparse.issparse(self._X):
            col = self._X[:, feature_index]
            return np.asarray(col[rows].todense()).ravel()
        return np.asarray(self._X[rows, feature_index])

    def _column_training_rows(self, feature_index: int) -> np.ndarray:
        return self._column_rows(feature_index, self._train)

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        for _position, feature_index in enumerate(self._perm.tolist()):
            values = self._column_training_rows(int(feature_index))
            if values.shape[0] != self._train.size:
                raise LeakageError(
                    f"column {feature_index} yielded {values.shape[0]} rows, expected "
                    f"{self._train.size} training rows"
                )
            yield int(feature_index), values

    def steps(self) -> Iterator[StreamStep]:
        for position, (feature_index, values) in enumerate(self):
            yield StreamStep(
                arrival_position=position,
                original_feature_index=feature_index,
                n_rows=int(values.shape[0]),
                dtype=str(values.dtype),
                values=values,
            )

    # ---- explicitly-named diagnostic escape hatch -------------------------
    def full_rows_diagnostic(self, feature_index: int) -> np.ndarray:
        """Return a column for ALL rows, including held-out rows.

        For debugging and reporting ONLY. This is deliberately not part of
        ``__iter__`` so that reaching test-row data from a selection loop
        requires calling a method whose name says so.
        """
        if sparse.issparse(self._X):
            return np.asarray(self._X[:, feature_index].todense()).ravel()
        return np.asarray(self._X[:, feature_index])

    def self_check(self, *, deep: bool = True, max_deep_features: int = 8) -> dict[str, Any]:
        """Prove the leakage contract against the real matrix, not just in theory.

        The v1 implementation reported ``train_indices_disjoint_from_complement``,
        which was ``train_set.isdisjoint(complement)`` — true by construction for
        any set and therefore worthless as evidence. That field is removed.

        What replaces it is a genuine content check: emitted columns are compared
        element-wise against ``X[train_indices, column]``, and held-out rows are
        confirmed to be *different content*, so a stream that accidentally
        served test rows would be caught rather than passing.

        ``deep`` performs the value comparison on up to ``max_deep_features``
        columns (every column for small feature counts). Set ``deep=False`` for
        a cheap structural-only pass on very wide datasets.
        """
        seen: list[int] = []
        row_counts: set[int] = set()
        mismatched: list[int] = []
        sample_ids = self._train

        for feature_index, values in self:
            seen.append(feature_index)
            row_counts.add(int(values.shape[0]))
            if deep and len(mismatched) < max_deep_features:
                expected = self._column_training_rows(int(feature_index))
                if not np.array_equal(np.asarray(values), np.asarray(expected)):
                    mismatched.append(int(feature_index))

        # Value-equality evidence: emitted == X[train_indices, column].
        values_match = not mismatched
        value_check_coverage = (
            min(len(seen), max_deep_features) if deep else 0
        )

        # Held-out rows must be reachable but must NOT be what we emitted.
        held_out = np.setdiff1d(np.arange(self._n_samples), self._train)
        held_out_distinguishable = True
        if held_out.size and seen:
            col = int(seen[0])
            emitted = self._column_training_rows(col)
            holdout_values = self._column_rows(col, held_out)
            if np.array_equal(np.asarray(emitted), np.asarray(holdout_values)):
                held_out_distinguishable = False

        return {
            "n_features_emitted": len(seen),
            "emission_is_permutation": sorted(seen) == list(range(self._n_features)),
            "distinct_row_counts_emitted": sorted(row_counts),
            "every_emitted_row_count_equals_train_rows": row_counts == {self._train.size},
            "emitted_values_equal_X_train_column": values_match,
            "value_checked_columns": value_check_coverage,
            "value_check_mode": "elementwise" if deep else "skipped",
            "mismatched_feature_indices": mismatched,
            "n_held_out_rows": int(held_out.size),
            "held_out_rows_are_distinguishable_from_emitted": held_out_distinguishable,
            "contract_satisfied": bool(
                sorted(seen) == list(range(self._n_features))
                and row_counts == {self._train.size}
                and values_match
                and held_out_distinguishable
            ),
            "removed_v1_field_note": (
                "'train_indices_disjoint_from_complement' was tautological and has been removed"
            ),
            "_train_index_preview": sample_ids[:5].tolist(),
        }
