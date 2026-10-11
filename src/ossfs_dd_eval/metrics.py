"""Classification metrics with explicit denominators and bounded values."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

from .classifiers import EMPTY_STATUS


def score_outcome(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    zero_division: int,
    fit_meta: dict[str, object] | None = None,
    standardized: bool = False,
) -> dict[str, object]:
    """Accuracy plus two supplementary metrics, all with recorded denominators."""
    n_test = int(y_true.shape[0])
    correct = int(np.sum(y_pred == y_true))
    outcome: dict[str, object] = {
        "classifier": name,
        "status": "COMPLETED",
        "n_test": n_test,
        "n_correct": correct,
        "accuracy": (correct / n_test) if n_test else None,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=zero_division)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "zero_division": zero_division,
        "preprocessing": "STANDARDIZED_SENSITIVITY" if standardized else "RAW_PRIMARY",
    }
    if fit_meta:
        outcome.update(fit_meta)
    return outcome


def empty_selection_outcome(name: str, n_selected: int) -> dict[str, object]:
    """Empty selection is a valid result, not an error.

    Accuracy is ``None``, never ``0.0``: substituting zero, all features or the
    majority class would silently change the question being asked.
    """
    return {
        "classifier": name,
        "status": EMPTY_STATUS,
        "n_test": None,
        "n_correct": None,
        "accuracy": None,
        "macro_f1": None,
        "balanced_accuracy": None,
        "n_selected": 0,
        "note": f"selector returned an empty subset ({n_selected}); not evaluable, not zero",
    }
