"""Conditional (partial-correlation) pair operator for continuous features.

Measures conditional LINEAR dependence only. Does not detect XOR-style
nonlinear synergy and is not equivalent to MI/CMI or the missing authors'
Fisher_Ztest helper. See standard_fisherz.pair_score_standard.
"""

from __future__ import annotations

from .standard_fisherz import (  # noqa: F401
    FisherMarginal,
    marginal_fisherz,
    pair_score_standard,
    partial_correlation,
)
