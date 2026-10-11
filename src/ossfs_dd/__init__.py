"""OSSFS-DD — online scalable streaming feature selection via dynamic decision.

Stage 2 of the `ossfs-dd-reproduction` project.

This package implements the three-way (select / discard / delay) streaming
feature-selection framework of Zhou, Zhao, Yan & Wu (ACM TKDD 16(5) art. 87,
2022), under two deliberately different modes:

``paper_spec_v1``
    Algorithm 1 and the equations read literally, with every place the paper is
    silent resolved by a declared, preregistered choice.

``matlab_compatible_v1``
    The observable behaviour of the authors' released MATLAB, which disagrees
    with the paper in several operational details.

**No classifier is fitted and no accuracy is computed anywhere in this
package.** Feature counts and runtimes are engineering diagnostics only.
"""

from .config import (
    MetricFamily,
    Mode,
    SelectionConfig,
    UnsupportedSemanticsError,
    load_stage2_config,
)
from .decisions import Decision, decide
from .metrics import (
    conditional_mutual_information,
    entropy,
    mutual_information,
    symmetrical_uncertainty,
)
from .moments import RunningMoments
from .pools import UndeterminedPool
from .reference_semantics import REFERENCE_SEMANTICS, lookup
from .selector import SelectionResult, Selector, SelectorState

__version__ = "0.2.0"

__all__ = [
    "REFERENCE_SEMANTICS",
    "Decision",
    "MetricFamily",
    "Mode",
    "RunningMoments",
    "SelectionConfig",
    "SelectionResult",
    "Selector",
    "SelectorState",
    "UndeterminedPool",
    "UnsupportedSemanticsError",
    "conditional_mutual_information",
    "decide",
    "entropy",
    "load_stage2_config",
    "lookup",
    "mutual_information",
    "symmetrical_uncertainty",
]
