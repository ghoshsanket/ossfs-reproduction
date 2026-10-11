"""Stable online mean and sample standard deviation.

Reproduces ``updateMeanStd`` from the authors' ``OSFS_DD_mi2.m:95-103``:

    mean_new = mean + (newVal - mean)/Num;
    if Num > 2:
        Sn = (1/(Num-1))*((Num-2)*std^2 + ((Num-1)/Num)*(newVal-mean)^2);
        std_new = sqrt(Sn);
    else:
        std_new = 0;

``Num`` is the **1-based arrival index**, i.e. the number of scores observed
after the update. The variance is the **sample** variance (``ddof=1``).

The stream is fed *every* incoming relevance score — accepted, rejected and
delayed alike — never only the accepted ones. Thresholds are never derived
from future scores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RunningMoments:
    """Online mean and sample standard deviation (ddof=1)."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0  # sum of squared deviations (Welford accumulator)

    @property
    def std(self) -> float:
        """Sample standard deviation; 0.0 when fewer than two observations."""
        if self.count < 2:
            return 0.0
        return math.sqrt(self.m2 / (self.count - 1))

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def degenerate(self) -> bool:
        """True when sigma collapsed to zero, making alpha == beta."""
        return self.count >= 2 and self.std == 0.0

    def update(self, value: float) -> tuple[float, float]:
        """Fold one observation; return the updated ``(mean, std)``.

        Uses the numerically stable two-pass-equivalent form, which is the same
        recurrence as the reference but avoids repeated square-root/round-trip.
        """
        if not math.isfinite(value):
            raise ValueError(f"non-finite score {value!r} cannot enter the running moments")
        self.count += 1
        n = self.count
        delta = value - self.mean
        self.mean += delta / n
        self.m2 += delta * (value - self.mean)
        # m2 can go microscopically negative through cancellation; clamp.
        if self.m2 < 0.0:
            self.m2 = 0.0
        return self.mean, self.std

    def thresholds(self, k1: float, k2: float) -> tuple[float, float]:
        """alpha = mu + k1*sigma, beta = mu + k2*sigma."""
        s = self.std
        return self.mean + k1 * s, self.mean + k2 * s

    def snapshot(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "variance": self.variance,
            "degenerate": self.degenerate,
        }


def seed_from_values(values: list[float]) -> tuple[float, float]:
    """Bootstrap used by the first two features.

    Matches MATLAB's ``mean(MIArray)`` / ``std(MIArray)`` over the scores seen
    so far (``mi2.m:38-39``), including MATLAB's sample standard deviation.
    With a single observation MATLAB's ``std`` returns 0.
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = math.fsum(values) / n
    if n < 2:
        return mean, 0.0
    var = math.fsum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var)


def moments_vs_numpy(values: list[float]) -> dict[str, float]:
    """Compare the online moments against NumPy at the final prefix.

    Used by tests and by the self-check script as an *independent* oracle;
    NumPy is not used inside the selector itself.
    """
    import numpy as np

    rm = RunningMoments()
    for v in values:
        rm.update(v)
    arr = np.asarray(values, dtype=np.float64)
    np_mean = float(arr.mean()) if arr.size else 0.0
    np_std = float(arr.std(ddof=1)) if arr.size >= 2 else 0.0
    return {
        "online_mean": rm.mean,
        "numpy_mean": np_mean,
        "mean_abs_diff": abs(rm.mean - np_mean) if arr.size else 0.0,
        "online_std": rm.std,
        "numpy_std": np_std,
        "std_abs_diff": abs(rm.std - np_std) if arr.size else 0.0,
    }
