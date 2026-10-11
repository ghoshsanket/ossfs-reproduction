"""Candidate subset and undetermined (uncertainty) pool.

Memory policy
-------------
Only feature *vectors* that are still needed are retained:

* a selected candidate keeps its vector while it remains a candidate;
* a delayed feature keeps its vector while it sits in the undetermined pool;
* a rejected feature keeps **no vector at all** — only a bounded scalar trace
  entry (score, arrival position, original feature ID).

Nothing is ever re-read from the Stage-1 global matrix after arrival, so a
removed feature cannot be resurrected by a later decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Candidate:
    """A selected feature still under consideration."""

    feature_id: int          # original zero-based Stage-1 feature ID
    arrival_position: int    # 0-based position in arrival_order
    score: float
    vector: list[float] | None = field(default=None, repr=False)


@dataclass
class UndeterminedPool:
    """The delay pool, in arrival order.

    Pair search order is significant: ``mi2.m:112-122`` walks
    ``i`` over the pool and ``j`` over ``(i+1)..end``, breaking the inner loop
    on the first qualifying pair while the outer loop continues.
    """

    pool_size: int = 100
    items: list[Candidate] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.pool_size < 1:
            raise ValueError(f"pool_size must be >= 1, got {self.pool_size}")

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, idx: int) -> Candidate:
        return self.items[idx]

    @property
    def full(self) -> bool:
        """``>=`` pool size, matching ``mi2.m:73``."""
        return len(self.items) >= self.pool_size

    def add(self, item: Candidate) -> None:
        self.items.append(item)

    def clear(self) -> list[Candidate]:
        taken, self.items = self.items, []
        return taken

    def iter_pairs(self):
        """Yield ``(i_index, j_index)`` in the reference's search order."""
        n = len(self.items)
        for i in range(n):
            for j in range(i + 1, n):
                yield i, j

    def peak(self) -> int:
        return len(self.items)


@dataclass
class CandidateSet:
    """The selected-feature candidate subset, in insertion order.

    ``mi2.m`` iterates ``AccIndexs(1..count-1)`` in insertion order, skipping
    the most recently appended feature, so insertion order is semantically
    load-bearing and must not be replaced by a set.
    """

    items: list[Candidate] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __contains__(self, feature_id: int) -> bool:
        return any(c.feature_id == feature_id for c in self.items)

    def add(self, item: Candidate) -> None:
        self.items.append(item)

    def remove(self, feature_id: int) -> bool:
        before = len(self.items)
        self.items = [c for c in self.items if c.feature_id != feature_id]
        return len(self.items) != before

    def duplicates(self) -> list[int]:
        """Feature IDs appearing more than once.

        The reference does not re-uniquify ``AccIndexs`` after concatenating a
        pool flush, so duplicates are possible. They are *counted and reported*,
        never silently removed.
        """
        seen: dict[int, int] = {}
        for c in self.items:
            seen[c.feature_id] = seen.get(c.feature_id, 0) + 1
        return sorted(fid for fid, n in seen.items() if n > 1)

    def feature_ids(self) -> list[int]:
        return [c.feature_id for c in self.items]

    def peak(self) -> int:
        return len(self.items)
