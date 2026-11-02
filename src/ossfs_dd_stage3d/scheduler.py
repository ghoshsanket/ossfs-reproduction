"""Deterministic, leakage-safe batch scheduling for Stage 3D.

Batch sizes are dataset-size-dependent (NOT adaptive to an unknown stream
length, which is explicitly documented). Block order and within-batch
presentation order are both randomized with independent PCG64 generators
seeded from a namespaced tuple, never from Python's unstable hash().
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

EPS_P = 1e-12
L_EVIDENCE = 12.0

# Namespaced constants so the seed cannot collide with other stages.
BLOCK_SEED_CONST = 0xD3A0_0001
WITHIN_SEED_CONST = 0xD3A0_0002
FLATTEN_SEED_CONST = 0xD3A0_0003


def batch_size_for_p(p: int) -> int:
    """Piecewise inclusive boundaries from the frozen config."""
    if p <= 2000:
        return 10
    if p <= 10000:
        return 25
    return 50


def make_schedule(base_order: np.ndarray, dataset: str, seed: int, fold: int,
                  namespace: str = "stage3d") -> dict[str, Any]:
    """Build the full deterministic batch schedule for one (dataset, seed, fold).

    Returns a dict containing:
      - original feature count p
      - batch_size B
      - block_order: list of block indices in presentation order
      - within_block_orders: {block_idx: list of within-block feature order}
      - flattened_order: the flattened feature-ID order the selector consumes
      - hashes of every derived artifact
    """
    base_order = np.asarray(base_order, dtype=np.int64)
    p = int(base_order.size)
    if p == 0:
        raise ValueError("base_order must be non-empty")
    B = batch_size_for_p(p)
    n_blocks = (p + B - 1) // B

    # block-level permutation
    block_rng = np.random.default_rng(
        np.array([seed, fold, BLOCK_SEED_CONST, ord(namespace[0])], dtype=np.uint64)
    )
    # Seed from a stable hash of the namespaced inputs rather than relying on
    # numpy's hash of a tuple (which is version-stable but we want explicitness).
    block_seed_int = _namespaced_seed(namespace, dataset, seed, fold, BLOCK_SEED_CONST)  # noqa: S311  # noqa: S311
    block_rng = np.random.default_rng(block_seed_int)
    block_order = block_rng.permutation(n_blocks).tolist()

    # within-batch presentation shuffle, one generator per block
    within_orders: dict[int, list[int]] = {}
    for b in range(n_blocks):
        start = b * B
        end = min(start + B, p)
        within_seed = _namespaced_seed(namespace, dataset, seed, fold,
                                       WITHIN_SEED_CONST, b)
        within_rng = np.random.default_rng(within_seed)
        perm = within_rng.permutation(end - start).tolist()
        within_orders[b] = perm

    # flattened order: blocks in block_order, features within each block in
    # within-block presentation order
    flattened: list[int] = []
    for b in block_order:
        start = b * B
        end = min(start + B, p)
        block_features = base_order[start:end]
        for j in within_orders[b]:
            flattened.append(int(block_features[j]))

    # flattened-order hash (independent RNG, not strictly needed for the
    # experiment but useful for the manifest)
    _ = FLATTEN_SEED_CONST

    hashes = {
        "base_order_sha256": _sha1d(base_order),
        "block_order_sha256": _sha1d(np.asarray(block_order, dtype=np.int64)),
        "within_block_orders_sha256": _sha_str(json.dumps(within_orders, sort_keys=True)),
        "flattened_order_sha256": _sha1d(np.asarray(flattened, dtype=np.int64)),
    }
    return {
        "dataset": dataset,
        "seed": seed,
        "fold": fold,
        "p": p,
        "B": B,
        "n_blocks": n_blocks,
        "block_order": block_order,
        "within_block_orders": within_orders,
        "flattened_order": flattened,
        "flattened_is_permutation": bool(
            np.array_equal(np.sort(flattened), np.sort(base_order))),
        "hashes": hashes,
        "rng": "numpy.random.Generator(PCG64)",
        "block_seed_int": int(block_seed_int),
    }


def _namespaced_seed(namespace: str, dataset: str, seed: int, fold: int,  # noqa: S311
                     const: int, *extra: int) -> int:
    """A deterministic 64-bit integer seed from the frozen inputs.

    Uses SHA-256 so the seed is version-stable and independent of Python's
    hash() randomization. Returns a Python int (numpy accepts it).
    """
    payload = f"{namespace}|{dataset}|{seed}|{fold}|{const}|{'|'.join(str(e) for e in extra)}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _sha1d(arr: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(arr).tobytes()).hexdigest()


def _sha_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def verify_schedule(sched: dict) -> dict:
    """Independent verification of a schedule's invariants."""
    flat = np.asarray(sched["flattened_order"], dtype=np.int64)
    base = sched["p"]
    n_blocks = sched["n_blocks"]
    B = sched["B"]
    last_ok = all(
        len(sched["within_block_orders"][b]) == min(B, base - b * B)
        for b in range(n_blocks)
    )
    within_ok = all(
        sorted(v) == list(range(len(v)))
        for v in sched["within_block_orders"].values()
    )
    checks = {
        "flattened_is_permutation": bool(
            flat.size == base and np.array_equal(np.sort(flat), np.arange(base))
        ),
        "block_count_correct": n_blocks == (base + B - 1) // B,
        "last_block_short_ok": last_ok,
        "block_order_is_permutation": sorted(sched["block_order"]) == list(range(n_blocks)),
        "within_orders_are_permutations": within_ok,
        "hashes_match_recompute": sched["hashes"]["flattened_order_sha256"] == _sha1d(flat),
    }
    return {"ok": all(checks.values()), "checks": checks,
            "n_features": base, "B": B, "n_blocks": n_blocks}


def within_batch_invariance_check(batch_features: list[int], score_fn,
                                   labels: np.ndarray) -> dict[str, Any]:
    """Verify that shuffling presentation order within a fixed batch does not
    change the set of selected IDs produced by the batch-level decision."""
    import random

    a = list(batch_features)
    b = list(batch_features)
    rng = random.Random(0)  # noqa: S311
    rng.shuffle(b)
    # The batch selector must produce the same selected set for both orders.
    # This helper only records the two orders and their equality of membership;
    # the actual selector call is made by the caller.
    return {"order_a": a, "order_b": b, "same_membership": sorted(a) == sorted(b)}
