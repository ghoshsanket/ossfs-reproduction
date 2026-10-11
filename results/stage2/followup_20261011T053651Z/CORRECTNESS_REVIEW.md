# Stage 2 Correctness Review — R1 to R6

Repaired 2026-10-11 against `doodzhou/OSFS@845a8e9d5625616b43e8cee16e8d22081dbe3f62`.
Evidence method: the pre-fix `selector.py` was restored from git HEAD and the same
test file was run against it, so every "fail before" below is a real observation,
not a recollection.

---

## R1 — Wrong threshold forwarded to the redundancy check (CONFIRMED, primary mode affected)

**Root cause.** `src/ossfs_dd/selector.py`, the ACCEPT branch of `consume()` called
`self._accept(cand, alpha, std)` while `_accept(self, cand, beta, std)` forwards its
second argument to `_redundancy_pass(new, beta)`, whose criterion is `joint < 2*beta`
(`OSFS_DD_mi2.m:138`). `alpha` therefore became `beta`, so the live selector compared
against `2*alpha`.

**Proof before fix.** With alpha=0.10, beta=0.20, joint=0.30:
`0.30 < 2*0.20 = 0.40` → redundant (**TRUE**, reference).
Observed: `removed=[]`; the forwarded value was `0.1`.

**Correction.** Pass `beta` at the call site. `test_R1_forwarded_threshold_is_beta`
pins the plumbing directly; `test_R1_redundancy_uses_beta_not_alpha` traverses
`consume → _accept → _redundancy_pass`.

**Impact.** Large on sparse data, where redundancy is the dominant filter:
DEXTER redundancy removals 25 → 352, selected 797 → 470.

---

## R2 — MATLAB's first-match inner break was missing (CONFIRMED)

**Root cause.** `_flush_pool()` iterated `self.pool.iter_pairs()` with **no break at
all**, despite a comment claiming one. `OSFS_DD_mi2.m:112-122` breaks the INNER loop
on the first qualifying pair for each outer `i`, and continues the outer loop.

**Proof before fix.** Oracle: `evaluated == [(0,1),(1,2),(2,3)]`; observed differed and
pair `(0,2)` — which qualifies but is unreachable after `(0,1)` matches — was evaluated.

**Correction.** True nested loops with an inner-only `break`. Also removes the wasted
`O(n²)` scan, which is why ARCENE fell from 12.3 s to 1.9 s.

---

## R3 — paper_spec warm-up double-counted scores (CONFIRMED, paper-spec ONLY)

**Root cause.** In the `pos <= 2` branch the moments were written directly from
`seed_from_values(...)` with `count = pos`, and for `paper_spec_v1` execution then
fell through to `self.moments.update(score)`, folding the same score in a **second**
time. The first score started at count 2 and the second at count 3, skewing every
subsequent threshold.

**Counterevidence — R3 does NOT affect the MATLAB route.** The MATLAB branch `return`s
inside the warm-up block. `test_R3_moment_count_matches_scores_seen[matlab_compatible_v1]`
passed both before and after. **No speculative edit was made to the MATLAB path.**

**Correction.** `paper_spec_v1` now folds each score exactly once through the normal
update and routes the feature through the ordinary decision. Algorithm 1 prescribes no
decision when one score exists and `sigma == 0`; the literal rule yields REJECT
(`score <= alpha`), which is recorded as an interpretation rather than forced into an
accept. No epsilon, no lookahead, `alpha < beta` never relaxed.
Versioned in `configs/stage2.yaml` as `paper_spec_v1.warmup_bootstrap_r2`; the previous
wording is retained as `warmup_bootstrap_superseded`.

---

## R4 — Promotion order followed feature IDs, not arrival positions (CONFIRMED)

**Root cause.** `_flush_pool()` applied `sorted(set(promoted))` to original
zero-based feature IDs. `OSFS_DD_mi2.m:124` applies `unique()` to the **1-based
arrival positions**. With a shuffled stream these differ, and candidate insertion
order drives every later redundancy traversal.

**Proof before fix.** Arrival positions 0,1,2 carrying IDs 900, 5, 100 produced
promoted arrival positions `[1, 2, 0]` — not ascending.

**Correction.** Sort and de-duplicate by arrival position, then map back to feature
IDs. The ID mapping is preserved exactly (`test_R4_arrival_position_to_id_mapping_is_preserved`).

---

## R5 — `x or nan` erased a legitimate alpha of 0.0 (CONFIRMED)

`self.last_alpha or float("nan")` treats a legitimate `0.0` as false and records NaN.
Both the ACCEPT event and the pool-flush event were affected. Replaced with an explicit
`is not None` check; two tests cover it.

---

## R6 — Continuous route remains fail-closed (no change, by design)

`Fisher_Ztest` is still absent from the pinned repository. S6/S7/S8 remain
`UNRESOLVED`, the four paper-`Real` datasets still return
`UNSUPPORTED_UNVERIFIED_SEMANTICS`, and no proxy statistic was introduced.

---

## Remaining uncertainties

1. The repaired code is verified against **source-derived oracles and documented
   MATLAB semantics**, not against executed MATLAB. `MIToolboxMex` ships only as a
   Windows `.mexw64` binary and no MATLAB/Octave exists on this host, so runtime
   parity stays `NOT_MEASURED`.
2. The joint quantity remains the source's scale-mixing expression
   (`SU + raw cmi`) in `matlab_compatible_v1`. That is faithful, not tidy, and is
   labelled as such.
3. A latent defect in the reference (with `P <= 2` features, `beta` is unassigned at
   the final flush) is recorded, not silently repaired.
