# Paper vs Authors' MATLAB — Auditable Discrepancy Table

Companion to `docs/stage2_source_audit.md`. Source: `doodzhou/OSFS` at commit
`845a8e9d5625616b43e8cee16e8d22081dbe3f62`.

Labels: `PAPER` (stated in the article) · `SOURCE_CONFIRMED` (read in the code) ·
`INTERPRETATION` (our documented choice where the source is silent) ·
`UNRESOLVED` (not determinable from available evidence).

---

## Q1 — Does the implementation act on the first two features?

| | |
|---|---|
| **Label** | `SOURCE_CONFIRMED` |
| **Paper** | Algorithm 1 (p. 12) gives no special case for the first two features. |
| **Source** | `mi2.m:37-41` scores them, updates mean/std, then `continue`. They are never added to `AccIndexs` or `DeterIndexs`. The comment says "default select"; the statements do not select. |
| **Consequence** | In `matlab_compatible_v1` the first two arrival positions are **always discarded**, whatever their scores. |
| **Also** | `z2.m:41-46` behaves **differently**: it *does* append them to `AccIndexs`. The two author files disagree with each other. |
| **Our handling** | `paper_spec_v1` processes the first two features through the normal decision path using a documented bootstrap (see Q1-bis). `matlab_compatible_v1` reproduces each file's own behaviour. |

### Q1-bis — Bootstrap under `paper_spec_v1`

`INTERPRETATION`. With two scores, `alpha == beta` when `sigma = 0`, which
violates the `alpha < beta` assumption the prompt forbids silently relaxing.
We **declare** the bootstrap: the first two features are scored, the running
mean/variance are initialised from them, and they are then routed through the
ordinary three-way decision using thresholds from the running statistics. No
relaxation of `alpha < beta` is performed and no feature is forced in.

---

## Q2 — Is the running variance sample variance, and how is `t=1,2` bootstrapped?

| | |
|---|---|
| **Label** | `SOURCE_CONFIRMED` |
| **Source** | `mi2.m:95-103`. `Sn = (1/(Num-1))*((Num-2)*std^2 + ((Num-1)/Num)*(newVal-mean)^2)` with `Num>2`, else `std_new = 0`. This is the Welford update for the **sample variance, `ddof=1`**. |
| **Bootstrap** | `Num = i`, the 1-based arrival index. At `i=3` the update consumes the two prior scores. At `i<=2` the main loop short-circuits (`mi2.m:37-41`) and mean/std come from `mean()`/`std()` of the scores seen so far. |
| **Our handling** | Reproduced exactly, and cross-checked against `numpy.std(ddof=1)` at every prefix. |

---

## Q3 — Is the score normalised MI, and in what units?

| | |
|---|---|
| **Label** | `SOURCE_CONFIRMED` |
| **Source** | `mi2.m:33` uses `SU(X,Y)`; `SU.m:3-7` computes `2*iXY/(hX+hY)`. |
| **Units** | **BITS.** `Entropy.c:50` computes `-= p*log(p)` then **line 54 `entropy /= log(2.0)`**. `MutualInformation.c:63` does the same. The `SU` ratio is base-independent, but `cmi` is reported in raw bits. |
| **Zero-entropy case** | `UNRESOLVED` in the source: if both entropies are 0 the C helper returns 0, so `SU = 0/0 = NaN` in MATLAB. `mi2.m` has **no** `isnan` guard on `mi_F` (unlike `z2.m:35-37`). |
| **Our handling** | `matlab_compatible_v1` reproduces the NaN-propagating behaviour and flags it; `paper_spec_v1` emits the documented neutral value 0. The difference is recorded per run. |

---

## Q4 — How is joint information `I({f_i,f_j};y)` computed?

| | |
|---|---|
| **Label** | `SOURCE_CONFIRMED` — **the source mixes a normalised score with a raw one** |
| **Source** | `mi2.m:137` `jointMI = cmi(X(:,ind),Y,X(:,newIndex)) + indMI` and `mi2.m:116` `jointMI = MIArray(ind) + cmi(X(:,ind), Y, X(:,ind_joint))`, where `indMI` is `SU`. |
| **Assessment** | `SU ∈ [0,1]` is **normalised**; `cmi` is **raw bits**, unbounded above by 1. The sum therefore compares against `2*beta`, a threshold built from `SU`-scale statistics. **The two operands are on different scales.** |
| **Explicitly not done** | We do **not** silently replace this with `SU(X,Y) + cmi`, `I({X,Z};Y)`, or any coherent joint information. The mixing is reproduced in `matlab_compatible_v1` and labelled. |
| **Our handling** | `paper_spec_v1` uses a coherent, preregistered definition: normalised MI of the **joint** feature variable, `SU({X,Z};Y) = 2·MI({X,Z};Y) / (H({X,Z})+H(Y))`, which is base-independent and scale-consistent with `2*beta`. The two modes are therefore **not** numerically interchangeable, and results are never compared across them as if they were. |

---

## Q5 — Redundancy comparator: `<2*beta`, weaker-of-two, which tie-break?

| | |
|---|---|
| **Label** | `SOURCE_CONFIRMED` |
| **Source** | `mi2.m:138` `if jointMI < 2*beta` (strict). `mi2.m:139-144`: `if newMI<indMI` delete the new feature and break, **else** delete the existing one and continue. |
| **Tie-break** | `newMI == indMI` falls to the `else` branch → the **existing** feature is removed. Ties favour the newer feature. |
| **Comparison order** | `AccIndexs(1..count-1)` in insertion order; the appended feature is skipped because the loop bound is `count-1` after it was added. |
| **Output** | `delIndx` may list several existing features plus possibly the new one; the caller removes all of them (`mi2.m:61-68`). |

---

## Q6 — Uncertain-pair promotion: `>` or `>=`? Stop on first match? Feature in two pairs?

| | |
|---|---|
| **Label** | `SOURCE_CONFIRMED` |
| **Source** | `mi2.m:117` `if jointMI>2*beta` — strict `>`, **not** `>=`. |
| **Pair search** | `mi2.m:112-122` — outer loop over `i=1..DeterCount`, inner `j=(i+1)..DeterCount`. The inner loop `break`s on the first qualifying pair, but the **outer loop continues**, so one feature **may** appear in several pairs. |
| **Promotion** | **Both** members are appended (`mi2.m:118`), not only one. |
| **Deduplication** | `mi2.m:124` `unique(selectedIndex)` → ascending, de-duplicated. |
| **Cost** | Worst case `O(DeterCount^2)` cmi evaluations per flush. |

---

## Q7 — Pool flush at `==N`, `>=N`, and again at stream end?

| | |
|---|---|
| **Label** | `SOURCE_CONFIRMED` |
| **Source** | `mi2.m:73` `if deterCount>=PoolSize` — **`>=`**. Flush at line 74-76, pool cleared. |
| **End of stream** | `mi2.m:82-86` — a final flush runs if `DeterIndexs` is non-empty, using the **last computed `beta`**. A trailing singleton is processed like any other (the inner loop simply never executes, so it is dropped unless an earlier `j` pair qualified). |
| **Latent defect** | If the stream has `P<=2`, `beta` is never assigned before line 83 references it → the original errors. Recorded; not silently "fixed" in `matlab_compatible_v1`, which returns the condition as a structured error. |

---

## Q8 — Is redundancy re-checked for promoted pairs?

| | |
|---|---|
| **Label** | `SOURCE_CONFIRMED` |
| **Source** | `mi2.m:74-76` — `flushDeterPool` results are concatenated straight into `AccIndexs`. `checkPoolRedundancy` is invoked **only** on the immediate-accept path (`mi2.m:60`). |
| **Consequence** | Promoted pairs can be mutually redundant and can duplicate an existing candidate. The final `AccIndexs` may therefore contain redundant members, and **may contain duplicates** if the same MATLAB index is promoted twice before `unique` merges them — note `unique` is applied *inside* `flushDeterPool` but `AccIndexs` concatenation is not re-uniqued. |
| **Our handling** | We reproduce this and **record duplicate occurrences in the trace** rather than hiding them. The Stage-2 gate "0 duplicate final selected IDs" is therefore evaluated on the *emitted* ID list; if the author semantics produce duplicates we report the count rather than silently de-duplicating. |

---

## Q9 — What does `OSFS_DD_z2.m` really calculate for continuous features?

| | |
|---|---|
| **Label** | **`UNRESOLVED`** |
| **Source** | `z2.m:34` `[~, fz_F] = Fisher_Ztest(X(:,i),Y)`; `z2.m:123` and `z2.m:151` `[CI_cmi,cmi] = Fisher_Ztest(...)`. |
| **Blocker** | `Fisher_Ztest.m` **does not exist anywhere in the repository**. An exhaustive tree walk over all 67 blobs returns zero matches for `fisher`/`ztest`. |
| **What is *not* determinable** | whether `fz_F` is a z-statistic, a p-value, a correlation or a rescaled score; its units; its sign convention; the conditional-independence test behind `CI_cmi`; behaviour on constant/singular features beyond the `isnan → 0` guard at `z2.m:35-37`. |
| **What *is* determinable** | `fz_F` must be a finite real scalar comparable across features (it is compared against thresholds from its own running mean/std); `cmi` is added to a marginal score before comparison with `2*beta`, reproducing the discrete file's scale mixing. |
| **Decision** | The continuous route returns `UNSUPPORTED_UNVERIFIED_SEMANTICS`. **No proxy is substituted.** `1-p`, `|r|`, `p`, correlation magnitude, discretised MI and `mutual_info_classif` are all explicitly rejected as substitutes. |

---

## Q10 — Does the code act on input column order and return 1-based positions?

| | |
|---|---|
| **Label** | `SOURCE_CONFIRMED` |
| **Source** | The loop is `for i=1:P` over `X(:,i)`, i.e. **input column order**. `MIArray(ind)` indexes by the same `i`, so the score history is aligned to input positions. `selectedFeatures=AccIndexs` is a list of **1-based input column indices**. |
| **Mapping to Stage 1** | Stage 1 delivers `arrival_order` (a permutation) and requires **original zero-based feature IDs**. Our mapping is `original_feature_id = arrival_order[matlab_index - 1]`. Arrival position and original feature ID are stored in **separate fields** throughout. |

---

## Summary of paper-vs-source divergences

| # | Issue | Paper | Source | Our position |
|---|---|---|---|---|
| 1 | First two features | unspecified | `mi2` skips them; `z2` selects them | mode-specific, documented |
| 2 | Sample variance | Eqs. 5–6 | `ddof=1` Welford | reproduced |
| 3 | Score units | unspecified | bits | documented |
| 4 | Joint information | `I({f_i,f_j};y)` | `SU + raw cmi` (mixed scales) | mode-specific; never silently tidied |
| 5 | Redundancy | — | `<2β`, weaker removed, tie favours new | reproduced |
| 6 | Promotion | — | `>2β`, both promoted, outer loop continues | reproduced |
| 7 | Flush | — | `>=N` and final flush | reproduced |
| 8 | Promoted-pair redundancy | — | not re-checked | reproduced, duplicates counted |
| 9 | Continuous score | Fisher Z-test | `Fisher_Ztest` absent | **`UNSUPPORTED_UNVERIFIED_SEMANTICS`** |
| 10 | Index convention | — | 1-based input positions | mapped to Stage-1 original IDs |

The `z2` accept-before-reject ordering also flips behaviour at `sigma=0`
(`alpha==beta`), where `mi2` rejects and `z2` accepts. This is recorded because
it is observable and would silently corrupt a comparison.