# Stage 2 Method Specification — OSSFS-DD

Implementation target: Zhou, Zhao, Yan & Wu, *Online Scalable Streaming Feature
Selection via Dynamic Decision*, ACM TKDD 16(5) art. 87 (2022).

This document states exactly what was implemented. Where the paper is silent or
disagrees with the authors' code, the resolution is named as an **interpretation**
rather than presented as fact.

---

## 1. Notation and the one mapping that matters

| Symbol | Meaning |
|---|---|
| `arrival_order[k]` | original **zero-based** feature ID arriving at position `k` |
| `feature_id` | the original Stage-1 feature ID — the only ID we emit |
| `arrival_position` | position in the stream — stored separately, never emitted as an ID |

The authors' MATLAB returns **1-based input column positions**
(`OSFS_DD_mi2.m:1,88`). Stage 1 delivers a shuffled `arrival_order` and
original zero-based IDs. The mapping is:

```
original_feature_id = arrival_order[matlab_index - 1]
```

`SelectorState` keeps the two apart; there is no path in the code that confuses
them, and `tests/test_stage2_algorithm.py::test_result_ids_are_original_zero_based_feature_ids`
asserts every emitted ID is in range.

---

## 2. Marginal relevance score

Discrete route (`Integer` datasets), from `OSFS_DD_mi2.m:33`:

```
SU(X, Y) = 2 * MI(X; Y) / (H(X) + H(Y))
```

**Units are bits.** `mi/Entropy.c:50` accumulates `-= p*log(p)` and line 54
applies `entropy /= log(2.0)`; `mi/MutualInformation.c:63` does the same. The
`SU` ratio is base-independent, but `cmi` is reported in raw bits.

Degenerate case: when both entropies are zero the MATLAB expression is `0/0` →
NaN. `matlab_compatible_v1` **propagates the NaN and records it in
`warnings`** (the reference file has no `isnan` guard, unlike `z2.m:35-37`).
`paper_spec_v1` emits the declared neutral value `0.0`. Neither mode ever
returns 1.0 for a constant feature.

Continuous route (`Real` datasets): **not implemented** — see §7.

---

## 3. Online moments

Reproduces `updateMeanStd` (`OSFS_DD_mi2.m:95-103`) exactly:

```
mean_n = mean_{n-1} + (x_n - mean_{n-1}) / n
Sn     = (1/(n-1)) * ((n-2) * S_{n-1} + ((n-1)/n) * (x_n - mean_{n-1})^2)     n > 2
std    = sqrt(Sn)        else 0
```

* sample variance, `ddof = 1`
* fed **every** incoming score — accepted, rejected and delayed alike
* `n` is the count of scores observed, i.e. the 1-based arrival index
* thresholds are never derived from future scores

Implemented in `RunningMoments` using the algebraically identical Welford
recurrence `m2 += (x - mean_old)(x - mean_new)`, which avoids a
square-root/round-trip per step. Validated against `numpy.mean` and
`numpy.std(ddof=1)` at **every prefix** to 1e-12 / 1e-10 absolute.

---

## 4. Three-way decision

```
alpha_t = mu_t + k1 * sigma_t          k1 = 1
beta_t  = mu_t + k2 * sigma_t          k2 = 2   (k1 < k2 enforced, never relaxed)
```

| Mode / entry | order | reject | accept |
|---|---|---|---|
| `matlab_compatible_v1` / `mi2` | reject first (`:52,56`) | `score <= alpha` | `score >= beta` |
| `matlab_compatible_v1` / `z2` | accept first (`:58,72`) | `score <= alpha` | `score >= beta` |
| `paper_spec_v1` | literal | `score <= alpha` | `score >= beta` |

**Observable divergence preserved:** when `sigma == 0` the score equals the
mean, so both conditions hold. `mi2` rejects; `z2` accepts. This is
reproduced rather than smoothed over, and is asserted by
`test_degenerate_differs_between_files`.

---

## 5. Redundancy and promotion

**Redundancy** (`mi2.m:128-150`), both modes:

```
for each existing candidate c (insertion order, newest excluded):
    joint = J(c, new)
    if joint < 2*beta:
        if new.score < c.score:  drop the NEW feature and stop
        else:                    drop c and continue
```

Strict `<`; the weaker member by marginal score goes; a tie takes the `else`
branch and removes the **existing** feature, so ties favour the newer one.

**Joint quantity — the two modes differ deliberately:**

| Mode | `J(A,B)` | Scale |
|---|---|---|
| `matlab_compatible_v1` | `SU(A;Y) + cmi(A, Y, B)` | **normalised + raw bits — mixed**, exactly as `mi2.m:116,137` |
| `paper_spec_v1` | `SU({A,B};Y)` = `2·MI({A,B};Y) / (H({A,B})+H(Y))` | scale-coherent |

The reference's mixing is not a reading error and is **not** tidied away. The
two modes are therefore not numerically interchangeable and results are never
compared across them as if they were.

**Promotion** (`mi2.m:107-125`):

```
for i in 1..D:                       # outer continues after the inner break
    for j in (i+1)..D:
        if J(D[i], D[j]) > 2*beta:  promote BOTH; break (inner only)
promoted = unique(promoted)         # ascending, de-duplicated
```

Strict `>` in `matlab_compatible_v1`; `>=` in `paper_spec_v1` (the paper writes
`>=`, the code writes `>`). One feature **may** appear in several discovered
pairs. Promoted features are **not** re-checked for redundancy, matching
`mi2.m:74-76`; duplicates are counted and reported, never silently removed.

**Pool flush:** when `len(pool) >= pool_size` (`:73`), and once more at stream
end if the pool is non-empty (`:82-86`) using the last computed `beta`.

---

## 6. Memory and state

* Only active candidates and pooled delays retain their column.
* A rejected feature retains **no vector** — only scalar trace metadata.
* Nothing is re-read from the Stage-1 matrix after arrival.
* `Selector` never calls `FeatureStream.full_rows_diagnostic`; asserted by
  `test_selector_has_no_held_out_accessor`.
* No `O(p²)` storage of pairwise scores; the only quadratic work is inside a
  pool/candidate flush, which is logged as `joint_evaluations`.
* No claim of guaranteed linear runtime is made; measured wall time is reported
  per dataset instead.

---

## 7. The continuous route — deliberately not implemented

`OSFS_DD_z2.m` calls `Fisher_Ztest` at lines 34, 123 and 151. That file **does
not exist in the pinned authors' repository** (67 blobs, zero matches for
`fisher`/`ztest`).

Without it we cannot determine the score's definition, units, sign convention,
the conditional-independence test, or behaviour on constant features.

`Selector.__init__` therefore raises `UnsupportedSemanticsError` for
`MetricFamily.GAUSSIAN_FISHER_Z`, and the runner records
`UNSUPPORTED_UNVERIFIED_SEMANTICS` for the affected datasets.

**Explicitly rejected substitutes** (per the Stage-2 prompt): `1 - p`, `|r|`,
`p`, correlation magnitude, discretised mutual information,
`sklearn.mutual_info_classif`, and any other heuristic proxy.

---

## 8. What is NOT here

No classifier. No accuracy. No KNN/SVM/CART. No hyperparameter search. No
baseline comparison. No 9×10×5 sweep. No claim of MATLAB runtime parity —
`MIToolboxMex` ships only as a Windows `.mexw64` binary and MATLAB/Octave is
absent from this AArch64 host, so parity is reported `NOT_MEASURED`.