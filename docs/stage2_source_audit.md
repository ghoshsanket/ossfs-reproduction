# Stage 2 Source Audit — Authors' MATLAB Implementation

**Audit performed:** 2026-10-10 (UTC), live on `spark-ef6f`
**Repository:** `https://github.com/doodzhou/OSFS`
**Pinned commit:** `845a8e9d5625616b43e8cee16e8d22081dbe3f62` (master, 2026-09-21T11:10:39Z)
**Local snapshots:** `$RESEARCH_OUTPUT_ROOT/ossfs-dd-reproduction/stage2/reference_snapshots/`

This audit was completed **before** any Stage-2 algorithm code was written and
before any real-data smoke run, as required by §6.1.

---

## 1. What was acquired, and from where

| File | SHA-256 | Bytes | Status |
|---|---|---:|---|
| `OSFS_DD_mi2.m` | `13f4227221559c39fb57a0a1376831c042b3e860616540c8be90fea44106390f` | 3492 | acquired |
| `OSFS_DD_z2.m` | `c14ce5dd99f235b352fa03affd403900b1937e5c645ba51bdc621447d2878ca1` | 4048 | acquired |
| `mi/SU.m` | `9a83c02ace6e9669447e1fad87def202557c2b21095287952b6068912d4dcea4` | 169 | acquired |
| `mi/mi.m` | `dc9d442107f845be3ca887292c04399d84926aae2dd4f850adef6aa8e21b1a8f` | 471 | acquired |
| `mi/cmi.m` | `7e51f32eb3a2b41ec1c5fa02429140d3655c8f8c5dd2ce1a9555549229041fe1` | 705 | acquired |
| `mi/h.m` | `177393a1a09804860353beac69cfd6b6dc006962afb305409661b63c32966baa` | 332 | acquired |
| `mi/ch.m` | `f9e54a95bff90aa3ffef30cbbcc1bbb0e5849f0ef37b7a293c39909b07d1df17` | 365 | acquired |
| `mi/mi3.m` | `9115ccca44d8308a6704ce1927cbf211205a7bd21e30064a73c17bac14572112` | 177 | acquired |
| `mi/Entropy.c` | `d5558cd419c8d46bdc958064cb97f963d1ea793866414c025906ec15033512ed` | 4083 | acquired |
| `mi/MutualInformation.c` | (acquired) | 3956 | acquired |
| `mi/MIToolboxMex.c` | `bd759781dd779374ae26746aa82ed2de8795a05795136e26a9988fd73367b2a1` | 15760 | acquired |
| `README.md` (folder) | `fbc449dfaeb44f4bbdca4e6c9995dd383967a30c51565ade6c28402b2dd2c660` | 2150 | acquired |

Fetched via `raw.githubusercontent.com` at the pinned commit. No third-party
code was copied into the project repository; it lives only under
`RESEARCH_OUTPUT_ROOT`, outside Git, with hashes recorded above.

---

## 2. The decisive negative result: `Fisher_Ztest` does not exist

`OSFS_DD_z2.m` calls `Fisher_Ztest` three times (lines 34, 123, 151). The
repository contains **67 blobs**; an exhaustive tree walk for `fisher` or
`ztest` returns:

```
files matching fisher/ztest: NONE
```

`Fisher_Ztest.m` is **not in the repository, not in `mi/`, and not anywhere
else in the project**. It is an unpublished external helper.

### What this means, stated precisely

**UNRESOLVED.** We cannot determine, from source:

* what quantity `fz_F` is (a Fisher z-statistic? a p-value? a correlation? a
  transformed score?),
* its units and sign convention,
* the conditional-independence test used for the `CI_cmi` output,
* its behaviour on constant or singular features (beyond the observed
  `if isnan(fz_F), fz_F=0; end` guard on line 35–37),
* whether it internally rescales to a `[0,1]` score.

Circumstantial evidence constrains but does not determine the answer: `fz_F` is
compared against dynamic thresholds built from **its own** running mean and
standard deviation, so it must be comparable across features; `isnan` is
guarded, so it is a finite real scalar; and the three-argument form returns
`[CI_cmi, cmi]` where `cmi` is added to a marginal score, the same
score-mixing pattern as the discrete file.

That is **not** enough to implement it. Per §6.2 of the Stage-2 prompt, we do
**not** substitute `1-p`, `|r|`, `p`, correlation magnitude, discretised MI or
`mutual_info_classif`. The continuous route therefore returns the named error
`UNSUPPORTED_UNVERIFIED_SEMANTICS`.

---

## 3. Settled facts from `OSFS_DD_mi2.m` (discrete, fully inspected)

All line references are to the pinned file.

### 3.1 Score definition and units — `SOURCE_CONFIRMED`

```
line 33:  mi_F = SU(X(:,i),Y);
```

`SU.m`:
```
score = (2 * iXY) / (hX + hY)
```

`h.m`/`mi.m` delegate to `MIToolboxMex`, whose C source is in the repository.
`Entropy.c` line 50 computes `-= p*log(p)` and **line 54 does
`entropy /= log(2.0)`**. `MutualInformation.c` line 63 does the same.
Therefore:

> **All MIToolbox quantities — entropy, joint entropy, conditional entropy,
> mutual information, conditional mutual information — are in BITS (base 2).**

So `SU ∈ [0,1]` (the ratio is base-independent), while `cmi` is a **raw,
non-negative, unbounded-in-practice quantity in bits**.

### 3.2 First two features — `SOURCE_CONFIRMED`

```
lines 37-41:
    if i<=2
        meanF=mean(MIArray);
        stdF=std(MIArray);
        continue;
    end
```

Both features are **scored and discarded**. They are never added to
`AccIndexs` nor `DeterIndexs`. The comment above them says "default select",
but the statements do not select. **Confirmed: the paper's pseudocode does not
specify this skip and the code performs it.**

### 3.3 Running moments — `SOURCE_CONFIRMED`

```
lines 95-103 (updateMeanStd, called with Num = i, the 1-based arrival index):
    mean_new = mean + (newVal-mean)/Num;
    if Num>2
        Sn = (1/(Num-1))*((Num-2)*std^2 + ((Num-1)/Num)*(newVal-mean)^2);
        std_new = sqrt(Sn);
    else
        std_new = 0;
    end
```

This is the standard Welford-style **sample variance** update (`ddof=1`), fed
every incoming score including rejected and delayed ones.

### 3.4 Three-way decision — `SOURCE_CONFIRMED`

```
line 49-50:  alpha = meanF + sig1*stdF;   beta = meanF + sig2*stdF;
line 52:      if (mi_F<=alpha)  -> Reject      (continue)
line 56:      elseif (mi_F>=beta) -> Accept
line 69:      else                -> Determined (delay)
```

`<=` and `>=` are **inclusive**. Reject is tested **first**.

### 3.5 Redundancy — `SOURCE_CONFIRMED`

```
lines 128-150:
    count = length(AccIndexs);        % NOTE: already includes newIndex
    if count>1
      for i=1:(count-1)               % previous features only
        jointMI = cmi(X(:,ind),Y,X(:,newIndex)) + indMI;
        if jointMI < 2*beta
            if newMI < indMI
                delIndx = [newIndex]; break;      % discard the NEW feature
            else
                delIndx = [delIndx,ind];          % discard the EXISTING one
            end
        end
      end
    end
```

* strict `< 2*beta`
* the **weaker** member is removed, where "weaker" means the smaller marginal
  score
* a tie (`newMI == indMI`) takes the `else` branch, so the **existing** feature
  is removed — ties favour the newer feature
* comparison order is the order of `AccIndexs`, skipping the appended feature
* `delIndx` may name several existing features; they are removed afterwards by
  the caller (lines 61–68)

### 3.6 Uncertain-pool promotion — `SOURCE_CONFIRMED`

```
lines 107-125:
    for i=1:DeterCount
        for j=(i+1):DeterCount
            jointMI = MIArray(ind) + cmi(X(:,ind), Y, X(:,ind_joint));
            if jointMI > 2*beta
                selectedIndex = [selectedIndex, ind, ind_joint];   % BOTH
                break;                                            % inner only
            end
        end
    end
    selectedIndex = unique(selectedIndex);
```

* strict `> 2*beta` (**not** `>=`)
* **both** members are promoted, not just one
* the inner loop breaks on the first qualifying pair; the **outer loop
  continues**, so one feature **can** appear in more than one discovered pair
* `unique` de-duplicates and returns ascending values, so the final promotion
  list is sorted

### 3.7 Pool flush and final flush — `SOURCE_CONFIRMED`

```
line 73:   if deterCount >= PoolSize     % >=, not ==
line 82-86: final flush if DeterIndexs is non-empty, using the LAST computed beta
```

**No redundancy pass is applied to promoted features** in either path; they are
concatenated straight into `AccIndexs` (line 75).

**Latent defect (recorded, not reproduced silently):** if the stream has
`P <= 2` features, `beta` is never assigned before line 83 references it, so
the original raises a MATLAB error. A third feature that is rejected leaves
`DeterIndexs` empty, so the final flush is skipped and `beta` is unused.

### 3.8 Output type — `SOURCE_CONFIRMED`

`AccIndexs` holds **1-based positions in the arrival order**, which coincide
with input column positions. Stage 1 stores original **zero-based** feature
IDs in a possibly shuffled `arrival_order`. The two must be mapped explicitly:
`selected_features[i] (1-based arrival position)` → `arrival_order[selected-1]`
(original zero-based feature ID). Arrival position and source column index are
stored separately in our implementation.

---

## 4. `OSFS_DD_z2.m` — what could be settled, and what could not

### 4.1 Divergences from `mi2.m` that ARE source-confirmed

| Behaviour | `mi2.m` | `z2.m` | Line refs |
|---|---|---|---|
| First two features | scored, **skipped** | scored, **`AccIndexs=[AccIndexs,i]`** — selected | mi2 37–41 / z2 41–46 |
| Decision test order | reject `<=alpha` first | **accept `>=beta` first** | mi2 52–69 / z2 58–74 |
| Behaviour when `sigma=0` (`alpha==beta`) | **Reject** | **Accept** | consequence of the order above |
| Conditional-independence guard | none | `if CI_cmi==1 \|\| isnan(cmi), continue` | z2 124, 152 |
| NaN guard on marginal score | none | `if isnan(fz_F), fz_F=0; end` | z2 35–37 |

The degenerate-`sigma` divergence is material: when the running standard
deviation collapses to zero, the same score produces **opposite** decisions in
the two files. `matlab_compatible_v1` must therefore be mode-specific and must
not share one decision routine blindly.

### 4.2 What remains UNRESOLVED

Everything about the score itself — see §2. `fz_F` is used as a marginal
relevance score and the 3-argument output `cmi` is added to it before being
compared with `2*beta`, reproducing the same normalised-plus-raw score mixing
as the discrete file. But **the numeric definition of either output is
unknown**, so the comparison cannot be reproduced.

---

## 5. Score-semantics ledger (machine-readable)

Machine-readable copy: `results/stage2/score_semantics_ledger.json`.

| ID | Quantity | Definition | Units | Evidence | Status |
|---|---|---|---|---|---|
| S1 | marginal score (discrete) | `SU(X,Y) = 2·MI(X;Y) / (H(X)+H(Y))` | bits inside, ratio in [0,1] | `SU.m:3-7`, `Entropy.c:50-54` | `SOURCE_CONFIRMED` |
| S2 | `mi` | `I(X;Y) = H(X)+H(Y)-H(X,Y)` via joint probability | bits | `MutualInformation.c:63` | `SOURCE_CONFIRMED` |
| S3 | `cmi` | `I(X;Y|Z)` | **raw bits, not normalised** | `cmi.m:26` → mex op 8 | `SOURCE_CONFIRMED` |
| S4 | joint quantity (discrete) | `SU(X_ind,Y) + cmi(X_ind,Y,X_joint)` | **normalised + raw — mixed** | `mi2.m:116`, `mi2.m:137` | `SOURCE_CONFIRMED` (mixing is real, not a reading error) |
| S5 | running mean / variance | Welford, sample variance `ddof=1` | same units as score | `mi2.m:95-103` | `SOURCE_CONFIRMED` |
| S6 | marginal score (continuous) | `Fisher_Ztest(X,Y)` output 2 | **unknown** | `z2.m:34` | **`UNRESOLVED`** |
| S7 | conditional quantity (continuous) | `Fisher_Ztest(X,Y,Z)` output 2 | **unknown** | `z2.m:123,151` | **`UNRESOLVED`** |
| S8 | conditional-independence flag | `Fisher_Ztest` output 1 | unknown | `z2.m:123` | **`UNRESOLVED`** |

### Consequence for the two data families

The Stage-2 prompt (§6.5) makes the paper's `Real`/`Integer` column the primary
preregistered metric route.

| Route | Author file | Helper availability | Status |
|---|---|---|---|
| **Integer** (ARCENE, DEXTER, MADELON, GINA, GISETTE) | `OSFS_DD_mi2.m` | `SU`,`mi`,`cmi`,`h` all present and inspected | **implementable with verified semantics** |
| **Real** (SRBCT, LYMPHOMA, PROSTATE, LEUKEMIA) | `OSFS_DD_z2.m` | **`Fisher_Ztest` absent from the repository** | **`UNSUPPORTED_UNVERIFIED_SEMANTICS`** |

No substitution is made for the Real route. This is a blocker to be reported,
not a licence to invent a proxy.