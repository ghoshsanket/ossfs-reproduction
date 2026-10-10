# Stage 1 Follow-up — DLBCL and BREAST Provenance Review

**Generated:** 2026-10-10 (UTC), live on `spark-ef6f`
**Scope:** documentary and numerical lineage corroboration only. No gate was
changed, no dataset was relabelled, and no author was contacted.

---

## 1. Summary judgement

Neither DLBCL nor BREAST can be promoted to `EXACT_VERIFIED`. The blocker is not
a lack of searching — it is a logical limit. `EXACT_VERIFIED` requires
demonstrating that *Zhou et al. used this exact binary*, and the paper discloses
no dataset URLs, versions or checksums beyond two collection URLs, neither of
which contains these two datasets.

Against the three-way distinction required by the follow-up:

| Level | DLBCL | BREAST |
|---|---|---|
| **Plausible alternate archive** | superseded — several candidates found and rejected (§3) | superseded (§4) |
| **Same named release independently corroborated** | **partially**: cohort and label semantics match the primary publication; **no independent archive exists to compare bytes against** | **partially**: two encodings of the *same* repository record agree exactly (§4.2); no independent archive exists |
| **Demonstrated to be exactly Zhou et al.'s variant** | **No** | **No** |

Status retained: `SHAPE_MATCH_PROVENANCE_UNVERIFIED` for both.

---

## 2. What is on disk (verified, not asserted)

From `results/stage1/followup_20261010/live_audit.json` and
`provenance_fingerprints.json`:

| | DLBCL | BREAST |
|---|---|---|
| shape | 77 × 7129 | 97 × 24481 |
| class labels | `DLBCL`, `FL` | `relapse`, `non-relapse` |
| class counts | 58 / 19 | 46 / 51 |
| raw SHA-256 | `30bc95073e27e05a…` | `9a4971e3bc260ff1…` |
| source | TU Berlin `dlbcl-tumor-from-harvard` (CSV) | TU Berlin `breast-cancer-kent-ridge-2` (CSV) |
| repository's own declared shape | "7130 attributes, 77 instances" | "24482 attributes, 97 instances" |

Fingerprints (for future comparison against an independently obtained archive;
**not** dataset bytes):

```
DLBCL  value_sum=427512912.0
       row_sums_sha256=c5aac3dfda0145923ee14cc87152fcc74120cb677e06ba94d999f13d1cbb7948
       col_sums_sha256=bbd9ff7791dbfd9faa4faa4c50d6671420036c1418b65a2f87052c104cadaf43
BREAST value_sum=4196496.077
       row_sums_sha256=0f57cec14a34d04775d31b06573724bea5ac3555fb31b6f95a3d811f717e6c13
       col_sums_sha256=1b44f66099a54bc9a152ff6da4bb1e7a77755247d8e040b62aeea0644bd98570
```

---

## 3. DLBCL — candidates examined

| Candidate | Observed | Verdict |
|---|---|---|
| **OpenML `DLBCL` did=45088** | downloaded and parsed: **77 × 5469** (5470 attributes incl. label) | **Definitively a different variant.** 77 samples agree, feature count does not. Cannot corroborate. |
| TU Berlin `dlbcl-outcome-from-harvard` | 58 × 7129 | Different classification problem (cured vs fatal), 58 ≠ 77 samples |
| TU Berlin `dlbcl-nih` | 240 × 7399 | Different cohort |
| TU Berlin `diffuse-large-b-cell-stanford` | 47 × 4027 | Different study |
| `ico2s.org` DLBCL | page text asserts "7129 genes, 77 samples" | **Dead link** — named in prose, no `href` in served HTML; all conventional paths HTTP 404 |
| `biostat.jhsph.edu/~rshepard` (Kent Ridge) | HTTP 404; the `web.archive.org` copy also 404s | Primary documentation unreachable |
| Bioconductor `DLBCL`, `DLBCLdata`, GDC DLBCL | unrelated cohorts / raw sequencing | Not the same variant |

**Documentary corroboration obtained.** The observed class balance (58 DLBCL /
19 FL) matches the first of the two classification problems described in
Shipp M.A. et al., *Nature Medicine* 8(1):68–74 (2002) — 58 DLBCL patients
against 19 follicular-lymphoma samples. The repository's own description
reproduces that framing. So the *cohort definition and label semantics* are
corroborated against the primary publication.

That corroborates what the dataset **is**. It does not establish which file Zhou
et al. opened.

---

## 4. BREAST — candidates examined

| Candidate | Observed | Verdict |
|---|---|---|
| ico2s `breast_*` (Naderi et al. 2006) | 128 samples × 47293 genes | Different cohort |
| `seventyGeneData` van't Veer | 117 × 24481 | **Coincidental 24481 collision**; rejected on sample count |
| `seventyGeneData` van de Vijver | 295 × 24481 | Same collision, different study |
| TU Berlin `uci-20070111-breast-cancer`, `breast-cancer-ida` | low-dimensional | Not the 24481-gene matrix |
| `csse.szu.edu.cn/staff/zhuzx/datasets.html` | lists "Breast Cancer 24481 97 2" | **Host unreachable** from this machine (curl exit, HTTP 000, http and https) |

### 4.2 New evidence obtained: CSV ↔ MATLAB equivalence

The TU Berlin `matlab/` variant of `breast-cancer-kent-ridge-2` returned a
server-side error during the original Stage 1 run but was reachable at follow-up
time. It is an mldata-obfuscated MATLAB v5 file (`mldata_descr_ordering` maps
722 obfuscated variables back to feature order).

Reassembled and compared against the stored CSV-derived matrix:

```
reassembled MATLAB (features × samples) : (24481, 97)
ours (samples × features)               : (97, 24481)
numpy.allclose(MATLAB.T, ours)          : True
max absolute difference                 : 0.0
```

**The two encodings of this record agree exactly, cell for cell.**

What this does and does not establish:

* **Does:** confirm our CSV parsing is correct, and rules out a conversion
  artefact in the CSV path. Independent check on our own pipeline.
* **Does not:** count as an independent archive. Both files come from the *same
  repository*. It is an internal-consistency check between two encodings, which
  is weaker evidence than a byte comparison against a different institution's
  copy.

---

## 5. New material finding: BREAST carries source-level sentinels

While fingerprinting BREAST, its value range came back as **[-2.0, 100.0]** with
a value-sum of 4,196,496.077 — not a log-ratio distribution. Investigated:

| Value | Occurrences | Share of cells |
|---|---:|---:|
| `100.0` | 42,026 | 1.770 % |
| `-2.0` | 4,686 | 0.197 % |
| any value > 20 | 42,026 | — i.e. *every* value above 20 is exactly 100 |

All 97 samples contain at least one `100.0` (up to 10,896 in a single sample).
Both classes are affected (51/51 non-relapse, 46/46 relapse), so it is not a
class-correlated artefact.

**Critically, the MATLAB variant carries exactly the same counts** —
42,026 × `100.0` and 4,686 × `-2.0`. The sentinels are therefore **inherent to
the source record**, not introduced by our CSV parsing.

A conservative detector has been added to `validate.py`
(`detect_sentinel_candidates`): a value is flagged when it occupies ≥ 1e-4 of
all cells **and** lies more than 10 IQR outside `[q1, q3]`. Run across all 11
stored datasets it flags **BREAST only** (3 candidates, 51,121 occurrences) and
nothing else — so this is not a false-positive generator.

**Stage 1 has not repaired anything.** No imputation, clipping or replacement was
performed, and the status was not relabelled. This is surfaced for a researcher
decision because the right handling (drop the cells, treat as missing, ignore)
changes what any Stage 2/3 measurement means, and only the researcher can decide
which is scientifically defensible. See §7.

---

## 6. Gate impact

**None.** `configs/stage1.yaml` is unchanged. The `exact_verified_provenance`
gate remains **11/11 required** and still fails at 9/11. No threshold was
lowered, widened or reinterpreted to accommodate this review.

---

## 7. Unresolved uncertainties

| # | Item | Why it cannot be closed here | What would close it |
|---|---|---|---|
| P1 | Which DLBCL release Zhou et al. used | Paper states only two collection URLs; neither contains DLBCL | Author confirmation, or a supplementary appendix naming versions/checksums |
| P2 | Which BREAST release Zhou et al. used | Same | Same |
| P3 | No independent archive of either exact variant exists publicly | Every candidate located is either a different cohort or a different feature count (§3, §4) | A mirror with a published checksum |
| P4 | Meaning of BREAST `100.0` / `-2.0` | Present in both encodings of the source record, so it is the source's convention, but no accessible documentation defines it | Data-owner documentation, or a researcher decision on handling |
| P5 | Kent Ridge primary documentation unreachable | `biostat.jhsph.edu/~rshepard` and its archive.org copy both 404 | Recovery of the original page |

**Recommendation.** Contacting the authors is the only route that can close P1,
P2 and P4. **No email has been sent and none will be without explicit approval.**
Until then `SHAPE_MATCH_PROVENANCE_UNVERIFIED` is the scientifically correct
answer for both datasets.
