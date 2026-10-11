"""Reference-semantics metadata: what is confirmed, what is an interpretation.

This is the machine-readable counterpart of ``docs/stage2_source_audit.md``.
Every mode records the source it mirrors and every unresolved item, so a run
artefact can never be read as more certain than its evidence.
"""

from __future__ import annotations

from typing import Any

from .config import Mode

COMMIT = "845a8e9d5625616b43e8cee16e8d22081dbe3f62"
REPO = "https://github.com/doodzhou/OSFS"
BASE = (
    f"https://raw.githubusercontent.com/doodzhou/OSFS/{COMMIT}/"
    "Online%20Scalable%20Streaming%20Feature%20Selection%20via%20Dynamic%20Decision"
)

REFERENCE_SEMANTICS: dict[str, dict[str, Any]] = {
    "S1": {
        "id": "S1",
        "quantity": "marginal relevance score (discrete)",
        "definition": "SU(X,Y) = 2*MI(X;Y) / (H(X)+H(Y))",
        "units": "bits internally; the ratio is base-independent and lies in [0,1]",
        "evidence": f"{BASE}/OSFS_DD_mi2.m:33 ; mi/SU.m:3-7",
        "log_base_evidence": "mi/Entropy.c:50 accumulates -p*log(p); :54 divides by log(2.0)",
        "status": "SOURCE_CONFIRMED",
    },
    "S2": {
        "id": "S2",
        "quantity": "mutual information",
        "definition": "I(X;Y) = H(X) + H(Y) - H(X,Y)",
        "units": "bits",
        "evidence": "mi/MutualInformation.c:63 divides by log(2.0)",
        "status": "SOURCE_CONFIRMED",
    },
    "S3": {
        "id": "S3",
        "quantity": "conditional mutual information",
        "definition": "I(X;Y|Z)",
        "units": "raw bits; NOT normalised, not bounded by 1",
        "evidence": f"{BASE}/OSFS_DD_mi2.m:116,137 via mi/cmi.m:26",
        "status": "SOURCE_CONFIRMED",
    },
    "S4": {
        "id": "S4",
        "quantity": "joint quantity (discrete)",
        "definition": "SU(X_ind,Y) + cmi(X_ind, Y, X_partner)",
        "units": "normalised [0,1] PLUS raw bits — MIXED SCALES",
        "evidence": f"{BASE}/OSFS_DD_mi2.m:116 (pool) and :137 (redundancy)",
        "assessment": (
            "The reference adds a normalised score to a raw information "
            "quantity, then compares the sum with 2*beta, a threshold built on "
            "the normalised scale. Reproduced deliberately in "
            "matlab_compatible_v1; paper_spec_v1 uses the scale-coherent "
            "SU({X,Z};Y) instead and the two are never compared as if equal."
        ),
        "status": "SOURCE_CONFIRMED",
    },
    "S5": {
        "id": "S5",
        "quantity": "running mean / sample variance",
        "definition": "Welford update, sample variance ddof=1, fed every score",
        "units": "same as the score",
        "evidence": f"{BASE}/OSFS_DD_mi2.m:95-103 (updateMeanStd), Num = 1-based arrival index",
        "status": "SOURCE_CONFIRMED",
    },
    "S6": {
        "id": "S6",
        "quantity": "marginal relevance score (continuous)",
        "definition": "second output of Fisher_Ztest(X,Y)",
        "units": "UNKNOWN",
        "evidence": f"{BASE}/OSFS_DD_z2.m:34",
        "blocker": (
            "Fisher_Ztest.m does not exist in the pinned repository. An exhaustive "
            "walk over all 67 blobs returns zero matches for 'fisher' or 'ztest'."
        ),
        "status": "UNRESOLVED",
    },
    "S7": {
        "id": "S7",
        "quantity": "conditional quantity (continuous)",
        "definition": "second output of Fisher_Ztest(X,Y,Z)",
        "units": "UNKNOWN",
        "evidence": f"{BASE}/OSFS_DD_z2.m:123,151",
        "status": "UNRESOLVED",
    },
    "S8": {
        "id": "S8",
        "quantity": "conditional-independence flag (continuous)",
        "definition": "first output of Fisher_Ztest(X,Y,Z)",
        "units": "UNKNOWN",
        "evidence": f"{BASE}/OSFS_DD_z2.m:123,124",
        "status": "UNRESOLVED",
    },
}

BEHAVIOUR_LEDGER: dict[str, dict[str, Any]] = {
    "first_two_features": {
        "paper": "unspecified",
        "matlab_mi2": "scored, moments updated, then skipped (never selected) — mi2.m:37-41",
        "matlab_z2": "scored, moments updated, AND selected — z2.m:41-46",
        "status": "SOURCE_CONFIRMED",
        "note": "The two author files disagree with each other; each mode mirrors its own file.",
    },
    "decision_order": {
        "matlab_mi2": "reject tested first (<=alpha), then accept (>=beta) — mi2.m:52,56",
        "matlab_z2": "accept tested first (>=beta), then reject (<=alpha) — z2.m:58,72",
        "status": "SOURCE_CONFIRMED",
        "note": (
            "When sigma==0 the two branches both hold, so the files give OPPOSITE "
            "decisions: mi2 rejects, z2 accepts."
        ),
    },
    "pool_flush": {
        "comparison": ">= PoolSize — mi2.m:73, z2.m:79",
        "final_flush": "once at stream end if the pool is non-empty — mi2.m:82-86, z2.m:88-92",
        "status": "SOURCE_CONFIRMED",
        "latent_defect": (
            "With P<=2 features, beta is never assigned before the final flush "
            "references it, so the original raises a MATLAB error."
        ),
    },
    "promotion": {
        "comparison": "strict > 2*beta in the reference (mi2.m:117); >= 2*beta in paper_spec_v1",
        "promotes": "BOTH members, not one (mi2.m:118)",
        "search": (
            "outer loop over i continues after the inner loop breaks, so one "
            "feature may appear in several pairs (mi2.m:112-122)"
        ),
        "dedup": "unique() inside flushDeterPool (mi2.m:124); AccIndexs is NOT re-uniquified",
        "status": "SOURCE_CONFIRMED",
    },
    "redundancy": {
        "comparison": "strict < 2*beta (mi2.m:138)",
        "removes": "the weaker member by marginal score (mi2.m:139-144)",
        "tie_break": "newMI == indMI takes the else branch, removing the EXISTING feature",
        "promoted_pairs": "NOT re-checked for redundancy (mi2.m:74-76)",
        "status": "SOURCE_CONFIRMED",
    },
    "index_convention": {
        "reference": "1-based input column positions, in input order",
        "stage1": "original zero-based feature IDs delivered via arrival_order",
        "mapping": "original_feature_id = arrival_order[matlab_index - 1]",
        "status": "SOURCE_CONFIRMED",
    },
    "matlab_runtime_parity": {
        "status": "NOT_MEASURED",
        "reason": (
            "MIToolboxMex ships only as MIToolboxMex.mexw64 (Windows x64) and "
            "MATLAB/Octave is not installed on this AArch64 host, so the reference "
            "cannot be executed for comparison."
        ),
    },
}


def lookup(semantic_id: str) -> dict[str, Any]:
    try:
        return REFERENCE_SEMANTICS[semantic_id]
    except KeyError as exc:
        raise KeyError(
            f"unknown reference semantic {semantic_id!r}; known: {sorted(REFERENCE_SEMANTICS)}"
        ) from exc


def mode_record(mode: Mode, *, metric_family: str, source_reference: str,
                joint_metric: str, entry_file: str) -> dict[str, Any]:
    """Metadata every result artefact carries about the mode it ran under."""
    unresolved = [
        v["id"] for v in REFERENCE_SEMANTICS.values() if v["status"] == "UNRESOLVED"
    ]
    return {
        "mode": mode.value,
        "metric_family": metric_family,
        "joint_metric": joint_metric,
        "source_reference": source_reference,
        "source_entry_file": entry_file,
        "source_commit": COMMIT,
        "source_repo": REPO,
        "unresolved_semantics": unresolved,
        "matlab_runtime_parity": BEHAVIOUR_LEDGER["matlab_runtime_parity"]["status"],
        "interpretations": _interpretations(mode),
    }


def _interpretations(mode: Mode) -> list[str]:
    out: list[str] = []
    if mode is Mode.PAPER_SPEC:
        out += [
            (
                "joint information defined as SU({X,Z};Y), scale-coherent with the "
                "marginal score"
            ),
            "uncertain promotion uses >= 2*beta (the reference uses strict >)",
            ("first two features are routed through the ordinary decision using a "
            "declared bootstrap; alpha < beta is never relaxed"),
            "zero entropy on both operands yields the neutral value 0.0",
        ]
    else:
        out += [
            ("first two features are scored, update the moments, and are then skipped "
            "(mirrors OSFS_DD_mi2.m:37-41)"),
            "reject is evaluated before accept (mirors mi2.m:52)",
            ("joint information is the source expression SU + raw cmi, mixing scales "
            "deliberately to mirror mi2.m:116,137"),
            "ties in redundancy comparison remove the existing feature (mi2.m:139-144)",
            "promoted pairs are not re-checked for redundancy (mi2.m:74-76)",
        ]
    return out
