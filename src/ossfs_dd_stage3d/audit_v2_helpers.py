"""Pure, testable validators for the Stage 3D V2 integration audit.

These are the functions the audit entrypoint calls. They are deliberately pure
(no I/O, no global state) so negative/mutation tests can invoke the *exact*
production validators on corrupted evidence copies.
"""

from __future__ import annotations

from typing import Any

FORBIDDEN = ("DLBCL", "BREAST")

DISCRETE = ("ARCENE", "DEXTER", "MADELON", "GINA", "GISETTE")
CONTINUOUS = ("SRBCT", "LYMPHOMA", "PROSTATE", "LEUKEMIA")
SEEDS_PILOT = (11, 17, 23)
SEEDS_ARCENE = (11, 17, 23, 31, 47, 59, 71, 83, 97, 109)
ARMS_DISCRETE = ("D1", "D2")
ARMS_CONTINUOUS = ("C1", "C2", "C3", "C4")
CLASSIFIERS = ("knn", "cart", "svm")


class ValidationFailure(RuntimeError):
    """Raised when an audit check fails. Carries the gate and the reason."""

    def __init__(self, gate: str, reason: str) -> None:
        super().__init__(f"{gate}: {reason}")
        self.gate = gate
        self.reason = reason


def expected_selection_keys() -> set[str]:
    """The exact 460-key selection cross-product from the frozen protocol."""
    keys: set[str] = set()
    for ds in DISCRETE:
        seeds = SEEDS_ARCENE if ds == "ARCENE" else SEEDS_PILOT
        for s in seeds:
            for f in range(5):
                for a in ARMS_DISCRETE:
                    keys.add(f"{ds}_seed{s}_fold{f}__{a}")
    for ds in CONTINUOUS:
        for s in SEEDS_PILOT:
            for f in range(5):
                for a in ARMS_CONTINUOUS:
                    keys.add(f"{ds}_seed{s}_fold{f}__{a}")
    return keys


def expected_classifier_keys(selection_keys: set[str]) -> set[str]:
    return {f"{k}__{c}" for k in selection_keys for c in CLASSIFIERS}


def selection_key(rec: dict[str, Any]) -> str:
    return f"{rec['dataset']}_seed{rec['seed']}_fold{rec['fold']}__{rec['arm']}"


def validate_selection_keys(records: list[dict[str, Any]]) -> set[str]:
    """G1: exact set equality, no duplicates, no forbidden datasets."""
    keys = [selection_key(r) for r in records]
    if len(keys) != len(set(keys)):
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        raise ValidationFailure("G1", f"duplicate selection keys: {dupes[:5]}")
    got = set(keys)
    want = expected_selection_keys()
    missing = sorted(want - got)
    extra = sorted(got - want)
    if missing:
        raise ValidationFailure("G1", f"{len(missing)} missing keys, e.g. {missing[:5]}")
    if extra:
        raise ValidationFailure("G1", f"{len(extra)} unexpected keys, e.g. {extra[:5]}")
    for r in records:
        if r["dataset"] in FORBIDDEN:
            raise ValidationFailure("G1", f"forbidden dataset {r['dataset']}")
    return got


def validate_schedule_hashes(records: list[dict[str, Any]]) -> None:
    """G1: every batch record must carry its schedule hashes."""
    for r in records:
        if r.get("batch_size", 1) > 1 or r.get("arm") in ("D2", "C2", "C4"):
            sh = r.get("schedule_hashes")
            if not sh or not sh.get("flattened_order_sha256"):
                raise ValidationFailure(
                    "G1", f"{selection_key(r)}: missing schedule hash")


def validate_feature_ids(record: dict[str, Any], n_features: int) -> None:
    """G1: selected IDs unique, in range, and consistent with the count."""
    ids = record.get("selected_feature_ids") or []
    if len(set(ids)) != len(ids):
        raise ValidationFailure("G1", f"{selection_key(record)}: duplicate selected IDs")
    if any((not isinstance(i, int)) or i < 0 or i >= n_features for i in ids):
        raise ValidationFailure("G1", f"{selection_key(record)}: feature ID out of range")
    if record.get("n_selected") != len(ids):
        raise ValidationFailure(
            "G1", f"{selection_key(record)}: n_selected != len(selected_feature_ids)")


def validate_classifier_linkage(classifiers: list[dict[str, Any]],
                                selection_keys: set[str]) -> None:
    """G1: every classifier record references a real selection, uniquely."""
    ids = [c["record_id"] for c in classifiers]
    if len(set(ids)) != len(ids):
        raise ValidationFailure("G1", "duplicate classifier record_id")
    for c in classifiers:
        sel = "__".join(c["record_id"].split("__")[:2])
        if sel not in selection_keys:
            raise ValidationFailure(
                "G1", f"classifier {c['record_id']} references unknown selection {sel}")


def validate_oracle(oracle: dict[str, Any]) -> None:
    """G2: a failing or absent numerical oracle blocks the audit."""
    if oracle.get("oracle_ok") is not True:
        raise ValidationFailure("G2", f"numerical oracle not passed: {oracle}")
    if oracle.get("import_ok") is not True:
        raise ValidationFailure("G2", f"import check failed: {oracle.get('import_error')}")


def validate_finalized(manifest: dict[str, Any]) -> None:
    """G7: V2 smoke must report the V2 method and a completed finalization."""
    if manifest.get("method_version") != "batch_score_shared_v2":
        raise ValidationFailure(
            "G7", f"method_version={manifest.get('method_version')!r} is not V2")
    if manifest.get("finalized") is not True:
        raise ValidationFailure("G7", "finalized is not True")
    if manifest.get("pending_pool_end") not in (0, None):
        raise ValidationFailure("G7", f"pending pool after finalization: {manifest['pending_pool_end']}")


def validate_v1_untouched(v1_hashes: dict[str, str],
                          v1_hashes_after: dict[str, str]) -> None:
    """G8: V1 evidence must be byte-identical."""
    changed = [k for k in v1_hashes if v1_hashes.get(k) != v1_hashes_after.get(k)]
    if changed:
        raise ValidationFailure("G8", f"V1 artifacts changed: {changed[:5]}")


def derive_verdict(gates: dict[str, dict[str, Any]]) -> str:
    """G10: verdict is the conjunction of mandatory gates."""
    mandatory = [f"G{i}" for i in range(1, 10)]
    failed = [g for g in mandatory if gates.get(g, {}).get("outcome") != "PASS"]
    return "PASS" if not failed else "PARTIAL"
