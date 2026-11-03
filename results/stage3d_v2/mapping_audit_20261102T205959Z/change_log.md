# Change log — Stage 3D V2 mapping and audit repair

## Bug fixed
- `scripts/stage3d_v2_run.py` mapped scheduled feature IDs to the WRONG matrix
  columns: `iter(enumerate([X[:, fid] for fid in flattened]))` bound position `f`
  to the column of `flattened[f]`, and the consumer then looked up by feature ID.
  Measured: **28/30 mismatches** on the SYNTH schedule (requested ID 25 received
  column 1). Replaced with a direct ID->column mapping.

## Added
- `scripts/stage3d_v2_real_smoke.py` — bounded MADELON train-only V2 smoke
- `scripts/stage3d_v2_mapping_audit.py` — this evidence bundle
- `src/ossfs_dd_stage3d/audit_v2_helpers.py` — pure, testable validators
- `tests/test_stage3d_v2_mapping.py`, `tests/test_stage3d_v2_audit_mutations.py`
- `configs/stage3d_v2_integration_repair.yaml` (write-once)

## Not changed
- Stage 1/2/3A/3B/3C, Stage 3D V1, its raw records, and the V1 auditor.
- No classifier training, no 460-run sweep.
