"""Stage 3A evaluation harness for the discrete OSFS-DD reproduction.

This package adds *only* evaluation. It does not modify ``ossfs_dd`` selector
semantics, ``configs/stage1.yaml``, ``configs/stage2.yaml`` or any Stage-1/2
report. Selection is delegated to the Stage-2 ``Selector`` through the Stage-1
``FeatureStream``, so a held-out row is still never visible to a score.
"""

__all__ = ["WORKSTREAM", "MODE_ID"]

WORKSTREAM = "A"
MODE_ID = "matlab_compatible_v1"
