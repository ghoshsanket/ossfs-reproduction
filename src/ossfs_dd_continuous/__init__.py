"""Workstream B: a RECONSTRUCTED continuous Fisher-Z estimator.

The authors' ``OSFS_DD_z2.m`` calls ``Fisher_Ztest``, which does not exist in
the pinned repository. This package is our declared reconstruction, labelled
``continuous_reconstructed_fisherz_v1``. It is not the authors' semantics, is not
source-verified, and does not unlock or replace Stage 2's fail-closed
``UNRESOLVED`` routing for the same datasets.
"""

MODE_ID = "continuous_reconstructed_fisherz_v1"
SEMANTICS_STATUS = "RECONSTRUCTED_NOT_SOURCE_VERIFIED"
ALLOWLIST = ("SRBCT", "LYMPHOMA", "PROSTATE", "LEUKEMIA")
FORBIDDEN = ("DLBCL", "BREAST")
ALPHA = 0.05
