"""Stage 3B follow-up: diagnostic repairs + two controlled studies.

Separate, newly versioned from Stage 3B. Carries no Stage 3C work and does not
alter any Stage 1/2/3A/3B artifact.
"""

STAGE = "3b_followup"
VERSION = "v1"
FORBIDDEN = ("DLBCL", "BREAST")
CONTINUOUS = ("SRBCT", "LYMPHOMA", "PROSTATE", "LEUKEMIA")

EXP_A_DATASETS = ("MADELON", "GINA", "GISETTE")
EXP_B_DATASET = "ARCENE"
