"""Stage 3B: controlled SVM conditioning and ARCENE feature-order sensitivity.

Two separable, preregistered experiments on Stage 3A evidence:

* Experiment A -- how training-only standardization and the user-declared
  ``max_iter`` cap affect ``SVC(kernel='linear')`` convergence on the **already
  selected** Stage 3A subsets. No feature selection is performed here.
* Experiment B -- how much the ARCENE feature *arrival order* alone changes the
  selected set, holding the Stage 1 seed-11 CV split fixed.

Neither experiment alters Stage 1/2/3A data, configs, selectors or results.
"""

STAGE = "3b"
FORBIDDEN = ("DLBCL", "BREAST")
CONTINUOUS = ("SRBCT", "LYMPHOMA", "PROSTATE", "LEUKEMIA")

EXP_A_DATASETS = ("ARCENE", "DEXTER", "MADELON", "GINA", "GISETTE")
EXP_B_DATASET = "ARCENE"
