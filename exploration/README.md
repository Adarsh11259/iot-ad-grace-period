# Exploration (not part of the final claims)

These two scripts were run while scoping the project. They're kept here for
transparency about what was tried, but **neither result is used as a claim
in the final writeup**, and here's why:

- **`four_model_comparison.py`** — runs the same held-out-device test from
  `scripts/gap_analysis.py` across four models (Random Forest, Decision
  Tree, k-NN, a small autoencoder). Random Forest is the only one of these
  confirmed to exist in the IoT-AD authors' own code
  (`DeviceProfileTrain.py`, `trainandtest.py`). Decision Tree, k-NN, and the
  autoencoder here are *our own* implementations inspired by the paper's
  model list — not verified against anything the authors actually ran — so
  their numbers describe our implementation, not a flaw in the authors' code.

- **`knn_scaling_test.py`** — found that adding `StandardScaler` before
  training fixes most of k-NN's generalization problem (FPR 26.5% → 3.9%
  on held-out devices). This is a real result for *our* k-NN implementation,
  but since there's no k-NN script in the authors' repo to compare it
  against, it can't be reported as "a bug we found in their code." Dropped
  from the final project for that reason.

The one claim in the main `results/` folder — the Random Forest
generalization gap — survives this scrutiny because it's the one model
directly traceable to the authors' own scripts.
