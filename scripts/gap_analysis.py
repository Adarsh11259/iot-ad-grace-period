"""
Reproduces IoT-AD's own packet-level detector (Random Forest, as implemented
in their DeviceProfileTrain.py / trainandtest.py) and then runs a harder,
held-out-device test their random-split evaluation structurally cannot do.

Experiment A: pooled random 75/25 split -- same method the authors use.
              Packets from the same device can land in both train and test.

Experiment B: held-out-device split -- entire device files are set aside
              before training, so the model is tested only against traffic
              from devices it has genuinely never seen.

Run from the repo root:
    python scripts/gap_analysis.py
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATA_PATH = "data/profile_traces_combined.csv"
FEATURES = ["DestinationPort", "Length", "FlagValue"]


def report(name, y_true, y_pred):
    print(f"--- {name} ---")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"F1       : {f1_score(y_true, y_pred):.4f}")
    print()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def main():
    df = pd.read_csv(DATA_PATH).dropna(subset=FEATURES)
    X, y = df[FEATURES], df["IsAnomaly"]

    # ---- Experiment A: original method ----
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=8)
    clf_a = RandomForestClassifier(random_state=8).fit(X_train, y_train)
    pred_a = clf_a.predict(X_test)
    metrics_a = report(
        "A) ORIGINAL METHOD: pooled random 75/25 split "
        "(same device's packets can appear in both train & test)",
        y_test, pred_a,
    )

    # ---- Experiment B: held-out-device generalization check ----
    normal_files = sorted(df[df.IsAnomaly == 0]["SourceFile"].unique())
    rng = np.random.RandomState(8)
    rng.shuffle(normal_files)
    n_test_files = max(1, int(len(normal_files) * 0.3))
    test_files = set(normal_files[:n_test_files])
    train_files = set(normal_files[n_test_files:])
    print(f"Held-out (never-seen) normal device files ({len(test_files)}): {sorted(test_files)}")
    print(f"Training normal device files ({len(train_files)}): {sorted(train_files)}")
    print()

    normal_train = df[(df.IsAnomaly == 0) & (df.SourceFile.isin(train_files))]
    normal_test = df[(df.IsAnomaly == 0) & (df.SourceFile.isin(test_files))]
    anomaly_df = df[df.IsAnomaly == 1]
    anom_train, anom_test = train_test_split(anomaly_df, test_size=0.25, random_state=8)

    train_b = pd.concat([normal_train, anom_train])
    test_b = pd.concat([normal_test, anom_test])

    clf_b = RandomForestClassifier(random_state=8).fit(train_b[FEATURES], train_b["IsAnomaly"])
    pred_b = clf_b.predict(test_b[FEATURES])
    metrics_b = report(
        "B) GENERALIZATION CHECK: trained on some devices, "
        "tested on ENTIRELY UNSEEN devices",
        test_b["IsAnomaly"], pred_b,
    )

    # False-positive rate on unseen-but-normal traffic only
    test_normal_only = test_b[test_b.IsAnomaly == 0]
    pred_normal_only = clf_b.predict(test_normal_only[FEATURES])
    fpr = (pred_normal_only == 1).mean()
    print(f"Of the packets from devices the model NEVER saw during training, "
          f"{fpr*100:.1f}% of their perfectly normal traffic was wrongly flagged as an anomaly.")

    return metrics_a, metrics_b, fpr


if __name__ == "__main__":
    main()
