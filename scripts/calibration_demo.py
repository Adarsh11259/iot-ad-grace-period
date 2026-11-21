"""
The new idea: a grace-period / calibration mechanism for new devices.

Instead of judging a newly-connected device's traffic against the one
shared global model (what the gap in gap_analysis.py exposes), each
held-out device is given its own private model, built only from its own
first slice of traffic (the "calibration window"). Everything after that
window is judged against its own baseline instead of the shared model.

Run from the repo root:
    python scripts/calibration_demo.py
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

DATA_PATH = "data/profile_traces_combined.csv"
FEATURES = ["DestinationPort", "Length", "FlagValue"]


def main():
    df = pd.read_csv(DATA_PATH).dropna(subset=FEATURES)

    # Same held-out-device split as gap_analysis.py, so results are comparable.
    normal_files = sorted(df[df.IsAnomaly == 0]["SourceFile"].unique())
    rng = np.random.RandomState(8)
    rng.shuffle(normal_files)
    n_test_files = max(1, int(len(normal_files) * 0.3))
    held_out_devices = sorted(normal_files[:n_test_files])
    train_files = set(normal_files[n_test_files:])

    # The "naive" shared global model -- what IoT-AD's approach does today.
    normal_train = df[(df.IsAnomaly == 0) & (df.SourceFile.isin(train_files))]
    anomaly_df = df[df.IsAnomaly == 1]
    anom_train, _ = train_test_split(anomaly_df, test_size=0.25, random_state=8)
    train_all = pd.concat([normal_train, anom_train])
    global_model = RandomForestClassifier(random_state=8).fit(train_all[FEATURES], train_all["IsAnomaly"])

    rows = []
    naive_total, calib_total, n_total = 0, 0, 0

    print(f"{'Device':22s} | {'#calib':>7s} | {'#test':>6s} | {'Naive FP%':>10s} | {'Calibrated FP%':>15s}")
    print("-" * 75)

    for device_file in held_out_devices:
        dev_df = df[(df.IsAnomaly == 0) & (df.SourceFile == device_file)].sort_values("Time").reset_index(drop=True)
        n = len(dev_df)
        n_calib = max(10, int(n * 0.3))
        if n <= n_calib:
            continue
        calib = dev_df.iloc[:n_calib]
        test = dev_df.iloc[n_calib:]

        # Naive: judge new device's traffic with the shared global model.
        naive_pred = global_model.predict(test[FEATURES])
        naive_fp = naive_pred.mean()

        # Calibrated: build a private per-device baseline from its own
        # calibration window only, then judge the rest of its traffic
        # against that private model instead of the global one.
        scaler = StandardScaler().fit(calib[FEATURES])
        iso = IsolationForest(random_state=8, contamination=0.05).fit(scaler.transform(calib[FEATURES]))
        calib_pred = (iso.predict(scaler.transform(test[FEATURES])) == -1).astype(int)
        calib_fp = calib_pred.mean()

        print(f"{device_file:22s} | {n_calib:7d} | {len(test):6d} | {naive_fp*100:9.1f}% | {calib_fp*100:14.1f}%")
        rows.append((device_file, n_calib, len(test), naive_fp, calib_fp))
        naive_total += naive_pred.sum()
        calib_total += calib_pred.sum()
        n_total += len(test)

    print("-" * 75)
    print(f"{'OVERALL':22s} | {'':7s} | {n_total:6d} | {naive_total/n_total*100:9.1f}% | {calib_total/n_total*100:14.1f}%")

    return rows, naive_total / n_total, calib_total / n_total


if __name__ == "__main__":
    main()
