import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("../data/profile_traces_combined.csv").dropna(subset=["DestinationPort","Length","FlagValue"])
FEATURES = ["DestinationPort","Length","FlagValue"]

def metrics(y_true, y_pred):
    return dict(acc=accuracy_score(y_true,y_pred), prec=precision_score(y_true,y_pred,zero_division=0),
                rec=recall_score(y_true,y_pred,zero_division=0), f1=f1_score(y_true,y_pred,zero_division=0))
def fmt(m): return f"acc={m['acc']:.3f} prec={m['prec']:.3f} rec={m['rec']:.3f} f1={m['f1']:.3f}"

X = df[FEATURES]; y = df["IsAnomaly"]
Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(X, y, test_size=0.25, random_state=8)

normal_files = sorted(df[df.IsAnomaly==0]["SourceFile"].unique())
rng = np.random.RandomState(8); rng.shuffle(normal_files)
n_test_files = max(1, int(len(normal_files)*0.3))
test_files = set(normal_files[:n_test_files]); train_files = set(normal_files[n_test_files:])
normal_train = df[(df.IsAnomaly==0) & (df.SourceFile.isin(train_files))]
normal_test  = df[(df.IsAnomaly==0) & (df.SourceFile.isin(test_files))]
anomaly_df   = df[df.IsAnomaly==1]
anom_train, anom_test = train_test_split(anomaly_df, test_size=0.25, random_state=8)
train_b = pd.concat([normal_train, anom_train]); test_b = pd.concat([normal_test, anom_test])
Xb_tr, yb_tr = train_b[FEATURES], train_b["IsAnomaly"]
Xb_te, yb_te = test_b[FEATURES], test_b["IsAnomaly"]
unseen_normal_X = normal_test[FEATURES]

print("=== k-NN WITHOUT scaling (what we ran before, for reference) ===")
m1 = KNeighborsClassifier(n_neighbors=5).fit(Xa_tr, ya_tr)
m2 = KNeighborsClassifier(n_neighbors=5).fit(Xb_tr, yb_tr)
print("A:", fmt(metrics(ya_te, m1.predict(Xa_te))))
print("B:", fmt(metrics(yb_te, m2.predict(Xb_te))), " FPR on unseen:", f"{(m2.predict(unseen_normal_X)==1).mean()*100:.1f}%")

print()
print("=== k-NN WITH proper feature scaling (StandardScaler fit on train only) ===")
scaler_a = StandardScaler().fit(Xa_tr)
m1s = KNeighborsClassifier(n_neighbors=5).fit(scaler_a.transform(Xa_tr), ya_tr)
pa = m1s.predict(scaler_a.transform(Xa_te))
print("A:", fmt(metrics(ya_te, pa)))

scaler_b = StandardScaler().fit(Xb_tr)
m2s = KNeighborsClassifier(n_neighbors=5).fit(scaler_b.transform(Xb_tr), yb_tr)
pb = m2s.predict(scaler_b.transform(Xb_te))
fpr_scaled = (m2s.predict(scaler_b.transform(unseen_normal_X))==1).mean()
print("B:", fmt(metrics(yb_te, pb)), " FPR on unseen:", f"{fpr_scaled*100:.1f}%")

print()
print("=== Bonus: Isolation Forest (unsupervised, trained ONLY on normal data) ===")
iso = IsolationForest(random_state=8, contamination=0.1).fit(scaler_b.transform(normal_train[FEATURES]))
# IsolationForest: -1 = anomaly, 1 = normal -> convert to 1=anomaly,0=normal
pred_iso = (iso.predict(scaler_b.transform(Xb_te)) == -1).astype(int)
fpr_iso = (iso.predict(scaler_b.transform(unseen_normal_X)) == -1).mean()
print("B:", fmt(metrics(yb_te, pred_iso)), " FPR on unseen:", f"{fpr_iso*100:.1f}%")
