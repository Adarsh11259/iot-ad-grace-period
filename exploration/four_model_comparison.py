import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("../data/profile_traces_combined.csv").dropna(subset=["DestinationPort","Length","FlagValue"])
FEATURES = ["DestinationPort","Length","FlagValue"]

def metrics(y_true, y_pred):
    return dict(
        acc=accuracy_score(y_true,y_pred),
        prec=precision_score(y_true,y_pred, zero_division=0),
        rec=recall_score(y_true,y_pred, zero_division=0),
        f1=f1_score(y_true,y_pred, zero_division=0),
    )

def fmt(m):
    return f"acc={m['acc']:.3f} prec={m['prec']:.3f} rec={m['rec']:.3f} f1={m['f1']:.3f}"

# ---- Build split A: pooled random 75/25 ----
X = df[FEATURES]; y = df["IsAnomaly"]
Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(X, y, test_size=0.25, random_state=8)

# ---- Build split B: held-out device files (same 6 held out as before) ----
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

results = {}

def run_supervised(name, model_ctor):
    m1 = model_ctor(); m1.fit(Xa_tr, ya_tr)
    pa = m1.predict(Xa_te)
    m2 = model_ctor(); m2.fit(Xb_tr, yb_tr)
    pb = m2.predict(Xb_te)
    fpr = (m2.predict(unseen_normal_X) == 1).mean()
    results[name] = (metrics(ya_te, pa), metrics(yb_te, pb), fpr)

run_supervised("Random Forest", lambda: RandomForestClassifier(random_state=8))
run_supervised("Decision Tree", lambda: DecisionTreeClassifier(random_state=8))
run_supervised("k-NN (k=5)",    lambda: KNeighborsClassifier(n_neighbors=5))

# ---- Autoencoder (reconstruction-error based, trained on NORMAL data only) ----
def run_autoencoder(Xtr_all, ytr_all, Xte_all, yte_all, unseen_X):
    scaler = StandardScaler().fit(Xtr_all[ytr_all==0])
    Xn = scaler.transform(Xtr_all[ytr_all==0])
    ae = MLPRegressor(hidden_layer_sizes=(8,3,8), max_iter=300, random_state=8)
    ae.fit(Xn, Xn)
    def recon_err(Xraw):
        Xs = scaler.transform(Xraw)
        rec = ae.predict(Xs)
        return np.mean((Xs-rec)**2, axis=1)
    train_err = recon_err(Xtr_all[ytr_all==0])
    threshold = np.percentile(train_err, 95)
    pred = (recon_err(Xte_all) > threshold).astype(int)
    fpr = (recon_err(unseen_X) > threshold).mean() if len(unseen_X) else float('nan')
    return metrics(yte_all, pred), fpr

m_a, _ = run_autoencoder(Xa_tr, ya_tr, Xa_te, ya_te, pd.DataFrame(columns=FEATURES))
m_b, fpr_ae = run_autoencoder(Xb_tr, yb_tr, Xb_te, yb_te, unseen_normal_X)
results["Autoencoder"] = (m_a, m_b, fpr_ae)

print(f"{'Model':15s} | {'A) Random split (reproduces paper)':38s} | {'B) Held-out-device split':38s} | FPR on unseen-normal devices")
print("-"*130)
for name,(ma,mb,fpr) in results.items():
    print(f"{name:15s} | {fmt(ma):38s} | {fmt(mb):38s} | {fpr*100:.1f}%")
