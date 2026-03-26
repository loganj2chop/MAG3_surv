import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

# ============================================================
# LOAD
# ============================================================
df = pd.read_csv("df3finaluse3_surv.csv")

# ============================================================
# FEATURE MATRIX & IDS
# ============================================================
id_cols     = ["pat_id", "group", "Time_to_event"]
feature_cols = [c for c in df.columns if c not in id_cols]

ids = df[id_cols].copy()
X   = df[feature_cols].values

# ============================================================
# SURVIVAL TARGET
# ============================================================
y = np.array(
    [(bool(g), t) for g, t in zip(df["group"], df["Time_to_event"])],
    dtype=[("fail", bool), ("Time_to_event", float)],
)

# ============================================================
# 2-FOLD STRATIFIED CV — RANDOM SURVIVAL FOREST
# ============================================================
kf  = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
rsf = RandomSurvivalForest(n_estimators=1600, random_state=42)

oof_risk = np.zeros(len(df))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, df["group"].values)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    rsf.fit(X_train, y_train)

    oof_risk[val_idx] = rsf.predict(X_val)
    c = rsf.score(X_val, y_val)

# ============================================================
# MERGE RESULTS
# ============================================================
df_merged = ids.reset_index(drop=True).copy()
df_merged["risk"] = oof_risk
df_merged["fail"] = df_merged["group"].astype(bool)

# overall OOF C-index
c_overall, *_ = concordance_index_censored(
    df_merged["fail"], df_merged["Time_to_event"], df_merged["risk"]
)
print(f"\nOverall OOF C-index: {c_overall:.3f}")
print(df_merged.head())

df_merged.to_csv("pmri_risk_scores.csv", index=False)
print("Saved pmri_risk_scores.csv")