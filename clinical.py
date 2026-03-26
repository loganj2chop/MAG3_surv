
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

# ============================================================
# LOAD & CLEAN
# ============================================================
df = pd.read_csv("crossfinal_mag3.csv")
df.columns = (
    df.columns.str.strip().str.lower()
    .str.replace(" ", "_")
    .str.replace("(", "")
    .str.replace(")", "")
)

# keep relevant groups and columns
df = df[df["group"].isin(["Complication", "No complication"])].dropna(subset=["group"]).copy()

KEEP_COLS = [
    "record_id", "group", "t_1/2_half",
    "differential_total_volume_right_1st_scan", "differential_total_volume_left",
    "initial_us_grade", "differential_per_unit_volume_right",
    "differential_per_unit_volume_left", "age_in_years", "sex", "time_to_event",
]
df = df[KEEP_COLS].copy()

# ============================================================
# FEATURE ENGINEERING
# ============================================================
df["differential_total_volume_left"] = df["differential_total_volume_left"].astype(float)

# absolute laterality difference — total volume
df["diff_total_volume"] = (
    df["differential_total_volume_right_1st_scan"] - df["differential_total_volume_left"]
).abs()

# smaller of the two per-unit volumes
df["diff_unit_volume"] = df[
    ["differential_per_unit_volume_right", "differential_per_unit_volume_left"]
].min(axis=1)

# ============================================================
# ENCODING
# ============================================================
df["sex"]             = df["sex"].map({"M": 0, "F": 1})
df["initial_us_grade"] = df["initial_us_grade"].map({"P1": 1, "P2": 2, "P3": 3})
df["group"]           = df["group"].map({"Complication": 1, "No complication": 0})

# ============================================================
# FINAL FEATURE MATRIX
# ============================================================
FEATURE_COLS = ["diff_unit_volume", "age_in_years", "diff_total_volume", "t_1/2_half"]

df = df[["record_id", "group", "time_to_event"] + FEATURE_COLS].dropna()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = df[FEATURE_COLS].values
ids = df[["record_id", "group", "time_to_event"]].copy()

# sksurv structured array
y = np.array(
    [(bool(g), t) for g, t in zip(df["group"], df["time_to_event"])],
    dtype=[("fail", bool), ("time_to_event", float)],
)

# ============================================================
# 2-FOLD STRATIFIED CV — RANDOM SURVIVAL FOREST
# ============================================================
kf  = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
rsf = RandomSurvivalForest(n_estimators=200, random_state=42)

oof_risk  = np.zeros(len(df))
oof_group = np.zeros(len(df))
oof_time  = np.zeros(len(df))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, df["group"].values)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    rsf.fit(X_train, y_train)

    risk = rsf.predict(X_val)
    c    = rsf.score(X_val, y_val)
    print(f"Fold {fold + 1} C-index: {c:.3f}")

    oof_risk[val_idx]  = risk
    oof_group[val_idx] = df["group"].values[val_idx]
    oof_time[val_idx]  = df["time_to_event"].values[val_idx]

# ============================================================
# MERGE RESULTS
# ============================================================
df_merged = ids.copy().reset_index(drop=True)
df_merged["risk"]  = oof_risk
df_merged["fail"]  = oof_group.astype(bool)

# overall OOF C-index
c_overall, *_ = concordance_index_censored(
    df_merged["fail"], df_merged["time_to_event"], df_merged["risk"]
)

print(df_merged.head())

df_merged.to_csv("clinical_risk_scores.csv", index=False)
print("Saved clinical_risk_scores.csv")