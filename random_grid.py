import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from itertools import combinations
from sklearn.model_selection import StratifiedKFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

# ============================================================
# CONFIG
# ============================================================
INPUT_CSV      = "df3finaluse3_surv.csv"
OUTPUT_CSV     = "pmri_feature_search_results.csv"
N_ITER         = 200       # number of random feature subsets to try
MIN_FEATURES   = 2         # minimum features per subset
MAX_FEATURES   = 10        # maximum features per subset
N_ESTIMATORS   = 200       # use fewer trees for speed during search
SEED           = 42

# ============================================================
# LOAD
# ============================================================
df = pd.read_csv(INPUT_CSV)

id_cols      = ["pat_id", "group", "Time_to_event"]
feature_cols = [c for c in df.columns if c not in id_cols]

X_df = df[feature_cols].copy()
y    = np.array(
    [(bool(g), t) for g, t in zip(df["group"], df["Time_to_event"])],
    dtype=[("fail", bool), ("Time_to_event", float)],
)

print(f"Total features available: {len(feature_cols)}")
print(f"Running {N_ITER} random feature subset evaluations …\n")

# ============================================================
# RANDOM FEATURE SEARCH
# ============================================================
rng = np.random.default_rng(SEED)
kf  = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)

results = []

for iteration in range(N_ITER):
    # sample a random subset of features
    n_feats  = rng.integers(MIN_FEATURES, min(MAX_FEATURES, len(feature_cols)) + 1)
    selected = rng.choice(feature_cols, size=n_feats, replace=False).tolist()

    X = X_df[selected].values

    fold_c = []
    for train_idx, val_idx in kf.split(X, df["group"].values):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        rsf = RandomSurvivalForest(n_estimators=N_ESTIMATORS, random_state=SEED)
        rsf.fit(X_train, y_train)
        fold_c.append(rsf.score(X_val, y_val))

    mean_c = np.mean(fold_c)
    results.append({
        "iteration":  iteration + 1,
        "n_features": n_feats,
        "mean_c":     round(mean_c, 4),
        "features":   ", ".join(selected),
    })

    if (iteration + 1) % 25 == 0:
        print(f"  [{iteration + 1}/{N_ITER}] best so far: "
              f"{max(r['mean_c'] for r in results):.4f}")

# ============================================================
# RESULTS
# ============================================================
results_df = pd.DataFrame(results).sort_values("mean_c", ascending=False).reset_index(drop=True)
results_df.to_csv(OUTPUT_CSV, index=False)

print(f"\nTop 10 feature subsets:")
print(results_df.head(10).to_string(index=False))
print(f"\nSaved full results to {OUTPUT_CSV}")

# ── Best subset ───────────────────────────────────────────────────────────────
best = results_df.iloc[0]
print(f"\nBest mean C-index: {best['mean_c']:.4f}")
print(f"Best features ({int(best['n_features'])}): {best['features']}")