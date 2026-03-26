from sksurv.metrics import concordance_index_censored
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import fisher_exact

df = pd.read_csv("EnsembleAUC2.csv")

from sksurv.metrics import concordance_index_censored
import numpy as np

# ── Observed C-index ──────────────────────────────────────────────────────────
y_struct = np.array(
    [(bool(e), t) for e, t in zip(df["hydro"], df["time_to_event"])],
    dtype=[("event", bool), ("time", float)]
)

c_index, concordant, discordant, tied_risk, tied_time = concordance_index_censored(
    y_struct["event"],
    y_struct["time"],
    df["ensemble_risk"]
)
print(f"Harrell's C-index: {c_index:.3f}")
print(f"Concordant pairs:  {concordant}")
print(f"Discordant pairs:  {discordant}")

# ── Bootstrap CI + p-value ────────────────────────────────────────────────────
N_BOOT = 1000
SEED   = 42
rng    = np.random.default_rng(SEED)
n      = len(df)

boot_c = []
for _ in range(N_BOOT):
    idx = rng.choice(n, size=n, replace=True)
    ys  = y_struct[idx]
    rs  = df["ensemble_risk"].values[idx]
    try:
        c, *_ = concordance_index_censored(ys["event"], ys["time"], rs)
        boot_c.append(c)
    except Exception:
        pass

boot_c = np.array(boot_c)

ci_lower = np.percentile(boot_c, 2.5)
ci_upper = np.percentile(boot_c, 97.5)

# p-value: proportion of bootstrap samples where C <= 0.5 (null = no discrimination)
p_value = np.mean(boot_c <= 0.5)

print(f"\n95% CI:  [{ci_lower:.3f}, {ci_upper:.3f}]")
print(f"p-value (vs C=0.5): {p_value:.4f}")



# ── Config — change this to 10, 20, 30, 40, or 50 ────────────────────────────
TOP_PCT = 50

# ── Risk grouping ─────────────────────────────────────────────────────────────
threshold = np.percentile(df["ensemble_risk"], 100 - TOP_PCT)
df["risk_group"] = (df["ensemble_risk"] >= threshold).astype(int)
df["risk_label"]  = df["risk_group"].map({1: "High Risk", 0: "Low Risk"})

# ── Confusion matrix values ───────────────────────────────────────────────────
ct = pd.crosstab(df["risk_label"], df["hydro"].map({1: "Event", 0: "No Event"}))
ct = ct.reindex(index=["High Risk", "Low Risk"], columns=["Event", "No Event"], fill_value=0)

# ── Fisher's exact p-value ────────────────────────────────────────────────────
_, p_value = fisher_exact(ct.values)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", ax=ax,
            linewidths=0.5, linecolor="gray")
ax.set_xlabel("Observed Outcome", fontsize=12)
ax.set_ylabel("Predicted Risk Group", fontsize=12)
ax.set_title(f"Risk Stratification (Top {TOP_PCT}% = High Risk)\np = {p_value:.4f}", fontsize=13)
plt.tight_layout()
plt.show()

