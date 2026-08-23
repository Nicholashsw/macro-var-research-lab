import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "font.family": "DejaVu Sans",
})
INK, BLUE, RED, GRAY, AMBER = "#1b2a41", "#2455c3", "#c0392b", "#8a97a8", "#c9922b"

t = pd.read_parquet("mp_compare.parquet")
ARMS = ["mp_raw", "mp_surp", "mp_orth"]
LABEL = {"mp_raw": "cheap dFFR", "mp_surp": "raw MPS", "mp_orth": "orthogonalized MPS"}
COLOR = {"mp_raw": AMBER, "mp_surp": BLUE, "mp_orth": RED}

# GDP at h=1 (conventional short-horizon), inflation at h=4 (transmission lag).
panels = [("gdp", 1, "GDP growth, h=1"), ("inf", 4, "Core inflation, h=4")]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
for ax, (target, h, ttl) in zip(axes, panels):
    samples = ["all", "excovid"]
    x = np.arange(len(samples))
    w = 0.24
    for k, arm in enumerate(ARMS):
        vals = []
        for sample in samples:
            pv = t[(t.arm == "pure_var") & (t.target == target) & (t.h == h) & (t["sample"] == sample)].iloc[0].rmse
            av = t[(t.arm == arm) & (t.target == target) & (t.h == h) & (t["sample"] == sample)].iloc[0].rmse
            vals.append(100 * (1 - av / pv))
        ax.bar(x + (k - 1) * w, vals, w, color=COLOR[arm], label=LABEL[arm])
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_xticks(x, ["full window", "ex COVID"], fontsize=7.5)
    ax.set_ylabel("RMSE reduction vs core, %")
    ax.set_title(ttl)
    if target == "gdp":
        ax.legend(frameon=False, fontsize=7, loc="lower left")
axes[0].text(0.5, -0.34,
             "core already carries stance endogenously; identified surprises add nothing significant",
             transform=axes[0].transAxes, ha="center", fontsize=6.8, color=GRAY)
fig.suptitle("Identified monetary surprises do not beat the endogenous stance proxy", fontsize=9.5)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("fig_mp.png", bbox_inches="tight")
plt.close(fig)
print("wrote fig_mp.png")
