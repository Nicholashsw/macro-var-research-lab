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

t = pd.read_parquet("id_compare.parquet")
ARM_ORDER = ["pure_var", "oil_raw", "oil_surp", "oil_news"]
ARM_LABEL = {"pure_var": "pure VAR", "oil_raw": "raw proxy",
             "oil_surp": "HF surprise", "oil_news": "news shock"}
ARM_COLOR = {"pure_var": GRAY, "oil_raw": BLUE, "oil_surp": AMBER, "oil_news": RED}

# Percentage RMSE change versus pure VAR, h=1, both windows, GDP and inflation.
fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
for ax, target, ttl in zip(axes, ["gdp", "inf"], ["GDP growth", "Core inflation"]):
    samples = ["all", "excovid"]
    x = np.arange(len(samples))
    w = 0.2
    for k, arm in enumerate(ARM_ORDER[1:]):  # skip pure_var, it is the 0% baseline
        vals = []
        for sample in samples:
            pv = t[(t.arm == "pure_var") & (t.target == target) & (t.h == 1) & (t["sample"] == sample)].iloc[0].rmse
            av = t[(t.arm == arm) & (t.target == target) & (t.h == 1) & (t["sample"] == sample)].iloc[0].rmse
            vals.append(100 * (1 - av / pv))
        ax.bar(x + (k - 1) * w, vals, w, color=ARM_COLOR[arm], label=ARM_LABEL[arm])
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_xticks(x, ["full window", "ex COVID"], fontsize=7.5)
    ax.set_ylabel("RMSE reduction vs pure VAR, %")
    ax.set_title(f"{ttl}, h=1")
    if target == "gdp":
        ax.legend(frameon=False, fontsize=7, loc="lower left", ncol=1)
axes[0].text(0.5, -0.34, "positive is better than the pure VAR; on the full window the raw proxy leads, none is significant",
             transform=axes[0].transAxes, ha="center", fontsize=6.8, color=GRAY)
fig.suptitle("External identification does not beat the cheap oil proxy out of sample", fontsize=9.5)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("fig_id.png", bbox_inches="tight")
plt.close(fig)
print("wrote fig_id.png")
