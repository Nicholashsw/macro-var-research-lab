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
INK, BLUE, RED, GRAY = "#1b2a41", "#2455c3", "#c0392b", "#8a97a8"

t = pd.read_parquet("bayes_density_compare.parquet")
t["cell"] = t.method + "+" + t.shocks
CELLS = ["plugin+gauss", "bayes+gauss", "plugin+emp"]
LABEL = {"plugin+gauss": "plug-in (baseline)",
         "bayes+gauss": "+ parameter uncertainty",
         "plugin+emp": "+ empirical tails"}
COLOR = {"plugin+gauss": GRAY, "bayes+gauss": BLUE, "plugin+emp": RED}

fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
hs = [1, 4, 8]
for ax, target, ttl in zip(axes, ["gdp", "inf"], ["GDP growth", "Core inflation"]):
    x = np.arange(len(hs)); w = 0.26
    for k, cell in enumerate(CELLS):
        vals = [t[(t.target == target) & (t["sample"] == "all") & (t.h == h) & (t.cell == cell)].iloc[0].cov90
                for h in hs]
        ax.bar(x + (k - 1) * w, vals, w, color=COLOR[cell], label=LABEL[cell])
    ax.axhline(0.90, color=INK, lw=1.0, ls="--")
    ax.text(2.4, 0.905, "nominal 90%", fontsize=6.5, color=INK, ha="right")
    ax.set_xticks(x, [f"h={h}" for h in hs])
    ax.set_ylabel("90% interval coverage")
    ax.set_ylim(0.6, 1.0)
    ax.set_title(ttl)
    if target == "gdp":
        ax.legend(frameon=False, fontsize=6.8, loc="lower right")
fig.suptitle("Pure VAR under-coverage is estimation noise: parameter uncertainty closes it, fat tails do not",
             fontsize=8.8)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("fig_bayes.png", bbox_inches="tight")
plt.close(fig)
print("wrote fig_bayes.png")
