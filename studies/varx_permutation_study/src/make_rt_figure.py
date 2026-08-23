import json
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

comp = pd.read_parquet("rt_compare.parquet")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))

# Panel A: final vs real-time RMSE, GDP specs, all horizons; 45-degree reference
ax = axes[0]
g = comp[(comp.target == "gdp") & (comp["sample"] == "all")]
col = {1: BLUE, 4: GRAY, 8: INK}
for h in [1, 4, 8]:
    gh = g[g.h == h]
    ax.scatter(gh.rmse_final, gh.rmse_rt, s=26, color=col[h], label=f"h={h}", zorder=3,
               edgecolor="white", linewidth=0.5)
# highlight the purge blow-up and the winner at h=1
gp = g[(g.h == 1) & (g.spec == "all4_gdp_resid")].iloc[0]
gw = g[(g.h == 1) & (g.spec == "all4_raw")].iloc[0]
ax.annotate("all four,\nGDP-purged", (gp.rmse_final, gp.rmse_rt), (gp.rmse_final - 1.4, gp.rmse_rt + 0.15),
            fontsize=6.5, color=RED, ha="right")
ax.annotate("all four, raw", (gw.rmse_final, gw.rmse_rt), (gw.rmse_final + 0.1, gw.rmse_rt - 0.5),
            fontsize=6.5, color=BLUE)
lo, hi = 5.4, 7.9
ax.plot([lo, hi], [lo, hi], color=GRAY, lw=0.8, ls="--", zorder=1)
ax.plot([lo, hi], [lo * 1.03, hi * 1.03], color=RED, lw=0.8, ls=":", zorder=1)
ax.text(7.55, 7.55 * 1.03 + 0.02, "+3%", color=RED, fontsize=6.5, rotation=38)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("Final-vintage RMSE, GDP")
ax.set_ylabel("Real-time RMSE, GDP")
ax.set_title("Revisions shift RMSE up almost uniformly")
ax.legend(frameon=False, fontsize=7, loc="lower right")

# Panel B: shock advantage (all4_raw vs pure_var), final vs real-time
ax = axes[1]
rows = []
for h in [1, 4, 8]:
    for sample in ["all", "excovid"]:
        pv = comp[(comp.spec == "pure_var") & (comp.target == "gdp") & (comp.h == h) & (comp["sample"] == sample)].iloc[0]
        a4 = comp[(comp.spec == "all4_raw") & (comp.target == "gdp") & (comp.h == h) & (comp["sample"] == sample)].iloc[0]
        rows.append((f"h={h}\n{'full' if sample=='all' else 'exCOVID'}",
                     100 * (1 - a4.rmse_final / pv.rmse_final),
                     100 * (1 - a4.rmse_rt / pv.rmse_rt)))
labels = [r[0] for r in rows]
fin = [r[1] for r in rows]
rt = [r[2] for r in rows]
x = np.arange(len(rows))
w = 0.38
ax.bar(x - w / 2, fin, w, color=BLUE, label="final vintage")
ax.bar(x + w / 2, rt, w, color=AMBER, label="real-time")
ax.axhline(0, color=GRAY, lw=0.6)
ax.set_xticks(x, labels, fontsize=7)
ax.set_ylabel("RMSE reduction vs pure VAR, %")
ax.set_title("All-four shock advantage survives real-time")
ax.legend(frameon=False, fontsize=7, loc="upper right")

fig.tight_layout()
fig.savefig("fig_rt.png", bbox_inches="tight")
plt.close(fig)
print("wrote fig_rt.png")
print(json.dumps(json.load(open("rt_stats.json"))["gdp"]["1"], indent=1))
