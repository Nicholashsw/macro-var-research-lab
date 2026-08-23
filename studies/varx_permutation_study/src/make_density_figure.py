import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import varx_lab as vl
import varx_density as vd

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "font.family": "DejaVu Sans",
})
INK, BLUE, RED, GRAY, AMBER = "#1b2a41", "#2455c3", "#c0392b", "#8a97a8", "#c9922b"

df = vl.build_panel()
store, origins = vd.run_density(df)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))

# Panel A: 90% coverage for GDP, all-four shocks, naive vs fair, across horizons.
ax = axes[0]
hs = [1, 4, 8]
cov_naive, cov_fair, cov_pure = [], [], []
for h in hs:
    for which, bucket in (("naive", cov_naive), ("fair", cov_fair)):
        _, _, vr = None, None, None
        dts, mu, v, y = vd.collect(store, origins, df, "all4_raw", "gdp", h, which)
        p = vd.pit(y, mu, v)
        mk = vl._mask(dts, "all")
        bucket.append(np.mean((p[mk] > 0.05) & (p[mk] < 0.95)))
    dts, mu, v, y = vd.collect(store, origins, df, "pure_var", "gdp", h, "naive")
    p = vd.pit(y, mu, v); mk = vl._mask(dts, "all")
    cov_pure.append(np.mean((p[mk] > 0.05) & (p[mk] < 0.95)))
x = np.arange(len(hs)); w = 0.26
ax.bar(x - w, cov_naive, w, color=RED, label="all four, naive density")
ax.bar(x, cov_fair, w, color=BLUE, label="all four, fair density")
ax.bar(x + w, cov_pure, w, color=GRAY, label="pure VAR")
ax.axhline(0.90, color=INK, lw=1.0, ls="--")
ax.text(2.35, 0.905, "nominal 90%", fontsize=6.5, color=INK, ha="right")
ax.set_xticks(x, [f"h={h}" for h in hs])
ax.set_ylabel("90% interval coverage, GDP")
ax.set_ylim(0.5, 1.0)
ax.set_title("Shocks look overconfident, until shock risk is priced in")
ax.legend(frameon=False, fontsize=6.8, loc="lower center")

# Panel B: PIT histogram for all-four shocks, GDP, h=1, naive vs fair.
ax = axes[1]
dts, mu, v, y = vd.collect(store, origins, df, "all4_raw", "gdp", 1, "naive")
mk = vl._mask(dts, "excovid")
p_naive = vd.pit(y[mk], mu[mk], v[mk])
dts, mu, v, y = vd.collect(store, origins, df, "all4_raw", "gdp", 1, "fair")
mk = vl._mask(dts, "excovid")
p_fair = vd.pit(y[mk], mu[mk], v[mk])
bins = np.linspace(0, 1, 11)
ax.hist(p_naive, bins=bins, color=RED, alpha=0.55, label="naive density", density=True)
ax.hist(p_fair, bins=bins, color=BLUE, alpha=0.55, label="fair density", density=True)
ax.axhline(1.0, color=INK, lw=1.0, ls="--")
ax.text(0.98, 1.06, "uniform", fontsize=6.5, color=INK, ha="right")
ax.set_xlabel("PIT value (all four shocks, GDP, h=1, ex COVID)")
ax.set_ylabel("density")
ax.set_title("Probability integral transform")
ax.legend(frameon=False, fontsize=6.8, loc="upper center")

fig.suptitle("Shock-augmented density is overconfident under the naive plug-in, calibrated once future-shock risk is added",
             fontsize=8.8)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("fig_density.png", bbox_inches="tight")
plt.close(fig)
print("wrote fig_density.png")
print("GDP 90% coverage naive:", [round(c, 3) for c in cov_naive],
      "fair:", [round(c, 3) for c in cov_fair])
