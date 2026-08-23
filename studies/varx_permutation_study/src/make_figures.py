import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import varx_lab as vl

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "font.family": "DejaVu Sans",
})
INK = "#1b2a41"
BLUE = "#2455c3"
RED = "#c0392b"
GRAY = "#8a97a8"

df = vl.build_panel()
res = pd.read_parquet("results.parquet")
V = res[res.valid]

# F1 data panel
fig, axes = plt.subplots(4, 3, figsize=(9, 7.5), sharex=True)
for ax, c in zip(axes.ravel(), ["gdp", "inf", "stance", "fx", "wage", "emp", "spread", "permits", "oil", "inv", "wealth", "fiscal"]):
    ax.plot(df.index, df[c], color=BLUE if c in vl.ENDOG_CORE + vl.ENDOG_ADDONS else RED, lw=0.9)
    ax.axhline(0, color=GRAY, lw=0.5)
    ax.set_title(vl.LABELS[c], color=INK)
fig.suptitle("Quarterly panel, 1990Q1 to 2026Q1. Blue endogenous candidates, red exogenous shocks.", y=1.0, fontsize=9)
fig.tight_layout()
fig.savefig("fig1_panel.png", bbox_inches="tight")
plt.close(fig)

# F2 shock count value, gdp h=1
s = V[(V.target == "gdp") & (V.h == 1) & (V.path == "uncond") & (V["sample"] == "all")].copy()
s["nshocks"] = s.shocks.apply(lambda x: 0 if x == "none" else len(x.split("+")))
fig, ax = plt.subplots(figsize=(4.2, 2.9))
groups = [s[s.nshocks == k].rmse.values for k in [0, 1, 2, 4]]
bp = ax.boxplot(groups, tick_labels=["0", "1", "2", "4"], showfliers=False, widths=0.55,
                medianprops=dict(color=RED, lw=1.4), boxprops=dict(color=INK), whiskerprops=dict(color=INK), capprops=dict(color=INK))
ax.set_xlabel("Number of exogenous shocks in the spec")
ax.set_ylabel("RMSE, GDP growth, h=1")
ax.set_title("More shocks help at h=1: no stacking penalty in this design")
fig.tight_layout()
fig.savefig("fig2_shockcount.png", bbox_inches="tight")
plt.close(fig)

# F3 treatment comparison per target h=1
fig, axes = plt.subplots(1, 4, figsize=(9, 2.7))
for ax, t in zip(axes, ["gdp", "inf", "stance", "fx"]):
    s = V[(V.target == t) & (V.h == 1) & (V.path == "uncond") & (V["sample"] == "all") & (V.shocks != "none")]
    groups = [s[s.treatment == tr].rmse.values for tr in ["raw", "gdp_resid", "full_resid"]]
    ax.boxplot(groups, tick_labels=["raw", "gdp", "full"], showfliers=False, widths=0.55,
               medianprops=dict(color=RED, lw=1.4), boxprops=dict(color=INK), whiskerprops=dict(color=INK), capprops=dict(color=INK))
    ax.set_title(t)
axes[0].set_ylabel("RMSE, h=1")
fig.suptitle("Shock treatment: raw vs GDP-purged vs fully purged", y=1.03, fontsize=9)
fig.tight_layout()
fig.savefig("fig3_treatment.png", bbox_inches="tight")
plt.close(fig)

# F4 lag rule medians heat-style table
s = V[(V.path == "uncond") & (V["sample"] == "all")]
piv = s.groupby(["target", "h", "lag_rule"]).rmse.median().unstack()
fig, ax = plt.subplots(figsize=(4.6, 3.2))
norm = piv.div(piv.min(axis=1), axis=0)
im = ax.imshow(norm.values, cmap="RdYlGn_r", vmin=1.0, vmax=1.12, aspect="auto")
ax.set_xticks(range(3), ["AIC", "BIC", "HQIC"])
ax.set_yticks(range(len(piv)), [f"{t} h={h}" for t, h in piv.index])
for i in range(len(piv)):
    for j in range(3):
        ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color=INK)
ax.set_title("Median RMSE by lag rule (green = best in row)")
fig.tight_layout()
fig.savefig("fig4_lagrules.png", bbox_inches="tight")
plt.close(fig)

# F5 value of knowing the shock path
s = V[(V.target == "gdp") & (V["sample"] == "all") & (V.shocks != "none")]
piv = s.groupby(["h", "treatment", "path"]).rmse.median().unstack()
fig, ax = plt.subplots(figsize=(4.6, 2.9))
xs = np.arange(3)
w = 0.13
for k, tr in enumerate(["raw", "gdp_resid", "full_resid"]):
    u = [piv.loc[(h, tr), "uncond"] for h in [1, 4, 8]]
    r = [piv.loc[(h, tr), "realized"] for h in [1, 4, 8]]
    ax.bar(xs + (k - 1) * 2.2 * w - w / 2, u, w, color=BLUE, alpha=0.4 + 0.3 * k)
    ax.bar(xs + (k - 1) * 2.2 * w + w / 2, r, w, color=RED, alpha=0.4 + 0.3 * k)
ax.set_xticks(xs, ["h=1", "h=4", "h=8"])
ax.set_ylabel("Median RMSE, GDP")
ax.set_ylim(5.4, None)
ax.set_title("Unconditional (blue) vs realized shock path (red)\npairs left to right: raw, GDP-purged, fully purged")
fig.tight_layout()
fig.savefig("fig5_paths.png", bbox_inches="tight")
plt.close(fig)

# F6 best gdp model forecast vs actual, h=1
best = V[(V.target == "gdp") & (V.h == 1) & (V.path == "uncond") & (V["sample"] == "all")].nsmallest(1, "rmse").iloc[0]
with open("payload.json") as f:
    payload = json.load(f)
sid = str(int(best.spec_id))
mid = str([s for s in payload["specs"] if s["sid"] == int(best.spec_id)][0]["mid"])
act = payload["actual"]["gdp_1"]
ser = payload["series"][mid]["uncond_gdp_1"]
dates = pd.PeriodIndex(act["dates"], freq="M").to_timestamp()
fig, axes = plt.subplots(1, 2, figsize=(9, 2.9))
axes[0].plot(dates, act["y"], color=INK, lw=1.1, label="actual")
axes[0].plot(dates, ser, color=RED, lw=1.1, label="forecast")
axes[0].legend(frameon=False)
axes[0].set_title(f"Best GDP spec, h=1: {best.endog} | shocks {best.shocks} ({best.treatment}, p={int(best.p)})")
axes[0].set_ylabel("GDP growth, saar")
mask = ~((dates >= "2020-01-01") & (dates <= "2021-03-31"))
axes[1].plot(dates[mask], np.array(act["y"])[mask], color=INK, lw=1.1)
axes[1].plot(dates[mask], np.array(ser)[mask], color=RED, lw=1.1)
axes[1].set_title("Same spec, COVID quarters removed")
fig.tight_layout()
fig.savefig("fig6_bestgdp.png", bbox_inches="tight")
plt.close(fig)

# paper stats
stats = {}
for t in ["gdp", "inf", "stance", "fx"]:
    for h in [1, 4, 8]:
        for sample in ["all", "excovid"]:
            s = V[(V.target == t) & (V.h == h) & (V.path == "uncond") & (V["sample"] == sample)]
            b = s.nsmallest(1, "rmse").iloc[0]
            ns = s[s.shocks == "none"].rmse.min()
            stats[f"{t}_{h}_{sample}"] = {
                "best": {k: (round(float(b[k]), 3) if isinstance(b[k], float) else (int(b[k]) if k == "p" else b[k]))
                         for k in ["endog", "shocks", "treatment", "lag_rule", "p", "rmse", "r2_oos", "dm_p"]},
                "noshock_best_rmse": round(float(ns), 3),
            }
s = V[(V.path == "uncond") & (V["sample"] == "all")]
stats["lagrule_median"] = {f"{t}_{h}": {r: round(float(x), 3) for r, x in
                            s[(s.target == t) & (s.h == h)].groupby("lag_rule").rmse.median().items()}
                           for t in ["gdp", "inf", "stance", "fx"] for h in [1, 4]}
sp = res.drop_duplicates("spec_id")
stats["p_by_rule"] = {r: sp[sp.lag_rule == r].p.value_counts().sort_index().to_dict() for r in vl.LAG_RULES}
s2 = V[(V.h == 1) & (V.path == "uncond") & (V["sample"] == "all") & (V.shocks != "none")]
stats["treat_median_h1"] = {t: {tr: round(float(x), 3) for tr, x in
                             s2[s2.target == t].groupby("treatment").rmse.median().items()}
                            for t in ["gdp", "inf", "stance", "fx"]}
s3 = V[(V.target == "gdp") & (V["sample"] == "all") & (V.shocks != "none")]
stats["path_median_gdp"] = {f"h{h}_{tr}": {p: round(float(x), 3) for p, x in
                             s3[(s3.h == h) & (s3.treatment == tr)].groupby("path").rmse.median().items()}
                            for h in [1, 4, 8] for tr in vl.TREATMENTS}
s4 = V[(V.target == "gdp") & (V.h == 1) & (V.path == "uncond") & (V["sample"] == "all")].copy()
s4["nshocks"] = s4.shocks.apply(lambda x: 0 if x == "none" else len(x.split("+")))
stats["shockcount_gdp_h1"] = {int(k): {"median": round(float(v["median"]), 3), "min": round(float(v["min"]), 3)}
                              for k, v in s4.groupby("nshocks").rmse.agg(["median", "min"]).iterrows()}
stats["counts"] = {"cells": 544, "spec_rows": int(res.spec_id.nunique()), "unique_models": int(res.model_id.nunique()),
                   "valid_share": round(float(res.drop_duplicates("spec_id").valid.mean()), 3),
                   "n_origins": 65, "result_rows": int(len(res))}
dmsig = V[(V.path == "uncond") & (V["sample"] == "all") & V.dm_p.notna()]
stats["dm_sig_share_5pct"] = round(float((dmsig.dm_p < 0.05).mean()), 4)
with open("paper_stats.json", "w") as f:
    json.dump(stats, f, indent=1)
print("figures + stats done")
print(json.dumps(stats["counts"]))
print("dm sig share:", stats["dm_sig_share_5pct"])
