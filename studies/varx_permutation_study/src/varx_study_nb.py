# %% [markdown]
# # VARX Permutation Study: US Quarterly Macro Forecasting
#
# A horse race over 1,632 spec cells: which endogenous variable set, exogenous shock
# subset, shock treatment and lag rule produces the best out-of-sample forecasts for
# GDP growth, core inflation, the policy stance and the dollar.
#
# Design:
#
# - Endogenous core in every spec: GDP growth, core PCE inflation, policy stance
#   defined as the real fed funds rate minus the Holston-Laubach-Williams r-star.
#   Up to two add-ons from: broad dollar, wages, payrolls, term spread, permits.
# - Exogenous shocks (VARX regressors, lags 0 and 1): real oil, real private
#   investment, real household net worth, real government spending growth.
#   Subsets: none, singles, pairs, all four.
# - Shock treatments: raw, purged of contemporaneous GDP, purged of all endogenous
#   variables and one lag. Purge coefficients are re-fit on each training window.
# - Lag order chosen by AIC, BIC and HQIC on the initial training window, p in 1..4.
# - Expanding-window walk-forward from 2009Q4, horizons 1, 4, 8. Two exogenous
#   paths at forecast time: shocks at training mean (unconditional) and realized.
# - Metrics: RMSE, MAE, out-of-sample R2 against an AR(BIC) benchmark,
#   Diebold-Mariano against the core three-variable VAR. All metrics reported for
#   the full window and excluding COVID target quarters (2020Q1 to 2021Q1).
# - Validity gate before ranking: companion stability, Ljung-Box residual
#   whiteness, observations per parameter of at least three.
#
# Data: FRED public CSV endpoint plus the NY Fed HLW workbook. Sample 1990Q1 to
# 2026Q1, 145 quarters, 65 forecast origins at h=1.

# %%
import itertools
import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import varx_lab as vl

plt.rcParams.update({
    "figure.dpi": 110, "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
})
INK, BLUE, RED = "#1b2a41", "#2455c3", "#c0392b"

# %%
# Pull the raw data if it is not already on disk. FRED needs no API key through
# the fredgraph CSV endpoint. The HLW workbook comes from the NY Fed site.
SERIES = ["GDPC1", "PCEPILFE", "FEDFUNDS", "TWEXBMTH", "DTWEXBGS", "AHETPI",
          "PAYEMS", "GS10", "TB3MS", "PERMIT", "WTISPLC", "CPIAUCSL",
          "GPDIC1", "TNWBSHNO", "GCEC1"]
HLW_URL = ("https://www.newyorkfed.org/medialibrary/media/research/economists/"
           "williams/data/Holston_Laubach_Williams_current_estimates.xlsx")
os.makedirs("data", exist_ok=True)
for s in SERIES:
    if not os.path.exists(f"data/{s}.csv"):
        subprocess.run(["curl", "-s", f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}",
                        "-o", f"data/{s}.csv"], check=True)
if not os.path.exists("data/hlw.xlsx"):
    subprocess.run(["curl", "-sL", HLW_URL, "-o", "data/hlw.xlsx"], check=True)
print("data ready")

# %% [markdown]
# ## 1. Panel construction
#
# All growth rates are annualized quarterly log differences, $400\,\Delta\ln x_t$.
# The stance variable is
#
# $$s_t = \left(i_t - \pi_t^{4q}\right) - r^{*}_t$$
#
# with $i_t$ the quarterly average fed funds rate, $\pi_t^{4q}$ trailing four-quarter
# core PCE inflation used only as a deflator, and $r^{*}_t$ the one-sided HLW estimate.
# The trailing four-quarter deflator is deliberate: modeling YoY transforms directly
# would push MA(3) autocorrelation into the residuals and fail the whiteness gate.
# The broad dollar splices the discontinued TWEXBMTH onto DTWEXBGS at 2006-01 with a
# level match so the log difference has no seam. Oil and net worth are deflated by
# headline CPI. COVID gets two impulse dummies, 2020Q2 and 2020Q3, in every model.

# %%
df = vl.build_panel()
print(df.shape, df.index[0].date(), "to", df.index[-1].date())
fig, axes = plt.subplots(4, 3, figsize=(11, 8), sharex=True)
for ax, c in zip(axes.ravel(), ["gdp", "inf", "stance", "fx", "wage", "emp",
                                "spread", "permits", "oil", "inv", "wealth", "fiscal"]):
    ax.plot(df.index, df[c], color=BLUE if c in vl.ENDOG_CORE + vl.ENDOG_ADDONS else RED, lw=0.9)
    ax.axhline(0, color="#8a97a8", lw=0.5)
    ax.set_title(vl.LABELS[c])
fig.suptitle("Endogenous candidates in blue, exogenous shocks in red")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Model and permutation space
#
# Each spec estimates
#
# $$Y_t = c + \sum_{i=1}^{p} A_i Y_{t-i} + B_0 X_t + B_1 X_{t-1} + \Gamma D_t + \varepsilon_t$$
#
# by equation-by-equation OLS, where $Y_t$ is the chosen endogenous vector, $X_t$
# the chosen (possibly purged) shock vector and $D_t$ the COVID dummies. Purging is
# a training-window projection: for the GDP purge, $x_t$ on $[1, g_t]$; for the full
# purge, $x_t$ on $[1, Y_t, Y_{t-1}]$. The residual replaces the raw shock everywhere,
# so the purge never sees evaluation data.
#
# Lag order minimizes, over $p \in \{1,\dots,4\}$ on the initial window,
#
# $$\mathrm{IC}(p) = \ln\left|\hat\Sigma_p\right| + \frac{\phi(T)\, n k_p}{T}, \qquad
# \phi_{\mathrm{AIC}} = 2,\; \phi_{\mathrm{BIC}} = \ln T,\; \phi_{\mathrm{HQ}} = 2\ln\ln T$$
#
# with a hard guard of at least three observations per estimated parameter. The
# order is held fixed across origins to avoid specification churn.

# %%
cells = vl.enumerate_specs()
n_endog_sets = len({tuple(c["endog"]) for c in cells})
n_shock_sets = len({tuple(c["shocks"]) for c in cells})
print(f"{n_endog_sets} endogenous sets x {n_shock_sets} shock subsets x treatments = {len(cells)} cells")
print(f"x 3 lag rules = {len(cells) * 3} spec rows before deduplication")

# %% [markdown]
# ## 3. Walk-forward study
#
# Roughly three minutes from scratch. Results are cached to parquet; delete the
# file to force a re-run.

# %%
if os.path.exists("results.parquet"):
    res = pd.read_parquet("results.parquet")
    print("loaded cache:", res.shape)
else:
    study = vl.run_study(df)
    train_end_i = df.index.get_loc(pd.Period(vl.TRAIN_END, freq="Q").to_timestamp())
    bench = vl.ar_benchmark(df, ["gdp", "inf", "stance", "fx"], train_end_i, study["origins"])
    res = vl.build_results(study, df, bench)
    res["valid"] = vl.validity(res)
    res.to_parquet("results.parquet")
    print("ran study:", res.shape)
V = res[res.valid]
print(f"validity gate keeps {res.drop_duplicates('spec_id').valid.mean():.1%} of specs")

# %% [markdown]
# ## 4. Leaderboards
#
# Per-target leaderboards, unconditional shock path, full evaluation window,
# validity-gated. DM p-values test squared-error loss against the core
# three-variable VAR under the same lag rule.

# %%
def leaderboard(target, h, path="uncond", sample="all", n=8):
    s = V[(V.target == target) & (V.h == h) & (V.path == path) & (V["sample"] == sample)]
    cols = ["addons", "shocks", "treatment", "lag_rule", "p", "rmse", "mae", "r2_oos", "dm_p"]
    out = s.nsmallest(n, "rmse")[cols].reset_index(drop=True)
    return out.round(3)

for t in ["gdp", "inf", "stance", "fx"]:
    print(f"\n=== {vl.LABELS[t]}, h=1 ===")
    print(leaderboard(t, 1).to_string())

# %%
for t in ["gdp", "inf", "stance", "fx"]:
    print(f"\n=== {vl.LABELS[t]}, h=4 ===")
    print(leaderboard(t, 4, n=5).to_string())

# %% [markdown]
# ## 5. Does adding shocks help at all?
#
# Best shocked spec against best pure VAR, per target and horizon. The follow-up
# question, whether the improvement is statistically distinguishable, is answered
# by the DM column above: it almost never is.

# %%
rows = []
for t in ["gdp", "inf", "stance", "fx"]:
    for h in [1, 4, 8]:
        s = V[(V.target == t) & (V.h == h) & (V.path == "uncond") & (V["sample"] == "all")]
        rows.append({"target": t, "h": h,
                     "pure_var_best": s[s.shocks == "none"].rmse.min(),
                     "shocked_best": s[s.shocks != "none"].rmse.min()})
tab = pd.DataFrame(rows)
tab["gain_pct"] = 100 * (1 - tab.shocked_best / tab.pure_var_best)
print(tab.round(3).to_string(index=False))

s = V[(V.target == "gdp") & (V.h == 1) & (V.path == "uncond") & (V["sample"] == "all")].copy()
s["nshocks"] = s.shocks.apply(lambda x: 0 if x == "none" else len(x.split("+")))
fig, ax = plt.subplots(figsize=(5, 3))
ax.boxplot([s[s.nshocks == k].rmse.values for k in [0, 1, 2, 4]],
           tick_labels=["0", "1", "2", "4"], showfliers=False,
           medianprops=dict(color=RED, lw=1.4))
ax.set_xlabel("number of shocks in spec")
ax.set_ylabel("RMSE, GDP h=1")
ax.set_title("Shock stacking does not overfit in this design")
plt.show()

# %% [markdown]
# The stacking result is the opposite of what shows up when shocks enter with deep
# lag structures or as identified instruments inside the system. Here each shock
# adds only two regressors per equation and the COVID dummies soak up the wild
# quarters, so extra shocks behave like extra leading indicators rather than extra
# noise. Do not read this as a license to stack shocks in richer settings.

# %% [markdown]
# ## 6. Residualization: extension 2
#
# Raw versus GDP-purged versus fully purged, shocked specs only.

# %%
s2 = V[(V.h == 1) & (V.path == "uncond") & (V["sample"] == "all") & (V.shocks != "none")]
print(s2.groupby(["target", "treatment"]).rmse.median().unstack().round(3).to_string())
fig, axes = plt.subplots(1, 4, figsize=(11, 3))
for ax, t in zip(axes, ["gdp", "inf", "stance", "fx"]):
    ss = s2[s2.target == t]
    ax.boxplot([ss[ss.treatment == tr].rmse.values for tr in ["raw", "gdp_resid", "full_resid"]],
               tick_labels=["raw", "gdp", "full"], showfliers=False,
               medianprops=dict(color=RED, lw=1.4))
    ax.set_title(t)
axes[0].set_ylabel("RMSE, h=1")
plt.show()

# %% [markdown]
# Purging is an identification tool, not a forecasting tool. For GDP, inflation and
# the dollar the raw shocks carry contemporaneous business-cycle information, and
# projecting it out removes exactly the signal that predicts next quarter. The
# stance equation is the exception: purged investment and fiscal shocks beat raw
# ones, consistent with the policy rate responding to the systematic component of
# demand while the residual component still helps.

# %% [markdown]
# ## 7. Lag rules
#
# AIC against BIC against HQIC, median RMSE across valid specs.

# %%
s3 = V[(V.path == "uncond") & (V["sample"] == "all")]
print(s3.groupby(["target", "h", "lag_rule"]).rmse.median().unstack().round(3).to_string())
sp = res.drop_duplicates("spec_id")
print("\nlag orders chosen:")
print(sp.groupby(["lag_rule", "p"]).size().unstack(fill_value=0).to_string())

# %% [markdown]
# BIC wins or ties for GDP, inflation and the dollar. AIC wins for the stance,
# where the extra persistence of a p=3 or p=4 system pays off. HQIC sits almost
# exactly at p=2 everywhere and is a reasonable single default. The usual warning
# that AIC overfits out of sample holds for three of the four targets.

# %% [markdown]
# ## 8. The value of knowing the shock path
#
# Unconditional forecasts set future shocks to their training mean. Conditional
# forecasts use the realized path, which is a scenario exercise rather than an
# honest forecast. The gap measures how much of the shocks' value lies in knowing
# their future realizations rather than their current values.

# %%
s4 = V[(V.target == "gdp") & (V["sample"] == "all") & (V.shocks != "none")]
print(s4.groupby(["h", "treatment", "path"]).rmse.median().unstack().round(3).to_string())

# %% [markdown]
# At h=1 the realized path adds little because the lagged shock is already known.
# By h=4 and h=8 the conditional forecasts pull ahead for every treatment. For raw
# shocks at h=8 the median gap is about 0.3 RMSE points of annualized GDP growth,
# which is the ceiling a perfect shock nowcast could buy in this design.

# %% [markdown]
# ## 9. Best GDP model against actuals

# %%
best = V[(V.target == "gdp") & (V.h == 1) & (V.path == "uncond")
         & (V["sample"] == "all")].nsmallest(1, "rmse").iloc[0]
print(best[["endog", "shocks", "treatment", "lag_rule", "p", "rmse", "r2_oos", "dm_p"]].to_string())

endog = best.endog.split("+")
shocks = best.shocks.split("+")
Y = df[endog].to_numpy()
D = df[[c for c in df.columns if c.startswith("d20")]].to_numpy()
train_end_i = df.index.get_loc(pd.Period(vl.TRAIN_END, freq="Q").to_timestamp())
origins = list(range(train_end_i, len(df) - 1))
fc, dates = [], []
for org in origins:
    Xp = vl.purge_shocks(df, endog, shocks, best.treatment, org)
    x_unc = Xp[:org + 1].mean(axis=0)
    fit = vl.fit_varx(Y[:org + 1], Xp[:org + 1], D[:org + 1], int(best.p))
    fc.append(vl.forecast(fit, Y, Xp, D, org, 1, vl.XLAGS, "uncond", x_unc)[0, endog.index("gdp")])
    dates.append(df.index[org + 1])
fc = np.array(fc)
act = df["gdp"].reindex(dates).to_numpy()
dts = pd.DatetimeIndex(dates)
mask = ~((dts >= "2020-01-01") & (dts <= "2021-03-31"))
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
axes[0].plot(dts, act, color=INK, lw=1.1, label="actual")
axes[0].plot(dts, fc, color=RED, lw=1.1, label="h=1 forecast")
axes[0].legend(frameon=False)
axes[0].set_title("Full window")
axes[1].plot(dts[mask], act[mask], color=INK, lw=1.1)
axes[1].plot(dts[mask], fc[mask], color=RED, lw=1.1)
axes[1].set_title("COVID quarters removed")
axes[0].set_ylabel("GDP growth, saar")
plt.show()
print(f"RMSE ex COVID: {np.sqrt(np.mean((fc[mask]-act[mask])**2)):.3f}")

# %% [markdown]
# ## 10. What to take away, and what not to do
#
# 1. The best GDP forecaster at h=1 is the core system plus the term spread with
#    all four raw shocks at p=2, cutting RMSE about 12 percent against the best
#    pure VAR and beating the AR benchmark by roughly a third on out-of-sample R2.
#    Most of the headline gain concentrates in the COVID reversal quarters; the
#    ex-COVID gain is real but modest.
# 2. Almost nothing survives a Diebold-Mariano test against the plain core VAR.
#    About 4 percent of specs clear 5 percent significance, which is what noise
#    alone would deliver. Rank models, but do not oversell the ranking.
# 3. Do not purge shocks because it feels rigorous. Purging costs forecast accuracy
#    for GDP, inflation and the dollar. It is the right tool when the question is
#    structural, and the wrong default when the question is predictive.
# 4. Do not use AIC as the house lag rule. It only pays for the persistent stance
#    variable. BIC or HQIC at p=2 is the robust default on this data.
# 5. Do not read the leaderboard shock combination as stable truth. With 65
#    evaluation points the top ten specs are statistically interchangeable, and
#    the honest summary is that oil plus wealth plus a real activity shock in raw
#    form spans the useful information.
#
# Single highest-value next step: wire the winning spec into a real-time vintage
# exercise with ALFRED data, because the end-of-quarter information set assumed
# here flatters every shock-augmented model.

# %% [markdown]
# ---
# Nicholas Hong | Built for educational and research purposes. Not financial advice.
