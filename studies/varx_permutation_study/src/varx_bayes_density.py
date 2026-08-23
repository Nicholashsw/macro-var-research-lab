"""
varx_bayes_density.py

The plug-in Gaussian density leaves even the pure VAR mildly under-covered. Two
things it omits could explain that: parameter uncertainty (the coefficients and
covariance are estimated, not known) and non-Gaussian tails (the true shocks are
fatter than a normal, COVID above all). This module separates them with a 2x2
predictive for the core pure VAR:

    plug-in  + Gaussian    fixed OLS coefficients, normal forward shocks (the
                           density module's baseline).
    plug-in  + empirical   fixed OLS coefficients, forward shocks resampled from
                           the training residuals: adds fat tails, no parameter
                           uncertainty.
    Bayesian + Gaussian    Normal-inverse-Wishart posterior (diffuse prior,
                           centered at OLS) with normal forward shocks: adds
                           parameter uncertainty, no fat tails.
    Bayesian + empirical   both.

Each predictive is a simulated ensemble per origin. Coverage and PIT come from
ensemble ranks; the score is the nonparametric CRPS. If moving from Gaussian to
empirical shocks closes the coverage gap while the posterior barely moves it, the
under-coverage is distributional misspecification, not estimation noise.

Nicholas Hong | Built for educational and research purposes. Not financial advice.
"""

import json
import numpy as np
import pandas as pd
from scipy import stats

import varx_lab as vl

CORE = vl.ENDOG_CORE
P = 2
TARGETS = ["gdp", "inf"]
N_PATHS = 600
SEED = 20260708


def fit_pieces(Y, D, p):
    """OLS fit plus the pieces needed for the NIW posterior. Dummy columns that
    are identically zero on the training window are dropped (unidentified, and
    future dummies are zero regardless), returning the kept-column mask."""
    active = D.any(axis=0)
    Dk = D[:, active]
    Z, Yt, t0 = vl._design(Y, None, Dk, p, vl.XLAGS)
    ZtZ = Z.T @ Z
    ZtZ_inv = np.linalg.inv(ZtZ)
    B = ZtZ_inv @ Z.T @ Yt
    E = Yt - Z @ B
    T_eff, k = Z.shape[0], Z.shape[1]
    Sigma = (E.T @ E) / T_eff
    return {"B": B, "E": E, "Sigma": Sigma, "ZtZ_inv": ZtZ_inv,
            "S": E.T @ E, "T_eff": T_eff, "k": k, "t0": t0, "nd": int(active.sum())}


def simulate(fit, Y, D, origin, h_max, n_paths, method, shocks, rng, n):
    """Ensemble of forward paths for the target vector, shape (n_paths, h_max, n)."""
    p = P
    k = fit["k"]
    hist0 = Y[origin - p + 1:origin + 1].copy()          # (p, n)
    E = fit["E"]
    nd = fit["nd"]
    out = np.empty((n_paths, h_max, n))

    if method == "plugin":
        Bs = [fit["B"]]
        pick = np.zeros(n_paths, int)                    # all share the one B
        chols = [np.linalg.cholesky(fit["Sigma"] + 1e-12 * np.eye(n))]
    else:
        # NIW posterior: Sigma ~ IW(S, T_eff-k); B|Sigma ~ MN(B_hat, Sigma (x) ZtZ_inv)
        nu = fit["T_eff"] - fit["k"]
        Sig_draws = stats.invwishart.rvs(df=nu, scale=fit["S"], size=n_paths, random_state=rng)
        L_ZtZ = np.linalg.cholesky(fit["ZtZ_inv"])
        Bs, chols, pick = [], [], np.arange(n_paths)
        for s in range(n_paths):
            Sig = Sig_draws[s]
            Ls = np.linalg.cholesky(Sig + 1e-12 * np.eye(n))
            W = rng.standard_normal((k, n))
            B_s = fit["B"] + L_ZtZ @ W @ Ls.T
            Bs.append(B_s)
            chols.append(Ls)

    for s in range(n_paths):
        B = Bs[pick[s]] if method == "plugin" else Bs[s]
        Ls = chols[pick[s]] if method == "plugin" else chols[s]
        hist = hist0.copy()
        for h in range(h_max):
            z = [1.0]
            for i in range(1, p + 1):
                z.extend(hist[-i])
            z.extend(np.zeros(nd))                       # future dummies = 0
            mu = np.asarray(z) @ B
            if shocks == "gauss":
                eps = Ls @ rng.standard_normal(n)
            else:
                eps = E[rng.integers(len(E))]
            y = mu + eps
            out[s, h] = y
            hist = np.vstack([hist, y])[-p:]
    return out


def crps_empirical(sample, y):
    s = np.sort(sample)
    m = len(s)
    t1 = np.mean(np.abs(s - y))
    # mean |X - X'| via sorted-sample formula
    t2 = (2.0 / (m * m)) * np.sum((2 * np.arange(1, m + 1) - m - 1) * s)
    return t1 - 0.5 * t2


def run():
    df = vl.build_panel()
    Y = df[CORE].to_numpy()
    n = Y.shape[1]
    D = df[[c for c in df.columns if c.startswith("d20")]].to_numpy()
    train_end_i = df.index.get_loc(pd.Period(vl.TRAIN_END, freq="Q").to_timestamp())
    origins = list(range(train_end_i, len(df) - 1))
    rng = np.random.default_rng(SEED)

    methods = [("plugin", "gauss"), ("plugin", "emp"), ("bayes", "gauss"), ("bayes", "emp")]
    # ensemble[(method,shocks)][oi] = (n_paths, h_max, n)
    ens = {ms: np.full((len(origins), vl.H_MAX, N_PATHS, n), np.nan) for ms in methods}
    for oi, org in enumerate(origins):
        fit = fit_pieces(Y[:org + 1], D[:org + 1], P)
        for method, shocks in methods:
            sim = simulate(fit, Y, D, org, vl.H_MAX, N_PATHS, method, shocks, rng, n)
            ens[(method, shocks)][oi] = np.transpose(sim, (1, 0, 2))  # (h, paths, n)

    rows = []
    for method, shocks in methods:
        arr = ens[(method, shocks)]
        for target in TARGETS:
            j = CORE.index(target)
            y = df[target].to_numpy()
            for h in vl.H_REPORT:
                dts, pit, crps = [], [], []
                for oi, org in enumerate(origins):
                    t = org + h
                    if t >= len(df.index):
                        continue
                    sample = arr[oi, h - 1, :, j]
                    a = y[t]
                    dts.append(df.index[t])
                    pit.append(np.mean(sample < a))
                    crps.append(crps_empirical(sample, a))
                dts = pd.DatetimeIndex(dts)
                pit = np.array(pit); crps = np.array(crps)
                for sample_name in ("all", "excovid"):
                    mk = vl._mask(dts, sample_name)
                    p = pit[mk]
                    rows.append({
                        "method": method, "shocks": shocks, "target": target, "h": h,
                        "sample": sample_name,
                        "cov50": round(float(np.mean((p > 0.25) & (p < 0.75))), 3),
                        "cov90": round(float(np.mean((p > 0.05) & (p < 0.95))), 3),
                        "crps": round(float(np.mean(crps[mk])), 4),
                        "n": int(mk.sum())})
    tab = pd.DataFrame(rows)
    tab.to_parquet("bayes_density_compare.parquet", index=False)

    def cov_table(target, sample):
        sub = tab[(tab.target == target) & (tab["sample"] == sample)].copy()
        sub["cell"] = sub.method + "+" + sub.shocks
        return sub.pivot_table(index="h", columns="cell", values="cov90")[
            ["plugin+gauss", "plugin+emp", "bayes+gauss", "bayes+emp"]]

    for target in TARGETS:
        for sample_name in ("all", "excovid"):
            print(f"\n=== {target} 90% coverage ({sample_name}, nominal 0.90) ===")
            print(cov_table(target, sample_name).round(3).to_string())

    stats_out = {}
    for target in TARGETS:
        stats_out[target] = {}
        for sample_name in ("all", "excovid"):
            stats_out[target][sample_name] = {}
            for h in vl.H_REPORT:
                sub = tab[(tab.target == target) & (tab["sample"] == sample_name) & (tab.h == h)]
                stats_out[target][sample_name][str(h)] = {
                    f"{r.method}+{r.shocks}": {"cov90": r.cov90, "cov50": r.cov50, "crps": r.crps}
                    for r in sub.itertuples()}
    with open("bayes_density_stats.json", "w") as f:
        json.dump(stats_out, f, indent=1)
    return tab


if __name__ == "__main__":
    run()
