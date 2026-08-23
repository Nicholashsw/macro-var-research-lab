"""
varx_density.py

The study scores point forecasts by RMSE, the margin where exogenous shocks are
weakest. This module moves to the conditional distribution. A shock-augmented
VARX absorbs in-sample variance, so its plug-in residual covariance is smaller
and its predictive density is tighter than the pure VAR's even when the mean
barely moves. That tightening helps the density only if it is calibrated. Here
we test it with the log predictive score, the continuous ranked probability
score, and the probability integral transform.

Predictive density: for a Gaussian VAR(p) with residual covariance Sigma, the
h-step iterated forecast is normal with mean from the usual iteration and
covariance sum_{j=0}^{h-1} Psi_j Sigma Psi_j', where Psi_j are the companion-form
MA coefficients. Future exogenous regressors are held at their training mean with
no added variance, which gives the shock models their tightest honest density, so
the comparison again favors shocks. Parameter uncertainty is ignored (plug-in);
adding it would widen every interval and narrow differences.

Arms, core endogenous set, fixed p = 2:
    pure_var    no shocks.
    oil_raw     the disciplined single-oil VARX.
    all4_raw    all four growth-rate shocks, the point-forecast winner.

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
ARMS = {"pure_var": [], "oil_raw": ["oil"], "all4_raw": ["oil", "inv", "wealth", "fiscal"]}


# ---------------------------------------------------------------- predictive covariance

def ma_coeffs(fit, n, h_max):
    p = fit["p"]
    A = fit["B"][1:1 + n * p].T                 # n x n*p = [A_1 ... A_p]
    C = np.zeros((n * p, n * p))
    C[:n, :] = A
    if p > 1:
        C[n:, :-n] = np.eye(n * (p - 1))
    Psis, Cj = [], np.eye(n * p)
    for _ in range(h_max):
        Psis.append(Cj[:n, :n].copy())
        Cj = C @ Cj
    return Psis


def pred_var(fit, n, h_max):
    """Diagonal predictive variances for horizons 1..h_max, shape (h_max, n).
    Endogenous residual uncertainty only (naive plug-in density)."""
    Sig = fit["Sigma"]
    Psis = ma_coeffs(fit, n, h_max)
    S = np.zeros((n, n))
    out = np.empty((h_max, n))
    for j in range(h_max):
        S = S + Psis[j] @ Sig @ Psis[j].T
        out[j] = np.diag(S)
    return out


def exog_var(fit, n, n_x, h_max, Sigma_x):
    """Added predictive variance from not knowing the future shock path, under the
    unconditional path. Each future shock innovation is mean-zero with covariance
    Sigma_x and propagates through the exogenous loadings and the MA structure."""
    p = fit["p"]
    Psis = ma_coeffs(fit, n, h_max)
    off = 1 + n * p
    B0 = fit["B"][off:off + n_x].T                 # n x n_x, contemporaneous
    B1 = fit["B"][off + n_x:off + 2 * n_x].T        # n x n_x, one lag
    out = np.zeros((h_max, n))
    for h in range(1, h_max + 1):
        Sx = np.zeros((n, n))
        for s in range(1, h + 1):                   # future shock at step s
            k = h - s
            L = Psis[k] @ B0
            if k - 1 >= 0:
                L = L + Psis[k - 1] @ B1
            Sx += L @ Sigma_x @ L.T
        out[h - 1] = np.diag(Sx)
    return out


# ---------------------------------------------------------------- scores

def log_score(y, mu, var):
    return stats.norm.logpdf(y, mu, np.sqrt(var))


def crps_gauss(y, mu, var):
    s = np.sqrt(var)
    z = (y - mu) / s
    return s * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


def pit(y, mu, var):
    return stats.norm.cdf((y - mu) / np.sqrt(var))


def hac_mean_p(d, h):
    """Two-sided p-value for E[d]=0 with Newey-West HAC(h-1). Positive mean of d
    (log-score differential) favors the first model."""
    d = d[np.isfinite(d)]
    T = len(d)
    if T < 10:
        return np.nan
    dbar = d.mean()
    g0 = ((d - dbar) ** 2).mean()
    v = g0
    for k in range(1, h):
        gk = ((d[k:] - dbar) * (d[:-k] - dbar)).mean()
        v += 2.0 * (1.0 - k / h) * gk
    if v <= 0:
        v = g0
    t = dbar / np.sqrt(v / T)
    return 2.0 * (1.0 - stats.norm.cdf(abs(t)))


# ---------------------------------------------------------------- walk-forward

def run_density(df):
    D = df[[c for c in df.columns if c.startswith("d20")]].to_numpy()
    train_end_i = df.index.get_loc(pd.Period(vl.TRAIN_END, freq="Q").to_timestamp())
    T = len(df)
    origins = list(range(train_end_i, T - 1))
    Y = df[CORE].to_numpy()
    n = Y.shape[1]

    store = {}
    for arm, shocks in ARMS.items():
        X = df[shocks].to_numpy() if shocks else None
        n_x = X.shape[1] if X is not None else 0
        mean = np.full((len(origins), vl.H_MAX, n), np.nan)
        var = np.full((len(origins), vl.H_MAX, n), np.nan)       # naive
        varf = np.full((len(origins), vl.H_MAX, n), np.nan)      # fair
        for oi, org in enumerate(origins):
            Ttr = org + 1
            Xtr = X[:Ttr] if X is not None else None
            x_unc = Xtr.mean(axis=0) if X is not None else None
            fit = vl.fit_varx(Y[:Ttr], Xtr, D[:Ttr], P)
            mean[oi] = vl.forecast(fit, Y, X, D, org, vl.H_MAX, vl.XLAGS, "uncond", x_unc)
            pv = pred_var(fit, n, vl.H_MAX)
            var[oi] = pv
            if n_x:
                Sigma_x = np.diag(Xtr.var(axis=0))
                varf[oi] = pv + exog_var(fit, n, n_x, vl.H_MAX, Sigma_x)
            else:
                varf[oi] = pv
        store[arm] = (mean, var, varf)
    return store, origins


def collect(store, origins, df, arm, target, h, which):
    j = CORE.index(target)
    mean, var, varf = store[arm]
    v = var if which == "naive" else varf
    y = df[target].to_numpy()
    idx = df.index
    dts, mu, vr, yy = [], [], [], []
    for oi, org in enumerate(origins):
        t = org + h
        if t >= len(idx):
            continue
        dts.append(idx[t]); mu.append(mean[oi, h - 1, j])
        vr.append(v[oi, h - 1, j]); yy.append(y[t])
    return (pd.DatetimeIndex(dts), np.array(mu), np.array(vr), np.array(yy))


def score(store, origins, df):
    rows = []
    for which in ("naive", "fair"):
        for target in TARGETS:
            for h in vl.H_REPORT:
                base = {arm: collect(store, origins, df, arm, target, h, which) for arm in ARMS}
                for arm in ARMS:
                    dts, mu, vr, y = base[arm]
                    ls = log_score(y, mu, vr)
                    cr = crps_gauss(y, mu, vr)
                    pt = pit(y, mu, vr)
                    for sample in ("all", "excovid"):
                        mk = vl._mask(dts, sample)
                        fin = mk & np.isfinite(ls)
                        lsm = float(np.mean(ls[fin]))
                        crm = float(np.mean(cr[fin]))
                        p = pt[fin]
                        cov50 = float(np.mean((p > 0.25) & (p < 0.75)))
                        cov90 = float(np.mean((p > 0.05) & (p < 0.95)))
                        if arm == "pure_var":
                            dp = np.nan
                        else:
                            lv = log_score(base["pure_var"][3], base["pure_var"][1], base["pure_var"][2])
                            dp = hac_mean_p((ls - lv)[fin], h)
                        rows.append({"density": which, "arm": arm, "target": target, "h": h,
                                     "sample": sample, "log_score": round(lsm, 4),
                                     "crps": round(crm, 4), "cov50": round(cov50, 3),
                                     "cov90": round(cov90, 3),
                                     "ls_dm_p_vs_var": None if not np.isfinite(dp) else round(dp, 3),
                                     "n": int(fin.sum())})
    return pd.DataFrame(rows)


def run():
    df = vl.build_panel()
    store, origins = run_density(df)
    tab = score(store, origins, df)
    tab.to_parquet("density_compare.parquet", index=False)

    stats_out = {}
    for which in ("naive", "fair"):
        stats_out[which] = {}
        for target in TARGETS:
            stats_out[which][target] = {}
            for h in vl.H_REPORT:
                stats_out[which][target][str(h)] = {}
                for sample in ("all", "excovid"):
                    sub = tab[(tab.density == which) & (tab.target == target)
                              & (tab.h == h) & (tab["sample"] == sample)]
                    stats_out[which][target][str(h)][sample] = {
                        r.arm: {"log_score": r.log_score, "crps": r.crps, "cov90": r.cov90,
                                "ls_dm_p_vs_var": r.ls_dm_p_vs_var} for r in sub.itertuples()}
    with open("density_stats.json", "w") as f:
        json.dump(stats_out, f, indent=1)

    for which in ("naive", "fair"):
        print(f"\n########## {which.upper()} density ##########")
        for target in TARGETS:
            print(f"\n=== {target}: 90% interval coverage (nominal 0.90) ===")
            piv = tab[(tab.density == which) & (tab.target == target)].pivot_table(
                index=["h", "sample"], columns="arm", values="cov90")
            print(piv[["pure_var", "oil_raw", "all4_raw"]].round(3).to_string())
    # validation: pure VAR identical across densities; fair coverage >= naive for shocks
    pv = tab[tab.arm == "pure_var"]
    ident = np.allclose(pv[pv.density == "naive"].cov90.values,
                        pv[pv.density == "fair"].cov90.values)
    a4 = tab[tab.arm == "all4_raw"]
    improved = (a4[a4.density == "fair"].cov90.values >= a4[a4.density == "naive"].cov90.values - 1e-9).all()
    print(f"\n[check] pure VAR naive==fair: {ident}; all4 fair coverage >= naive everywhere: {improved}")
    return tab


if __name__ == "__main__":
    run()
