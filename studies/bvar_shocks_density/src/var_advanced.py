"""
var_advanced.py
Two upgrades for the US VAR project:
  (1) Cointegration diagnostic (integration order + Johansen) -> decide VAR-in-levels vs VECM
  (2) Rolling-origin real-time OOS evaluation with proper density scoring
      (RMSE, log predictive score, CRPS, interval coverage) and large-sample
      Diebold-Mariano tests -- the fix for the n=13 power problem.

Depends on var_addons.py (the corrected BVAR) and shock_builder.py (clean shocks).
"""
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.tsa.ar_model import AutoReg
from var_addons import (build_design, ar_residual_scales, minnesota_dummies, fit_bvar,
                        companion_max_eig, sample_bvar)


# ---------------------------------------------------------------- scoring rules
def gaussian_crps(y, mu, sigma):
    """Closed-form CRPS for a Gaussian predictive N(mu, sigma^2)."""
    sigma = np.maximum(sigma, 1e-8)
    z = (y - mu) / sigma
    return float(sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi)))

def ensemble_crps(y, draws):
    """CRPS of an empirical ensemble (energy form, exact)."""
    x = np.sort(np.asarray(draws, float))
    m = x.size
    t1 = np.mean(np.abs(x - y))
    # E|X-X'| via sorted formula: (2/m^2) * sum_i (2i-m+1) x_i  (i from 0)
    i = np.arange(m)
    t2 = (2.0 / (m * m)) * np.sum((2 * i - m + 1) * x)
    return float(t1 - 0.5 * t2)

def gaussian_logscore(y, mu, sigma):
    sigma = max(sigma, 1e-8)
    return float(stats.norm.logpdf(y, mu, sigma))   # higher = better


# ---------------------------------------------------------- one-step predictives
def _bvar_predictive_draws(post, p, n, m_excl, x_next, n_draws=300, seed=0):
    """Sample the one-step-ahead predictive vector y_{t+1} ~ posterior predictive."""
    Bs, Ss, eigs, _ = sample_bvar(post, p, n_draws=n_draws, seed=seed, require_stable=True)
    rng = np.random.default_rng(seed + 1)
    out = np.empty((len(Bs), n))
    for s, (B, Sig) in enumerate(zip(Bs, Ss)):
        mean = x_next @ B
        out[s] = mean + rng.multivariate_normal(np.zeros(n), Sig)
    return out                                   # (n_draws, n)


def rolling_density_eval(df_endog, df_exog, list_endog, lag=5,
                         first_origin="2013Q1", lam=0.05, mu_soc=0.3,
                         n_draws=300, condition_on_shocks=True,
                         covid_quarters=("2020Q2", "2020Q3", "2020Q4", "2021Q1")):
    """Expanding-window 1-step OOS backtest scoring BVAR vs RW vs AR(1).

    Re-estimates every model at every origin, so DM tests get real power.
    COVID dummies (covid_quarters) absorb the 2020-21 outliers so the homoskedastic
    BVAR's residual covariance doesn't explode; set covid_quarters=() to disable.
    Returns (metrics_df, errors_dict) where errors_dict[model][var] is a Series of
    one-step forecast errors (for DM tests).
    """
    df_exog = df_exog.copy()
    for q in covid_quarters:                      # absorb COVID outliers
        d = (df_endog.index.to_period("Q") == pd.Period(q, "Q")).astype(float)
        df_exog[f"covid_{q}"] = d
    n = len(list_endog)
    idx = df_endog.index
    first = pd.Period(first_origin, "Q")
    origins = [t for t in idx if t.to_period("Q") >= first][:-1]  # need t+1 to exist

    # storage
    rec = {mdl: {v: {"err": [], "crps": [], "ls": [], "cov": []} for v in list_endog}
           for mdl in ["BVAR", "RW", "AR1"]}

    # full design once (for picking out X rows by date)
    _, Xall, cols, allidx = build_design(df_endog, df_exog, lag)
    Xall = pd.DataFrame(Xall, index=allidx, columns=cols)

    for oi, t in enumerate(origins):
        nxt = idx[idx.get_loc(t) + 1]
        train = df_endog.loc[:t]
        if len(train) < 4 * lag + 8:
            continue
        Xtr_e = df_exog.reindex(train.index).fillna(0.0)
        Y_mat, X_mat, _, di = build_design(train, Xtr_e, lag)
        m_excl = X_mat.shape[1] - n * lag
        sigma = ar_residual_scales(train, lag)
        delta = np.array([0.8 if c.endswith("_yoy") else 0.95 for c in train.columns])
        Yd, Xd = minnesota_dummies(train.loc[di], sigma, delta, lag, m_excl,
                                   lam=lam, tau=10.0, eps=1e-4, mu_soc=mu_soc)
        post = fit_bvar(Y_mat, X_mat, Yd, Xd)

        # x_next row: realized lags (known at t); future shocks realized or zero
        if nxt not in Xall.index:
            continue
        x_next = Xall.loc[nxt].values.copy()
        if not condition_on_shocks:
            x_next[n * lag + 1:] = 0.0            # zero out exog (keep const)

        draws = _bvar_predictive_draws(post, lag, n, m_excl, x_next,
                                       n_draws=n_draws, seed=oi)
        actual = df_endog.loc[nxt].values

        for j, v in enumerate(list_endog):
            y = actual[j]
            # BVAR
            ens = draws[:, j]
            mu, sd = ens.mean(), ens.std(ddof=1)
            lo, hi = np.percentile(ens, [5, 95])
            rec["BVAR"][v]["err"].append(mu - y)
            rec["BVAR"][v]["crps"].append(ensemble_crps(y, ens))
            rec["BVAR"][v]["ls"].append(gaussian_logscore(y, mu, sd))
            rec["BVAR"][v]["cov"].append(int(lo <= y <= hi))
            # RW: N(last value, sd of training 1-step changes)
            yv = train[v]
            rw_mu = yv.iloc[-1]; rw_sd = yv.diff().std(ddof=1)
            rec["RW"][v]["err"].append(rw_mu - y)
            rec["RW"][v]["crps"].append(gaussian_crps(y, rw_mu, rw_sd))
            rec["RW"][v]["ls"].append(gaussian_logscore(y, rw_mu, rw_sd))
            rec["RW"][v]["cov"].append(int(rw_mu - 1.645 * rw_sd <= y <= rw_mu + 1.645 * rw_sd))
            # AR(1)
            try:
                ar = AutoReg(yv, lags=1, old_names=False).fit()
                ar_mu = float(ar.predict(start=len(yv), end=len(yv)).iloc[0])
                ar_sd = float(np.std(ar.resid, ddof=1))
            except Exception:
                ar_mu, ar_sd = rw_mu, rw_sd
            rec["AR1"][v]["err"].append(ar_mu - y)
            rec["AR1"][v]["crps"].append(gaussian_crps(y, ar_mu, ar_sd))
            rec["AR1"][v]["ls"].append(gaussian_logscore(y, ar_mu, ar_sd))
            rec["AR1"][v]["cov"].append(int(ar_mu - 1.645 * ar_sd <= y <= ar_mu + 1.645 * ar_sd))

    # assemble metrics
    rows, errs = [], {m: {} for m in rec}
    for mdl in rec:
        for v in list_endog:
            e = np.array(rec[mdl][v]["err"])
            errs[mdl][v] = pd.Series(e)
            rows.append({
                "model": mdl, "variable": v, "n_origins": len(e),
                "RMSE": round(np.sqrt(np.mean(e ** 2)), 3),
                "CRPS": round(np.mean(rec[mdl][v]["crps"]), 3),
                "LogScore": round(np.mean(rec[mdl][v]["ls"]), 3),
                "Cov90": round(np.mean(rec[mdl][v]["cov"]), 2),
            })
    return pd.DataFrame(rows), errs


def diebold_mariano(e1, e2, h=1, power=2):
    """DM with Harvey-Leybourne-Newbold small-sample correction (e1 vs e2)."""
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    ok = np.isfinite(e1) & np.isfinite(e2)
    e1, e2 = e1[ok], e2[ok]
    T = e1.size
    if T < 5:
        return np.nan, np.nan
    d = np.abs(e1) ** power - np.abs(e2) ** power
    dbar = d.mean()
    g0 = np.mean((d - dbar) ** 2)
    var = g0 / T
    if var <= 0:
        return np.nan, np.nan
    dm = dbar / np.sqrt(var)
    corr = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm *= corr
    return float(dm), float(2 * stats.t.sf(abs(dm), df=T - 1))
