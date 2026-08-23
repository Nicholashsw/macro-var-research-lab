"""
eval_multi.py
Multi-horizon (h = 1,2,4,8) expanding-window density evaluation with an expanded
benchmark set:  Random Walk, AR(1), unrestricted VAR (OLS), no-shock BVAR, and the
full shock-augmented BVAR. Reuses var_addons (corrected BVAR) end to end.

Scores per (model, variable, horizon): RMSE, CRPS, log score.
Multi-step forecasts are ex-ante (future exogenous shocks set to 0), which is the
honest forecasting setup -- constructed shocks known only contemporaneously cannot
aid multi-step forecasts unless the shocks themselves are forecast.
"""
import numpy as np, pandas as pd
from numpy.linalg import inv, pinv
from scipy import stats
from scipy.special import polygamma
from statsmodels.tsa.ar_model import AutoReg
from var_addons import (build_design, ar_residual_scales, minnesota_dummies, fit_bvar,
                        companion_max_eig, sample_bvar)


def _gauss_crps(y, mu, sd):
    sd = max(sd, 1e-8); z = (y - mu) / sd
    return float(sd * (z * (2*stats.norm.cdf(z) - 1) + 2*stats.norm.pdf(z) - 1/np.sqrt(np.pi)))

def _ens_crps(y, x):
    x = np.sort(np.asarray(x, float)); m = x.size
    t1 = np.mean(np.abs(x - y))
    i = np.arange(m); t2 = (2.0/(m*m)) * np.sum((2*i - m + 1) * x)
    return float(t1 - 0.5*t2)

def _ls(y, mu, sd):
    return float(stats.norm.logpdf(y, mu, max(sd, 1e-8)))


def _bvar_hstep_ensemble(Bs, Ss, y_init, p, n, m_excl, H, rng):
    """h-step predictive ensembles (ex-ante, future exog=0) for h=1..H.
    Returns dict h -> (n_draws, n) array."""
    out = {h: [] for h in range(1, H+1)}
    for B, Sig in zip(Bs, Ss):
        cholS = np.linalg.cholesky((Sig + Sig.T)/2)
        win = y_init.copy()
        for h in range(1, H+1):
            x = np.hstack([win[::-1].flatten(), 1.0, np.zeros(m_excl-1)])
            yt = x @ B + cholS @ rng.standard_normal(n)
            out[h].append(yt.copy())
            win = np.vstack([win[1:], yt])
    return {h: np.array(v) for h, v in out.items()}


def _fit_bvar_block(train, exog, lag, n, lam, mu_soc):
    Xe = exog.reindex(train.index).fillna(0.0)
    Y, X, cols, di = build_design(train, Xe, lag)
    m_excl = X.shape[1] - n*lag
    sigma = ar_residual_scales(train, lag)
    delta = np.array([0.8 if str(c).endswith('_yoy') or 'gdp' in str(c) or 'inf' in str(c)
                      else 0.95 for c in train.columns])
    Yd, Xd = minnesota_dummies(train.loc[di], sigma, delta, lag, m_excl, lam=lam, mu_soc=mu_soc)
    post = fit_bvar(Y, X, Yd, Xd)
    return post, m_excl


def rolling_eval_multi(df_endog, df_exog, list_endog, horizons=(1,2,4,8), lag=5,
                       first_origin="2013Q1", lam=0.05, mu_soc=0.3, n_draws=250,
                       covid_quarters=("2020Q2","2020Q3","2020Q4","2021Q1"), seed=0):
    n = len(list_endog); H = max(horizons)
    dfx = df_exog.copy()
    for q in covid_quarters:
        dfx[f"covid_{q}"] = (df_endog.index.to_period("Q") == pd.Period(q,"Q")).astype(float)
    empty_exog = pd.DataFrame(index=df_endog.index)            # for no-shock BVAR (const only)
    idx = df_endog.index
    origins = [t for t in idx if t.to_period("Q") >= pd.Period(first_origin,"Q")]
    origins = [t for t in origins if idx.get_loc(t) + H < len(idx)]
    rng = np.random.default_rng(seed)

    rec = {(mdl,v,h): {"e":[], "crps":[], "ls":[]} for mdl in
           ["RW","AR1","VAR_OLS","BVAR_noshock","BVAR"] for v in list_endog for h in horizons}

    for oi, t in enumerate(origins):
        train = df_endog.loc[:t]
        if len(train) < 4*lag + 8: continue
        y_init = train.values[-lag:]
        # --- BVAR (full) and no-shock BVAR ---
        post, m_excl = _fit_bvar_block(train, dfx, lag, n, lam, mu_soc)
        Bs, Ss, eigs, _ = sample_bvar(post, lag, n_draws=n_draws, seed=oi, require_stable=True)
        ens = _bvar_hstep_ensemble(Bs, Ss, y_init, lag, n, m_excl, H, rng)
        post0, m0 = _fit_bvar_block(train, empty_exog, lag, n, lam, mu_soc)
        Bs0, Ss0, _, _ = sample_bvar(post0, lag, n_draws=n_draws, seed=oi+999, require_stable=True)
        ens0 = _bvar_hstep_ensemble(Bs0, Ss0, y_init, lag, n, m0, H, rng)
        # --- unrestricted VAR (OLS) point+Gaussian ---
        Xe = empty_exog.reindex(train.index)
        Yo, Xo, _, _ = build_design(train, Xe, lag)
        Bo = pinv(Xo) @ Yo; resid = Yo - Xo @ Bo; Sigo = resid.T@resid/(Yo.shape[0]-Xo.shape[1])
        sdo = np.sqrt(np.maximum(np.diag(Sigo), 1e-10))
        # iterate OLS forward
        win = y_init.copy(); ols_fc = {}
        for h in range(1, H+1):
            x = np.hstack([win[::-1].flatten(), 1.0]); yt = x @ Bo
            ols_fc[h] = yt.copy(); win = np.vstack([win[1:], yt])

        for h in horizons:
            nxt = idx[idx.get_loc(t) + h]
            actual = df_endog.loc[nxt].values
            for j, v in enumerate(list_endog):
                y = actual[j]
                # BVAR full
                e = ens[h][:, j]; mu, sd = e.mean(), e.std(ddof=1)
                rec[("BVAR",v,h)]["e"].append(mu - y); rec[("BVAR",v,h)]["crps"].append(_ens_crps(y,e)); rec[("BVAR",v,h)]["ls"].append(_ls(y,mu,sd))
                # BVAR no-shock
                e0 = ens0[h][:, j]; mu0, sd0 = e0.mean(), e0.std(ddof=1)
                rec[("BVAR_noshock",v,h)]["e"].append(mu0 - y); rec[("BVAR_noshock",v,h)]["crps"].append(_ens_crps(y,e0)); rec[("BVAR_noshock",v,h)]["ls"].append(_ls(y,mu0,sd0))
                # VAR OLS (Gaussian, var grows ~h)
                muo = ols_fc[h][j]; sdh = sdo[j]*np.sqrt(h)
                rec[("VAR_OLS",v,h)]["e"].append(muo - y); rec[("VAR_OLS",v,h)]["crps"].append(_gauss_crps(y,muo,sdh)); rec[("VAR_OLS",v,h)]["ls"].append(_ls(y,muo,sdh))
                # RW
                yv = train[v]; rwm = yv.iloc[-1]; rws = yv.diff().std(ddof=1)*np.sqrt(h)
                rec[("RW",v,h)]["e"].append(rwm - y); rec[("RW",v,h)]["crps"].append(_gauss_crps(y,rwm,rws)); rec[("RW",v,h)]["ls"].append(_ls(y,rwm,rws))
                # AR(1) h-step
                try:
                    ar = AutoReg(yv, lags=1, old_names=False).fit()
                    c0, phi = ar.params.iloc[0], ar.params.iloc[1]; s1 = np.std(ar.resid, ddof=1)
                    mh = yv.iloc[-1]
                    for _ in range(h): mh = c0 + phi*mh
                    sh = s1*np.sqrt(np.sum([phi**(2*k) for k in range(h)]))
                except Exception:
                    mh, sh = rwm, rws
                rec[("AR1",v,h)]["e"].append(mh - y); rec[("AR1",v,h)]["crps"].append(_gauss_crps(y,mh,sh)); rec[("AR1",v,h)]["ls"].append(_ls(y,mh,sh))

    rows, errs = [], {}
    for (mdl,v,h), d in rec.items():
        e = np.array(d["e"]); errs[(mdl,v,h)] = e
        if len(e):
            rows.append({"model":mdl,"variable":v,"h":h,"n":len(e),
                         "RMSE":round(np.sqrt(np.mean(e**2)),3),
                         "CRPS":round(np.mean(d["crps"]),3),
                         "LogScore":round(np.mean(d["ls"]),3)})
    return pd.DataFrame(rows), errs


def dm_test(e1, e2, h=1, power=2):
    e1, e2 = np.asarray(e1,float), np.asarray(e2,float)
    ok = np.isfinite(e1)&np.isfinite(e2); e1,e2 = e1[ok],e2[ok]; T=e1.size
    if T < 5: return np.nan, np.nan
    d = np.abs(e1)**power - np.abs(e2)**power; db = d.mean()
    var = np.mean((d-db)**2)/T
    if var <= 0: return np.nan, np.nan
    dm = db/np.sqrt(var); corr = np.sqrt((T+1-2*h+h*(h-1)/T)/T); dm*=corr
    return float(dm), float(2*stats.t.sf(abs(dm), df=T-1))
