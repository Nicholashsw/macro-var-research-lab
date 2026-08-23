"""
sv_univariate.py
Univariate stochastic volatility (Kim, Shephard & Chib 1998) for a single residual
series:  u_t = exp(h_t/2) * eps_t,  eps_t ~ N(0,1),  h_t = mu + phi(h_{t-1}-mu) + eta_t.

Estimated by Gibbs with the KSC 7-component normal mixture for log(eps^2) and FFBS
for the volatility states. Diagnosis on this project showed the COVID variance spike
is concentrated in GDP (≈4x) and absent elsewhere, so SV here should be
VARIABLE-SPECIFIC, not the common factor tried earlier.
"""
import numpy as np
from numpy.linalg import inv

# KSC (1998) 7-component mixture approximating log(chi^2_1) = log(eps^2), eps~N(0,1)
_KSC_Q = np.array([0.00730, 0.10556, 0.00002, 0.04395, 0.34001, 0.24566, 0.25750])
_KSC_M = np.array([-10.12999, -3.97281, -8.56686, 2.77786, 0.61942, 1.79518, -1.08819])
_KSC_V = np.array([5.79596, 2.61369, 5.17950, 0.16735, 0.64009, 0.34023, 1.26261])


def _ffbs_ar1(ystar, m_s, v_s, mu, phi, sig2, rng):
    """FFBS for h_t = mu + phi(h_{t-1}-mu) + N(0,sig2); obs ystar_t = h_t + N(m_s,v_s)."""
    T = len(ystar)
    a = np.empty(T); P = np.empty(T); mflt = np.empty(T); C = np.empty(T)
    # stationary init
    a[0] = mu; P[0] = sig2 / max(1e-6, 1 - phi**2)
    R0 = v_s[0]
    K = P[0] / (P[0] + R0)
    mflt[0] = a[0] + K * (ystar[0] - m_s[0] - a[0]); C[0] = (1 - K) * P[0]
    for t in range(1, T):
        a[t] = mu + phi * (mflt[t-1] - mu); P[t] = phi**2 * C[t-1] + sig2
        R = v_s[t]
        K = P[t] / (P[t] + R)
        mflt[t] = a[t] + K * (ystar[t] - m_s[t] - a[t]); C[t] = (1 - K) * P[t]
    h = np.empty(T)
    h[T-1] = mflt[T-1] + np.sqrt(max(C[T-1], 1e-12)) * rng.standard_normal()
    for t in range(T-2, -1, -1):
        Pn = phi**2 * C[t] + sig2
        J = phi * C[t] / Pn
        mean = mflt[t] + J * (h[t+1] - (mu + phi * (mflt[t] - mu)))
        var = C[t] - J**2 * Pn
        h[t] = mean + np.sqrt(max(var, 1e-12)) * rng.standard_normal()
    return h


def sv_gibbs(u, n_iter=4000, burn=1500, seed=0, phi_fixed=None):
    """Return posterior draws of the log-vol path h (n_keep, T) and params."""
    rng = np.random.default_rng(seed)
    u = np.asarray(u, float)
    T = len(u)
    ystar = np.log(u**2 + 1e-7)                 # offset for zeros
    h = np.full(T, np.log(np.var(u) + 1e-6))
    mu = h[0]; phi = 0.95 if phi_fixed is None else phi_fixed; sig2 = 0.05
    hs, mus, phis, s2s = [], [], [], []
    comp = np.full(T, 4)

    for it in range(n_iter):
        # 1. mixture indicators s_t | h
        for t in range(T):
            r = ystar[t] - h[t]
            logp = np.log(_KSC_Q) - 0.5*np.log(_KSC_V) - 0.5*(r - _KSC_M)**2/_KSC_V
            p = np.exp(logp - logp.max()); p /= p.sum()
            comp[t] = rng.choice(7, p=p)
        m_s = _KSC_M[comp]; v_s = _KSC_V[comp]
        # 2. h | indicators (FFBS)
        h = _ffbs_ar1(ystar, m_s, v_s, mu, phi, sig2, rng)
        # 3. mu | h, phi, sig2  (Gaussian)
        if phi_fixed is None or True:
            denom = (1 - phi)**2 * (T - 1) + (1 - phi**2)
            num = (1 - phi**2) * h[0] + (1 - phi) * np.sum(h[1:] - phi * h[:-1])
            mu_var = sig2 / max(denom, 1e-6); mu_mean = num / max(denom, 1e-6)
            mu = mu_mean + np.sqrt(mu_var) * rng.standard_normal()
        # 4. phi | h, mu, sig2  (random-walk MH, stay stationary)
        if phi_fixed is None:
            z0 = h[:-1] - mu; z1 = h[1:] - mu
            phi_hat = np.sum(z0 * z1) / max(np.sum(z0**2), 1e-6)
            phi_v = sig2 / max(np.sum(z0**2), 1e-6)
            prop = phi_hat + np.sqrt(max(phi_v, 1e-8)) * rng.standard_normal()
            if abs(prop) < 0.999:
                phi = prop
        # 5. sig2 | h, mu, phi  (inverse-gamma)
        resid = (h[1:] - mu) - phi * (h[:-1] - mu)
        a_post = 2.5 + (T - 1) / 2
        b_post = 0.025 + 0.5 * np.sum(resid**2)
        sig2 = 1.0 / rng.gamma(a_post, 1.0 / b_post)
        sig2 = min(max(sig2, 1e-5), 3.0)

        if it >= burn:
            hs.append(h.copy()); mus.append(mu); phis.append(phi); s2s.append(sig2)
    return np.array(hs), np.array(mus), np.array(phis), np.array(s2s)


if __name__ == "__main__":
    import os, pathlib
    import pandas as pd
    from statsmodels.tsa.api import VAR
    # Spreadsheet exports are not redistributed; see DATA.md for the series and layout.
    # Default location is <study>/data/; override with BVAR_DATA_DIR.
    BASE = os.environ.get(
        "BVAR_DATA_DIR",
        str(pathlib.Path(__file__).resolve().parent.parent / "data"),
    )
    m = pd.read_excel(f"{BASE}/main_var.xlsx", sheet_name="Sheet1")
    m["Date"] = pd.to_datetime(m["Date"]); m = m.set_index("Date").sort_index()
    endog = ['gdp_yoy', 'inflation_yoy', 'reer_yoy', '10yTbill_m_avg', 'nir']
    df = m[endog].dropna()
    U = VAR(df).fit(maxlags=4, ic=None).resid

    print("Univariate SV on GDP residuals (KSC mixture sampler)...")
    hs, mus, phis, s2s = sv_gibbs(U['gdp_yoy'].values, n_iter=4000, burn=1500, seed=1)
    h_mean = hs.mean(0)
    vol = np.exp(h_mean / 2)                    # time-varying residual std
    idx = U.index
    svol = pd.Series(vol, index=idx)
    print(f"persistence phi (post mean): {phis.mean():.3f}   vol-of-vol sig2: {s2s.mean():.3f}")
    print("Estimated GDP residual volatility (std), key quarters:")
    for q in ["2007Q1", "2008Q4", "2009Q2", "2015Q1", "2020Q2", "2021Q2", "2024Q4"]:
        t = pd.Period(q, "Q").to_timestamp(how="end")
        nearest = svol.index[np.argmin(np.abs(svol.index - t))]
        print(f"  {q}: vol = {svol.loc[nearest]:.2f}")
    base = svol[svol.index.year.isin([2015, 2016, 2017])].mean()
    gfc = svol[svol.index.to_period('Q').isin(pd.period_range('2008Q3', '2009Q2', freq='Q'))].max()
    cov = svol[svol.index.to_period('Q').isin(pd.period_range('2020Q2', '2021Q2', freq='Q'))].max()
    print(f"\nvol ratio  GFC peak / calm: {gfc/base:.1f}x      COVID peak / calm: {cov/base:.1f}x")
    print("RECOVERED both crisis spikes." if (gfc/base > 1.5 and cov/base > 1.5) else "weak recovery.")
