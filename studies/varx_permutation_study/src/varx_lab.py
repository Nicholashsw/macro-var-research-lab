"""
varx_lab.py

VARX permutation study on US quarterly macro data.

Pipeline:
    1. Load FRED CSVs and the NY Fed HLW r-star file, build a quarterly panel
       of stationary transforms (1990Q1 onward).
    2. Enumerate specs: endogenous sets (core + up to two add-ons), exogenous
       shock subsets, shock treatments (raw, GDP-purged, fully purged), and
       lag orders selected by AIC, BIC and HQIC on the initial training window.
    3. Expanding-window walk-forward forecasts at h = 1..8 under two exogenous
       paths: unconditional (shocks at training mean) and realized.
    4. Metrics per target: RMSE, MAE, out-of-sample R2 vs an AR benchmark,
       Diebold-Mariano vs the core three-variable VAR.

All purge regressions are re-estimated on each training window only. No
full-sample coefficients touch the evaluation period.

Nicholas Hong | Built for educational and research purposes. Not financial advice.
"""

import itertools
import json
import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

ENDOG_CORE = ["gdp", "inf", "stance"]
ENDOG_ADDONS = ["fx", "wage", "emp", "spread", "permits"]
SHOCKS = ["oil", "inv", "wealth", "fiscal"]
TREATMENTS = ["raw", "gdp_resid", "full_resid"]
LAG_RULES = ["aic", "bic", "hqic"]
P_GRID = [1, 2, 3, 4]
XLAGS = 2               # shock enters with lags 0 and 1
H_MAX = 8
H_REPORT = [1, 4, 8]
TRAIN_END = "2009Q4"    # last quarter of the initial training window
COVID_DUMMIES = ["2020Q2", "2020Q3"]
COVID_EVAL_EXCL = ("2020-01-01", "2021-03-31")  # target dates dropped in ex-COVID metrics
MIN_OBS_PER_PARAM = 3.0

LABELS = {
    "gdp": "Real GDP growth", "inf": "Core PCE inflation", "stance": "Policy stance (real FFR - r*)",
    "fx": "Broad dollar growth", "wage": "AHE wage growth", "emp": "Payrolls growth",
    "spread": "Term spread 10y-3m", "permits": "Building permits growth",
    "oil": "Real oil price growth", "inv": "Real investment growth",
    "wealth": "Real net worth growth", "fiscal": "Real gov spending growth",
}


# ---------------------------------------------------------------- data

def _fred(series, data_dir=DATA_DIR):
    df = pd.read_csv(f"{data_dir}/{series}.csv", parse_dates=[0], index_col=0)
    s = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    s.name = series
    return s


def _to_q(s):
    return s.resample("QS").mean()


def _dlog(s):
    return 400.0 * np.log(s).diff()


def load_hlw(data_dir=DATA_DIR):
    """US one-sided r-star from the NY Fed HLW workbook (column K, rows from 7)."""
    import openpyxl
    wb = openpyxl.load_workbook(f"{data_dir}/hlw.xlsx", read_only=True)
    ws = wb["HLW Estimates"]
    dates, vals = [], []
    for row in ws.iter_rows(min_row=7, values_only=True):
        if row[0] is None:
            continue
        dates.append(pd.Timestamp(row[0]))
        v = row[10]
        vals.append(np.nan if v in (None, "NA") else float(v))
    return pd.Series(vals, index=pd.DatetimeIndex(dates), name="rstar")


def build_panel(data_dir=DATA_DIR):
    """Quarterly panel of stationary transforms plus COVID dummies."""
    gdp = _fred("GDPC1", data_dir)
    pce = _to_q(_fred("PCEPILFE", data_dir))
    ffr = _to_q(_fred("FEDFUNDS", data_dir))
    ahe = _to_q(_fred("AHETPI", data_dir))
    pay = _to_q(_fred("PAYEMS", data_dir))
    gs10 = _to_q(_fred("GS10", data_dir))
    tb3 = _to_q(_fred("TB3MS", data_dir))
    perm = _to_q(_fred("PERMIT", data_dir))
    wti = _to_q(_fred("WTISPLC", data_dir))
    cpi = _to_q(_fred("CPIAUCSL", data_dir))
    inv = _fred("GPDIC1", data_dir)
    nw = _fred("TNWBSHNO", data_dir)
    gov = _fred("GCEC1", data_dir)

    # Broad dollar: splice the discontinued broad index (TWEXBMTH) onto the
    # goods-and-services index (DTWEXBGS) at 2006-01, scaling the old series
    # to the new level so the log-diff has no seam.
    old = _fred("TWEXBMTH", data_dir)
    new = _fred("DTWEXBGS", data_dir).resample("MS").mean()
    cut = new.dropna().index[0]
    factor = new.loc[cut] / old.loc[cut]
    fx_m = pd.concat([old.loc[:cut - pd.offsets.MonthBegin(1)] * factor, new.loc[cut:]])
    fx = _to_q(fx_m)

    rstar = load_hlw(data_dir)
    pi4 = 100.0 * (np.log(pce) - np.log(pce.shift(4)))  # trailing 4q core inflation, deflator for the real rate only

    df = pd.DataFrame({
        "gdp": _dlog(gdp),
        "inf": _dlog(pce),
        "stance": (ffr - pi4) - rstar,
        "fx": _dlog(fx),
        "wage": _dlog(ahe),
        "emp": _dlog(pay),
        "spread": gs10 - tb3,
        "permits": _dlog(perm),
        "oil": _dlog(wti / cpi),
        "inv": _dlog(inv),
        "wealth": _dlog(nw / cpi),
        "fiscal": _dlog(gov),
    })
    df = df.loc["1990-01-01":].dropna()
    for q in COVID_DUMMIES:
        ts = pd.Period(q, freq="Q").to_timestamp()
        df[f"d{q}"] = (df.index == ts).astype(float)
    return df


# ---------------------------------------------------------------- specs

def enumerate_specs():
    """Every (endog set, shock subset, treatment) cell before lag selection."""
    endog_sets = []
    for k in (0, 1, 2):
        for combo in itertools.combinations(ENDOG_ADDONS, k):
            endog_sets.append(ENDOG_CORE + list(combo))
    shock_sets = [[]]
    shock_sets += [[s] for s in SHOCKS]
    shock_sets += [list(c) for c in itertools.combinations(SHOCKS, 2)]
    shock_sets += [list(SHOCKS)]
    cells = []
    for ev in endog_sets:
        for sv in shock_sets:
            treats = ["none"] if not sv else TREATMENTS
            for tr in treats:
                cells.append({"endog": ev, "shocks": sv, "treatment": tr})
    return cells


# ---------------------------------------------------------------- purge

def purge_shocks(df, endog, shocks, treatment, train_end_i):
    """Return the shock matrix under a treatment, coefficients fit on the
    training window only and applied to the full sample.

    raw        : the series as is
    gdp_resid  : residual from x_t on [1, gdp_t]
    full_resid : residual from x_t on [1, Y_t, Y_{t-1}] for the spec's endog set
    """
    if not shocks:
        return None
    X = df[shocks].to_numpy()
    if treatment == "raw":
        return X
    if treatment == "gdp_resid":
        Z = np.column_stack([np.ones(len(df)), df["gdp"].to_numpy()])
        fit_rows = slice(0, train_end_i + 1)
    else:
        Y = df[endog].to_numpy()
        Ylag = np.vstack([np.full((1, Y.shape[1]), np.nan), Y[:-1]])
        Z = np.column_stack([np.ones(len(df)), Y, Ylag])
        fit_rows = slice(1, train_end_i + 1)
    beta, *_ = np.linalg.lstsq(Z[fit_rows], X[fit_rows], rcond=None)
    out = X - Z @ beta
    if treatment == "full_resid":
        out[0] = X[0] - X[fit_rows].mean(axis=0)  # first row has no lag; fall back to demeaning
    return out


# ---------------------------------------------------------------- estimation

def _design(Y, X, D, p, xlags):
    """Stack the VARX regression: rows t = max(p, xlags-1) .. T-1."""
    T, n = Y.shape
    t0 = max(p, xlags - 1)
    rows = T - t0
    blocks = [np.ones((rows, 1))]
    for i in range(1, p + 1):
        blocks.append(Y[t0 - i:T - i])
    if X is not None:
        for j in range(xlags):
            blocks.append(X[t0 - j:T - j])
    if D is not None:
        blocks.append(D[t0:])
    return np.column_stack(blocks), Y[t0:], t0


def fit_varx(Y, X, D, p, xlags=XLAGS):
    Z, Yt, t0 = _design(Y, X, D, p, xlags)
    B, *_ = np.linalg.lstsq(Z, Yt, rcond=None)
    E = Yt - Z @ B
    T_eff = Z.shape[0]
    Sigma = (E.T @ E) / T_eff
    return {"B": B, "Sigma": Sigma, "E": E, "T_eff": T_eff, "k": Z.shape[1], "p": p, "t0": t0}


def info_criteria(fit, n):
    T, k = fit["T_eff"], fit["k"]
    sign, logdet = np.linalg.slogdet(fit["Sigma"])
    ld = logdet if sign > 0 else np.inf
    m = n * k
    return {"aic": ld + 2.0 * m / T,
            "bic": ld + np.log(T) * m / T,
            "hqic": ld + 2.0 * np.log(np.log(T)) * m / T}


def select_lags(Y, X, D, n, T_train):
    """AIC, BIC, HQIC choices over P_GRID with a degrees-of-freedom guard."""
    ics = {}
    for p in P_GRID:
        k = 1 + n * p + (X.shape[1] * XLAGS if X is not None else 0) + (D.shape[1] if D is not None else 0)
        if (T_train - max(p, XLAGS - 1)) / k < MIN_OBS_PER_PARAM:
            continue
        fit = fit_varx(Y[:T_train], X[:T_train] if X is not None else None,
                       D[:T_train] if D is not None else None, p)
        ics[p] = info_criteria(fit, n)
    if not ics:
        return {r: 1 for r in LAG_RULES}
    return {r: min(ics, key=lambda p: ics[p][r]) for r in LAG_RULES}


def forecast(fit, Y, X, D, origin, h_max, xlags, x_path, x_uncond):
    """Iterated forecasts from `origin` (last in-sample row index).

    x_path 'uncond': future shocks at their training mean.
    x_path 'realized': future shocks at realized values.
    Future dummies are zero.
    """
    n = Y.shape[1]
    p = fit["p"]
    B = fit["B"]
    hist = Y[origin - p + 1:origin + 1].copy()
    out = np.empty((h_max, n))
    for h in range(1, h_max + 1):
        t = origin + h
        z = [1.0]
        for i in range(1, p + 1):
            z.extend(hist[-i])
        if X is not None:
            for j in range(xlags):
                tj = t - j
                if tj <= origin:
                    z.extend(X[tj])
                elif x_path == "realized" and tj < len(X):
                    z.extend(X[tj])
                else:
                    z.extend(x_uncond)
        if D is not None:
            z.extend(np.zeros(D.shape[1]))
        yhat = np.asarray(z) @ B
        out[h - 1] = yhat
        hist = np.vstack([hist, yhat])
    return out


# ---------------------------------------------------------------- diagnostics

def stability(fit, n):
    p = fit["p"]
    A = fit["B"][1:1 + n * p].T
    comp = np.zeros((n * p, n * p))
    comp[:n, :] = A
    if p > 1:
        comp[n:, :-n] = np.eye(n * (p - 1))
    return np.abs(np.linalg.eigvals(comp)).max()


def ljung_box_p(e, lags=8):
    """Portmanteau p-value for one residual series."""
    from scipy import stats
    e = e - e.mean()
    T = len(e)
    denom = e @ e
    q = 0.0
    for k in range(1, lags + 1):
        r = (e[k:] @ e[:-k]) / denom
        q += r * r / (T - k)
    q *= T * (T + 2)
    return 1.0 - stats.chi2.cdf(q, lags)


def dm_test(e1, e2, h):
    """Diebold-Mariano on squared errors, HAC with h-1 lags. Negative stat
    favors model 1."""
    from scipy import stats
    d = e1 ** 2 - e2 ** 2
    d = d[np.isfinite(d)]
    T = len(d)
    if T < 10:
        return np.nan, np.nan
    dbar = d.mean()
    g0 = ((d - dbar) ** 2).mean()
    v = g0
    for k in range(1, h):
        gk = ((d[k:] - dbar) * (d[:-k] - dbar)).mean()
        v += 2.0 * (1.0 - k / h) * gk
    if v <= 0:
        v = g0
    stat = dbar / np.sqrt(v / T)
    return stat, 2.0 * (1.0 - stats.norm.cdf(abs(stat)))


# ---------------------------------------------------------------- study

def run_study(df, verbose=True):
    idx = df.index
    D_all = df[[c for c in df.columns if c.startswith("d20")]].to_numpy()
    train_end_i = idx.get_loc(pd.Period(TRAIN_END, freq="Q").to_timestamp())
    T = len(df)
    origins = list(range(train_end_i, T - 1))

    cells = enumerate_specs()
    specs, models, forecasts = [], {}, {}
    for ci, cell in enumerate(cells):
        endog, shocks, treat = cell["endog"], cell["shocks"], cell["treatment"]
        Y = df[endog].to_numpy()
        n = Y.shape[1]
        Xp = purge_shocks(df, endog, shocks, treat, train_end_i)
        p_by_rule = select_lags(Y, Xp, D_all, n, train_end_i + 1)
        for rule in LAG_RULES:
            p = p_by_rule[rule]
            key = (tuple(endog), tuple(shocks), treat, p)
            specs.append({"spec_id": len(specs), "endog": endog, "shocks": shocks,
                          "treatment": treat, "lag_rule": rule, "p": p, "model_key": key})
        if verbose and ci % 100 == 0:
            print(f"lag selection {ci}/{len(cells)}")

    unique_keys = sorted({s["model_key"] for s in specs})
    if verbose:
        print(f"{len(cells)} cells, {len(specs)} spec rows, {len(unique_keys)} unique models")

    for mi, key in enumerate(unique_keys):
        endog, shocks, treat, p = list(key[0]), list(key[1]), key[2], key[3]
        Y = df[endog].to_numpy()
        n = Y.shape[1]
        fc = {path: np.full((len(origins), H_MAX, n), np.nan)
              for path in (("uncond", "realized") if shocks else ("uncond",))}
        diag = {}
        for oi, org in enumerate(origins):
            Ttr = org + 1
            Xp = purge_shocks(df, endog, shocks, treat, org) if shocks else None
            x_unc = Xp[:Ttr].mean(axis=0) if shocks else None
            fit = fit_varx(Y[:Ttr], Xp[:Ttr] if shocks else None, D_all[:Ttr], p)
            if oi == 0:
                diag["stab"] = float(stability(fit, n))
                diag["lb_min_p"] = float(min(ljung_box_p(fit["E"][:, j]) for j in range(n)))
                diag["dof_ratio"] = float(fit["T_eff"] / fit["k"])
            for path in fc:
                fc[path][oi] = forecast(fit, Y, Xp, D_all, org, H_MAX, XLAGS, path, x_unc)
        models[key] = {"model_id": mi, "endog": endog, "shocks": shocks,
                       "treatment": treat, "p": p, **diag}
        forecasts[key] = fc
        if verbose and mi % 50 == 0:
            print(f"walk-forward {mi}/{len(unique_keys)}")

    return {"specs": specs, "models": models, "forecasts": forecasts,
            "origins": origins, "index": idx}


# ---------------------------------------------------------------- benchmarks

def ar_benchmark(df, targets, train_end_i, origins):
    """AR(p by BIC on the initial window) iterated forecasts per target."""
    out = {}
    for v in targets:
        y = df[v].to_numpy().reshape(-1, 1)
        D = df[[c for c in df.columns if c.startswith("d20")]].to_numpy()
        p_by = select_lags(y, None, D, 1, train_end_i + 1)
        p = p_by["bic"]
        fc = np.full((len(origins), H_MAX), np.nan)
        for oi, org in enumerate(origins):
            fit = fit_varx(y[:org + 1], None, D[:org + 1], p)
            fc[oi] = forecast(fit, y, None, D, org, H_MAX, XLAGS, "uncond", None)[:, 0]
        out[v] = {"fc": fc, "p": p}
    return out


# ---------------------------------------------------------------- metrics

def _errors(fc, df, endog, origins, target, h):
    """Forecast errors for one target at one horizon, aligned to target dates."""
    j = endog.index(target)
    idx = df.index
    y = df[target].to_numpy()
    dates, e = [], []
    for oi, org in enumerate(origins):
        t = org + h
        if t >= len(idx):
            continue
        dates.append(idx[t])
        e.append(fc[oi, h - 1, j] - y[t])
    return pd.Series(e, index=pd.DatetimeIndex(dates))


def _mask(dates, sample):
    if sample == "all":
        return np.ones(len(dates), bool)
    lo, hi = pd.Timestamp(COVID_EVAL_EXCL[0]), pd.Timestamp(COVID_EVAL_EXCL[1])
    return ~((dates >= lo) & (dates <= hi))


def build_results(study, df, bench, targets=("gdp", "inf", "stance", "fx")):
    """Long results table: one row per spec x horizon x path x sample x target."""
    idx = df.index
    origins = study["origins"]
    ar_err = {}
    for v in bench:
        y = df[v].to_numpy()
        for h in H_REPORT:
            dates, e = [], []
            for oi, org in enumerate(origins):
                t = org + h
                if t >= len(idx):
                    continue
                dates.append(idx[t])
                e.append(bench[v]["fc"][oi, h - 1] - y[t])
            ar_err[(v, h)] = pd.Series(e, index=pd.DatetimeIndex(dates))

    core_key = {}
    for s in study["specs"]:
        if s["endog"] == ENDOG_CORE and not s["shocks"]:
            core_key[s["lag_rule"]] = s["model_key"]
    core_err = {}
    for rule, key in core_key.items():
        fc = study["forecasts"][key]["uncond"]
        for v in targets[:3]:
            for h in H_REPORT:
                core_err[(rule, v, h)] = _errors(fc, df, ENDOG_CORE, origins, v, h)

    rows = []
    for s in study["specs"]:
        key = s["model_key"]
        m = study["models"][key]
        fc_all = study["forecasts"][key]
        for path, fc in fc_all.items():
            for v in targets:
                if v not in s["endog"]:
                    continue
                for h in H_REPORT:
                    e = _errors(fc, df, s["endog"], origins, v, h)
                    ea = ar_err[(v, h)].reindex(e.index)
                    ec = core_err.get((s["lag_rule"], v, h))
                    for sample in ("all", "excovid"):
                        mk = _mask(e.index, sample)
                        ee = e[mk].to_numpy()
                        rmse = float(np.sqrt(np.mean(ee ** 2)))
                        mae = float(np.mean(np.abs(ee)))
                        r2 = float(1.0 - np.mean(ee ** 2) / np.mean(ea[mk].to_numpy() ** 2))
                        if ec is not None and v in ENDOG_CORE:
                            ecc = ec.reindex(e.index)[mk].to_numpy()
                            dm, dmp = dm_test(ee, ecc, h)
                        else:
                            dm, dmp = np.nan, np.nan
                        rows.append({
                            "spec_id": s["spec_id"], "model_id": m["model_id"],
                            "endog": "+".join(s["endog"]), "addons": "+".join(s["endog"][3:]) or "none",
                            "shocks": "+".join(s["shocks"]) or "none",
                            "treatment": s["treatment"], "lag_rule": s["lag_rule"], "p": s["p"],
                            "path": path, "target": v, "h": h, "sample": sample,
                            "rmse": rmse, "mae": mae, "r2_oos": r2,
                            "dm_stat": dm, "dm_p": dmp,
                            "stab": m["stab"], "lb_min_p": m["lb_min_p"], "dof_ratio": m["dof_ratio"],
                        })
    return pd.DataFrame(rows)


def validity(res, stab_max=1.0, lb_min=0.01, dof_min=3.0):
    return (res["stab"] < stab_max) & (res["lb_min_p"] > lb_min) & (res["dof_ratio"] >= dof_min)
