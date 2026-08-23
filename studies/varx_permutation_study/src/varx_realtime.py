"""
varx_realtime.py

Real-time (ALFRED vintage) replication of the VARX study.

The main study forecasts on final-vintage data with an end-of-quarter
information set. This module re-runs the core comparison as a forecaster would
have faced it: at each origin the panel is rebuilt from the data vintage that
was current roughly 45 days after the quarter closed, i.e. just after the
advance GDP print. That vintage carries the publication lags (a ragged edge
where net worth trails the NIPA block by a quarter) and the pre-revision values
of every national-accounts series.

Scope. Revisions are concentrated in the NIPA and Financial Accounts block, so
GDP, core PCE, real investment, real government spending and household net worth
are pulled at their contemporaneous vintages through the keyless alfredgraph
endpoint. Series that are not meaningfully revised (fed funds, oil, CPI) are
held at final values, and the HLW natural rate is held at its final one-sided
estimate because no keyless real-time vintage exists for it. The exercise
therefore isolates the effect of national-accounts data revisions and
publication timing on the shock-augmentation advantage, holding the trend
components fixed. Forecasts are scored against final-vintage actuals so the
numbers line up directly with the final-data leaderboard.

Nicholas Hong | Built for educational and research purposes. Not financial advice.
"""

import os
import subprocess
import datetime as dt
import numpy as np
import pandas as pd

import varx_lab as vl

VLAG_DAYS = 45                       # forecast standpoint: advance GDP just released
VDIR = os.path.join(vl.DATA_DIR, "vintages")
REVISED = {"gdp": "GDPC1", "inf": "PCEPILFE", "inv": "GPDIC1",
           "fiscal": "GCEC1", "wealth": "TNWBSHNO"}
CORE = ["gdp", "inf", "stance"]
RT_SHOCKS = ["oil", "inv", "wealth", "fiscal"]
P_FIXED = 2                          # BIC/HQIC choice for the core specs in the main study


# ---------------------------------------------------------------- vintage IO

def _vintage_date(qend, days=VLAG_DAYS):
    return (qend + dt.timedelta(days=days)).strftime("%Y-%m-%d")


def fetch_vintage(series, vdate):
    """Levels of `series` as known on `vdate`, cached to disk. Returns an empty
    Series if that vintage predates the series' real-time record."""
    os.makedirs(VDIR, exist_ok=True)
    path = f"{VDIR}/{series}_{vdate}.csv"
    if not os.path.exists(path):
        subprocess.run(["curl", "-s",
                        f"https://alfred.stlouisfed.org/graph/alfredgraph.csv?id={series}&vintage_date={vdate}",
                        "-o", path, "--max-time", "30"], check=True)
    try:
        df = pd.read_csv(path, parse_dates=[0], index_col=0)
    except (pd.errors.EmptyDataError, ValueError):
        return pd.Series(dtype=float, name=series)          # vintage predates the series' record
    s = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
    s.name = series
    return s


# ---------------------------------------------------------------- finals

def load_finals():
    """Non-revised inputs and scoring actuals, all at final vintage."""
    ffr = vl._to_q(vl._fred("FEDFUNDS"))
    rstar = vl.load_hlw()
    cpi_q = vl._to_q(vl._fred("CPIAUCSL"))
    wti_q = vl._to_q(vl._fred("WTISPLC"))
    oil = vl._dlog(wti_q / cpi_q)
    panel = vl.build_panel()                       # final transforms for scoring and rstar/ffr grid
    return {"ffr": ffr, "rstar": rstar, "cpi_q": cpi_q, "oil": oil, "panel": panel}


# ---------------------------------------------------------------- vintage panel

def _q_index(s):
    """Collapse a level series (monthly or quarterly) to quarter-start stamps."""
    return vl._to_q(s) if s.index.freqstr not in ("QS", "QS-OCT") else s


def build_vintage_transforms(vdate, finals):
    """Stationary transforms from a single data vintage. Each column runs to its
    own last available quarter; columns beyond that are absent (NaN after
    reindexing to the common grid)."""
    gdp_l = fetch_vintage("GDPC1", vdate)
    inv_l = fetch_vintage("GPDIC1", vdate)
    gov_l = fetch_vintage("GCEC1", vdate)
    pce_m = fetch_vintage("PCEPILFE", vdate)
    nw_l = fetch_vintage("TNWBSHNO", vdate)

    pce_q = _q_index(pce_m)
    ffr = finals["ffr"]
    rstar = finals["rstar"]
    cpi = finals["cpi_q"]

    pi4 = 100.0 * (np.log(pce_q) - np.log(pce_q.shift(4)))
    cols = {
        "gdp": vl._dlog(gdp_l),
        "inf": vl._dlog(pce_q),
        "stance": (ffr - pi4) - rstar,
        "oil": finals["oil"],
        "inv": vl._dlog(inv_l),
        "fiscal": vl._dlog(gov_l),
        "wealth": vl._dlog(nw_l / cpi) if len(nw_l) else pd.Series(dtype=float, index=pd.DatetimeIndex([])),
    }
    df = pd.DataFrame(cols)
    df.index = pd.DatetimeIndex(df.index)
    df = df.sort_index().loc[pd.Timestamp("1990-01-01"):]
    for q in vl.COVID_DUMMIES:
        ts = pd.Period(q, freq="Q").to_timestamp()
        df[f"d{q}"] = (df.index == ts).astype(float)
    return df


# ---------------------------------------------------------------- rt forecast

def _last_valid_q(df, col, upto_i):
    """Index of the last non-NaN quarter of `col` at or before row upto_i."""
    v = df[col].to_numpy()
    for i in range(upto_i, -1, -1):
        if np.isfinite(v[i]):
            return i
    return -1


def rt_forecast(df, endog, shocks, treat, p, o_idx):
    """One real-time forecast bundle from origin quarter o_idx. Coefficients are
    estimated on the balanced sub-sample ending at the earliest edge across the
    endogenous block and the selected shocks; the level path is iterated from
    the endogenous edge with each shock held at its vintage value through its own
    edge and at the training mean thereafter."""
    D = df[[c for c in df.columns if c.startswith("d20")]].to_numpy()
    Y = df[endog].to_numpy()
    n = Y.shape[1]

    shock_edges = {}
    for s in shocks:
        e = _last_valid_q(df, s, o_idx)
        if e < 0:
            return None                       # shock unpublished at this vintage
        shock_edges[s] = e
    est_end = o_idx if not shocks else min(o_idx, min(shock_edges.values()))

    Xp = vl.purge_shocks(df, endog, shocks, treat, est_end) if shocks else None
    fit = vl.fit_varx(Y[:est_end + 1], Xp[:est_end + 1] if shocks else None,
                      D[:est_end + 1], p)
    x_unc = Xp[:est_end + 1].mean(axis=0) if shocks else None

    if shocks:
        Xf = Xp.copy()
        for j, s in enumerate(shocks):
            Xf[shock_edges[s] + 1:, j] = x_unc[j]        # mean beyond each shock's edge
            bad = ~np.isfinite(Xf[:, j])
            Xf[bad, j] = x_unc[j]
    else:
        Xf = None

    fc = vl.forecast(fit, Y, Xf, D, o_idx, vl.H_MAX, vl.XLAGS, "uncond", x_unc)
    return {"fc": fc, "endog": endog, "p": p, "est_end": est_end, "o_idx": o_idx}


# ---------------------------------------------------------------- study

def rt_ar_benchmark(finals, origins_q, targets, p_by_target):
    """Real-time AR(p) forecasts per target, p taken from the main study."""
    out = {}
    for v in targets:
        p = p_by_target[v]
        series_id = REVISED[v]
        rows = {}
        for oq in origins_q:
            vdate = _vintage_date(oq.to_timestamp(how="end").date())
            lv = fetch_vintage(series_id, vdate)
            y = (vl._dlog(_q_index(lv)) if v in ("gdp", "inf") else lv).loc["1990-01-01":]
            ots = oq.to_timestamp()
            if ots not in y.index or not np.isfinite(y.loc[ots]):
                continue                     # aligns with the endogenous-edge skip in run_realtime
            yv = y.to_numpy().reshape(-1, 1)
            oi = y.index.get_loc(ots)
            D = np.zeros((len(y), 1))            # dummies immaterial for the AR level path here
            for q in vl.COVID_DUMMIES:
                ts = pd.Period(q, freq="Q").to_timestamp()
                if ts in y.index:
                    D[y.index.get_loc(ts), 0] = 1.0
            fit = vl.fit_varx(yv[:oi + 1], None, D[:oi + 1], p)
            f = vl.forecast(fit, yv, None, D, oi, vl.H_MAX, vl.XLAGS, "uncond", None)[:, 0]
            rows[oq] = f
        out[v] = rows
    return out


def run_realtime(finals, specs, targets=("gdp", "inf"), verbose=True):
    """Walk-forward every spec across origins on real-time vintages.
    Returns a long DataFrame of forecasts keyed by spec, target, horizon."""
    panel = finals["panel"]
    q_all = panel.index.to_period("Q")
    start = pd.Period(vl.TRAIN_END, freq="Q")
    origins_q = [q for q in q_all if q >= start and q < q_all[-1]]

    recs = []
    for k, oq in enumerate(origins_q):
        vdate = _vintage_date(oq.to_timestamp(how="end").date())
        df_v = build_vintage_transforms(vdate, finals)
        gidx = df_v.index
        if oq.to_timestamp() not in gidx:
            continue
        o_idx = gidx.get_loc(oq.to_timestamp())
        if not np.all(np.isfinite(df_v[CORE].to_numpy()[o_idx])):
            continue                         # own-quarter NIPA not yet published at this vintage
        for sp in specs:
            res = rt_forecast(df_v, CORE, sp["shocks"], sp["treat"], P_FIXED, o_idx)
            if res is None:
                continue
            for v in targets:
                j = CORE.index(v)
                for h in vl.H_REPORT:
                    t_idx = o_idx + h
                    if t_idx >= len(gidx):
                        continue
                    recs.append({"spec": sp["name"], "target": v, "h": h,
                                 "date": gidx[t_idx], "yhat": float(res["fc"][h - 1, j])})
        if verbose and k % 10 == 0:
            print(f"origin {k}/{len(origins_q)}  {oq}  vintage {vdate}")
    return pd.DataFrame(recs), origins_q


def score(rt_df, finals, ar_rt, origins_q, p_by_target):
    """RMSE and out-of-sample R2 against the real-time AR, full window and
    ex-COVID, scored on final-vintage actuals."""
    panel = finals["panel"]
    lo, hi = pd.Timestamp(vl.COVID_EVAL_EXCL[0]), pd.Timestamp(vl.COVID_EVAL_EXCL[1])
    ar_err = {}
    for v, rows in ar_rt.items():
        for h in vl.H_REPORT:
            e, d = [], []
            for oq, f in rows.items():
                t = oq.to_timestamp() + pd.offsets.QuarterBegin(h, startingMonth=1)
                if t not in panel.index:
                    continue
                d.append(t)
                e.append(f[h - 1] - panel.loc[t, v])
            ar_err[(v, h)] = pd.Series(e, index=pd.DatetimeIndex(d))

    out = []
    for (spec, v, h), g in rt_df.groupby(["spec", "target", "h"]):
        g = g.set_index("date")
        act = panel[v].reindex(g.index)
        err = (g["yhat"] - act)
        aer = ar_err[(v, h)].reindex(g.index)
        for sample in ("all", "excovid"):
            keep = np.isfinite(err.to_numpy()) & np.isfinite(aer.to_numpy())
            if sample == "excovid":
                keep &= ~((g.index >= lo) & (g.index <= hi))
            ee = err.to_numpy()[keep]
            ae = aer.to_numpy()[keep]
            rmse = float(np.sqrt(np.mean(ee ** 2)))
            r2 = float(1.0 - np.mean(ee ** 2) / np.mean(ae ** 2))
            out.append({"spec": spec, "target": v, "h": h, "sample": sample,
                        "rmse": round(rmse, 3), "r2_oos": round(r2, 3), "n": int(keep.sum())})
    return pd.DataFrame(out)


SPECS = [
    {"name": "pure_var", "shocks": [], "treat": "none"},
    {"name": "oil", "shocks": ["oil"], "treat": "raw"},
    {"name": "inv", "shocks": ["inv"], "treat": "raw"},
    {"name": "wealth", "shocks": ["wealth"], "treat": "raw"},
    {"name": "fiscal", "shocks": ["fiscal"], "treat": "raw"},
    {"name": "all4_raw", "shocks": RT_SHOCKS, "treat": "raw"},
    {"name": "all4_gdp_resid", "shocks": RT_SHOCKS, "treat": "gdp_resid"},
]
