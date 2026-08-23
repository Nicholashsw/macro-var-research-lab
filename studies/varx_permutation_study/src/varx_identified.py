"""
varx_identified.py

External identification check for the VARX study. The permutation study uses
cheap growth-rate shock proxies. This module asks whether an externally
identified oil shock, built from high-frequency OPEC announcement surprises,
buys any forecast accuracy that the raw real-oil-price growth proxy does not.

Three exogenous regressors are compared inside the same core VARX (GDP growth,
core PCE inflation, policy stance), fixed at p = 2, in the same expanding-window
walk-forward as the main study:

    oil_raw   400 dln(real WTI), the study's own proxy: leak-free, persistent.
    oil_surp  Kaenzig (2021) oil supply surprise instrument, the first principal
              component of oil-futures moves in a tight window around OPEC
              announcements: leak-free, but zero in non-announcement quarters.
    oil_news  Kaenzig (2021) oil supply news shock recovered from his full-sample
              SVAR-IV: dense, but its construction uses the whole sample, so as a
              regressor it carries look-ahead the other two do not.

The look-ahead in oil_news stacks the deck in favor of identification. If it
still fails to beat the raw proxy out of sample, the null is decisive.

Data: github.com/dkaenzig/oilsupplynews, vintage 2025M12, "Monthly" sheet,
monthly shocks summed to quarters. Reference: Kaenzig (2021), AER 111(4).

Nicholas Hong | Built for educational and research purposes. Not financial advice.
"""

import json
import numpy as np
import pandas as pd

import os
import varx_lab as vl

VINTAGE = os.path.join(vl.DATA_DIR, "identified", "oilsupplynews-master", "oilSupplyNewsShocks_2025M12.xlsx")
CORE = vl.ENDOG_CORE
P = 2
TARGETS = ["gdp", "inf"]
ARMS = {
    "pure_var": [],
    "oil_raw": ["oil"],
    "oil_surp": ["oil_id_surp"],
    "oil_news": ["oil_id_news"],
}


# ---------------------------------------------------------------- data

def load_oil_identified(path=VINTAGE):
    """Monthly Kaenzig surprise instrument and news shock, summed to quarters."""
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["Monthly"]
    dates, surp, news = [], [], []
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r[0] is None:
            continue
        ym = str(r[0])
        dates.append(pd.Timestamp(int(ym[:4]), int(ym[5:]), 1))
        surp.append(np.nan if r[1] is None else float(r[1]))
        news.append(np.nan if r[2] is None else float(r[2]))
    m = pd.DataFrame({"oil_id_surp": surp, "oil_id_news": news},
                     index=pd.DatetimeIndex(dates)).sort_index()
    return m.resample("QS").sum(min_count=1)


def build_identified_panel():
    """Base study panel with the two identified oil regressors joined on."""
    df = vl.build_panel().copy()
    q = load_oil_identified()
    for c in ("oil_id_surp", "oil_id_news"):
        df[c] = q[c].reindex(df.index)
    return df


# ---------------------------------------------------------------- walk-forward

def run_identified(df):
    D = df[[c for c in df.columns if c.startswith("d20")]].to_numpy()
    train_end_i = df.index.get_loc(pd.Period(vl.TRAIN_END, freq="Q").to_timestamp())
    T = len(df)
    origins = list(range(train_end_i, T - 1))
    Y = df[CORE].to_numpy()
    n = Y.shape[1]

    fcs = {}
    for arm, shocks in ARMS.items():
        X = df[shocks].to_numpy() if shocks else None
        fc = np.full((len(origins), vl.H_MAX, n), np.nan)
        for oi, org in enumerate(origins):
            Ttr = org + 1
            Xtr = X[:Ttr] if X is not None else None
            x_unc = np.nanmean(Xtr, axis=0) if X is not None else None
            fit = vl.fit_varx(Y[:Ttr], Xtr, D[:Ttr], P)
            fc[oi] = vl.forecast(fit, Y, X, D, org, vl.H_MAX, vl.XLAGS, "uncond", x_unc)
        fcs[arm] = fit_diag(df, Y, X, D, train_end_i, n), fc
    bench = vl.ar_benchmark(df, TARGETS, train_end_i, origins)
    return fcs, bench, origins


def fit_diag(df, Y, X, D, org, n):
    """Stability and residual whiteness on the initial training window."""
    Xtr = X[:org + 1] if X is not None else None
    fit = vl.fit_varx(Y[:org + 1], Xtr, D[:org + 1], P)
    return {"stab": float(vl.stability(fit, n)),
            "lb_min_p": float(min(vl.ljung_box_p(fit["E"][:, j]) for j in range(n)))}


# ---------------------------------------------------------------- scoring

def _err(fc, df, origins, target, h):
    j = CORE.index(target)
    y = df[target].to_numpy()
    idx = df.index
    d, e = [], []
    for oi, org in enumerate(origins):
        t = org + h
        if t >= len(idx):
            continue
        d.append(idx[t])
        e.append(fc[oi, h - 1, j] - y[t])
    return pd.Series(e, index=pd.DatetimeIndex(d))


def score(fcs, bench, origins, df):
    rows = []
    var_err = {}
    for target in TARGETS:
        for h in vl.H_REPORT:
            var_err[(target, h)] = _err(fcs["pure_var"][1], df, origins, target, h)
    ar_err = {}
    y = {v: df[v].to_numpy() for v in TARGETS}
    for v in TARGETS:
        for h in vl.H_REPORT:
            d, e = [], []
            for oi, org in enumerate(origins):
                t = org + h
                if t >= len(df.index):
                    continue
                d.append(df.index[t])
                e.append(bench[v]["fc"][oi, h - 1] - y[v][t])
            ar_err[(v, h)] = pd.Series(e, index=pd.DatetimeIndex(d))

    for arm, (diag, fc) in fcs.items():
        for target in TARGETS:
            for h in vl.H_REPORT:
                e = _err(fc, df, origins, target, h)
                ea = ar_err[(target, h)].reindex(e.index)
                ev = var_err[(target, h)].reindex(e.index)
                for sample in ("all", "excovid"):
                    mk = vl._mask(e.index, sample)
                    ee = e[mk].to_numpy()
                    fin = np.isfinite(ee)
                    ee = ee[fin]
                    rmse = float(np.sqrt(np.mean(ee ** 2)))
                    mae = float(np.mean(np.abs(ee)))
                    r2 = float(1.0 - np.mean(ee ** 2)
                               / np.mean(ea[mk].to_numpy()[fin] ** 2))
                    if arm == "pure_var":
                        dmp = np.nan
                    else:
                        dm, dmp = vl.dm_test(ee, ev[mk].to_numpy()[fin], h)
                    rows.append({"arm": arm, "target": target, "h": h, "sample": sample,
                                 "rmse": round(rmse, 4), "mae": round(mae, 4),
                                 "r2_oos": round(r2, 4),
                                 "dm_p_vs_var": None if not np.isfinite(dmp) else round(dmp, 3),
                                 "n": int(fin.sum()),
                                 "stab": round(diag["stab"], 3), "lb_min_p": round(diag["lb_min_p"], 3)})
    return pd.DataFrame(rows)


def run():
    df = build_identified_panel()
    cov = df["oil_id_news"].dropna()
    print(f"panel {df.index[0].date()}..{df.index[-1].date()}, "
          f"oil identified {cov.index[0].date()}..{cov.index[-1].date()}, "
          f"in-sample surprise nonzero quarters "
          f"{int((df['oil_id_surp'].fillna(0) != 0).sum())}/{len(df)}")
    fcs, bench, origins = run_identified(df)
    tab = score(fcs, bench, origins, df)
    tab.to_parquet("id_compare.parquet", index=False)

    stats = {}
    for target in TARGETS:
        stats[target] = {}
        for h in vl.H_REPORT:
            stats[target][str(h)] = {}
            for sample in ("all", "excovid"):
                sub = tab[(tab.target == target) & (tab.h == h) & (tab["sample"] == sample)]
                d = {r.arm: {"rmse": r.rmse, "r2_oos": r.r2_oos, "dm_p_vs_var": r.dm_p_vs_var}
                     for r in sub.itertuples()}
                stats[target][str(h)][sample] = d
    with open("id_stats.json", "w") as f:
        json.dump(stats, f, indent=1)

    for target in TARGETS:
        print(f"\n=== {target} ===")
        sub = tab[(tab.target == target)].pivot_table(
            index=["h", "sample"], columns="arm", values="rmse")
        sub = sub[["pure_var", "oil_raw", "oil_surp", "oil_news"]]
        print(sub.round(3).to_string())
    return tab


if __name__ == "__main__":
    run()
