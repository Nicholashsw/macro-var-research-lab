"""
varx_monetary.py

Monetary-margin counterpart to varx_identified.py. The core VARX already carries
monetary conditions through the endogenous stance variable (real fed funds minus
the HLW natural rate), so stance is the built-in raw proxy. The question is
whether an externally identified FOMC surprise, entered as an exogenous
regressor, adds forecast information the stance dynamics miss. This is the one
margin where identification might plausibly win: stance is the weakest of the
three core proxies, and a high-frequency surprise moves even at the zero lower
bound, where the cheap rate-change proxy is degenerate.

Four arms, core endogenous set (GDP, inflation, stance), fixed p = 2:

    pure_var  the core alone; stance is the endogenous monetary proxy.
    mp_raw    exogenous change in the fed funds rate: cheap, but flat at the ZLB.
    mp_surp   Bauer-Swanson raw monetary policy surprise (MPS): the first
              principal component of 30-minute money-market futures moves around
              FOMC announcements. Leak-free, zero outside announcement quarters.
    mp_orth   Bauer-Swanson orthogonalized surprise (MPS_ORTH): MPS purged of
              predictable variation in pre-announcement public data, the cleanest
              identified shock on offer.

Identified coverage is 1988-2023, so all arms are capped to training windows
ending by the last identified quarter and evaluated on the same origins.

Data: FRBSF Center for Monetary Research, "Monetary Policy Surprises" workbook,
sheet "Monthly (update 2023)", monthly surprises summed to quarters.
Reference: Bauer and Swanson (2023), NBER Macroeconomics Annual.

Nicholas Hong | Built for educational and research purposes. Not financial advice.
"""

import json
import numpy as np
import pandas as pd

import os
import varx_lab as vl

MPS_FILE = os.path.join(vl.DATA_DIR, "identified", "mps.xlsx")
MPS_SHEET = "Monthly (update 2023)"
CORE = vl.ENDOG_CORE
P = 2
TARGETS = ["gdp", "inf"]
ARMS = {
    "pure_var": [],
    "mp_raw": ["mp_raw"],
    "mp_surp": ["mp_surp"],
    "mp_orth": ["mp_orth"],
}


# ---------------------------------------------------------------- data

def load_mps(path=MPS_FILE, sheet=MPS_SHEET):
    """Monthly MPS and MPS_ORTH summed to quarters."""
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet]
    dates, mps, orth = [], [], []
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r[0] is None or r[1] is None:
            continue
        dates.append(pd.Timestamp(int(r[0]), int(r[1]), 1))
        mps.append(np.nan if r[2] is None else float(r[2]))
        orth.append(np.nan if r[3] is None else float(r[3]))
    m = pd.DataFrame({"mp_surp": mps, "mp_orth": orth},
                     index=pd.DatetimeIndex(dates)).sort_index()
    return m.resample("QS").sum(min_count=1)


def build_monetary_panel():
    df = vl.build_panel().copy()
    ffr = vl._to_q(vl._fred("FEDFUNDS"))
    df["mp_raw"] = ffr.diff().reindex(df.index)
    q = load_mps()
    for c in ("mp_surp", "mp_orth"):
        df[c] = q[c].reindex(df.index)
    return df


# ---------------------------------------------------------------- walk-forward

def run_monetary(df):
    D = df[[c for c in df.columns if c.startswith("d20")]].to_numpy()
    train_end_i = df.index.get_loc(pd.Period(vl.TRAIN_END, freq="Q").to_timestamp())
    # cap origins so every training window ends where all identified series exist
    id_ok = df[["mp_surp", "mp_orth"]].notna().all(axis=1).to_numpy()
    last_id = np.max(np.where(id_ok))
    origins = list(range(train_end_i, last_id))     # last training row <= last_id
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
        diag = _diag(df, Y, X, D, last_id, n)
        fcs[arm] = (diag, fc)
    bench = vl.ar_benchmark(df, TARGETS, train_end_i, origins)
    return fcs, bench, origins


def _diag(df, Y, X, D, org, n):
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
    var_err = {(t, h): _err(fcs["pure_var"][1], df, origins, t, h)
               for t in TARGETS for h in vl.H_REPORT}
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
                    r2 = float(1.0 - np.mean(ee ** 2) / np.mean(ea[mk].to_numpy()[fin] ** 2))
                    if arm == "pure_var":
                        dmp = np.nan
                    else:
                        _, dmp = vl.dm_test(ee, ev[mk].to_numpy()[fin], h)
                    rows.append({"arm": arm, "target": target, "h": h, "sample": sample,
                                 "rmse": round(rmse, 4), "mae": round(mae, 4),
                                 "r2_oos": round(r2, 4),
                                 "dm_p_vs_var": None if not np.isfinite(dmp) else round(dmp, 3),
                                 "n": int(fin.sum()),
                                 "stab": round(diag["stab"], 3), "lb_min_p": round(diag["lb_min_p"], 3)})
    return pd.DataFrame(rows)


def run():
    df = build_monetary_panel()
    fcs, bench, origins = run_monetary(df)
    o0, o1 = df.index[origins[0]].date(), df.index[origins[-1]].date()
    zlb = int((df["mp_raw"].abs() < 0.02).reindex(df.index[origins[0]:origins[-1] + 1]).sum())
    print(f"origins {o0}..{o1} ({len(origins)}), cheap proxy near-zero in "
          f"{zlb}/{len(origins)} origin quarters (ZLB)")
    tab = score(fcs, bench, origins, df)
    tab.to_parquet("mp_compare.parquet", index=False)

    stats = {}
    for target in TARGETS:
        stats[target] = {}
        for h in vl.H_REPORT:
            stats[target][str(h)] = {}
            for sample in ("all", "excovid"):
                sub = tab[(tab.target == target) & (tab.h == h) & (tab["sample"] == sample)]
                stats[target][str(h)][sample] = {
                    r.arm: {"rmse": r.rmse, "r2_oos": r.r2_oos, "dm_p_vs_var": r.dm_p_vs_var}
                    for r in sub.itertuples()}
    with open("mp_stats.json", "w") as f:
        json.dump(stats, f, indent=1)

    for target in TARGETS:
        print(f"\n=== {target} ===")
        piv = tab[tab.target == target].pivot_table(
            index=["h", "sample"], columns="arm", values="rmse")
        print(piv[["pure_var", "mp_raw", "mp_surp", "mp_orth"]].round(3).to_string())
    return tab


if __name__ == "__main__":
    run()
