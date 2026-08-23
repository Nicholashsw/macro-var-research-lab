"""
varx_combine.py

Capstone for the external-identification thread. The oil and monetary checks
each showed an identified shock with, at best, a small horizon-specific edge over
the cheap proxy and a loss elsewhere. This module asks whether those isolated
edges are additive or redundant: does combining the proxy forecast with the
identified-shock forecast beat either component out of sample, and does the
identified forecast carry any information the proxy lacks?

For each margin the proxy-based forecast is paired with the identified-shock
forecast, at the target variable and horizon, over the same origins the modules
already evaluate:

    oil        proxy = raw real-oil-price growth (oil_raw)
               identified = Kaenzig news shock (oil_news)
    monetary   proxy = the core, whose stance variable is the endogenous
               monetary proxy (pure_var)
               identified = raw Bauer-Swanson FOMC surprise (mp_surp)

Three questions, all leak-free out of sample:
    equal       0.5/0.5 combination RMSE against each component.
    inv_mse     Bates-Granger inverse-MSE weights, estimated only on realized
                errors available at each origin, expanding.
    encompass   regression y - f_proxy = beta (f_id - f_proxy) + u over the OOS
                sample, HAC(h-1) standard errors. beta near zero means the proxy
                forecast encompasses the identified one: identification adds
                nothing conditional on the proxy. The RMSE at the fitted beta is
                the best any static combination could do, an upper bound.

Nicholas Hong | Built for educational and research purposes. Not financial advice.
"""

import json
import numpy as np
import pandas as pd

import varx_lab as vl
import varx_identified as vi
import varx_monetary as vm

MIN_TRAIN = 8   # minimum realized pairs before estimating combination weights


def hac_beta(y, x, h):
    """OLS slope of y on [1, x] with a Newey-West HAC(h-1) t-test on the slope."""
    from scipy import stats
    X = np.column_stack([np.ones_like(x), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ b
    T = len(y)
    XtX_inv = np.linalg.inv(X.T @ X)
    S = (X * u[:, None]).T @ (X * u[:, None])
    for k in range(1, h):
        w = 1.0 - k / h
        Xu = X * u[:, None]
        S += w * (Xu[k:].T @ Xu[:-k] + Xu[:-k].T @ Xu[k:])
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(max(cov[1, 1], 1e-18))
    t = b[1] / se
    p = 2.0 * (1.0 - stats.norm.cdf(abs(t)))
    return b[1], t, p


def series(fc, df, origins, target, h):
    """Forecast of `target` at horizon h, aligned to realized target dates."""
    j = vi.CORE.index(target)
    idx = df.index
    y = df[target].to_numpy()
    dts, f, yy = [], [], []
    for oi, org in enumerate(origins):
        t = org + h
        if t >= len(idx):
            continue
        dts.append(idx[t])
        f.append(fc[oi, h - 1, j])
        yy.append(y[t])
    return pd.DataFrame({"f": f, "y": yy}, index=pd.DatetimeIndex(dts))


def inv_mse_combo(fp, fi, y, origins_dates, h):
    """Leak-free inverse-MSE combination. At each target date, weights come from
    squared errors of pairs whose targets were realized at least h steps earlier."""
    n = len(y)
    out = np.empty(n)
    ep = (fp - y) ** 2
    ei = (fi - y) ** 2
    for k in range(n):
        # realized-by-now training set: targets dated on or before the origin of k
        cut = origins_dates[k]  # forecast k was made h quarters before its target
        avail = origins_dates <= cut
        avail[k:] = False       # strictly earlier realized targets only
        if avail.sum() < MIN_TRAIN:
            out[k] = 0.5 * (fp[k] + fi[k])
            continue
        mp = ep[avail].mean()
        mi = ei[avail].mean()
        wp = (1.0 / mp) / (1.0 / mp + 1.0 / mi)
        out[k] = wp * fp[k] + (1.0 - wp) * fi[k]
    return out


PAIRS = {
    "oil": ("oil_raw", "oil_news"),
    "monetary": ("pure_var", "mp_surp"),
}


def run():
    df_o = vi.build_identified_panel()
    fco, _, oo = vi.run_identified(df_o)
    df_m = vm.build_monetary_panel()
    fcm, _, om = vm.run_monetary(df_m)
    margins = {
        "oil": (df_o, fco, oo),
        "monetary": (df_m, fcm, om),
    }

    rows = []
    for margin, (df, fcs, origins) in margins.items():
        proxy_arm, id_arm = PAIRS[margin]
        fp_arm = fcs[proxy_arm][1]
        fi_arm = fcs[id_arm][1]
        # target date for each origin index (origin + h handled inside series)
        for target in vi.TARGETS:
            for h in vl.H_REPORT:
                sp = series(fp_arm, df, origins, target, h)
                si = series(fi_arm, df, origins, target, h)
                j = sp.join(si, lsuffix="_p", rsuffix="_i")
                # origin date = target date minus h quarters, used for leak-free cut
                odates = j.index - pd.offsets.QuarterBegin(startingMonth=1) * h
                for sample in ("all", "excovid"):
                    mk = vl._mask(j.index, sample)
                    sub = j[mk]
                    od = odates[mk]
                    fp = sub["f_p"].to_numpy()
                    fi = sub["f_i"].to_numpy()
                    y = sub["y_p"].to_numpy()
                    fin = np.isfinite(fp) & np.isfinite(fi) & np.isfinite(y)
                    fp, fi, y, od = fp[fin], fi[fin], y[fin], od[fin]
                    rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))
                    r_p = rmse(fp - y)
                    r_i = rmse(fi - y)
                    r_eq = rmse(0.5 * (fp + fi) - y)
                    fc_iv = inv_mse_combo(fp, fi, y, od.to_numpy(), h)
                    r_iv = rmse(fc_iv - y)
                    beta, tstat, p = hac_beta(y - fp, fi - fp, h)
                    bc = min(max(beta, 0.0), 1.0)               # sensible convex weight
                    r_enc = rmse((fp + bc * (fi - fp)) - y)     # best static convex blend
                    rows.append({
                        "margin": margin, "target": target, "h": h, "sample": sample,
                        "rmse_proxy": round(r_p, 4), "rmse_id": round(r_i, 4),
                        "rmse_equal": round(r_eq, 4), "rmse_invmse": round(r_iv, 4),
                        "rmse_bestconvex": round(r_enc, 4),
                        "beta_id": round(float(beta), 3), "beta_p": round(float(p), 3),
                        "n": int(fin.sum())})
    tab = pd.DataFrame(rows)
    tab.to_parquet("combine_compare.parquet", index=False)

    # headline: does any combination beat the better component, and is beta_id ever significant?
    tab["best_component"] = tab[["rmse_proxy", "rmse_id"]].min(axis=1)
    tab["equal_vs_best"] = 100 * (tab["rmse_equal"] / tab["best_component"] - 1)
    tab["invmse_vs_best"] = 100 * (tab["rmse_invmse"] / tab["best_component"] - 1)
    print("combination RMSE relative to the better component, percent (negative = combo wins):")
    print(tab.groupby(["margin", "target"])[["equal_vs_best", "invmse_vs_best"]]
          .mean().round(2).to_string())
    print("\nencompassing: beta on the identified forecast given the proxy")
    print(tab[["margin", "target", "h", "sample", "beta_id", "beta_p"]].to_string(index=False))
    sig = tab[tab.beta_p < 0.05]
    print(f"\nbeta_id significant at 5% in {len(sig)}/{len(tab)} cells")
    if len(sig):
        print(sig[["margin", "target", "h", "sample", "beta_id", "beta_p",
                   "rmse_proxy", "rmse_id", "rmse_bestconvex"]].to_string(index=False))

    stats = {}
    for margin in margins:
        stats[margin] = {}
        for target in vi.TARGETS:
            sub = tab[(tab.margin == margin) & (tab.target == target)]
            stats[margin][target] = {
                "equal_vs_best_mean_pct": round(float(sub.equal_vs_best.mean()), 2),
                "invmse_vs_best_mean_pct": round(float(sub.invmse_vs_best.mean()), 2),
                "beta_id_significant_cells": int((sub.beta_p < 0.05).sum()),
                "n_cells": int(len(sub))}
    with open("combine_stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    return tab


if __name__ == "__main__":
    run()
