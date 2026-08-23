"""
panel.py
Modular G7 panel layer over the existing engine. One entry point per country
(run_country) feeds the same multi-horizon evaluation used for the U.S.; the
aggregator pools results across countries with panel Diebold-Mariano tests, win
counts, and a Hansen Model Confidence Set.

The common exogenous shock is the global oil-price surprise (fetch_g7.oil_shock).
"""
import numpy as np, pandas as pd
from scipy import stats
from fetch_g7 import build_country, oil_shock, SERIES
from eval_multi import rolling_eval_multi, dm_test

ENDOG = ["gdp_yoy", "inflation_yoy", "reer_yoy", "y10", "policy"]


def run_country(cc, horizons=(1, 2, 4, 8), n_draws=150, first_origin=None):
    """Build country data + global oil shock, run the multi-horizon eval."""
    df = build_country(cc)
    osh = oil_shock()
    exog = pd.DataFrame(osh.reindex(df.index.to_period("Q")).values,
                        index=df.index, columns=["shock_oil"]).fillna(0.0)
    if first_origin is None:
        # start evaluation after ~10 yrs of training, leaving room for h=8
        first_origin = (df.index[min(40, len(df)//2)]).to_period("Q").strftime("%YQ%q")
    metrics, errs = rolling_eval_multi(df, exog, ENDOG, horizons=horizons,
                                       first_origin=first_origin, n_draws=n_draws)
    metrics["country"] = cc
    return metrics, errs


# ---------------------------------------------------------------- aggregation
def win_table(all_metrics, benchmark="RW", metric="CRPS", h=1):
    """For each variable: in how many countries does BVAR beat the benchmark at horizon h."""
    rows = []
    for v in ENDOG:
        wins, tot = 0, 0
        for cc in all_metrics["country"].unique():
            sub = all_metrics[(all_metrics.country == cc) & (all_metrics.variable == v) & (all_metrics.h == h)]
            b = sub[sub.model == "BVAR"][metric]
            k = sub[sub.model == benchmark][metric]
            if len(b) and len(k):
                tot += 1; wins += int(b.iloc[0] < k.iloc[0])
        rows.append({"variable": v, f"BVAR<{benchmark}": f"{wins}/{tot}"})
    return pd.DataFrame(rows)


def pooled_dm(all_errs, benchmark="RW", variable="gdp_yoy", h=1):
    """Pool one-step loss differentials across countries -> a panel DM p-value."""
    e1 = np.concatenate([all_errs[cc][("BVAR", variable, h)]
                         for cc in all_errs if ("BVAR", variable, h) in all_errs[cc]])
    e2 = np.concatenate([all_errs[cc][(benchmark, variable, h)]
                         for cc in all_errs if (benchmark, variable, h) in all_errs[cc]])
    return dm_test(e1, e2, h=h)


def model_confidence_set(all_errs, variable, h=1, models=("BVAR","BVAR_noshock","VAR_OLS","RW","AR1"),
                         alpha=0.10, B=1000, seed=0):
    """Hansen (2011) Model Confidence Set via the T_max statistic and a stationary
    bootstrap on pooled squared-error losses. Returns the surviving model set."""
    rng = np.random.default_rng(seed)
    # pooled loss matrix L (T x M)
    L = {}
    for m in models:
        e = np.concatenate([all_errs[cc][(m, variable, h)]
                            for cc in all_errs if (m, variable, h) in all_errs[cc]])
        L[m] = e**2
    T = min(len(v) for v in L.values())
    Lm = np.column_stack([L[m][:T] for m in models])
    surv = list(range(len(models)))
    # stationary bootstrap indices (mean block length 4)
    def boot_idx():
        idx = np.empty(T, int); i = 0
        while i < T:
            start = rng.integers(0, T); ln = rng.geometric(0.25)
            for k in range(ln):
                if i >= T: break
                idx[i] = (start + k) % T; i += 1
        return idx
    boots = [boot_idx() for _ in range(B)]
    while len(surv) > 1:
        sub = Lm[:, surv]
        dbar = sub.mean(0)                                  # mean loss per model
        # pairwise relative losses vs average
        meanall = sub.mean(1, keepdims=True)
        d = sub - meanall                                   # (T x k)
        dmean = d.mean(0)
        # bootstrap variance of dmean
        var = np.zeros(len(surv))
        for bi in boots:
            var += (d[bi].mean(0) - dmean)**2
        var /= B
        var = np.maximum(var, 1e-12)
        tstat = dmean / np.sqrt(var)
        Tmax = tstat.max()
        # bootstrap distribution of Tmax under equal predictive ability
        tmax_b = np.empty(B)
        for j, bi in enumerate(boots):
            db = d[bi].mean(0) - dmean
            tmax_b[j] = (db / np.sqrt(var)).max()
        pval = (tmax_b >= Tmax).mean()
        if pval > alpha:
            break
        worst = surv[int(np.argmax(tstat))]                 # eliminate worst (highest excess loss)
        surv.remove(worst)
    return [models[i] for i in surv]


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    countries = ["US", "UK", "CA", "DE", "JP"]
    all_m, all_e = [], {}
    for cc in countries:
        print(f"running {cc} ...", flush=True)
        try:
            m, e = run_country(cc, horizons=(1, 4, 8), n_draws=120)
            all_m.append(m); all_e[cc] = e
        except Exception as ex:
            print(f"  {cc} failed: {repr(ex)[:120]}")
    M = pd.concat(all_m, ignore_index=True)
    M.to_csv("panel_metrics.csv", index=False)
    print("\n=== Win table (CRPS, h=1): BVAR vs benchmarks across countries ===")
    for bm in ["RW", "AR1", "VAR_OLS", "BVAR_noshock"]:
        wt = win_table(M, benchmark=bm, h=1)
        print(f"\n  vs {bm}:"); print(wt.to_string(index=False))
    print("\n=== Pooled (panel) DM p-values, h=1, BVAR vs RW ===")
    for v in ENDOG:
        _, p = pooled_dm(all_e, "RW", v, 1)
        print(f"  {v:16s} p={p:.3f}")
    print("\n=== Model Confidence Set (CRPS-eq via SE loss, h=1, 90%) ===")
    for v in ENDOG:
        mcs = model_confidence_set(all_e, v, h=1)
        print(f"  {v:16s} -> {mcs}")
