"""
var_horserace.py
================
Baseline classical-VAR specification horse race on real US FRED data.

Implements the methodology in var_horserace_methodology.md:
  - economic-BLOCK forward selection (not 2^N variable search)
  - a-priori exogenous/endogenous split (oil tested as VARX exog with AR projection)
  - per-target RELATIVE RMSE vs RW + AR(BIC) benchmarks (never raw cross-target avg)
  - hard validity filters: stability / residual whiteness / dof / conditioning
  - significance: Clark-West (nested vs benchmark), Diebold-Mariano (vs best),
    Model Confidence Set (Hansen-Lunde-Nason) per target + pooled
  - all lag selection done INSIDE each expanding training window (no leakage)

Data: FRED public no-key CSV endpoint (fetched separately into ./fred_data).
For research / education. Not investment advice.
"""

import warnings, io, json
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen

ROOT = Path("/home/claude")
DATA = ROOT / "fred_data"
OUT  = Path("/mnt/user-data/outputs")
OUT.mkdir(parents=True, exist_ok=True)
FIG  = ROOT / "figs"; FIG.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
# 1. SERIES MAP  (id -> how to make it stationary)
#    growth = 400*dlog (annualized %), oilret = 100*dlog (q %), level = as-is
# ----------------------------------------------------------------------------
CORE = {
    "GDP":    ("GDPC1",    "growth"),   # real GDP
    "INF":    ("PCEPILFE", "growth"),   # core PCE price index  (CPILFESL is a 1-line swap)
    "POLICY": ("FEDFUNDS", "level"),    # effective fed funds rate
    "FX":     ("RBUSBIS",  "growth"),   # real broad USD (BIS); binds sample to 1994+
}
BLOCKS = {                              # candidate shock blocks (one representative each)
    "INV":     ("GPDIC1",          "growth"),  # real gross private domestic investment
    "CONS":    ("PCECC96",         "growth"),  # real personal consumption expenditures
    "FISCAL":  ("GCEC1",           "growth"),  # real govt consumption+investment (CAPB proxy)
    "EXPORTS": ("EXPGSC1",         "growth"),  # real exports G&S (foreign-demand proxy; note: GDP component)
    "WEALTH":  ("TNWBSHNO",        "growth"),  # household net worth, Z.1 (nominal; ends 2025Q4 -> binds sample end)
    "HOUSING": ("USSTHPI",         "growth"),  # FHFA all-transactions house price index
    "OIL":     ("WTISPLC",         "oilret"),  # WTI spot crude        -> a-priori EXOGENOUS candidate
    "TFP":     ("FERNALD_TFPUTIL", "level"),   # Fernald util-adj TFP growth (already a rate) -> EXOG supply shock
}
EXOG_BLOCKS = {"OIL", "TFP"}            # tested as VARX exogenous (AR-projected OOS)
TARGETS = ["GDP", "INF", "POLICY", "FX"]   # always rank on these 4, comparable across specs

H_LIST   = [1, 4]        # forecast horizons (quarters). h=8 is a 1-line addition.
H_SELECT = 1             # horizon that drives forward selection
MAXLAG_VAR = 4
MAXLAG_AR  = 8
OOS_START_FRAC = 0.55    # first OOS origin at ~55% of sample

# ----------------------------------------------------------------------------
# 2. LOAD + BUILD STATIONARY QUARTERLY PANEL
# ----------------------------------------------------------------------------
def load_q(series_id):
    df = pd.read_csv(DATA / f"{series_id}.csv")
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.dropna().set_index("date")["value"].sort_index()
    return s.resample("QS").mean()      # monthly->quarterly mean; quarterly passes through

def transform(level, kind):
    if kind == "level":  return level
    if kind == "growth": return 400.0 * np.log(level).diff()
    if kind == "oilret": return 100.0 * np.log(level).diff()
    raise ValueError(kind)

raw_levels, modeled = {}, {}
for name, (sid, kind) in {**CORE, **BLOCKS}.items():
    lv = load_q(sid)
    raw_levels[name] = lv
    modeled[name] = transform(lv, kind)

panel = pd.DataFrame(modeled).dropna()
panel = panel.loc["1994-04-01":]        # FX binds start; first dlog at 1994Q2
print(f"[panel] {panel.shape[0]} quarters  {panel.index.min().date()} -> {panel.index.max().date()}")
print(f"[panel] columns: {list(panel.columns)}\n")

# ----------------------------------------------------------------------------
# 3. STAGE-0 DIAGNOSTICS  (run once; fix transform strategy, do NOT search it)
# ----------------------------------------------------------------------------
print("="*78); print("STAGE 0  DIAGNOSTICS"); print("="*78)
diag_rows = []
for c in panel.columns:
    x = panel[c].dropna().values
    adf_p = adfuller(x, autolag="AIC")[1]
    try:
        kpss_p = kpss(x, regression="c", nlags="auto")[1]
    except Exception:
        kpss_p = np.nan
    verdict = "I(0)" if (adf_p < 0.05 and (np.isnan(kpss_p) or kpss_p > 0.05)) else "check"
    diag_rows.append((c, adf_p, kpss_p, verdict))
    print(f"  {c:8s}  ADF p={adf_p:6.3f}   KPSS p={kpss_p:6.3f}   -> {verdict}")

# Johansen on the I(1) LEVELS of the core block (decide levels-VAR vs VECM)
print("\n  Johansen cointegration on core LEVELS [logGDP, logP, fedfunds, logFX]:")
core_lv = pd.DataFrame({
    "logGDP": np.log(raw_levels["GDP"]),
    "logP":   np.log(raw_levels["INF"]),
    "ff":     raw_levels["POLICY"],
    "logFX":  np.log(raw_levels["FX"]),
}).dropna().loc["1994-01-01":]
joh = coint_johansen(core_lv.values, det_order=0, k_ar_diff=4)
r = int((joh.lr1 > joh.cvt[:, 1]).sum())   # trace vs 5% crit
print(f"    trace stats : {np.round(joh.lr1,2)}")
print(f"    5% crit     : {np.round(joh.cvt[:,1],2)}")
print(f"    => rank r = {r}  ({'some cointegration; VECM possible' if r>0 else 'no cointegration at 5%'})")
print("    DECISION: model growth rates / level rate (stationary VAR). VECM left as robustness.\n")

# ----------------------------------------------------------------------------
# 4. OOS SCHEME + BENCHMARKS
# ----------------------------------------------------------------------------
N = len(panel)
i0 = int(N * OOS_START_FRAC)
print(f"[oos] expanding window; first origin idx={i0} ({panel.index[i0].date()}); "
      f"origins end at N-h\n")

def ar_bic_path(y, h, maxlag=MAXLAG_AR):
    """AR(p) with BIC-selected p (manual OLS), iterated h steps. Returns length-h path."""
    y = np.asarray(y, float); n = len(y)
    best = (np.inf, 1, None)
    for p in range(1, maxlag + 1):
        if n - p < p + 5: break
        Y = y[p:]
        X = np.column_stack([np.ones(n - p)] + [y[p - 1 - j:n - 1 - j] for j in range(p)])
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        resid = Y - X @ beta; s2 = max(resid @ resid / len(Y), 1e-12)
        bic = len(Y) * np.log(s2) + (p + 1) * np.log(len(Y))
        if bic < best[0]: best = (bic, p, beta)
    _, p, beta = best
    hist = list(y[-p:]); out = []
    for _ in range(h):
        x = np.array([1.0] + [hist[-1 - j] for j in range(p)])
        yh = float(x @ beta); out.append(yh); hist.append(yh)
    return np.array(out)

# benchmark forecasts: dict[target][h] -> dict origin_idx -> {'rw','ar','actual'}
bench = {t: {h: {} for h in H_LIST} for t in TARGETS}
for t in TARGETS:
    y = panel[t].values
    for h in H_LIST:
        for i in range(i0, N - h):
            ytr = y[: i + 1]
            bench[t][h][i] = {
                "rw": ytr[-1],
                "ar": ar_bic_path(ytr, h)[-1],
                "actual": y[i + h],
            }

# ----------------------------------------------------------------------------
# 5. VAR / VARX SPEC ENGINE  (lag selection inside each training window)
# ----------------------------------------------------------------------------
def fit_var(endog_arr, exog_arr=None):
    m = VAR(endog_arr, exog=exog_arr)
    res = m.fit(maxlags=MAXLAG_VAR, ic="bic", trend="c")
    if res.k_ar == 0:
        res = m.fit(1, trend="c")
    return res

def spec_oos(endog_names, exog_names, h):
    """Return dict[target] -> dict origin_idx -> prediction (core targets only)."""
    cols = endog_names
    preds = {t: {} for t in TARGETS}
    for i in range(i0, N - h):
        try:
            endog = panel[cols].values[: i + 1]
            if exog_names:
                exog = panel[exog_names].values[: i + 1]
                res = fit_var(endog, exog)
                # project exog forward by AR(BIC) (exogenous => unknown future OOS)
                ef = np.column_stack([ar_bic_path(panel[e].values[: i + 1], h)
                                      for e in exog_names])
                last = endog[-res.k_ar:]
                fc = res.forecast(last, steps=h, exog_future=ef)
            else:
                res = fit_var(endog)
                last = endog[-res.k_ar:]
                fc = res.forecast(last, steps=h)
            row = fc[h - 1]
            for t in TARGETS:
                preds[t][i] = row[cols.index(t)]
        except Exception:
            for t in TARGETS:
                preds[t][i] = np.nan
    return preds

def rel_rmse(preds, h):
    """relative RMSE vs RW and vs AR, per target, on common non-NaN origins."""
    out = {}
    for t in TARGETS:
        idx = [i for i in preds[t] if np.isfinite(preds[t][i])]
        if len(idx) < 8:
            out[t] = {"rmse": np.nan, "vs_rw": np.nan, "vs_ar": np.nan, "n": len(idx)}
            continue
        e  = np.array([bench[t][h][i]["actual"] - preds[t][i] for i in idx])
        er = np.array([bench[t][h][i]["actual"] - bench[t][h][i]["rw"] for i in idx])
        ea = np.array([bench[t][h][i]["actual"] - bench[t][h][i]["ar"] for i in idx])
        rmse = np.sqrt((e**2).mean())
        out[t] = {"rmse": rmse,
                  "vs_rw": rmse / np.sqrt((er**2).mean()),
                  "vs_ar": rmse / np.sqrt((ea**2).mean()),
                  "n": len(idx)}
    return out

def mean_relrmse(rr):
    v = [rr[t]["vs_rw"] for t in TARGETS if np.isfinite(rr[t]["vs_rw"])]
    return np.mean(v) if v else np.inf

# ----------------------------------------------------------------------------
# 6. STAGE 1-3  ANCHOR CORE + BLOCK FORWARD SELECTION (greedy, by mean relRMSE@h1)
# ----------------------------------------------------------------------------
print("="*78); print("STAGE 1-3  ANCHOR + BLOCK FORWARD SELECTION (h=1, score = mean relRMSE vs RW)")
print("="*78)
specs = {}   # spec_id -> dict(endog, exog, label)

def register(label, endog, exog):
    specs[label] = {"endog": endog, "exog": exog}

# anchor
core_preds = spec_oos(list(CORE), [], H_SELECT)
core_rr = rel_rmse(core_preds, H_SELECT)
register("CORE", list(CORE), [])
current = {"endog": list(CORE), "exog": [], "score": mean_relrmse(core_rr), "used": set()}
print(f"  CORE                         mean relRMSE@1 = {current['score']:.4f}")

# greedy forward selection over blocks
endo_candidates = [b for b in BLOCKS if b not in EXOG_BLOCKS]   # INV, CONS, FISCAL endog
exo_candidates  = [b for b in BLOCKS if b in EXOG_BLOCKS]       # OIL exog
sel_log = [("CORE", current["score"])]
improved = True
while improved:
    improved = False
    best_step = None
    for b in endo_candidates:
        if b in current["used"]: continue
        cand_endog = current["endog"] + [b]
        rr = rel_rmse(spec_oos(cand_endog, current["exog"], H_SELECT), H_SELECT)
        sc = mean_relrmse(rr)
        print(f"    try +{b:6s} endog          mean relRMSE@1 = {sc:.4f}")
        if best_step is None or sc < best_step[1]:
            best_step = (("endog", b, cand_endog, current["exog"]), sc)
    for b in exo_candidates:
        if b in current["used"]: continue
        cand_exog = current["exog"] + [b]
        rr = rel_rmse(spec_oos(current["endog"], cand_exog, H_SELECT), H_SELECT)
        sc = mean_relrmse(rr)
        print(f"    try +{b:6s} exog (VARX)    mean relRMSE@1 = {sc:.4f}")
        if best_step is None or sc < best_step[1]:
            best_step = (("exog", b, current["endog"], cand_exog), sc)
    if best_step and best_step[1] < current["score"] - 1e-4:
        (kind, b, en, ex), sc = best_step
        current = {"endog": en, "exog": ex, "score": sc, "used": current["used"] | {b}}
        improved = True
        lbl = "CORE+" + "+".join([x for x in en if x not in CORE] +
                                 [x + "(x)" for x in ex])
        register(lbl, en, ex)
        sel_log.append((lbl, sc))
        print(f"  >> ACCEPT +{b} ({kind}); new best mean relRMSE@1 = {sc:.4f}  [{lbl}]\n")
    else:
        print("  >> no block improves the score; forward selection stops.\n")

# also register every single-block variant for the results table (transparency)
for b in endo_candidates:
    register(f"CORE+{b}", list(CORE) + [b], [])
for b in exo_candidates:
    register(f"CORE+{b}(x)", list(CORE), [b])

print(f"[selection path] " + " -> ".join(f"{l}({s:.3f})" for l, s in sel_log))
print(f"[forward-selection winner] {sel_log[-1][0]}\n")

# ----------------------------------------------------------------------------
# 7. FULL-SAMPLE DIAGNOSTICS + HARD VALIDITY FILTERS (per spec)
# ----------------------------------------------------------------------------
def companion_max_modulus(res):
    p, k = res.k_ar, res.neqs
    if p == 0: return 0.0
    C = np.zeros((k * p, k * p))
    C[:k, :] = np.hstack([res.coefs[i] for i in range(p)])
    if p > 1: C[k:, :-k] = np.eye(k * (p - 1))
    return float(np.max(np.abs(np.linalg.eigvals(C))))

def full_diag(endog_names, exog_names):
    endog = panel[endog_names].values
    exog  = panel[exog_names].values if exog_names else None
    res = fit_var(endog, exog)
    k, p = res.neqs, res.k_ar
    n_params = k * (k * p + 1) + (k * len(exog_names) if exog_names else 0)
    T = res.nobs
    try:
        stable = bool(res.is_stable(verbose=False))
    except Exception:
        stable = companion_max_modulus(res) < 1.0
    maxmod = companion_max_modulus(res)
    try:
        wh = res.test_whiteness(nlags=max(p + 4, 10), adjusted=True).pvalue
    except Exception:
        wh = np.nan
    # conditioning of the stacked design (Z'Z) via lagged regressors
    Z = []
    for i in range(p, len(endog)):
        row = [1.0]
        for L in range(1, p + 1): row += list(endog[i - L])
        if exog_names: row += list(exog[i])
        Z.append(row)
    Z = np.array(Z)
    cond = float(np.linalg.cond(Z.T @ Z)) if Z.size else np.nan
    dof_ratio = T / n_params
    return {"var_lag": p, "n_params": n_params, "T": T, "dof_ratio": dof_ratio,
            "stable": stable, "max_modulus": maxmod, "whiteness_p": wh, "cond": cond,
            "aic": float(res.aic), "bic": float(res.bic), "hqic": float(res.hqic)}

# ----------------------------------------------------------------------------
# 8. SCORE ALL SPECS (relRMSE per target, both horizons) + assemble table
# ----------------------------------------------------------------------------
print("="*78); print("STAGE 4-5  SCORE ALL SPECS + RANK"); print("="*78)
records, oos_store = [], {}
for label, sp in specs.items():
    rr = {h: rel_rmse(spec_oos(sp["endog"], sp["exog"], h), h) for h in H_LIST}
    oos_store[label] = {h: spec_oos(sp["endog"], sp["exog"], h) for h in H_LIST}  # reuse
    fd = full_diag(sp["endog"], sp["exog"])
    rec = {
        "model_id": label,
        "endogenous_set": "+".join(sp["endog"]),
        "exogenous_set": "+".join(sp["exog"]) if sp["exog"] else "-",
        "var_lag_order": fd["var_lag"],
        "n_params": fd["n_params"], "T_train": fd["T"],
        "T_over_params": round(fd["dof_ratio"], 2),
        "stability_pass": fd["stable"], "max_eig_modulus": round(fd["max_modulus"], 4),
        "whiteness_pvalue": round(fd["whiteness_p"], 4) if np.isfinite(fd["whiteness_p"]) else np.nan,
        "cond_number": f"{fd['cond']:.2e}",
        "AIC": round(fd["aic"], 3), "BIC": round(fd["bic"], 3), "HQIC": round(fd["hqic"], 3),
    }
    for h in H_LIST:
        for t in TARGETS:
            rec[f"relRMSE_{t}_h{h}"] = round(rr[h][t]["vs_rw"], 4) if np.isfinite(rr[h][t]["vs_rw"]) else np.nan
        vals = [rr[h][t]["vs_rw"] for t in TARGETS if np.isfinite(rr[h][t]["vs_rw"])]
        rec[f"mean_relRMSE_h{h}"] = round(np.mean(vals), 4) if vals else np.nan
    # hard filters -> eligibility
    dof_ok = fd["dof_ratio"] >= 3.0 and fd["T"] > fd["n_params"]
    cond_ok = (not np.isfinite(fd["cond"])) or fd["cond"] < 1e10
    rec["passes_hard_filters"] = bool(fd["stable"] and dof_ok and cond_ok
                                      and (np.isnan(fd["whiteness_p"]) or fd["whiteness_p"] > 0.01))
    records.append(rec)

tab = pd.DataFrame(records)

# rank by avg relative-RMSE RANK across the 4 targets (eligible specs), at h=1
def add_rank(tab, h):
    elig = tab[tab["passes_hard_filters"]].copy()
    for t in TARGETS:
        tab[f"rank_{t}_h{h}"] = tab[f"relRMSE_{t}_h{h}"].rank(method="min")
    rk_cols = [f"rank_{t}_h{h}" for t in TARGETS]
    tab[f"avg_rank_h{h}"] = tab[rk_cols].mean(axis=1)
    return tab
for h in H_LIST:
    tab = add_rank(tab, h)

tab = tab.sort_values("avg_rank_h1").reset_index(drop=True)
print(tab[["model_id", "var_lag_order", "T_over_params", "stability_pass",
           "whiteness_pvalue", "mean_relRMSE_h1", "avg_rank_h1",
           "mean_relRMSE_h4", "passes_hard_filters"]].to_string(index=False))

eligible = tab[tab["passes_hard_filters"]].copy().sort_values("avg_rank_h1")
winner = eligible.iloc[0]["model_id"]
print(f"\n[BASELINE VAR WINNER among eligible] {winner}")

# ----------------------------------------------------------------------------
# 9. SIGNIFICANCE: Clark-West (vs RW), Diebold-Mariano (vs winner), MCS
# ----------------------------------------------------------------------------
print("\n" + "="*78); print("STAGE 5  SIGNIFICANCE TESTS"); print("="*78)

def dm_test(e1, e2, h):
    d = e1**2 - e2**2; n = len(d); dbar = d.mean()
    g0 = np.dot(d - dbar, d - dbar) / n; var = g0
    for L in range(1, h):
        c = np.dot(d[L:] - dbar, d[:-L] - dbar) / n
        var += 2 * c
    var /= n
    if var <= 0: return np.nan, np.nan
    dm = dbar / np.sqrt(var)
    adj = np.sqrt(max((n + 1 - 2*h + h*(h-1)/n) / n, 1e-9))
    dm *= adj
    p = 2 * (1 - stats.t.cdf(abs(dm), df=n - 1))
    return float(dm), float(p)

def clark_west(actual, f_small, f_large, h):
    es = actual - f_small; el = actual - f_large
    fhat = es**2 - (el**2 - (f_small - f_large)**2)
    n = len(fhat); fbar = fhat.mean()
    g0 = np.dot(fhat - fbar, fhat - fbar) / n; var = g0
    for L in range(1, h):
        c = np.dot(fhat[L:] - fbar, fhat[:-L] - fbar) / n
        var += 2 * c
    var /= n
    if var <= 0: return np.nan, np.nan
    cw = fbar / np.sqrt(var)
    return float(cw), float(1 - stats.norm.cdf(cw))   # one-sided

# Clark-West winner vs RW, per target, h=1
print(f"\nClark-West: '{winner}' vs Random Walk (one-sided; small p => VAR beats RW), h=1")
wsp = specs[winner]; wp = oos_store[winner][1]
for t in TARGETS:
    idx = [i for i in wp[t] if np.isfinite(wp[t][i])]
    act = np.array([bench[t][1][i]["actual"] for i in idx])
    frw = np.array([bench[t][1][i]["rw"] for i in idx])
    fvar = np.array([wp[t][i] for i in idx])
    cw, p = clark_west(act, frw, fvar, 1)
    star = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
    print(f"  {t:7s}  CW={cw:6.2f}  p={p:6.3f} {star}")

# Diebold-Mariano: winner vs CORE (non-nested-ish info comparison), per target, h=1
print(f"\nDiebold-Mariano: '{winner}' vs CORE, h=1 (small p => different accuracy)")
cp = oos_store["CORE"][1]
for t in TARGETS:
    idx = [i for i in wp[t] if np.isfinite(wp[t][i]) and np.isfinite(cp[t][i])]
    act = np.array([bench[t][1][i]["actual"] for i in idx])
    e_w = act - np.array([wp[t][i] for i in idx])
    e_c = act - np.array([cp[t][i] for i in idx])
    dm, p = dm_test(e_c, e_w, 1)   # positive dm => winner better
    print(f"  {t:7s}  DM={dm:6.2f}  p={p:6.3f}")

# Model Confidence Set per target (h=1) across all eligible specs + benchmarks
print(f"\nModel Confidence Set (90%) per target, h=1  [included models]:")
mcs_results = {}
try:
    from arch.bootstrap import MCS
    elig_models = list(eligible["model_id"])
    for t in TARGETS:
        # common origins across all eligible models + benchmarks
        common = None
        for m in elig_models:
            ok = {i for i in oos_store[m][1][t] if np.isfinite(oos_store[m][1][t][i])}
            common = ok if common is None else (common & ok)
        common = sorted(common)
        if len(common) < 20:
            print(f"  {t:7s}  (too few common origins)"); continue
        act = np.array([bench[t][1][i]["actual"] for i in common])
        loss = {}
        for m in elig_models:
            loss[m] = (act - np.array([oos_store[m][1][t][i] for i in common]))**2
        loss["RW"] = (act - np.array([bench[t][1][i]["rw"] for i in common]))**2
        loss["AR(BIC)"] = (act - np.array([bench[t][1][i]["ar"] for i in common]))**2
        L = pd.DataFrame(loss)
        mcs = MCS(L, size=0.10, reps=1000, block_size=4, method="R")
        mcs.compute()
        inc = list(mcs.included)
        mcs_results[t] = inc
        print(f"  {t:7s}  {inc}")
except Exception as e:
    print(f"  MCS skipped: {type(e).__name__}: {e}")

# ----------------------------------------------------------------------------
# 10. SAVE TABLE + CHARTS
# ----------------------------------------------------------------------------
col_order = (["model_id", "endogenous_set", "exogenous_set", "var_lag_order",
              "n_params", "T_train", "T_over_params", "stability_pass",
              "max_eig_modulus", "whiteness_pvalue", "cond_number",
              "AIC", "BIC", "HQIC"]
             + [f"relRMSE_{t}_h1" for t in TARGETS] + ["mean_relRMSE_h1", "avg_rank_h1"]
             + [f"relRMSE_{t}_h4" for t in TARGETS] + ["mean_relRMSE_h4", "avg_rank_h4"]
             + ["passes_hard_filters"])
tab_out = tab[[c for c in col_order if c in tab.columns]]
csv_path = OUT / "var_horserace_results_augmented.csv"
tab_out.to_csv(csv_path, index=False)
print(f"\n[saved] {csv_path}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# chart 1: relRMSE vs RW heatmap (specs x targets, h=1)
fig, ax = plt.subplots(figsize=(8, 0.5 * len(tab) + 1.5))
M = tab.set_index("model_id")[[f"relRMSE_{t}_h1" for t in TARGETS]].astype(float)
M.columns = TARGETS
im = ax.imshow(M.values, aspect="auto", cmap="RdYlGn_r", vmin=0.7, vmax=1.3)
ax.set_xticks(range(len(TARGETS))); ax.set_xticklabels(TARGETS)
ax.set_yticks(range(len(M))); ax.set_yticklabels(M.index, fontsize=8)
for (i, j), v in np.ndenumerate(M.values):
    if np.isfinite(v): ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
ax.set_title("Relative RMSE vs Random Walk (h=1)\n<1 beats RW (green), >1 worse (red)", fontsize=10)
plt.colorbar(im, ax=ax, shrink=0.6, label="RMSE / RW")
plt.tight_layout(); plt.savefig(FIG / "relrmse_heatmap.png", dpi=140, bbox_inches="tight"); plt.close()

# chart 2: winner forecast vs actual, 4 targets, h=1
fig, axes = plt.subplots(2, 2, figsize=(13, 7))
for ax, t in zip(axes.ravel(), TARGETS):
    idx = sorted(i for i in wp[t] if np.isfinite(wp[t][i]))
    dts = [panel.index[i + 1] for i in idx]
    ax.plot(dts, [bench[t][1][i]["actual"] for i in idx], "k-", lw=1.4, label="actual")
    ax.plot(dts, [wp[t][i] for i in idx], "C0--", lw=1.2, label=winner)
    ax.plot(dts, [bench[t][1][i]["rw"] for i in idx], "C3:", lw=1.0, label="RW")
    ax.set_title(f"{t}  (h=1 OOS)", fontsize=10); ax.legend(fontsize=7)
fig.suptitle(f"Out-of-sample 1Q forecasts: winner [{winner}] vs RW", fontsize=12, fontweight="bold")
plt.tight_layout(); plt.savefig(FIG / "winner_forecasts.png", dpi=140, bbox_inches="tight"); plt.close()

for f in ["relrmse_heatmap.png", "winner_forecasts.png"]:
    (OUT / f.replace(".png", "_augmented.png")).write_bytes((FIG / f).read_bytes())
print(f"[saved] {OUT/'relrmse_heatmap_augmented.png'}\n[saved] {OUT/'winner_forecasts_augmented.png'}")

# machine-readable summary
summary = {
    "sample": f"{panel.index.min().date()} to {panel.index.max().date()}",
    "n_quarters": int(N), "oos_origins_from": str(panel.index[i0].date()),
    "forward_selection_path": [l for l, _ in sel_log],
    "forward_selection_winner": sel_log[-1][0],
    "eligible_winner_by_avg_rank": winner,
    "winner_spec": {"endog": specs[winner]["endog"], "exog": specs[winner]["exog"]},
    "mcs_included_h1": mcs_results,
}
(OUT / "var_horserace_summary_augmented.json").write_text(json.dumps(summary, indent=2))
print(f"[saved] {OUT/'var_horserace_summary_augmented.json'}")
print("\nDONE.")
