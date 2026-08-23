"""
shock_builder.py
Rebuild the ExoVar exogenous shocks WITHOUT look-ahead leakage.

ExoVar's leakage: the AR(1) is fit on the full sample and the z-score uses
full-sample mean/std, so 2022-2025 information bleeds into the standardisation.

Three modes (same growth/diff transforms as ExoVar; only the surprise + scaling change):
  'fullsample'   : original ExoVar behaviour (for sanity-checking the reproduction)
  'train_frozen' : AR(1) coeffs and z-score mean/std estimated on TRAIN only,
                   then applied out-of-sample with those frozen parameters
  'rolling'      : real-time. surprise from an expanding AR(1) using only data up
                   to t; standardised by a trailing W-quarter rolling mean/std
"""
import numpy as np, pandas as pd
from statsmodels.tsa.ar_model import AutoReg

# ---- raw files on disk (underscored names) ----
FILES = {
    "oil":         ("Crude_Oil_Spot_Price_West_Texas_Intermediate_Cushing__oil_.xlsx", "ar"),
    "wealth":      ("US_Composite_Index_SampP_100__wealth_shock_.xlsx", "ar"),
    "consumption": ("Personal_Consumption_Expenditure_PCE_Quantity_Index_QI_sa__consumption_shock_.xlsx", "ar"),
    "investment":  ("Private_Fixed_Investment_PFI_2017p_saar__investment_shock_.xlsx", "ar"),
    "credit":      ("Debt_Outs_saar_Domestic_Nonfinancial_Sectors_DN__credit_impulse_.xlsx", "diff"),
}
FISCAL_CSV = "cyclically_adjusted_primary_balance__fiscal_impulse_.csv"

# ---------- ExoVar's robust reader / transforms (verbatim logic) ----------
def _pick_date_col(raw):
    best_col, best_score = None, -1
    for col in raw.columns:
        ser = raw[col]
        num_share = pd.to_numeric(ser, errors="coerce").notna().mean()
        ts_share  = ser.apply(lambda x: isinstance(x, pd.Timestamp)).mean()
        str_share = ser.apply(lambda x: isinstance(x, str)).mean()
        parsed = pd.to_datetime(ser, errors="coerce", dayfirst=False)
        valid = parsed.notna()
        if valid.sum() < 20:
            continue
        yrs = parsed[valid].dt.year
        if (yrs.between(1900, 2100)).mean() < 0.95:
            continue
        pv = parsed[valid].sort_values()
        mono = (pv.diff().dropna() >= pd.Timedelta(0)).mean() if len(pv) > 1 else 0
        score = valid.sum() * (0.5 + 0.5 * mono) * (1 + 2 * ts_share + 1 * str_share) * (1 - 0.9 * num_share)
        if score > best_score:
            best_score, best_col = score, col
    return best_col

def read_myseries_excel(path):
    xl = pd.ExcelFile(path); best = None
    for sheet in xl.sheet_names:
        raw = xl.parse(sheet, header=None)
        dc = _pick_date_col(raw)
        if dc is None:
            continue
        dates = pd.to_datetime(raw[dc], errors="coerce")
        start = dates.first_valid_index()
        cand = [c for c in raw.columns if c != dc]
        best_v, best_n = None, -1
        for c in cand:
            vals = pd.to_numeric(raw.loc[start:, c], errors="coerce")
            if vals.notna().sum() > best_n:
                best_n, best_v = vals.notna().sum(), c
        vals = pd.to_numeric(raw.loc[start:, best_v], errors="coerce")
        out = pd.DataFrame({"date": dates.loc[start:], "value": vals}).dropna().sort_values("date")
        out = out[~out["date"].duplicated()]
        s = out.set_index("date")["value"]
        if best is None or len(s) > len(best):
            best = s
    if best is None:
        raise ValueError(f"no date/value in {path}")
    return best

def to_quarterly_mean(s):
    s = s.sort_index()
    q = s.groupby(s.index.to_period("Q")).mean()
    q.index = q.index.astype("period[Q]")
    return q

def qoq_logdiff(q):
    q = q.replace(0, np.nan)
    return 100 * np.log(q).diff()

def fiscal_q_diff(path):
    f = pd.read_csv(path, header=None)
    years = pd.to_numeric(f.iloc[0, 7:], errors="coerce").dropna().astype(int)
    vals  = pd.to_numeric(f.iloc[1, 7:], errors="coerce")
    ann = pd.Series(vals.values[:len(years)], index=years.values)
    q = pd.Series([ann[y] for y in ann.index for _ in range(4)],
                  index=pd.period_range(f"{ann.index.min()}Q1", f"{ann.index.max()}Q4", freq="Q"))
    return q.diff()

# ---------- leak-free surprise + scaling ----------
def _ar1_resid_frozen(x, split):
    """AR(1) residual with coeffs estimated on x[:split] only, applied to all t."""
    x = x.dropna()
    tr = x.loc[:split]
    if len(tr) < 12:
        return x - x.shift(1)
    res = AutoReg(tr, lags=1, old_names=False).fit()
    c, phi = res.params.iloc[0], res.params.iloc[1]
    return x - (c + phi * x.shift(1))

def _ar1_resid_rolling(x, win):
    """Expanding/rolling-window AR(1) surprise: residual uses only data up to t-1."""
    x = x.dropna()
    out = pd.Series(index=x.index, dtype=float)
    vals = x.values
    for i in range(2, len(x)):
        lo = max(0, i - win)
        y = vals[lo:i]
        if len(y) < 8:
            continue
        # AR(1) by OLS on the window: y_t = c + phi y_{t-1}
        Y, X = y[1:], np.column_stack([np.ones(len(y) - 1), y[:-1]])
        try:
            beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        except Exception:
            continue
        out.iloc[i] = vals[i] - (beta[0] + beta[1] * vals[i - 1])
    return out

def _z_frozen(s, split, ddof=0):
    tr = s.loc[:split].dropna()
    return (s - tr.mean()) / tr.std(ddof=ddof)

def _z_rolling(s, win):
    m = s.rolling(win, min_periods=8).mean()
    sd = s.rolling(win, min_periods=8).std(ddof=0)
    return (s - m) / sd

def build_shocks(mode="train_frozen", split="2021Q4", win=20, base="."):
    """Return a quarterly DataFrame of the six shocks for the chosen mode."""
    split = pd.Period(split, "Q")
    cols = {}
    for name, (fname, kind) in FILES.items():
        raw = read_myseries_excel(f"{base}/{fname}")
        q = to_quarterly_mean(raw)
        series = qoq_logdiff(q) if kind == "ar" else q.diff()
        if kind == "ar":
            if mode == "fullsample":
                resid = AutoReg(series.dropna(), lags=1, old_names=False).fit().resid
            elif mode == "train_frozen":
                resid = _ar1_resid_frozen(series, split)
            else:  # rolling
                resid = _ar1_resid_rolling(series, win)
        else:
            resid = series  # credit/fiscal: the differenced series is the "impulse"
        # standardise
        if mode == "fullsample":
            z = (resid - resid.mean()) / resid.std(ddof=0)
        elif mode == "train_frozen":
            z = _z_frozen(resid, split)
        else:
            z = _z_rolling(resid, win)
        cols[f"shock_{name}"] = z

    # fiscal
    fq = fiscal_q_diff(f"{base}/{FISCAL_CSV}")
    if mode == "fullsample":
        cols["shock_fiscal"] = (fq - fq.mean()) / fq.std(ddof=0)
    elif mode == "train_frozen":
        cols["shock_fiscal"] = _z_frozen(fq, split)
    else:
        cols["shock_fiscal"] = _z_rolling(fq, win)

    df = pd.concat(cols, axis=1)
    df.index = df.index.astype("period[Q]")
    order = ["shock_oil", "shock_wealth", "shock_consumption", "shock_investment", "shock_credit", "shock_fiscal"]
    return df[order].sort_index()


if __name__ == "__main__":
    import os, pathlib
    # Spreadsheet exports are not redistributed; see DATA.md for the series and layout.
    # Default location is <study>/data/; override with BVAR_DATA_DIR.
    base = os.environ.get(
        "BVAR_DATA_DIR",
        str(pathlib.Path(__file__).resolve().parent.parent / "data"),
    )
    full = build_shocks("fullsample", base=base)
    # sanity: does fullsample reproduce the shipped shocks?
    shipped = pd.read_excel(f"{base}/main_var_with_shocks.xlsx", sheet_name="Sheet1")
    shipped = shipped.drop(columns=[c for c in shipped.columns if str(c).startswith("Unnamed")])
    shipped["q"] = pd.to_datetime(shipped["Date"]).dt.to_period("Q")
    shipped = shipped.set_index("q")
    print("Correlation of rebuilt 'fullsample' vs shipped shocks (should be ~1.0):")
    for c in full.columns:
        a = full[c].reindex(shipped.index); b = shipped[c]
        m = a.notna() & b.notna()
        if m.sum() > 5:
            print(f"  {c:20s} corr={np.corrcoef(a[m], b[m])[0,1]:.3f}  (n={m.sum()})")
