"""
fetch_g7.py
Assemble comparable quarterly macro datasets for G7 economies from FRED's public
CSV endpoint (no API key). Each country gets five endogenous series matching the
U.S. study -- GDP growth (YoY), CPI inflation (YoY), REER (YoY), 10y govt yield,
short/policy rate -- plus a single GLOBAL oil-price surprise as the common
exogenous shock (oil is genuinely exogenous to each country and was the most
useful shock in the U.S. analysis, so it is the natural panel choice).
"""
import pandas as pd, numpy as np

FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

SERIES = {            # country -> (real GDP, CPI, 10y yield, 3m rate, REER)
 "US": ("GDPC1",          "CPIAUCSL",        "IRLTLT01USM156N", "IR3TIB01USM156N", "RBUSBIS"),
 "UK": ("NGDPRSAXDCGBQ",  "GBRCPIALLMINMEI", "IRLTLT01GBM156N", "IR3TIB01GBM156N", "RBGBBIS"),
 "CA": ("NGDPRSAXDCCAQ",  "CANCPIALLMINMEI", "IRLTLT01CAM156N", "IR3TIB01CAM156N", "RBCABIS"),
 "DE": ("CLVMNACSCAB1GQDE","DEUCPIALLMINMEI","IRLTLT01DEM156N", "IR3TIB01DEM156N", "RBDEBIS"),
 "JP": ("JPNRGDPEXP",     "JPNCPIALLMINMEI", "IRLTLT01JPM156N", "IR3TIB01JPM156N", "RBJPBIS"),
}
OIL = "WTISPLC"


def _grab(sid):
    d = pd.read_csv(FRED.format(sid))
    d.columns = ["date", "v"]
    d["date"] = pd.to_datetime(d["date"]); d["v"] = pd.to_numeric(d["v"], errors="coerce")
    return d.dropna().set_index("date")["v"]


def _to_q(s, how="mean"):
    g = s.groupby(s.index.to_period("Q"))
    return g.mean() if how == "mean" else g.last()


def oil_shock(split="2021Q4"):
    """Global oil-price AR(1) surprise, z-scored, train-frozen (no leakage)."""
    from statsmodels.tsa.ar_model import AutoReg
    o = _to_q(_grab(OIL))
    g = 100 * np.log(o.replace(0, np.nan)).diff()
    g = g.dropna()
    tr = g.loc[:pd.Period(split, "Q")]
    res = AutoReg(tr, lags=1, old_names=False).fit()
    c, phi = res.params.iloc[0], res.params.iloc[1]
    resid = g - (c + phi * g.shift(1))
    rtr = resid.loc[:pd.Period(split, "Q")].dropna()
    z = (resid - rtr.mean()) / rtr.std()
    z.name = "shock_oil"
    return z


def build_country(cc, start="2000Q1", end="2025Q1"):
    """Return (df_endog quarterly, available range) for country code cc."""
    gdp_id, cpi_id, y10_id, m3_id, reer_id = SERIES[cc]
    gdp = _to_q(_grab(gdp_id))
    cpi = _to_q(_grab(cpi_id))
    y10 = _to_q(_grab(y10_id))
    m3  = _to_q(_grab(m3_id))
    reer = _to_q(_grab(reer_id))
    df = pd.DataFrame({
        "gdp_yoy":       100 * (gdp / gdp.shift(4) - 1),
        "inflation_yoy": 100 * (cpi / cpi.shift(4) - 1),
        "reer_yoy":      100 * (reer / reer.shift(4) - 1),
        "y10":           y10,
        "policy":        m3,
    })
    df = df.loc[pd.Period(start, "Q"):pd.Period(end, "Q")].dropna()
    df.index = df.index.to_timestamp(how="end")
    return df


if __name__ == "__main__":
    osh = oil_shock()
    print("oil shock:", osh.dropna().index.min(), "->", osh.dropna().index.max(), "n=", osh.notna().sum())
    for cc in SERIES:
        df = build_country(cc)
        print(f"{cc}: {df.shape[0]} quarters  {df.index.min().date()}..{df.index.max().date()}  "
              f"cols ok={list(df.columns)==['gdp_yoy','inflation_yoy','reer_yoy','y10','policy']}")
