"""
demo_hp_lookahead.py
Quantify the HP-filter look-ahead problem on the US fed funds rate:
  - full-sample (two-sided) HP cycle  vs  real-time (one-sided, recursive) HP cycle
  - endpoint revision: how much the recent-quarter "gap" changes as data arrives
  - Hamilton (2018) regression filter as the recommended alternative
For research/education. Not investment advice.
"""
import os
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from statsmodels.tsa.filters.hp_filter import hpfilter

# repo-relative paths; override the root with REPRO_ROOT if you run from elsewhere.
# fred_data/ and outputs/ are gitignored, so a re-run never clobbers committed artifacts.
ROOT = Path(os.environ.get("REPRO_ROOT", Path(__file__).resolve().parent.parent))
DATA = ROOT / "fred_data"; DATA.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "outputs"; OUT.mkdir(parents=True, exist_ok=True)
FIG = OUT / "figs"; FIG.mkdir(parents=True, exist_ok=True)

def load_q(sid):
    df = pd.read_csv(DATA/f"{sid}.csv"); df.columns=["date","value"]
    df["date"]=pd.to_datetime(df["date"]); df["value"]=pd.to_numeric(df["value"],errors="coerce")
    return df.dropna().set_index("date")["value"].sort_index().resample("QS").mean()

ff = load_q("FEDFUNDS").loc["1994-01-01":]
LAMB = 1600

# two-sided (full-sample) HP cycle -- what you get if you filter the whole series once
cyc_full, _ = hpfilter(ff, lamb=LAMB)

# one-sided / real-time: at each quarter t, refit HP on data up to t, keep the last cycle value
rt = {}
for i in range(20, len(ff)+1):
    c, _ = hpfilter(ff.iloc[:i], lamb=LAMB)
    rt[ff.index[i-1]] = c.iloc[-1]
cyc_rt = pd.Series(rt)

df = pd.DataFrame({"ff": ff, "two_sided": cyc_full, "one_sided": cyc_rt}).dropna()
df["revision"] = df["two_sided"] - df["one_sided"]

# Hamilton (2018) regression filter: y_t = b0 + sum_{j=0..p-1} b_j y_{t-h-j} + e_t ; cycle = resid
h, p = 8, 4
X = pd.concat([ff.shift(h+j) for j in range(p)], axis=1); X.columns=[f"l{h+j}" for j in range(p)]
dat = pd.concat([ff.rename("y"), X], axis=1).dropna()
A = np.column_stack([np.ones(len(dat))] + [dat[c].values for c in X.columns])
beta, *_ = np.linalg.lstsq(A, dat["y"].values, rcond=None)
ham = pd.Series(dat["y"].values - A @ beta, index=dat.index)

corr   = df["two_sided"].corr(df["one_sided"])
mar    = df["revision"].abs().mean()
mar8   = df["revision"].abs().tail(8).mean()
maxrev = df["revision"].abs().max(); maxdt = df["revision"].abs().idxmax()
sd_cyc = df["two_sided"].std()

print("="*70); print("HP-FILTER LOOK-AHEAD ON US FED FUNDS  (lambda=1600, 1994Q1+)"); print("="*70)
print(f"  full-sample cycle std dev ................ {sd_cyc:.3f} pp")
print(f"  corr(two-sided, real-time one-sided) ..... {corr:.3f}")
print(f"  mean |revision| (all quarters) ........... {mar:.3f} pp")
print(f"  mean |revision| (last 8 quarters) ........ {mar8:.3f} pp")
print(f"  max  |revision| .......................... {maxrev:.3f} pp  ({maxdt.date()})")
print(f"  => last-8-qtr revision is {100*mar8/sd_cyc:.0f}% of the cycle's own std dev\n")
print("  recent quarters: two-sided vs real-time gap (pp)")
print(df[["two_sided","one_sided","revision"]].tail(8).round(3).to_string())

# chart
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(2,1, figsize=(12,7), height_ratios=[2,1], sharex=True)
sub = df.loc["2004-01-01":]
ax[0].axhline(0, color="0.6", lw=0.8)
ax[0].plot(sub.index, sub["two_sided"], "C0-",  lw=2.0, label="two-sided HP cycle (full sample, uses future data)")
ax[0].plot(sub.index, sub["one_sided"], "C3--", lw=1.8, label="one-sided HP cycle (real-time, data up to t only)")
ham_s = ham.loc["2004-01-01":]
ax[0].plot(ham_s.index, ham_s.values, "C2:", lw=1.6, label="Hamilton (2018) filter cycle")
ax[0].set_ylabel("fed funds gap (pp)"); ax[0].legend(fontsize=8, loc="upper left")
ax[0].set_title("Fed funds 'rate gap': two-sided HP leaks the future and is revised heavily near the endpoint",
                fontsize=11, fontweight="bold")
ax[1].bar(sub.index, sub["revision"], width=70, color="C1", alpha=0.8)
ax[1].axhline(0, color="0.6", lw=0.8)
ax[1].set_ylabel("two-sided minus\nreal-time (pp)"); ax[1].set_xlabel("")
ax[1].set_title("Endpoint revision: what the backtest 'knew' that you would not have known in real time", fontsize=9)
plt.tight_layout(); plt.savefig(FIG/"hp_lookahead.png", dpi=140, bbox_inches="tight"); plt.close()
(OUT/"hp_lookahead.png").write_bytes((FIG/"hp_lookahead.png").read_bytes())
print(f"\n[saved] {OUT/'hp_lookahead.png'}")
