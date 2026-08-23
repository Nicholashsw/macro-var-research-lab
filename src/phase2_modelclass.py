"""
phase2_modelclass.py
====================
Model-class horse race on the Phase-1 winning specification.

Estimators (all VAR(p=2), iterated multistep, oil/tfp exog projected by AR-BIC):
  OLS-VAR, BVAR-Minnesota (tight & loose), Ridge-VAR, LASSO-VAR, ElasticNet-VAR
Information sets:
  small = {GDP, INF, POLICY, FX} + OIL exog     (the Phase-1 winner CORE+OIL(x))
  wide  = small + {INV,CONS,FISCAL,EXPORTS,WEALTH,HOUSING} + {OIL,TFP} exog
Benchmarks: RW, AR(BIC).

Fairness: identical 1994Q2-2025Q4 sample, expanding-window OOS, lag p=2,
relRMSE vs RW per target, one joint Model Confidence Set (Hansen-Lunde-Nason).
Penalties tuned by TimeSeriesSplit CV INSIDE each training window (no leakage).
BVAR Minnesota via dummy observations (Banbura-Giannone-Reichlin 2010); point
forecast = posterior mean = OLS on data augmented with prior dummies.
Research/education. Not investment advice.
"""
import os
import warnings, json; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit

# repo-relative paths; override the root with REPRO_ROOT if you run from elsewhere.
# fred_data/ and outputs/ are gitignored, so a re-run never clobbers committed artifacts.
ROOT = Path(os.environ.get("REPRO_ROOT", Path(__file__).resolve().parent.parent))
DATA = ROOT / "fred_data"; DATA.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "outputs"; OUT.mkdir(parents=True, exist_ok=True)
FIG = OUT / "figs"; FIG.mkdir(parents=True, exist_ok=True)
P = 2  # VAR lag order, fixed at Phase-1-selected value for every model class

# ---------- design / estimator helpers (importable without running) ----------
def build_design(endog, exog, p):
    T, n = endog.shape; m = exog.shape[1] if exog is not None else 0
    Y, X = [], []
    for t in range(p, T):
        xr = []
        for l in range(1, p+1): xr += list(endog[t-l])
        if m: xr += list(exog[t])
        xr += [1.0]
        Y.append(endog[t]); X.append(xr)
    return np.array(Y), np.array(X), n, m

def ar_resid_std(y, p):
    y = np.asarray(y, float); n = len(y)
    if n - p < p + 3: return np.std(y) or 1.0
    Y = y[p:]; X = np.column_stack([np.ones(n-p)] + [y[p-1-j:n-1-j] for j in range(p)])
    b, *_ = np.linalg.lstsq(X, Y, rcond=None); r = Y - X @ b
    return float(np.std(r)) or 1.0

def bvar_B(endog, exog, p, lam, delta):
    """Minnesota posterior-mean coefficient matrix via dummy observations.
    delta: array len n, prior mean of own first lag (1 level, 0 growth).
    Flat prior on constant and exogenous (no dummies for them)."""
    Y, X, n, m = build_design(endog, exog, p)
    k = X.shape[1]
    sig = np.array([ar_resid_std(endog[:, i], p) for i in range(n)])
    # lag-coefficient dummies: n*p rows
    J = np.diag(np.arange(1, p+1).astype(float))           # lag decay
    Xd_lags = np.kron(J, np.diag(sig)) / lam               # (n*p x n*p)
    Xd_lags = np.hstack([Xd_lags, np.zeros((n*p, m+1))])   # pad exog+const cols
    Yd_lags = np.zeros((n*p, n))
    Yd_lags[:n, :] = np.diag(delta * sig) / lam            # only lag-1 block nonzero
    # covariance-scale dummies: n rows (x=0, so don't affect posterior mean of B)
    Xd_cov = np.zeros((n, k)); Yd_cov = np.diag(sig)
    Xs = np.vstack([X, Xd_lags, Xd_cov]); Ys = np.vstack([Y, Yd_lags, Yd_cov])
    B, *_ = np.linalg.lstsq(Xs, Ys, rcond=None)
    return B, n, m

def iterate_B(B, endog, exog_future, p, n, m, h):
    hist = [endog[-p+i] for i in range(p)]  # last p rows
    out = []
    for s in range(h):
        xr = []
        for l in range(1, p+1): xr += list(hist[-l])
        if m: xr += list(exog_future[s])
        xr += [1.0]
        yn = np.array(xr) @ B; out.append(yn); hist.append(yn)
    return np.array(out)

def penalized_state(endog, exog, p, kind):
    Y, X, n, m = build_design(endog, exog, p)
    Xnc = X[:, :-1]                                  # drop const (sklearn intercept)
    mu = Xnc.mean(0); sd = Xnc.std(0); sd[sd == 0] = 1.0
    Xs = (Xnc - mu) / sd
    tscv = TimeSeriesSplit(n_splits=3)
    models = []
    for kcol in range(n):
        yk = Y[:, kcol]
        if kind == "ridge":
            mdl = RidgeCV(alphas=np.logspace(-2, 3, 12), cv=tscv)
        elif kind == "lasso":
            mdl = LassoCV(n_alphas=12, cv=tscv, max_iter=3000, n_jobs=1)
        else:  # enet
            mdl = ElasticNetCV(l1_ratio=0.5, n_alphas=12, cv=tscv, max_iter=3000, n_jobs=1)
        mdl.fit(Xs, yk); models.append(mdl)
    return {"models": models, "mu": mu, "sd": sd, "p": p, "n": n, "m": m}

def iterate_pen(state, endog, exog_future, h):
    p, n, m = state["p"], state["n"], state["m"]
    hist = [endog[-p+i] for i in range(p)]
    out = []
    for s in range(h):
        xr = []
        for l in range(1, p+1): xr += list(hist[-l])
        if m: xr += list(exog_future[s])
        x = (np.array(xr) - state["mu"]) / state["sd"]
        yn = np.array([mdl.predict(x.reshape(1, -1))[0] for mdl in state["models"]])
        out.append(yn); hist.append(yn)
    return np.array(out)

# ============================== main run =====================================
if __name__ == "__main__":
    CORE_COMMON = {"GDP": ("GDPC1","growth"), "INF": ("PCEPILFE","growth"),
                   "POLICY": ("FEDFUNDS","level"), "FX": ("RBUSBIS","growth")}
    EXTRA = {"INV":("GPDIC1","growth"), "CONS":("PCECC96","growth"), "FISCAL":("GCEC1","growth"),
             "EXPORTS":("EXPGSC1","growth"), "WEALTH":("TNWBSHNO","growth"), "HOUSING":("USSTHPI","growth"),
             "OIL":("WTISPLC","oilret"), "TFP":("FERNALD_TFPUTIL","level")}
    TARGETS = ["GDP","INF","POLICY","FX"]
    H_LIST = [1, 4]; OOS_START_FRAC = 0.55; MAXLAG_AR = 8

    def load_q(sid):
        df=pd.read_csv(DATA/f"{sid}.csv"); df.columns=["date","value"]
        df["date"]=pd.to_datetime(df["date"]); df["value"]=pd.to_numeric(df["value"],errors="coerce")
        return df.dropna().set_index("date")["value"].sort_index().resample("QS").mean()
    def transform(lv,k):
        return lv if k=="level" else (400.0*np.log(lv).diff() if k=="growth" else 100.0*np.log(lv).diff())

    modeled={}; tmap={}
    for nm,(sid,k) in {**CORE_COMMON, **EXTRA}.items():
        modeled[nm]=transform(load_q(sid),k); tmap[nm]=k
    panel=pd.DataFrame(modeled).dropna().loc["1994-04-01":]
    N=len(panel); i0=int(N*OOS_START_FRAC)
    delta_of=lambda names: np.array([1.0 if tmap[x]=="level" else 0.0 for x in names])
    print(f"[panel] {N} q  {panel.index.min().date()}->{panel.index.max().date()}  oos from {panel.index[i0].date()}\n")

    def ar_bic_path(y,h,maxlag=MAXLAG_AR):
        y=np.asarray(y,float); n=len(y); best=(np.inf,1,None)
        for p in range(1,maxlag+1):
            if n-p<p+5: break
            Y=y[p:]; X=np.column_stack([np.ones(n-p)]+[y[p-1-j:n-1-j] for j in range(p)])
            b,*_=np.linalg.lstsq(X,Y,rcond=None); r=Y-X@b; s2=max(r@r/len(Y),1e-12)
            bic=len(Y)*np.log(s2)+(p+1)*np.log(len(Y))
            if bic<best[0]: best=(bic,p,b)
        _,p,b=best; hist=list(y[-p:]); out=[]
        for _ in range(h):
            x=np.array([1.0]+[hist[-1-j] for j in range(p)]); v=float(x@b); out.append(v); hist.append(v)
        return np.array(out)

    bench={t:{h:{} for h in H_LIST} for t in TARGETS}
    for t in TARGETS:
        y=panel[t].values
        for h in H_LIST:
            for i in range(i0,N-h):
                yt=y[:i+1]; bench[t][h][i]={"rw":yt[-1],"ar":ar_bic_path(yt,h)[-1],"actual":y[i+h]}

    SMALL=["GDP","INF","POLICY","FX"]; SMALL_X=["OIL"]
    WIDE=SMALL+["INV","CONS","FISCAL","EXPORTS","WEALTH","HOUSING"]; WIDE_X=["OIL","TFP"]
    SPECS=[
        ("OLS:small","ols",SMALL,SMALL_X,{}),
        ("BVAR-Tight:small","bvar",SMALL,SMALL_X,{"lam":0.1}),
        ("BVAR-Loose:small","bvar",SMALL,SMALL_X,{"lam":0.5}),
        ("Ridge:small","ridge",SMALL,SMALL_X,{}),
        ("LASSO:small","lasso",SMALL,SMALL_X,{}),
        ("ENet:small","enet",SMALL,SMALL_X,{}),
        ("OLS:wide(overfit)","ols",WIDE,WIDE_X,{}),
        ("BVAR-Tight:wide","bvar",WIDE,WIDE_X,{"lam":0.1}),
        ("Ridge:wide","ridge",WIDE,WIDE_X,{}),
        ("LASSO:wide","lasso",WIDE,WIDE_X,{}),
        ("ENet:wide","enet",WIDE,WIDE_X,{}),
    ]

    h_max=max(H_LIST)
    def run_spec(family,endog_names,exog_names,hp):
        store={h:{} for h in H_LIST}
        delta=delta_of(endog_names); tgt_idx={t:endog_names.index(t) for t in TARGETS}
        for i in range(i0,N-h_max):
            en=panel[endog_names].values[:i+1]
            ex=panel[exog_names].values[:i+1] if exog_names else None
            exf={h:np.column_stack([ar_bic_path(panel[e].values[:i+1],h) for e in exog_names]) if exog_names
                   else None for h in H_LIST}
            try:
                if family=="ols":
                    Y,X,n,m=build_design(en,ex,P); B,*_=np.linalg.lstsq(X,Y,rcond=None)
                    fc={h:iterate_B(B,en,exf[h],P,n,m,h)[-1] for h in H_LIST}
                elif family=="bvar":
                    B,n,m=bvar_B(en,ex,P,hp["lam"],delta)
                    fc={h:iterate_B(B,en,exf[h],P,n,m,h)[-1] for h in H_LIST}
                else:
                    st=penalized_state(en,ex,P,family)
                    fc={h:iterate_pen(st,en,exf[h],h)[-1] for h in H_LIST}
                for h in H_LIST:
                    row=fc[h]
                    for t in TARGETS: store[h].setdefault(t,{})[i]=row[tgt_idx[t]]
            except Exception:
                for h in H_LIST:
                    for t in TARGETS: store[h].setdefault(t,{})[i]=np.nan
        return store

    results={}
    for label,fam,en,ex,hp in SPECS:
        print(f"  running {label} ...", flush=True)
        results[label]=run_spec(fam,en,ex,hp)

    def relrmse(store,h,t):
        idx=[i for i in store[h].get(t,{}) if np.isfinite(store[h][t][i])]
        if len(idx)<8: return np.nan
        e=np.array([bench[t][h][i]["actual"]-store[h][t][i] for i in idx])
        er=np.array([bench[t][h][i]["actual"]-bench[t][h][i]["rw"] for i in idx])
        return float(np.sqrt((e**2).mean())/np.sqrt((er**2).mean()))

    rows=[]
    for label,fam,en,ex,hp in SPECS:
        rec={"model_id":label,"family":fam,"info_set":("wide" if "wide" in label else "small"),
             "n_endog":len(en)}
        for h in H_LIST:
            for t in TARGETS: rec[f"relRMSE_{t}_h{h}"]=round(relrmse(results[label],h,t),4)
            vals=[relrmse(results[label],h,t) for t in TARGETS]
            rec[f"mean_relRMSE_h{h}"]=round(np.nanmean(vals),4)
        rows.append(rec)
    # benchmarks as rows (AR(BIC) vs RW; RW=1 by definition)
    for label in ["AR(BIC)","RW"]:
        rec={"model_id":label,"family":"benchmark","info_set":"-","n_endog":1}
        for h in H_LIST:
            for t in TARGETS:
                if label=="RW": v=1.0
                else:
                    idx=list(bench[t][h]); 
                    e=np.array([bench[t][h][i]["actual"]-bench[t][h][i]["ar"] for i in idx])
                    er=np.array([bench[t][h][i]["actual"]-bench[t][h][i]["rw"] for i in idx])
                    v=float(np.sqrt((e**2).mean())/np.sqrt((er**2).mean()))
                rec[f"relRMSE_{t}_h{h}"]=round(v,4)
            rec[f"mean_relRMSE_h{h}"]=round(np.nanmean([rec[f"relRMSE_{t}_h{h}"] for t in TARGETS]),4)
        rows.append(rec)
    tab=pd.DataFrame(rows).sort_values("mean_relRMSE_h1").reset_index(drop=True)

    print("\n"+"="*100); print("PHASE 2 RESULTS  (relRMSE vs RW; <1 beats RW)"); print("="*100)
    show=["model_id","family","info_set"]+[f"relRMSE_{t}_h1" for t in TARGETS]+["mean_relRMSE_h1","mean_relRMSE_h4"]
    print(tab[show].to_string(index=False))

    # joint MCS per target (h=1), all models with finite losses + benchmarks
    print("\nJoint Model Confidence Set (90%) per target, h=1 [included]:")
    mcs_inc={}
    try:
        from arch.bootstrap import MCS
        model_labels=[s[0] for s in SPECS]
        for t in TARGETS:
            common=None
            for m in model_labels:
                ok={i for i in results[m][1].get(t,{}) if np.isfinite(results[m][1][t][i])}
                common=ok if common is None else common&ok
            common=sorted(common)
            act=np.array([bench[t][1][i]["actual"] for i in common])
            L={}
            for m in model_labels:
                arr=np.array([results[m][1][t][i] for i in common]); 
                if np.all(np.isfinite(arr)): L[m]=(act-arr)**2
            L["RW"]=(act-np.array([bench[t][1][i]["rw"] for i in common]))**2
            L["AR(BIC)"]=(act-np.array([bench[t][1][i]["ar"] for i in common]))**2
            Ldf=pd.DataFrame(L)
            # drop exploding columns so MCS scaling is sane (keep for table, not MCS)
            keep=[c for c in Ldf.columns if Ldf[c].mean()<25*Ldf["RW"].mean()]
            mcs=MCS(Ldf[keep],size=0.10,reps=1000,block_size=4,method="R"); mcs.compute()
            mcs_inc[t]=list(mcs.included)
            dropped=[c for c in Ldf.columns if c not in keep]
            print(f"  {t:7s} {sorted(mcs_inc[t])}" + (f"   (excluded as blown-up: {dropped})" if dropped else ""))
    except Exception as e:
        print(f"  MCS skipped: {type(e).__name__}: {e}")

    tab.to_csv(OUT/"phase2_modelclass_results.csv",index=False)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    M=tab[tab["family"]!="benchmark"].set_index("model_id")[[f"relRMSE_{t}_h1" for t in TARGETS]].astype(float)
    M.columns=TARGETS
    fig,ax=plt.subplots(figsize=(7.5,0.5*len(M)+1.4))
    im=ax.imshow(M.values,aspect="auto",cmap="RdYlGn_r",vmin=0.6,vmax=1.6)
    ax.set_xticks(range(len(TARGETS))); ax.set_xticklabels(TARGETS)
    ax.set_yticks(range(len(M))); ax.set_yticklabels(M.index,fontsize=8)
    for (i,j),v in np.ndenumerate(M.values):
        if np.isfinite(v): ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=7)
    ax.set_title("Phase 2: relative RMSE vs RW (h=1)\nmodel class x information set",fontsize=10)
    plt.colorbar(im,ax=ax,shrink=0.6,label="RMSE / RW"); plt.tight_layout()
    plt.savefig(FIG/"phase2_heatmap.png",dpi=140,bbox_inches="tight"); plt.close()
    (OUT/"phase2_heatmap.png").write_bytes((FIG/"phase2_heatmap.png").read_bytes())
    (OUT/"phase2_summary.json").write_text(json.dumps(
        {"sample":f"{panel.index.min().date()} to {panel.index.max().date()}","n":N,
         "best_mean_relRMSE_h1":tab.iloc[0]["model_id"],"mcs_h1":mcs_inc},indent=2,default=str))
    print(f"\n[saved] {OUT/'phase2_modelclass_results.csv'}\n[saved] {OUT/'phase2_heatmap.png'}\n[saved] {OUT/'phase2_summary.json'}\nDONE.")
