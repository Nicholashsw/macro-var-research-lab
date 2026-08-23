"""
reproduce.py  --  single-execution reproducibility driver for
"Disciplined Specification Search and Model-Class Evaluation for
Macroeconomic VAR Forecasting: A Leakage-Free Study on US Quarterly Data"

Runs every phase from one seed on one sample and emits:
  - repro/results.json      : every number cited in the paper, exact
  - outputs/reproduce_results.json : the same payload, next to the figures
  - outputs/figs/fig_*.png  : the three paper figures, regenerated
  - console "PAPER NUMBERS" : labelled values for transcription

Outputs go to gitignored directories (repro/, outputs/, fred_data/) so a re-run
never overwrites the committed reference copies in results/ or the typeset
figures in papers/var_selection_horserace/figs/.

Data: cached FRED CSVs in fred_data/ (frozen vintage 2026-07-10); a curl
fallback re-fetches if a file is missing. Fernald TFP cleaned in
fred_data/FERNALD_TFPUTIL.csv.

For educational and research purposes. Not investment advice.
"""
import os, warnings, json, sys, subprocess, platform
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.filters.hp_filter import hpfilter
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
import statsmodels, sklearn, arch

SEED = 20260710
np.random.seed(SEED)

ROOT = Path(os.environ.get("REPRO_ROOT", Path(__file__).resolve().parent.parent)); DATA = ROOT/"fred_data"
REPRO = ROOT/"repro"; REPRO.mkdir(exist_ok=True)
OUT = ROOT/"outputs"; OUT.mkdir(parents=True, exist_ok=True)
PFIG = OUT/"figs"; PFIG.mkdir(parents=True, exist_ok=True)

P = 2                      # Phase-2 fixed lag (model-class comparison)
MAXLAG_VAR = 4             # baseline/gap/YoY: BIC lag selection up to 4
MAXLAG_AR = 8
H_LIST = [1, 4]
OOS_START_FRAC = 0.55
R = {}                     # results collector -> results.json

# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
SERIES = {"GDPC1","PCEPILFE","FEDFUNDS","RBUSBIS","GPDIC1","PCECC96","WTISPLC",
          "GCEC1","EXPGSC1","TNWBSHNO","USSTHPI","FERNALD_TFPUTIL"}
DATA.mkdir(exist_ok=True)
def build_fernald():
    f=DATA/"FERNALD_TFPUTIL.csv"
    if f.exists(): return
    xl=DATA/"fernald_tfp.xlsx"
    subprocess.run(["curl","-sSL","--max-time","90","-o",str(xl),
        "https://www.frbsf.org/wp-content/uploads/quarterly_tfp.xlsx"],check=True)
    df=pd.read_excel(xl,sheet_name="quarterly",header=1)[["date","dtfp_util"]]
    df=df[df["date"].astype(str).str.match(r"^\d{4}:Q[1-4]$")].copy()
    q={"1":1,"2":4,"3":7,"4":10}
    df["date"]=df["date"].map(lambda s:(lambda y,qq:pd.Timestamp(int(y),q[qq],1))(*str(s).split(":Q")))
    df["dtfp_util"]=pd.to_numeric(df["dtfp_util"],errors="coerce")
    df.dropna().set_index("date").sort_index().to_csv(f,header=["value"])
def ensure(sid):
    if sid=="FERNALD_TFPUTIL": build_fernald(); return
    f=DATA/f"{sid}.csv"
    if not f.exists():
        subprocess.run(["curl","-sS","--max-time","60","-o",str(f),
                        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"], check=True)
for s in SERIES: ensure(s)

def load_q(sid):
    df = pd.read_csv(DATA/f"{sid}.csv"); df.columns=["date","value"]
    df["date"]=pd.to_datetime(df["date"]); df["value"]=pd.to_numeric(df["value"],errors="coerce")
    return df.dropna().set_index("date")["value"].sort_index().resample("QS").mean()

def tf(lv, k):
    if k=="level":  return lv
    if k=="growth": return 400.0*np.log(lv).diff()
    if k=="oilret": return 100.0*np.log(lv).diff()
    if k=="yoy":    return 100.0*(lv/lv.shift(4)-1.0)
    raise ValueError(k)

SPEC = {  # name -> (fred id, level-transform)
    "GDP":("GDPC1","growth"), "INF":("PCEPILFE","growth"), "POLICY":("FEDFUNDS","level"),
    "FX":("RBUSBIS","growth"), "INV":("GPDIC1","growth"), "CONS":("PCECC96","growth"),
    "FISCAL":("GCEC1","growth"), "EXPORTS":("EXPGSC1","growth"), "WEALTH":("TNWBSHNO","growth"),
    "HOUSING":("USSTHPI","growth"), "OIL":("WTISPLC","oilret"), "TFP":("FERNALD_TFPUTIL","level")}
YOY_CORE = {"GDP","INF","FX"}   # YoY panel switches these to yoy; POLICY stays level

raw = {n: load_q(sid) for n,(sid,_) in SPEC.items()}
panelL = pd.DataFrame({n: tf(raw[n], k) for n,(sid,k) in SPEC.items()}).dropna().loc["1994-04-01":]
def yoy_kind(n,k): return "yoy" if n in YOY_CORE else k
panelY = pd.DataFrame({n: tf(raw[n], yoy_kind(n,k)) for n,(sid,k) in SPEC.items()}).dropna().loc["1994-04-01":]
tmap = {n:k for n,(sid,k) in SPEC.items()}

for tagp,pan in [("level",panelL),("yoy",panelY)]:
    R[f"sample_{tagp}"] = {"n":int(len(pan)),"start":str(pan.index.min().date()),
                           "end":str(pan.index.max().date())}
print(f"[panelL] {len(panelL)}q {panelL.index.min().date()}..{panelL.index.max().date()}")
print(f"[panelY] {len(panelY)}q {panelY.index.min().date()}..{panelY.index.max().date()}")

# ----------------------------------------------------------------------------
# shared estimator + evaluation machinery
# ----------------------------------------------------------------------------
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

def make_bench(pan, targets, i0):
    N=len(pan); bench={t:{h:{} for h in H_LIST} for t in targets}
    for t in targets:
        y=pan[t].values
        for h in H_LIST:
            for i in range(i0,N-h):
                yt=y[:i+1]; bench[t][h][i]={"rw":yt[-1],"ar":ar_bic_path(yt,h)[-1],"actual":y[i+h]}
    return bench

def build_design(endog, exog, p):
    T,n=endog.shape; m=exog.shape[1] if exog is not None else 0
    Y,X=[],[]
    for t in range(p,T):
        xr=[]
        for l in range(1,p+1): xr+=list(endog[t-l])
        if m: xr+=list(exog[t])
        xr+=[1.0]; Y.append(endog[t]); X.append(xr)
    return np.array(Y),np.array(X),n,m

def ar_resid_std(y, p):
    y=np.asarray(y,float); n=len(y)
    if n-p<p+3: return float(np.std(y)) or 1.0
    Y=y[p:]; X=np.column_stack([np.ones(n-p)]+[y[p-1-j:n-1-j] for j in range(p)])
    b,*_=np.linalg.lstsq(X,Y,rcond=None); r=Y-X@b
    return float(np.std(r)) or 1.0

def fit_var(endog, exog=None):
    m=VAR(endog,exog=exog); r=m.fit(maxlags=MAXLAG_VAR,ic="bic",trend="c")
    return m.fit(1,trend="c") if r.k_ar==0 else r

def companion_max(res):
    p,k=res.k_ar,res.neqs
    if p==0: return 0.0
    C=np.zeros((k*p,k*p)); C[:k,:]=np.hstack([res.coefs[i] for i in range(p)])
    if p>1: C[k:,:-k]=np.eye(k*(p-1))
    return float(np.max(np.abs(np.linalg.eigvals(C))))

def full_diag(pan,endog,exog):
    en=pan[endog].values; ex=pan[exog].values if exog else None
    res=fit_var(en,ex); k,p=res.neqs,res.k_ar
    npar=k*(k*p+1)+(k*len(exog) if exog else 0); T=res.nobs
    try: stable=bool(res.is_stable(verbose=False))
    except Exception: stable=companion_max(res)<1.0
    try: wh=float(res.test_whiteness(nlags=max(p+4,10),adjusted=True).pvalue)
    except Exception: wh=float("nan")
    return {"var_lag":p,"n_params":int(npar),"T":int(T),"dof":T/npar,"stable":stable,
            "maxmod":companion_max(res),"wh":wh}

def var_oos(pan,endog,exog,i0,targets):
    N=len(pan); store={h:{t:{} for t in targets} for h in H_LIST}
    for i in range(i0,N-max(H_LIST)):
        try:
            en=pan[endog].values[:i+1]
            if exog:
                ex=pan[exog].values[:i+1]; res=fit_var(en,ex)
                for h in H_LIST:
                    ef=np.column_stack([ar_bic_path(pan[e].values[:i+1],h) for e in exog])
                    fc=res.forecast(en[-res.k_ar:],steps=h,exog_future=ef)
                    for t in targets: store[h][t][i]=fc[h-1][endog.index(t)]
            else:
                res=fit_var(en)
                for h in H_LIST:
                    fc=res.forecast(en[-res.k_ar:],steps=h)
                    for t in targets: store[h][t][i]=fc[h-1][endog.index(t)]
        except Exception:
            for h in H_LIST:
                for t in targets: store[h][t][i]=np.nan
    return store

def relrmse(store,bench,h,t):
    idx=[i for i in store[h][t] if np.isfinite(store[h][t][i])]
    if len(idx)<8: return float("nan")
    e=np.array([bench[t][h][i]["actual"]-store[h][t][i] for i in idx])
    er=np.array([bench[t][h][i]["actual"]-bench[t][h][i]["rw"] for i in idx])
    return float(np.sqrt((e**2).mean())/np.sqrt((er**2).mean()))

def clark_west(actual,f_small,f_large,h):
    es=actual-f_small; el=actual-f_large
    fhat=es**2-(el**2-(f_small-f_large)**2); n=len(fhat); fb=fhat.mean()
    g0=np.dot(fhat-fb,fhat-fb)/n; var=g0
    for L in range(1,h): var+=2*np.dot(fhat[L:]-fb,fhat[:-L]-fb)/n
    var/=n
    if var<=0: return float("nan"),float("nan")
    cw=fb/np.sqrt(var); return float(cw),float(1-stats.norm.cdf(cw))

def run_mcs(loss_df, seed=SEED, size=0.10, reps=1000, block=4):
    from arch.bootstrap import MCS
    np.random.seed(seed)
    try:    mcs=MCS(loss_df,size=size,reps=reps,block_size=block,method="R",seed=seed)
    except TypeError:
        np.random.seed(seed); mcs=MCS(loss_df,size=size,reps=reps,block_size=block,method="R")
    mcs.compute(); return sorted(map(str,mcs.included))

# ============================================================================
# PHASE A -- baseline block forward selection (level panel, 4 targets)
# ============================================================================
print("\n=== PHASE A: baseline specification search ===")
TARG=["GDP","INF","POLICY","FX"]
N=len(panelL); i0=int(N*OOS_START_FRAC)
R["oos"]={"first_origin":str(panelL.index[i0].date()),
          "n_origins_h1":int(N-1-i0),"n_origins_h4":int(N-4-i0),"start_frac":OOS_START_FRAC,"lag_p":P}
benchL=make_bench(panelL,TARG,i0)

# Johansen on core levels
coreL=pd.DataFrame({"lg":np.log(raw["GDP"]),"lp":np.log(raw["INF"]),
                    "ff":raw["POLICY"],"lx":np.log(raw["FX"])}).dropna().loc["1994-01-01":]
joh=coint_johansen(coreL.values,det_order=0,k_ar_diff=4)
R["johansen"]={"rank":int((joh.lr1>joh.cvt[:,1]).sum()),
               "trace":[round(x,2) for x in joh.lr1],"crit5":[round(x,2) for x in joh.cvt[:,1]]}

CORE=["GDP","INF","POLICY","FX"]
endo=["INV","CONS","FISCAL","EXPORTS","WEALTH","HOUSING"]; exo=["OIL","TFP"]
specs={"CORE":(CORE,[])}
for b in endo: specs[f"CORE+{b}"]=(CORE+[b],[])
for b in exo:  specs[f"CORE+{b}(x)"]=(CORE,[b])
specs["CORE+WEALTH+OIL(x)"]=(CORE+["WEALTH"],["OIL"])

def mean_rr(store,bench,targets,h=1):
    v=[relrmse(store,bench,h,t) for t in targets]; v=[x for x in v if np.isfinite(x)]
    return float(np.mean(v)) if v else float("inf")

# greedy forward selection (score = mean relRMSE h1 over 4 targets)
cur_en,cur_ex,used=list(CORE),[],set()
cur_store=var_oos(panelL,cur_en,cur_ex,i0,TARG); cur=mean_rr(cur_store,benchL,TARG)
path=[("CORE",round(cur,4))]
while True:
    best=None
    for b in endo:
        if b in used: continue
        s=var_oos(panelL,cur_en+[b],cur_ex,i0,TARG); sc=mean_rr(s,benchL,TARG)
        if best is None or sc<best[1]: best=(("en",b),sc)
    for b in exo:
        if b in used: continue
        s=var_oos(panelL,cur_en,cur_ex+[b],i0,TARG); sc=mean_rr(s,benchL,TARG)
        if best is None or sc<best[1]: best=(("ex",b),sc)
    if best and best[1]<cur-1e-4:
        (kind,b),sc=best
        if kind=="en": cur_en=cur_en+[b]
        else: cur_ex=cur_ex+[b]
        used.add(b); cur=sc; path.append((b,round(sc,4)))
    else: break
R["phaseA_forward_path"]=path

store_all={}; rowsA=[]
for name,(en,ex) in specs.items():
    st=var_oos(panelL,en,ex,i0,TARG); store_all[name]=st
    fd=full_diag(panelL,en,ex)
    rr={f"{t}_h{h}":round(relrmse(st,benchL,h,t),4) for h in H_LIST for t in TARG}
    dof_ok=fd["dof"]>=3.0 and fd["T"]>fd["n_params"]
    passes=bool(fd["stable"] and dof_ok and (np.isnan(fd["wh"]) or fd["wh"]>0.01))
    rowsA.append({"spec":name,"mean_h1":round(mean_rr(st,benchL,TARG,1),4),
                  "mean_h4":round(mean_rr(st,benchL,TARG,4),4),
                  "T_over_k":round(fd["dof"],2),"stable":fd["stable"],
                  "whiteness_p":round(fd["wh"],4),"passes":passes,**rr})
tabA=pd.DataFrame(rowsA)
elig=tabA[tabA["passes"]].copy()
# rank eligible by avg rank of relRMSE across targets @h1
for t in TARG: tabA[f"rk_{t}"]=tabA[f"{t}_h1"].rank(method="min")
tabA["avg_rank"]=tabA[[f"rk_{t}" for t in TARG]].mean(axis=1)
elig=tabA[tabA["passes"]].sort_values("avg_rank")
winnerA=elig.iloc[0]["spec"]
R["phaseA_winner_eligible"]=winnerA
R["phaseA_table"]=tabA[["spec","mean_h1","mean_h4","T_over_k","stable","whiteness_p","passes",
                        *[f"{t}_h1" for t in TARG],*[f"{t}_h4" for t in TARG]]].to_dict("records")

# CW for CORE+OIL(x) vs RW, per target
cwo={}; st=store_all["CORE+OIL(x)"]
for t in TARG:
    idx=[i for i in st[1][t] if np.isfinite(st[1][t][i])]
    act=np.array([benchL[t][1][i]["actual"] for i in idx])
    frw=np.array([benchL[t][1][i]["rw"] for i in idx]); fv=np.array([st[1][t][i] for i in idx])
    cw,pv=clark_west(act,frw,fv,1); cwo[t]={"CW":round(cw,3),"p":round(pv,4)}
R["phaseA_CW_COREOIL_vs_RW"]=cwo

# MCS per target (eligible + benchmarks)
mcsA={}
for t in TARG:
    em=list(elig["spec"]); common=None
    for m in em:
        ok={i for i in store_all[m][1][t] if np.isfinite(store_all[m][1][t][i])}
        common=ok if common is None else common&ok
    common=sorted(common); act=np.array([benchL[t][1][i]["actual"] for i in common])
    L={m:(act-np.array([store_all[m][1][t][i] for i in common]))**2 for m in em}
    L["RW"]=(act-np.array([benchL[t][1][i]["rw"] for i in common]))**2
    L["AR(BIC)"]=(act-np.array([benchL[t][1][i]["ar"] for i in common]))**2
    mcsA[t]=run_mcs(pd.DataFrame(L))
R["phaseA_MCS"]=mcsA
print(f"  winner={winnerA}  path={path}")

# ============================================================================
# PHASE B -- HP look-ahead + gap-vs-level tie
# ============================================================================
print("=== PHASE B: HP look-ahead + gap-vs-level ===")
ff=load_q("FEDFUNDS").loc["1994-01-01":]
cyc_full,_=hpfilter(ff,lamb=1600)
ffALL=load_q("FEDFUNDS")            # anchor one-sided on full history
rt={}
for i in range(12,len(ffALL)+1):
    c,_=hpfilter(ffALL.iloc[:i],lamb=1600); rt[ffALL.index[i-1]]=c.iloc[-1]
cyc_rt=pd.Series(rt)
dfh=pd.DataFrame({"two":cyc_full,"one":cyc_rt}).dropna().loc["1994-01-01":]
dfh["rev"]=dfh["two"]-dfh["one"]
R["hp"]={"corr":round(float(dfh["two"].corr(dfh["one"])),3),
         "mean_abs_rev":round(float(dfh["rev"].abs().mean()),3),
         "last8_abs_rev":round(float(dfh["rev"].abs().tail(8).mean()),3),
         "cyc_std":round(float(dfh["two"].std()),3),
         "pct_of_std":round(100*float(dfh["rev"].abs().tail(8).mean())/float(dfh["two"].std()),1)}

# gap-vs-level on GDP/INF/FX (leakage-free one-sided gap anchored on full history)
panelL2=panelL.copy(); panelL2["POLICY_GAP"]=cyc_rt.reindex(panelL.index)
panelL2=panelL2.dropna()
N2=len(panelL2); i02=int(N2*OOS_START_FRAC); RANK3=["GDP","INF","FX"]
bench2=make_bench(panelL2,["GDP","INF","FX","POLICY","POLICY_GAP"],i02)
def spec_mean3(en,ex,ratevar):
    st=var_oos(panelL2,en,ex,i02,["GDP","INF","FX",ratevar])
    m3=float(np.mean([relrmse(st,bench2,1,t) for t in RANK3]))
    rown=relrmse(st,bench2,1,ratevar)
    return round(m3,4),round(rown,4)
lvl_m,lvl_rate=spec_mean3(["GDP","INF","POLICY","FX"],["OIL"],"POLICY")
gap_m,gap_rate=spec_mean3(["GDP","INF","POLICY_GAP","FX"],["OIL"],"POLICY_GAP")
R["gap_vs_level"]={"level_mean3_h1":lvl_m,"gap_mean3_h1":gap_m,
                   "level_rate_own":lvl_rate,"gap_rate_own":gap_rate}
print(f"  hp corr={R['hp']['corr']} last8rev={R['hp']['last8_abs_rev']}pp ({R['hp']['pct_of_std']}% std)")
print(f"  gap tie: level {lvl_m} vs gap {gap_m}")

# ============================================================================
# PHASE C -- model-class horse race on CORE+OIL(x) (small) and wide
# ============================================================================
print("=== PHASE C: model-class comparison ===")
def bvar_B(endog,exog,p,lam,delta):
    Y,X,n,m=build_design(endog,exog,p); k=X.shape[1]
    sig=np.array([ar_resid_std(endog[:,i],p) for i in range(n)])
    Jm=np.diag(np.arange(1,p+1).astype(float))
    Xdl=np.kron(Jm,np.diag(sig))/lam; Xdl=np.hstack([Xdl,np.zeros((n*p,m+1))])
    Ydl=np.zeros((n*p,n)); Ydl[:n,:]=np.diag(delta*sig)/lam
    Xdc=np.zeros((n,k)); Ydc=np.diag(sig)
    Xs=np.vstack([X,Xdl,Xdc]); Ys=np.vstack([Y,Ydl,Ydc])
    B,*_=np.linalg.lstsq(Xs,Ys,rcond=None); return B,n,m
def iterate_B(B,endog,exf,p,n,m,h):
    hist=[endog[-p+i] for i in range(p)]; out=[]
    for s in range(h):
        xr=[]
        for l in range(1,p+1): xr+=list(hist[-l])
        if m: xr+=list(exf[s])
        xr+=[1.0]; yn=np.array(xr)@B; out.append(yn); hist.append(yn)
    return np.array(out)
def pen_state(endog,exog,p,kind):
    Y,X,n,m=build_design(endog,exog,p); Xnc=X[:,:-1]
    mu=Xnc.mean(0); sd=Xnc.std(0); sd[sd==0]=1.0; Xs=(Xnc-mu)/sd
    cv=TimeSeriesSplit(n_splits=3); models=[]
    for kc in range(n):
        if kind=="ridge": mdl=RidgeCV(alphas=np.logspace(-2,3,12),cv=cv)
        elif kind=="lasso": mdl=LassoCV(n_alphas=12,cv=cv,max_iter=3000,n_jobs=1,random_state=SEED)
        else: mdl=ElasticNetCV(l1_ratio=0.5,n_alphas=12,cv=cv,max_iter=3000,n_jobs=1,random_state=SEED)
        mdl.fit(Xs,Y[:,kc]); models.append(mdl)
    return {"models":models,"mu":mu,"sd":sd,"p":p,"n":n,"m":m}
def iterate_pen(st,endog,exf,h):
    p,n,m=st["p"],st["n"],st["m"]; hist=[endog[-p+i] for i in range(p)]; out=[]
    for s in range(h):
        xr=[]
        for l in range(1,p+1): xr+=list(hist[-l])
        if m: xr+=list(exf[s])
        x=(np.array(xr)-st["mu"])/st["sd"]
        yn=np.array([md.predict(x.reshape(1,-1))[0] for md in st["models"]]); out.append(yn); hist.append(yn)
    return np.array(out)

SMALL=["GDP","INF","POLICY","FX"]; SMALLX=["OIL"]
WIDE=SMALL+["INV","CONS","FISCAL","EXPORTS","WEALTH","HOUSING"]; WIDEX=["OIL","TFP"]
PSPECS=[("OLS:small","ols",SMALL,SMALLX,{}),("BVAR-Tight:small","bvar",SMALL,SMALLX,{"lam":0.1}),
        ("BVAR-Loose:small","bvar",SMALL,SMALLX,{"lam":0.5}),("Ridge:small","ridge",SMALL,SMALLX,{}),
        ("LASSO:small","lasso",SMALL,SMALLX,{}),("ENet:small","enet",SMALL,SMALLX,{}),
        ("OLS:wide(overfit)","ols",WIDE,WIDEX,{}),("BVAR-Tight:wide","bvar",WIDE,WIDEX,{"lam":0.1}),
        ("Ridge:wide","ridge",WIDE,WIDEX,{}),("LASSO:wide","lasso",WIDE,WIDEX,{}),("ENet:wide","enet",WIDE,WIDEX,{})]
def delta_of(names): return np.array([1.0 if tmap[x]=="level" else 0.0 for x in names])
def pc_oos(fam,en,ex,hp):
    st={h:{t:{} for t in SMALL} for h in H_LIST}; delta=delta_of(en); ti={t:en.index(t) for t in SMALL}
    for i in range(i0,N-max(H_LIST)):
        E=panelL[en].values[:i+1]; X=panelL[ex].values[:i+1] if ex else None
        exf={h:np.column_stack([ar_bic_path(panelL[e].values[:i+1],h) for e in ex]) if ex else None for h in H_LIST}
        try:
            if fam=="ols":
                Y,Xd,n,m=build_design(E,X,P); B,*_=np.linalg.lstsq(Xd,Y,rcond=None)
                fc={h:iterate_B(B,E,exf[h],P,n,m,h)[-1] for h in H_LIST}
            elif fam=="bvar":
                B,n,m=bvar_B(E,X,P,hp["lam"],delta); fc={h:iterate_B(B,E,exf[h],P,n,m,h)[-1] for h in H_LIST}
            else:
                s=pen_state(E,X,P,fam); fc={h:iterate_pen(s,E,exf[h],h)[-1] for h in H_LIST}
            for h in H_LIST:
                for t in SMALL: st[h][t][i]=fc[h][ti[t]]
        except Exception:
            for h in H_LIST:
                for t in SMALL: st[h][t][i]=np.nan
    return st
pc_store={}
for name,fam,en,ex,hp in PSPECS:
    print(f"  {name}"); pc_store[name]=pc_oos(fam,en,ex,hp)
rowsC=[]
for name,fam,en,ex,hp in PSPECS:
    rowsC.append({"model":name,"mean_h1":round(mean_rr(pc_store[name],benchL,SMALL,1),4),
                  "mean_h4":round(mean_rr(pc_store[name],benchL,SMALL,4),4),
                  "GDP_h1":round(relrmse(pc_store[name],benchL,1,"GDP"),4)})
# benchmarks
def bench_mean(kind,h):
    v=[]
    for t in SMALL:
        idx=list(benchL[t][h])
        e=np.array([benchL[t][h][i]["actual"]-benchL[t][h][i][kind] for i in idx])
        er=np.array([benchL[t][h][i]["actual"]-benchL[t][h][i]["rw"] for i in idx])
        v.append(np.sqrt((e**2).mean())/np.sqrt((er**2).mean()))
    return round(float(np.mean(v)),4)
rowsC.append({"model":"AR(BIC)","mean_h1":bench_mean("ar",1),"mean_h4":bench_mean("ar",4),"GDP_h1":None})
rowsC.append({"model":"RW","mean_h1":1.0,"mean_h4":1.0,"GDP_h1":1.0})
R["phaseC_table"]=sorted(rowsC,key=lambda r:r["mean_h1"])
# joint MCS per target
mcsC={}
for t in SMALL:
    ml=[s[0] for s in PSPECS]; common=None
    for m in ml:
        ok={i for i in pc_store[m][1][t] if np.isfinite(pc_store[m][1][t][i])}
        common=ok if common is None else common&ok
    common=sorted(common); act=np.array([benchL[t][1][i]["actual"] for i in common])
    L={m:(act-np.array([pc_store[m][1][t][i] for i in common]))**2 for m in ml}
    L["RW"]=(act-np.array([benchL[t][1][i]["rw"] for i in common]))**2
    L["AR(BIC)"]=(act-np.array([benchL[t][1][i]["ar"] for i in common]))**2
    mcsC[t]=run_mcs(pd.DataFrame(L))
R["phaseC_MCS"]=mcsC
R["phaseC_all_in_mcs"]=all(len(mcsC[t])==len(PSPECS)+2 for t in SMALL)

# ============================================================================
# PHASE D -- YoY transform robustness (Table IV) + whiteness collapse
# ============================================================================
print("=== PHASE D: YoY transform robustness ===")
NY=len(panelY); i0Y=int(NY*OOS_START_FRAC)
benchY=make_bench(panelY,TARG,i0Y)
storeY_core=var_oos(panelY,CORE,[],i0Y,TARG)
storeL_core=store_all["CORE"]
tab4={}
for t in TARG:
    tab4[t]={"aq_h1":round(relrmse(storeL_core,benchL,1,t),4),"aq_h4":round(relrmse(storeL_core,benchL,4,t),4),
             "yoy_h1":round(relrmse(storeY_core,benchY,1,t),4),"yoy_h4":round(relrmse(storeY_core,benchY,4,t),4)}
R["phaseD_table4_CORE"]=tab4
# whiteness of all specs on YoY panel
whY={}
for name,(en,ex) in specs.items():
    whY[name]=round(full_diag(panelY,en,ex)["wh"],4)
R["phaseD_yoy_whiteness"]=whY
R["phaseD_yoy_all_fail_whiteness"]=all((np.isnan(v) or v<=0.01) for v in whY.values())
R["phaseD_adf"]={"FX_yoy":round(float(adfuller(panelY["FX"].dropna().values,autolag="AIC")[1]),3),
                 "INF_yoy":round(float(adfuller(panelY["INF"].dropna().values,autolag="AIC")[1]),3)}

# ============================================================================
# FIGURES  (regenerated from this single run)
# ============================================================================
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def heatmap(df, cols, title, path, vmin=0.7, vmax=1.3):
    M=df[cols].astype(float)
    fig,ax=plt.subplots(figsize=(7,0.5*len(M)+1.4))
    im=ax.imshow(M.values,aspect="auto",cmap="RdYlGn_r",vmin=vmin,vmax=vmax)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c.split("_")[0] for c in cols])
    ax.set_yticks(range(len(M))); ax.set_yticklabels(df.index,fontsize=8)
    for (i,j),v in np.ndenumerate(M.values):
        if np.isfinite(v): ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=7)
    ax.set_title(title,fontsize=10); plt.colorbar(im,ax=ax,shrink=0.6,label="RMSE / RW")
    plt.tight_layout(); plt.savefig(path,dpi=140,bbox_inches="tight"); plt.close()
# fig_baseline
dfB=tabA.set_index("spec")[[f"{t}_h1" for t in TARG]]; dfB.columns=TARG
heatmap(dfB,TARG,"Relative RMSE vs Random Walk (h=1)\nbaseline specification search",PFIG/"fig_baseline.png")
# fig_phase2
dfC=pd.DataFrame([{ "model":r["model"],**{t:relrmse(pc_store[r["model"]],benchL,1,t) if r["model"] in pc_store else np.nan for t in SMALL}} for r in rowsC if r["model"] in pc_store]).set_index("model")
heatmap(dfC,SMALL,"Phase 2: relative RMSE vs RW (h=1)\nmodel class x information set",PFIG/"fig_phase2.png",0.6,1.6)
# fig_hp
fig,ax=plt.subplots(2,1,figsize=(12,7),height_ratios=[2,1],sharex=True)
sub=dfh.loc["2004-01-01":]
ax[0].axhline(0,color="0.6",lw=0.8)
ax[0].plot(sub.index,sub["two"],"C0-",lw=2.0,label="two-sided HP cycle (full sample, uses future data)")
ax[0].plot(sub.index,sub["one"],"C3--",lw=1.8,label="one-sided HP cycle (real-time, data up to t only)")
ax[0].set_ylabel("fed funds gap (pp)"); ax[0].legend(fontsize=8,loc="upper left")
ax[0].set_title("Fed funds rate gap: two-sided HP leaks the future and is revised heavily near the endpoint",fontsize=11,fontweight="bold")
ax[1].bar(sub.index,sub["rev"],width=70,color="C1",alpha=0.8); ax[1].axhline(0,color="0.6",lw=0.8)
ax[1].set_ylabel("two-sided minus\nreal-time (pp)")
ax[1].set_title("Endpoint revision: what the backtest knew that you would not have known in real time",fontsize=9)
plt.tight_layout(); plt.savefig(PFIG/"fig_hp.png",dpi=140,bbox_inches="tight"); plt.close()

# ============================================================================
# provenance + dump
# ============================================================================
R["provenance"]={"seed":SEED,"python":platform.python_version(),
    "numpy":np.__version__,"pandas":pd.__version__,"statsmodels":statsmodels.__version__,
    "sklearn":sklearn.__version__,"arch":arch.__version__,
    "data_vintage":"2026-07-10 (FRED public CSV, cached)"}
(REPRO/"results.json").write_text(json.dumps(R,indent=2,default=str))
for f in ["fig_baseline.png","fig_hp.png","fig_phase2.png"]:
    (OUT/f).write_bytes((PFIG/f).read_bytes())
(OUT/"reproduce_results.json").write_text(json.dumps(R,indent=2,default=str))

# ============================================================================
# PAPER NUMBERS block
# ============================================================================
print("\n"+"="*70); print("PAPER NUMBERS (transcribe exactly)"); print("="*70)
print(f"sample level: {R['sample_level']['n']}q {R['sample_level']['start']}..{R['sample_level']['end']}")
print(f"sample yoy  : {R['sample_yoy']['n']}q {R['sample_yoy']['start']}..{R['sample_yoy']['end']}")
print(f"oos first origin {R['oos']['first_origin']}, origins h1={R['oos']['n_origins_h1']} h4={R['oos']['n_origins_h4']}")
print(f"johansen rank {R['johansen']['rank']}")
co=[r for r in R['phaseA_table'] if r['spec']=='CORE+OIL(x)'][0]
print(f"CORE+OIL(x): mean h1={co['mean_h1']} h4={co['mean_h4']}  per-target h1: "
      f"GDP={co['GDP_h1']} INF={co['INF_h1']} POLICY={co['POLICY_h1']} FX={co['FX_h1']}")
print(f"CW CORE+OIL(x) vs RW: {R['phaseA_CW_COREOIL_vs_RW']}")
print(f"phaseA winner(eligible)={R['phaseA_winner_eligible']}")
for r in R['phaseA_table']:
    if r['spec'] in ['CORE','CORE+OIL(x)','CORE+TFP(x)','CORE+WEALTH','CORE+WEALTH+OIL(x)']:
        print(f"  TabII {r['spec']:20s} mean_h1={r['mean_h1']} T/k={r['T_over_k']} stable={r['stable']} wh={r['whiteness_p']} pass={r['passes']}")
print(f"HP: {R['hp']}")
print(f"gap-vs-level: {R['gap_vs_level']}")
print("phaseC (sorted by mean_h1):")
for r in R['phaseC_table']: print(f"  {r['model']:20s} h1={r['mean_h1']} h4={r['mean_h4']} GDP_h1={r['GDP_h1']}")
mcs_sizes = ", ".join(f"{t}:{len(R['phaseC_MCS'][t])}" for t in SMALL)
print(f"phaseC all-in-MCS={R['phaseC_all_in_mcs']}  (per-target MCS sizes: {mcs_sizes})")
print(f"Table IV (CORE): {R['phaseD_table4_CORE']}")
print(f"YoY all fail whiteness={R['phaseD_yoy_all_fail_whiteness']}  ADF={R['phaseD_adf']}")
print("\n[saved] repro/results.json + paper/figs/*.png")
print("DONE.")
