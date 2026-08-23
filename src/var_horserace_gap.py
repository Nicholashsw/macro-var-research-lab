"""
var_horserace_gap.py
Tests the monetary-stance variable as LEVEL (fed funds) vs a leakage-free
one-sided HP rate GAP, inside the augmented block forward-selection.

Leakage control: POLICY_GAP_t = last value of HP filter run on fed funds
data up to t only (expanding/one-sided), anchored on history from 1954.
No data after a forecast origin ever enters that origin's regressors.

Ranking + MCS on the three targets identical across both representations:
GDP, INF, FX. The rate's own forecast is reported separately (level-RMSE and
gap-RMSE are not comparable, so the rate is never mixed into the cross-model rank).
Research/education. Not investment advice.
"""
import os
import warnings, json; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tsa.filters.hp_filter import hpfilter

# repo-relative paths; override the root with REPRO_ROOT if you run from elsewhere.
# fred_data/ and outputs/ are gitignored, so a re-run never clobbers committed artifacts.
ROOT = Path(os.environ.get("REPRO_ROOT", Path(__file__).resolve().parent.parent))
DATA = ROOT / "fred_data"; DATA.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "outputs"; OUT.mkdir(parents=True, exist_ok=True)
FIG = OUT / "figs"; FIG.mkdir(parents=True, exist_ok=True)

CORE_COMMON = {"GDP": ("GDPC1","growth"), "INF": ("PCEPILFE","growth"), "FX": ("RBUSBIS","growth")}
RATE_LEVEL  = ("FEDFUNDS","level")
BLOCKS = {
    "INV":("GPDIC1","growth"), "CONS":("PCECC96","growth"), "FISCAL":("GCEC1","growth"),
    "EXPORTS":("EXPGSC1","growth"), "WEALTH":("TNWBSHNO","growth"), "HOUSING":("USSTHPI","growth"),
    "OIL":("WTISPLC","oilret"), "TFP":("FERNALD_TFPUTIL","level"),
}
EXOG_BLOCKS = {"OIL","TFP"}
RANK_TARGETS = ["GDP","INF","FX"]          # apples-to-apples across both cores
MAXLAG_VAR, MAXLAG_AR = 4, 8
H_LIST=[1,4]; H_SELECT=1; OOS_START_FRAC=0.55

def load_q(sid):
    df=pd.read_csv(DATA/f"{sid}.csv"); df.columns=["date","value"]
    df["date"]=pd.to_datetime(df["date"]); df["value"]=pd.to_numeric(df["value"],errors="coerce")
    return df.dropna().set_index("date")["value"].sort_index().resample("QS").mean()

def transform(lv,k):
    if k=="level": return lv
    if k=="growth": return 400.0*np.log(lv).diff()
    if k=="oilret": return 100.0*np.log(lv).diff()
    raise ValueError(k)

# ---- build modeled series ----
modeled={}; 
for nm,(sid,k) in {**CORE_COMMON, "POLICY":RATE_LEVEL, **BLOCKS}.items():
    modeled[nm]=transform(load_q(sid),k)

# ---- leakage-free one-sided HP gap of fed funds (anchored on full 1954+ history) ----
ff_full=load_q("FEDFUNDS")
rt={}
for i in range(12,len(ff_full)+1):
    c,_=hpfilter(ff_full.iloc[:i],lamb=1600); rt[ff_full.index[i-1]]=c.iloc[-1]
modeled["POLICY_GAP"]=pd.Series(rt)

panel=pd.DataFrame(modeled).dropna().loc["1994-04-01":]
N=len(panel); i0=int(N*OOS_START_FRAC)
print(f"[panel] {N} quarters  {panel.index.min().date()} -> {panel.index.max().date()}  oos from {panel.index[i0].date()}\n")

# ---- benchmarks (RW, AR-BIC) for all evaluated targets ----
EVAL=["GDP","INF","FX","POLICY","POLICY_GAP"]
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
bench={t:{h:{} for h in H_LIST} for t in EVAL}
for t in EVAL:
    y=panel[t].values
    for h in H_LIST:
        for i in range(i0,N-h):
            yt=y[:i+1]; bench[t][h][i]={"rw":yt[-1],"ar":ar_bic_path(yt,h)[-1],"actual":y[i+h]}

# ---- VAR/VARX engine ----
def fit_var(endog,exog=None):
    m=VAR(endog,exog=exog); r=m.fit(maxlags=MAXLAG_VAR,ic="bic",trend="c")
    return m.fit(1,trend="c") if r.k_ar==0 else r
def spec_oos(endog_names,exog_names,h):
    preds={t:{} for t in EVAL if t in endog_names}
    for i in range(i0,N-h):
        try:
            en=panel[endog_names].values[:i+1]
            if exog_names:
                ex=panel[exog_names].values[:i+1]; res=fit_var(en,ex)
                ef=np.column_stack([ar_bic_path(panel[e].values[:i+1],h) for e in exog_names])
                fc=res.forecast(en[-res.k_ar:],steps=h,exog_future=ef)
            else:
                res=fit_var(en); fc=res.forecast(en[-res.k_ar:],steps=h)
            row=fc[h-1]
            for t in preds: preds[t][i]=row[endog_names.index(t)]
        except Exception:
            for t in preds: preds[t][i]=np.nan
    return preds
def rel_rmse(preds,h,targets):
    out={}
    for t in targets:
        if t not in preds: out[t]={"vs_rw":np.nan,"n":0}; continue
        idx=[i for i in preds[t] if np.isfinite(preds[t][i])]
        if len(idx)<8: out[t]={"vs_rw":np.nan,"n":len(idx)}; continue
        e=np.array([bench[t][h][i]["actual"]-preds[t][i] for i in idx])
        er=np.array([bench[t][h][i]["actual"]-bench[t][h][i]["rw"] for i in idx])
        out[t]={"vs_rw":np.sqrt((e**2).mean())/np.sqrt((er**2).mean()),"n":len(idx)}
    return out
def mean_rank_score(rr): 
    v=[rr[t]["vs_rw"] for t in RANK_TARGETS if np.isfinite(rr[t]["vs_rw"])]; return np.mean(v) if v else np.inf

def companion_max(res):
    p,k=res.k_ar,res.neqs
    if p==0: return 0.0
    C=np.zeros((k*p,k*p)); C[:k,:]=np.hstack([res.coefs[i] for i in range(p)])
    if p>1: C[k:,:-k]=np.eye(k*(p-1))
    return float(np.max(np.abs(np.linalg.eigvals(C))))
def full_diag(endog_names,exog_names):
    en=panel[endog_names].values; ex=panel[exog_names].values if exog_names else None
    res=fit_var(en,ex); k,p=res.neqs,res.k_ar
    nparm=k*(k*p+1)+(k*len(exog_names) if exog_names else 0); T=res.nobs
    try: stable=bool(res.is_stable(verbose=False))
    except Exception: stable=companion_max(res)<1.0
    try: wh=res.test_whiteness(nlags=max(p+4,10),adjusted=True).pvalue
    except Exception: wh=np.nan
    return {"var_lag":p,"n_params":nparm,"T":T,"dof":T/nparm,"stable":stable,
            "maxmod":companion_max(res),"wh":wh,"aic":float(res.aic),"bic":float(res.bic)}

# ---- forward selection for a given core ----
def forward_select(core_endog,label_core):
    print(f"--- forward selection: {label_core}  (score=mean relRMSE@1 over GDP/INF/FX) ---")
    cur={"endog":list(core_endog),"exog":[],"score":None,"used":set()}
    cur["score"]=mean_rank_score(rel_rmse(spec_oos(cur["endog"],[],H_SELECT),H_SELECT,RANK_TARGETS))
    print(f"  {label_core:14s} {cur['score']:.4f}")
    path=[(label_core,cur["score"])]
    endo=[b for b in BLOCKS if b not in EXOG_BLOCKS]; exo=[b for b in BLOCKS if b in EXOG_BLOCKS]
    improved=True
    while improved:
        improved=False; best=None
        for b in endo:
            if b in cur["used"]: continue
            sc=mean_rank_score(rel_rmse(spec_oos(cur["endog"]+[b],cur["exog"],H_SELECT),H_SELECT,RANK_TARGETS))
            if best is None or sc<best[1]: best=(("endo",b,cur["endog"]+[b],cur["exog"]),sc)
        for b in exo:
            if b in cur["used"]: continue
            sc=mean_rank_score(rel_rmse(spec_oos(cur["endog"],cur["exog"]+[b],H_SELECT),H_SELECT,RANK_TARGETS))
            if best is None or sc<best[1]: best=(("exo",b,cur["endog"],cur["exog"]+[b]),sc)
        if best and best[1]<cur["score"]-1e-4:
            (kind,b,en,ex),sc=best
            cur={"endog":en,"exog":ex,"score":sc,"used":cur["used"]|{b}}; improved=True
            path.append((b,sc)); print(f"  >> +{b} ({kind})  {sc:.4f}")
    return cur,path

specs={}
def reg(label,en,ex): specs[label]={"endog":en,"exog":ex}

CORE_LVL=["GDP","INF","POLICY","FX"]; CORE_GAP=["GDP","INF","POLICY_GAP","FX"]
reg("LVL:core",CORE_LVL,[]); reg("GAP:core",CORE_GAP,[])
wl,_=forward_select(CORE_LVL,"LVL:core"); print()
wg,_=forward_select(CORE_GAP,"GAP:core"); print()
# register winners + single-block oil/tfp variants for both cores (transparency)
for core,tag in [(CORE_LVL,"LVL"),(CORE_GAP,"GAP")]:
    reg(f"{tag}:core+OIL(x)",core,["OIL"]); reg(f"{tag}:core+TFP(x)",core,["TFP"])
reg("LVL:"+ "+".join(["core"]+[x for x in wl["endog"] if x not in CORE_LVL]+[x+"(x)" for x in wl["exog"]]), wl["endog"],wl["exog"])
reg("GAP:"+ "+".join(["core"]+[x for x in wg["endog"] if x not in CORE_GAP]+[x+"(x)" for x in wg["exog"]]), wg["endog"],wg["exog"])

# ---- score all specs ----
rows=[]; store={}
for label,sp in specs.items():
    store[label]={h:spec_oos(sp["endog"],sp["exog"],h) for h in H_LIST}
    rr1=rel_rmse(store[label][1],1,RANK_TARGETS); rr4=rel_rmse(store[label][4],4,RANK_TARGETS)
    fd=full_diag(sp["endog"],sp["exog"])
    ratevar="POLICY_GAP" if "POLICY_GAP" in sp["endog"] else "POLICY"
    rr_rate=rel_rmse(store[label][1],1,[ratevar])[ratevar]["vs_rw"]
    rec={"model_id":label,"rate_repr":("gap" if ratevar=="POLICY_GAP" else "level"),
         "endogenous_set":"+".join(sp["endog"]),"exogenous_set":"+".join(sp["exog"]) or "-",
         "var_lag":fd["var_lag"],"n_params":fd["n_params"],"T_over_params":round(fd["dof"],2),
         "stability_pass":fd["stable"],"max_eig_modulus":round(fd["maxmod"],4),
         "whiteness_p":round(fd["wh"],4) if np.isfinite(fd["wh"]) else np.nan,
         "AIC":round(fd["aic"],2),"BIC":round(fd["bic"],2)}
    for t in RANK_TARGETS:
        rec[f"relRMSE_{t}_h1"]=round(rr1[t]["vs_rw"],4) if np.isfinite(rr1[t]["vs_rw"]) else np.nan
    rec["mean_relRMSE_h1"]=round(np.mean([rr1[t]["vs_rw"] for t in RANK_TARGETS]),4)
    rec["mean_relRMSE_h4"]=round(np.mean([rr4[t]["vs_rw"] for t in RANK_TARGETS]),4)
    rec["rate_own_relRMSE_h1"]=round(rr_rate,4) if np.isfinite(rr_rate) else np.nan
    dof_ok=fd["dof"]>=3.0 and fd["T"]>fd["n_params"]
    rec["passes_hard_filters"]=bool(fd["stable"] and dof_ok and (np.isnan(fd["wh"]) or fd["wh"]>0.01))
    rows.append(rec)
tab=pd.DataFrame(rows)
for t in RANK_TARGETS: tab[f"rk_{t}"]=tab[f"relRMSE_{t}_h1"].rank(method="min")
tab["avg_rank_h1"]=tab[[f"rk_{t}" for t in RANK_TARGETS]].mean(axis=1)
tab=tab.sort_values("avg_rank_h1").reset_index(drop=True)

print("="*96); print("RESULTS  (rank/MCS on GDP, INF, FX -- rate shown separately, not comparable across reprs)"); print("="*96)
show=["model_id","rate_repr","var_lag","T_over_params","stability_pass","whiteness_p",
      "relRMSE_GDP_h1","relRMSE_INF_h1","relRMSE_FX_h1","mean_relRMSE_h1","mean_relRMSE_h4",
      "rate_own_relRMSE_h1","passes_hard_filters"]
print(tab[show].to_string(index=False))
elig=tab[tab["passes_hard_filters"]].sort_values("avg_rank_h1")
print(f"\n[best eligible by avg rank on GDP/INF/FX] {elig.iloc[0]['model_id']}")

# ---- MCS on GDP/INF/FX (h=1) across eligible specs + benchmarks ----
print("\nModel Confidence Set (90%) per target, h=1 [included]:")
mcs_inc={}
try:
    from arch.bootstrap import MCS
    em=list(elig["model_id"])
    for t in RANK_TARGETS:
        common=None
        for m in em:
            if t not in store[m][1]: continue
            ok={i for i in store[m][1][t] if np.isfinite(store[m][1][t][i])}
            common=ok if common is None else common&ok
        common=sorted(common)
        act=np.array([bench[t][1][i]["actual"] for i in common])
        L={m:(act-np.array([store[m][1][t][i] for i in common]))**2 for m in em if t in store[m][1]}
        L["RW"]=(act-np.array([bench[t][1][i]["rw"] for i in common]))**2
        L["AR(BIC)"]=(act-np.array([bench[t][1][i]["ar"] for i in common]))**2
        mcs=MCS(pd.DataFrame(L),size=0.10,reps=1000,block_size=4,method="R"); mcs.compute()
        mcs_inc[t]=list(mcs.included); print(f"  {t:5s}  {mcs_inc[t]}")
except Exception as e:
    print(f"  MCS skipped: {type(e).__name__}: {e}")

# ---- save ----
tab.to_csv(OUT/"var_horserace_gap_results.csv",index=False)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
M=tab.set_index("model_id")[[f"relRMSE_{t}_h1" for t in RANK_TARGETS]].astype(float); M.columns=RANK_TARGETS
fig,ax=plt.subplots(figsize=(7,0.5*len(M)+1.4))
im=ax.imshow(M.values,aspect="auto",cmap="RdYlGn_r",vmin=0.7,vmax=1.3)
ax.set_xticks(range(len(RANK_TARGETS))); ax.set_xticklabels(RANK_TARGETS)
ax.set_yticks(range(len(M))); ax.set_yticklabels(M.index,fontsize=8)
for (i,j),v in np.ndenumerate(M.values):
    if np.isfinite(v): ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=7)
ax.set_title("Relative RMSE vs RW (h=1) on comparable targets\nlevel-rate vs leakage-free gap-rate cores",fontsize=10)
plt.colorbar(im,ax=ax,shrink=0.6,label="RMSE / RW"); plt.tight_layout()
plt.savefig(FIG/"gap_vs_level_heatmap.png",dpi=140,bbox_inches="tight"); plt.close()
(OUT/"gap_vs_level_heatmap.png").write_bytes((FIG/"gap_vs_level_heatmap.png").read_bytes())
summary={"sample":f"{panel.index.min().date()} to {panel.index.max().date()}","n":N,
         "lvl_forward_winner":[x for x in wl["endog"] if x not in CORE_LVL]+[x+"(x)" for x in wl["exog"]],
         "gap_forward_winner":[x for x in wg['endog'] if x not in CORE_GAP]+[x+'(x)' for x in wg['exog']],
         "best_eligible":elig.iloc[0]["model_id"],"mcs_h1":mcs_inc}
(OUT/"var_horserace_gap_summary.json").write_text(json.dumps(summary,indent=2,default=str))
print(f"\n[saved] {OUT/'var_horserace_gap_results.csv'}\n[saved] {OUT/'gap_vs_level_heatmap.png'}\n[saved] {OUT/'var_horserace_gap_summary.json'}\nDONE.")
