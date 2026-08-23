import json
import time
import numpy as np
import pandas as pd
import varx_lab as vl
import varx_realtime as rt

t0 = time.time()
finals = rt.load_finals()

# AR lag orders used in the main study (BIC on the initial window)
train_end_i = finals["panel"].index.get_loc(pd.Period(vl.TRAIN_END, freq="Q").to_timestamp())
bench = vl.ar_benchmark(finals["panel"], ["gdp", "inf"], train_end_i,
                        list(range(train_end_i, len(finals["panel"]) - 1)))
p_by = {v: bench[v]["p"] for v in ["gdp", "inf"]}
print("AR p:", p_by)

rt_df, origins_q = rt.run_realtime(finals, rt.SPECS, targets=("gdp", "inf"))
print("forecast rows:", len(rt_df), f"{time.time()-t0:.0f}s")

ar_rt = rt.rt_ar_benchmark(finals, origins_q, ("gdp", "inf"), p_by)
sc = rt.score(rt_df, finals, ar_rt, origins_q, p_by)
sc.to_parquet("rt_results.parquet")

# ---- final-data RMSE for the same core specs, pulled from the main study ----
res = pd.read_parquet("results.parquet")
name_map = {
    "pure_var": ("gdp+inf+stance", "none", "none"),
    "oil": ("gdp+inf+stance", "oil", "raw"),
    "inv": ("gdp+inf+stance", "inv", "raw"),
    "wealth": ("gdp+inf+stance", "wealth", "raw"),
    "fiscal": ("gdp+inf+stance", "fiscal", "raw"),
    "all4_raw": ("gdp+inf+stance", "oil+inv+wealth+fiscal", "raw"),
    "all4_gdp_resid": ("gdp+inf+stance", "oil+inv+wealth+fiscal", "gdp_resid"),
}
fin_rows = []
for name, (endog, shocks, treat) in name_map.items():
    sub = res[(res.endog == endog) & (res.shocks == shocks) & (res.treatment == treat)
              & (res.path == "uncond")]
    for v in ["gdp", "inf"]:
        for h in vl.H_REPORT:
            for sample in ["all", "excovid"]:
                r = sub[(sub.target == v) & (sub.h == h) & (sub["sample"] == sample)]
                if len(r):
                    fin_rows.append({"spec": name, "target": v, "h": h, "sample": sample,
                                     "rmse_final": round(float(r.rmse.min()), 3)})
fin = pd.DataFrame(fin_rows)

comp = sc.merge(fin, on=["spec", "target", "h", "sample"], how="left")
comp = comp.rename(columns={"rmse": "rmse_rt"})
comp["degr_pct"] = (100 * (comp.rmse_rt / comp.rmse_final - 1)).round(1)
comp.to_parquet("rt_compare.parquet")

# headline: shock advantage, final vs real-time, gdp
print("\n=== GDP: pure VAR vs all4_raw, final vs real-time ===")
for h in [1, 4, 8]:
    for sample in ["all", "excovid"]:
        pv = comp[(comp.spec == "pure_var") & (comp.target == "gdp") & (comp.h == h) & (comp["sample"] == sample)].iloc[0]
        a4 = comp[(comp.spec == "all4_raw") & (comp.target == "gdp") & (comp.h == h) & (comp["sample"] == sample)].iloc[0]
        fin_adv = 100 * (1 - a4.rmse_final / pv.rmse_final)
        rt_adv = 100 * (1 - a4.rmse_rt / pv.rmse_rt)
        print(f"h={h:<2} {sample:<8} final: VAR {pv.rmse_final:.3f} all4 {a4.rmse_final:.3f} adv {fin_adv:+.1f}%"
              f" | rt: VAR {pv.rmse_rt:.3f} all4 {a4.rmse_rt:.3f} adv {rt_adv:+.1f}%")

print("\n=== full comparison (gdp) ===")
print(comp[(comp.target == "gdp") & (comp["sample"] == "all")].sort_values(["h", "rmse_rt"])[
    ["spec", "h", "rmse_final", "rmse_rt", "degr_pct", "r2_oos", "n"]].to_string(index=False))
print("\n=== full comparison (inf, all) ===")
print(comp[(comp.target == "inf") & (comp["sample"] == "all")].sort_values(["h", "rmse_rt"])[
    ["spec", "h", "rmse_final", "rmse_rt", "degr_pct", "r2_oos", "n"]].to_string(index=False))

# stats for paper/notebook
out = {"gdp": {}, "inf": {}}
for v in ["gdp", "inf"]:
    for h in [1, 4, 8]:
        row = {}
        for sample in ["all", "excovid"]:
            c = comp[(comp.target == v) & (comp.h == h) & (comp["sample"] == sample)]
            pv = c[c.spec == "pure_var"].iloc[0]
            a4 = c[c.spec == "all4_raw"].iloc[0]
            row[sample] = {
                "var_final": pv.rmse_final, "var_rt": pv.rmse_rt,
                "all4_final": a4.rmse_final, "all4_rt": a4.rmse_rt,
                "adv_final_pct": round(100 * (1 - a4.rmse_final / pv.rmse_final), 1),
                "adv_rt_pct": round(100 * (1 - a4.rmse_rt / pv.rmse_rt), 1),
                "all4_r2_rt": a4.r2_oos, "n": int(a4.n),
            }
        out[v][h] = row
with open("rt_stats.json", "w") as f:
    json.dump(out, f, indent=1)
print(f"\ntotal {time.time()-t0:.0f}s")
