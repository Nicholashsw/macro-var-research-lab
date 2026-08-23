import json
import time
import numpy as np
import pandas as pd
import varx_lab as vl

t0 = time.time()
df = vl.build_panel()          # varx_lab.DATA_DIR is anchored to this file, not the cwd
print("panel", df.shape, df.index[0].date(), "to", df.index[-1].date())

study = vl.run_study(df)
train_end_i = df.index.get_loc(pd.Period(vl.TRAIN_END, freq="Q").to_timestamp())
bench = vl.ar_benchmark(df, ["gdp", "inf", "stance", "fx"], train_end_i, study["origins"])
res = vl.build_results(study, df, bench)
res["valid"] = vl.validity(res)
res.to_parquet("results.parquet")
print("results", res.shape, f"{time.time()-t0:.0f}s")

# quick sanity leaderboard
# NB: res["sample"] not res.sample -- DataFrame.sample is a method and shadows the column
sub = res[(res.target == "gdp") & (res.h == 1) & (res.path == "uncond") & (res["sample"] == "all") & res.valid]
print(sub.nsmallest(5, "rmse")[["endog", "shocks", "treatment", "lag_rule", "p", "rmse", "r2_oos"]].to_string())

# dashboard payload: metrics for every spec row plus forecast series at h in {1,4}
idx = study["index"]
origins = study["origins"]
payload = {
    "meta": {
        "sample": [str(idx[0].date()), str(idx[-1].date())],
        "train_end": vl.TRAIN_END, "n_origins": len(origins),
        "targets": ["gdp", "inf", "stance", "fx"], "h": vl.H_REPORT,
        "labels": vl.LABELS,
    },
    "specs": [], "metrics": {}, "series": {}, "actual": {},
}

for v in ["gdp", "inf", "stance", "fx"]:
    for h in [1, 4]:
        dates, ys = [], []
        for oi, org in enumerate(origins):
            t = org + h
            if t >= len(idx):
                continue
            dates.append(str(idx[t].date())[:7])
            ys.append(round(float(df[v].to_numpy()[t]), 2))
        payload["actual"][f"{v}_{h}"] = {"dates": dates, "y": ys}

seen = {}
for s in study["specs"]:
    key = s["model_key"]
    m = study["models"][key]
    payload["specs"].append({
        "sid": s["spec_id"], "mid": m["model_id"],
        "endog": s["endog"], "addons": s["endog"][3:],
        "shocks": s["shocks"], "treat": s["treatment"],
        "rule": s["lag_rule"], "p": s["p"],
        "stab": round(m["stab"], 3), "lb": round(m["lb_min_p"], 4),
        "dof": round(m["dof_ratio"], 2),
    })
    if m["model_id"] not in seen:
        seen[m["model_id"]] = True
        fc_all = study["forecasts"][key]
        ser = {}
        for path, fc in fc_all.items():
            for v in ["gdp", "inf", "stance", "fx"]:
                if v not in s["endog"]:
                    continue
                j = s["endog"].index(v)
                for h in [1, 4]:
                    vals = []
                    for oi, org in enumerate(origins):
                        if org + h >= len(idx):
                            continue
                        vals.append(round(float(fc[oi, h - 1, j]), 2))
                    ser[f"{path}_{v}_{h}"] = vals
        payload["series"][str(m["model_id"])] = ser

rk = res.set_index(["spec_id", "path", "target", "h", "sample"])
met = {}
for (sid, path, v, h, sample), row in rk.iterrows():
    met.setdefault(str(sid), {})[f"{path}_{v}_{h}_{sample}"] = [
        round(row["rmse"], 3), round(row["mae"], 3), round(row["r2_oos"], 3),
        None if not np.isfinite(row["dm_p"]) else round(row["dm_p"], 3),
        bool(row["valid"]),
    ]
payload["metrics"] = met

with open("payload.json", "w") as f:
    json.dump(payload, f, separators=(",", ":"))
import os
print("payload MB", os.path.getsize("payload.json") / 1e6)
print(f"total {time.time()-t0:.0f}s")
