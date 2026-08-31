import pandas as pd, numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location("rcn", "./runs/exp-rank04/venue-reassess/recompute_new_numbers.py")
# just reuse constants
SN_INT = "data/raw_cache/sn_datadecide_intermediate.parquet"
def final_common_step(g, need=3):
    cnt = g.groupby("step")["seed"].nunique(); common = cnt[cnt>=need].index
    return None if len(common)==0 else common.max()
d = pd.read_parquet(SN_INT)
sizes = ["4M","6M","8M","10M","14M","16M","20M","60M","90M","150M","300M"]
d = d[d["size"].isin(sizes)]
mets = ["primary_metric","acc_raw","acc_per_char","acc_uncond","bits_per_byte_corr",
        "correct_prob_per_char","correct_logit_per_char","norm_correct_prob_per_char","margin_per_char"]
rows=[]
for (sz,mx,tk), g in d.groupby(["size","mix","task"]):
    s = final_common_step(g)
    if s is None: continue
    gg = g[g.step==s]; r = {"size":sz,"mix":mx,"task":tk}
    for m in mets:
        v = gg[m].values
        if len(v)==3 and np.all(np.isfinite(v)) and not np.all(v==0):
            r[m+"_mu"], r[m+"_sd"] = v.mean(), v.std(ddof=1)
    rows.append(r)
T = pd.DataFrame(rows)
out_obs, out_dec = {}, {}
for sz in sizes:
    g = T[T["size"]==sz]
    for m in mets:
        if m+"_mu" not in g: continue
        per = g.dropna(subset=[m+"_mu", m+"_sd"]).groupby("task").apply(
            lambda x: pd.Series({"sig": x[m+"_mu"].std(ddof=1),
                                 "noi_med": x[m+"_sd"].median(),
                                 "noi_rms": np.sqrt(np.mean(x[m+"_sd"]**2)),
                                 "dec": np.sqrt(max(x[m+"_mu"].var(ddof=1) - np.mean(x[m+"_sd"]**2)/3, 0.0))}),
            include_groups=False)
        if len(per):
            out_obs[(sz,m)] = float((per["sig"]/per["noi_med"]).median())
            out_dec[(sz,m)] = float((per["dec"]/per["noi_rms"]).median())
import json
print(f"{'metric':28s}", " ".join(f"{s:>6}" for s in sizes))
for m in mets:
    print(f"{m:28s}", " ".join(f"{out_dec.get((s,m),float('nan')):6.2f}" for s in sizes))
print("\nobserved (as currently printed):")
for m in mets:
    print(f"{m:28s}", " ".join(f"{out_obs.get((s,m),float('nan')):6.2f}" for s in sizes))
print("\nbpb/primary ratio: observed vs deconvolved:")
for s in sizes:
    o = out_obs.get((s,"bits_per_byte_corr")); p = out_obs.get((s,"primary_metric"))
    od = out_dec.get((s,"bits_per_byte_corr")); pd_ = out_dec.get((s,"primary_metric"))
    print(f"  {s}: {o/p:.2f} -> {od/pd_:.2f}")
json.dump({"obs":{f"{k[0]}|{k[1]}":v for k,v in out_obs.items()},
           "dec":{f"{k[0]}|{k[1]}":v for k,v in out_dec.items()}}, open("appg_deconv.json","w"))
