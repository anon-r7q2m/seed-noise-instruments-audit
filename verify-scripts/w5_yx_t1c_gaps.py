import pandas as pd, numpy as np
from scipy.special import digamma
def b_log(nu):
    return 0.5*(digamma(nu/2.0) + np.log(2.0) - np.log(nu))
c = pd.read_parquet("data/analysis/t1b_cells.parquet")
ORDER = ["4M","6M","8M","10M","14M","16M","20M","60M","90M","150M","300M","530M","750M","1B"]
print("== y/x per size, paper aggregation (median of per-recipe medians) ==")
rows=[]
for p in ORDER:
    s = c[c.params==p].dropna(subset=["x","y"]); s=s[(s.x>0)&(s.y>0)]
    per=[]
    for mix,g in s.groupby("data"):
        corr = np.exp(b_log(int(g.n_ckpt.median())-1) - b_log(2))
        per.append(((g.y/g.x).median(), ((g.y/g.x)*corr).median()))
    raw = np.median([a for a,_ in per]); cor = np.median([b for _,b in per])
    rows.append((p, round(raw,2), round(cor,2)))
print(pd.DataFrame(rows, columns=["size","yx_raw","yx_corr"]).set_index("size").to_string())
s = c.dropna(subset=["x","y"]); s=s[(s.x>0)&(s.y>0)]
corr = np.array([np.exp(b_log(int(mm)-1)-b_log(2)) for mm in s.n_ckpt])
print("pooled cell-level: raw", round(float((s.y/s.x).median()),2), "-> corrected", round(float(((s.y/s.x)*corr).median()),2))

# --- t1c non-monotonicity diagnostic
t = pd.read_parquet("data/analysis/t1c_ppl_cells.parquet")
print("\n== t1c ppl: n_ckpt and detrended ratio by size ==")
for p in ORDER:
    sp = t[t.params==p].dropna(subset=["x","x_dt","y"]); sp=sp[(sp.x>0)&(sp.y>0)&(sp.x_dt>0)]
    if len(sp)<10: continue
    print(f"{p}: n_ckpt med {int(sp.n_ckpt.median())}, cells {len(sp)}, "
          f"raw y/x {float((sp.y/sp.x).median()):.2f}, detrended {float((sp.y/sp.x_dt).median()):.2f}, "
          f"within2x_dt {float((((sp.y/sp.x_dt)>=0.5)&((sp.y/sp.x_dt)<=2)).mean()*100):.0f}%")

# --- ablation gaps: adjacent-recipe gaps in final log-ppl (c4_en) and in macro acc
print("\n== adjacent-recipe gaps (sorted recipe means, successive diffs), by scale ==")
dd = pd.read_parquet("data/analysis/dd_tidy.parquet")
def final_common_step(g, need=3):
    cnt = g.groupby("step")["seed"].nunique(); common = cnt[cnt>=need].index
    return None if len(common)==0 else common.max()
for p in ["4M","10M","20M","150M"]:
    g = dd[dd.params==p]
    mus = []
    for mix, gg in g.groupby("data"):
        s_ = final_common_step(gg)
        if s_ is None: continue
        v = gg[gg.step==s_].groupby("seed")["primary_metric"].mean()
        if len(v)>=3: mus.append(v.mean())
    mus = np.sort(mus); gaps = np.diff(mus)
    print(f"{p}: n={len(mus)} adjacent-gap median {np.median(gaps):.4f}, IQR [{np.percentile(gaps,25):.4f}, {np.percentile(gaps,75):.4f}] (macro acc units)")
# log-ppl gaps from t1c cells (y is seed SD; need means - recompute from dd? use c4_en final means)
tt = t[t.dom=="c4_en"]
print("\n== log-ppl adjacent gaps across recipes (c4_en), by scale ==")
# per (params,data): need recipe mean log-ppl at final step -> not in t1c (only SDs). Use dd logppl? check columns
print("dd cols:", [x for x in dd.columns])
