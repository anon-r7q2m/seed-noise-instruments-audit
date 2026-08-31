import itertools, numpy as np, pandas as pd
from scipy.special import gamma as gammafn
DD = "data/analysis/dd_tidy.parquet"
ORDER = ["4M","6M","8M","10M","14M","16M","20M","60M","90M","150M","300M","530M","750M","1B"]
C4_3 = np.sqrt(2.0/2)*gammafn(1.5)/gammafn(1.0)
def final_common_step(g, need=3):
    cnt = g.groupby("step")["seed"].nunique(); common = cnt[cnt>=need].index
    return None if len(common)==0 else common.max()
d = pd.read_parquet(DD)
rng = np.random.default_rng(0)
rows=[]
for p in ORDER:
    g = d[d["params"]==p]; per={}
    for mix, gg in g.groupby("data"):
        s = final_common_step(gg)
        if s is None: continue
        v = gg[gg.step==s].groupby("seed")["primary_metric"].mean()
        if len(v)>=3: per[mix]=(float(v.mean()), float(v.std(ddof=1)))
    if len(per)<5: continue
    mu = np.array([per[m][0] for m in per]); sds = np.array([per[m][1] for m in per])
    n = len(mu)
    def stats(mu_, sds_):
        so = mu_.std(ddof=1); nm = float(np.median(sds_))
        st = np.sqrt(max(so**2 - np.mean(sds_**2)/3.0, 0.0))
        return so/nm, st/nm, st/(nm/C4_3)
    obs = stats(mu, sds)
    B=2000; boot=np.empty((B,3))
    for b in range(B):
        idx = rng.integers(0, n, n)
        boot[b] = stats(mu[idx], sds[idx])
    lo = np.percentile(boot, 2.5, axis=0); hi = np.percentile(boot, 97.5, axis=0)
    frac0 = float((boot[:,1]<=1e-9).mean())
    rows.append(dict(size=p, snr_obs=round(obs[0],2),
        snr_dc=round(obs[1],2), dc_lo=round(lo[1],2), dc_hi=round(hi[1],2),
        snr_dc_c4=round(obs[2],2), c4_lo=round(lo[2],2), c4_hi=round(hi[2],2),
        frac_boot_zero=round(frac0,3)))
print(pd.DataFrame(rows).set_index("size").to_string())
