import numpy as np, pandas as pd
DD = "data/analysis/dd_tidy.parquet"
ORDER = ["4M","6M","8M","10M","14M","16M","20M","60M","90M","150M","300M","530M","750M","1B"]
MED3 = float(np.sqrt(np.log(2)))
def final_common_step(g, need=3):
    cnt = g.groupby("step")["seed"].nunique(); common = cnt[cnt>=need].index
    return None if len(common)==0 else common.max()
d = pd.read_parquet(DD)
rng = np.random.default_rng(0); B=2000
rows=[]
for p in ORDER:
    g = d[d["params"]==p]; per={}
    for mix, gg in g.groupby("data"):
        s = final_common_step(gg)
        if s is None: continue
        v = gg[gg.step==s].groupby("seed")["primary_metric"].mean()
        if len(v)>=3: per[mix]=(float(v.mean()), float(v.std(ddof=1)))
    names=list(per); mu=np.array([per[m][0] for m in per]); sds=np.array([per[m][1] for m in per])
    n=len(mu); S2=mu.var(ddof=1); meanv=np.mean(sds**2)
    sig_obs=np.sqrt(S2); sig_dec=np.sqrt(max(S2-meanv/3,0)); rms=np.sqrt(meanv); med=np.median(sds)
    db=[]; rb=[]
    for b in range(B):
        idx=rng.integers(0,n,n)
        S2b=mu[idx].var(ddof=1); mvb=np.mean(sds[idx]**2)
        db.append(S2b-mvb/3); rb.append(np.sqrt(max(S2b-mvb/3,0))/np.sqrt(mvb))
    db=np.array(db); rb=np.array(rb)
    keep=np.argsort(sds)[:-2]
    S2t=mu[keep].var(ddof=1); mvt=np.mean(sds[keep]**2)
    rows.append(dict(size=p,
        signal_obs=f"{sig_obs:.5f}", signal_dec=f"{sig_dec:.5f}",
        noise_med=f"{med:.5f}", noise_rms=f"{rms:.5f}",
        snr_obs=f"{sig_obs/med:.2f}", snr_dec=f"{sig_dec/rms:.2f}",
        ci_lo=f"{np.percentile(rb,2.5):.2f}", ci_hi=f"{np.percentile(rb,97.5):.2f}",
        het=f"{meanv/med**2:.2f}",
        untrunc=f"{S2-meanv/3:.2e}", ut_lo=f"{np.percentile(db,2.5):.2e}", ut_hi=f"{np.percentile(db,97.5):.2e}",
        trim=f"{np.sqrt(max(S2t-mvt/3,0))/np.sqrt(mvt):.2f}",
        snr_med0833=f"{sig_dec/(med/MED3):.2f}"))
df=pd.DataFrame(rows).set_index("size")
df.to_csv("w1_v2_final.csv")
pd.set_option("display.width",250)
print(df.to_string())
