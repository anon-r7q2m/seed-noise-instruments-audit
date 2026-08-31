# W1 v2: PI-adjudicated estimands.
# Main: signal_dec / RMS(noise), RMS = sqrt(mean sd^2)  [no c4 needed]
# Robustness: signal_dec / (median sd / sqrt(ln2))     [median correction 0.8326]
# Diagnostics: mean(sd^2)/median^2, top-3 noisy recipes, trimmed (drop top-2 sd),
#              untruncated S^2 - mean(sd^2)/2... wait /3 with its bootstrap CI (signed),
#              paired bootstrap of SNR_90M/SNR_60M.
import numpy as np, pandas as pd
DD = "data/analysis/dd_tidy.parquet"
ORDER = ["4M","6M","8M","10M","14M","16M","20M","60M","90M","150M","300M","530M","750M","1B"]
MED3 = float(np.sqrt(np.log(2)))   # median of sqrt(chi2_2/2)
def final_common_step(g, need=3):
    cnt = g.groupby("step")["seed"].nunique(); common = cnt[cnt>=need].index
    return None if len(common)==0 else common.max()
d = pd.read_parquet(DD)
cells = {}
for p in ORDER:
    g = d[d["params"]==p]; per={}
    for mix, gg in g.groupby("data"):
        s = final_common_step(gg)
        if s is None: continue
        v = gg[gg.step==s].groupby("seed")["primary_metric"].mean()
        if len(v)>=3: per[mix]=(float(v.mean()), float(v.std(ddof=1)))
    if len(per)>=5: cells[p]=per

rng = np.random.default_rng(0); B=2000
rows=[]
boot_store={}
for p, per in cells.items():
    names=list(per); mu=np.array([per[m][0] for m in per]); sds=np.array([per[m][1] for m in per])
    n=len(mu)
    S2 = mu.var(ddof=1); meanv = np.mean(sds**2)
    sig_obs = np.sqrt(S2)
    sig_dec = np.sqrt(max(S2-meanv/3,0))
    rms = np.sqrt(meanv); med = np.median(sds)
    snr_obs = sig_obs/med
    snr_rms = sig_dec/rms
    snr_med = sig_dec/(med/MED3)
    # untruncated signed diff and its bootstrap
    d0 = S2 - meanv/3
    db = []
    rb = []
    for b in range(B):
        idx = rng.integers(0,n,n)
        S2b = mu[idx].var(ddof=1); mvb = np.mean(sds[idx]**2)
        db.append(S2b - mvb/3)
        rb.append(np.sqrt(max(S2b-mvb/3,0))/np.sqrt(mvb))
    db=np.array(db); rb=np.array(rb)
    # trimmed: drop top-2 sd
    keep = np.argsort(sds)[:-2]
    S2t = mu[keep].var(ddof=1); mvt = np.mean(sds[keep]**2)
    sig_dec_t = np.sqrt(max(S2t-mvt/3,0)); snr_rms_t = sig_dec_t/np.sqrt(mvt)
    top3 = sorted(per.items(), key=lambda kv:-kv[1][1])[:3]
    boot_store[p]=rb
    rows.append(dict(size=p, het=round(meanv/med**2,2),
        snr_obs=round(snr_obs,2), snr_rms=round(snr_rms,2), snr_med0833=round(snr_med,2),
        rms=round(rms,5),
        untrunc=round(d0,8), ut_lo=round(np.percentile(db,2.5),8), ut_hi=round(np.percentile(db,97.5),8),
        rms_lo=round(np.percentile(rb,2.5),2), rms_hi=round(np.percentile(rb,97.5),2),
        trim_snr=round(snr_rms_t,2),
        top3="; ".join(f"{k}={v[1]:.4f}" for k,v in top3)))
df = pd.DataFrame(rows).set_index("size")
pd.set_option("display.width",250); pd.set_option("display.max_colwidth",80)
print(df.to_string())
# paired bootstrap 90M vs 60M (same 25 recipes)
a60, a90 = cells["60M"], cells["90M"]
common = [k for k in a60 if k in a90]
m60 = np.array([a60[k][0] for k in common]); s60 = np.array([a60[k][1] for k in common])
m90 = np.array([a90[k][0] for k in common]); s90 = np.array([a90[k][1] for k in common])
def snr_rms_(m,s):
    S2=m.var(ddof=1); mv=np.mean(s**2); return np.sqrt(max(S2-mv/3,0))/np.sqrt(mv)
ratio=[]
n=len(common)
for b in range(B):
    idx=rng.integers(0,n,n)
    r60=snr_rms_(m60[idx],s60[idx]); r90=snr_rms_(m90[idx],s90[idx])
    if r60>1e-9: ratio.append(r90/r60)
ratio=np.array(ratio)
print(f"\npaired bootstrap SNR_90M/SNR_60M (RMS def): point={snr_rms_(m90,s90)/snr_rms_(m60,s60):.2f}  "
      f"95% CI=[{np.percentile(ratio,2.5):.2f}, {np.percentile(ratio,97.5):.2f}]  P(ratio<=1)={float((ratio<=1).mean()):.4f}")
