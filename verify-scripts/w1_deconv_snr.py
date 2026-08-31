# W1 verification: reproduce Table tab:snr exactly (N1 protocol), then deconvolve.
# signal_obs = SD over 25 recipe 3-seed means; noise = median within-recipe 3-seed SD.
# Deconvolution: signal_true^2 = signal_obs^2 - mean_i(sd_i^2)/3   (variance-scale, unbiased)
# c4(3) = sqrt(2/2)*Gamma(3/2)/Gamma(2/2)... for n=3: c4 = sqrt(2/(n-1)) * Gamma(n/2)/Gamma((n-1)/2)
import itertools
import numpy as np
import pandas as pd
from scipy.special import gamma as gammafn

DD = "data/analysis/dd_tidy.parquet"
ORDER = ["4M","6M","8M","10M","14M","16M","20M","60M","90M","150M","300M","530M","750M","1B"]

def c4(n):
    return np.sqrt(2.0/(n-1)) * gammafn(n/2.0) / gammafn((n-1)/2.0)

C4_3 = c4(3)   # ~0.8862

def final_common_step(g, need=3):
    cnt = g.groupby("step")["seed"].nunique()
    common = cnt[cnt >= need].index
    return None if len(common) == 0 else common.max()

d = pd.read_parquet(DD)
print("columns:", list(d.columns))
rows = []
for p in ORDER:
    g = d[d["params"] == p]
    per = {}
    for mix, gg in g.groupby("data"):
        s = final_common_step(gg)
        if s is None:
            continue
        v = gg[gg.step == s].groupby("seed")["primary_metric"].mean()
        if len(v) >= 3:
            per[mix] = (float(v.mean()), float(v.std(ddof=1)))
    if len(per) < 5:
        continue
    mu  = np.array([per[m][0] for m in per])
    sds = np.array([per[m][1] for m in per])
    signal_obs = float(mu.std(ddof=1))
    noise_med  = float(np.median(sds))
    # --- deconvolution ---
    deconv_term = float(np.mean(sds**2)) / 3.0          # mean per-recipe 3-seed-mean variance
    sig2_true = max(signal_obs**2 - deconv_term, 0.0)
    signal_true = np.sqrt(sig2_true)
    snr_obs   = signal_obs / noise_med
    snr_dc    = signal_true / noise_med                 # deconvolved signal, same noise column
    snr_dc_c4 = signal_true / (noise_med / C4_3)        # + c4-unbiasing of the noise SD
    rows.append(dict(size=p, n_recipes=len(per),
        signal_obs=round(signal_obs,5), noise_med=round(noise_med,5),
        snr_obs=round(snr_obs,3),
        mean_sd2_over3=round(deconv_term,7),
        signal_true=round(float(signal_true),5),
        snr_deconv=round(float(snr_dc),3),
        snr_deconv_c4=round(float(snr_dc_c4),3),
        mean_sd2_vs_med_sd2=round(float(np.mean(sds**2)/np.median(sds)**2),3)))

df = pd.DataFrame(rows).set_index("size")
pd.set_option("display.width", 200)
print(df.to_string())
print("\nc4(3) =", round(C4_3,4))
