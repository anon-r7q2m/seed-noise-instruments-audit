import pandas as pd, numpy as np
pp = pd.read_parquet("data/analysis/pp_tidy.parquet")
print("cols:", list(pp.columns))
b = pp[(pp.task=="blimp") & (pp.metric=="acc")]
FINAL = b.step.max()
for size in ["14m","31m","70m","160m","410m"]:
    v = b[(b["size"]==size) & (b.step==FINAL)].set_index("seed")["value"].dropna()
    if len(v)==0: continue
    sd_all = v.std(ddof=1)/abs(v.mean())
    med, mad = v.median(), (v-v.median()).abs().median()
    z = 0.6745*(v-med)/mad if mad>0 else v*0
    keep = z.abs() <= 3
    sd_clean = v[keep].std(ddof=1)/abs(v[keep].mean()) if keep.sum()>=4 else np.nan
    print(f"{size}: n={len(v)} relsd_all={sd_all:.5f} relsd_excl_outlier={sd_clean:.5f}  outliers={list(v.index[~keep])} z={z.round(1).to_dict()}")
