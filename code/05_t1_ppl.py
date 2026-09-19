#!/usr/bin/env python3
"""T1 supplement — perplexity (continuous-metric) version of the proxy test on DataDecide.

Reads tmp/data/dd_ppl.parquet (per-seed per-step perplexity on 11 validation domains);
writes data/analysis/t1c_ppl_cells.parquet and data/tables/t1c_ppl_summary_by_size.csv.
Statistics are on log(ppl): absolute SD of log ppl equals relative SD of ppl.
Includes a detrended-x variant (SD of residuals from a linear fit in step).
"""
import pandas as pd, numpy as np
from scipy import stats

TMP = "./tmp"
OUT = "data"

ppl = pd.read_parquet(f"{TMP}/data/dd_ppl.parquet")
DOMS = [c for c in ppl.columns if c.startswith("eval/")]
SIZES = ["4M","6M","8M","10M","14M","16M","20M","60M","90M","150M","300M","530M","750M","1B"]

cells = []
for (p, d), g in ppl.groupby(["params","data"]):
    seeds = sorted(g.seed.unique())
    if len(seeds) < 3: continue
    common = sorted(set.intersection(*[set(g[g.seed==s].step) for s in seeds]))
    if len(common) < 4: continue
    n_ckpt = max(3, min(5, len(common)//2))
    late = common[-n_ckpt:]
    final = common[-1]; l2 = common[-2:]
    gd = {s: g[g.seed==s].set_index("step") for s in seeds}
    for dom in DOMS:
        xv = np.log(gd["default"].loc[[s for s in late if s in gd["default"].index], dom].astype(float))
        x = np.std(xv, ddof=1) if len(xv) >= 3 else np.nan
        # detrended x: SD of residuals from a linear fit in step over the same window
        if len(xv) >= 3:
            st = np.array([s for s in late if s in gd["default"].index], float)
            res = xv.to_numpy() - np.polyval(np.polyfit(st, xv.to_numpy(), 1), st)
            x_dt = np.std(res, ddof=1) * np.sqrt(len(xv)/max(len(xv)-2,1))  # dof adj
        else:
            x_dt = np.nan
        yv = [np.log(gd[s].loc[final, dom]) if final in gd[s].index else np.nan for s in seeds]
        yv = np.array(yv, float)
        y = np.std(yv[np.isfinite(yv)], ddof=1) if np.isfinite(yv).sum() >= 3 else np.nan
        yv2 = [np.log(gd[s].loc[[t for t in l2 if t in gd[s].index], dom].astype(float)).mean() for s in seeds]
        yv2 = np.array(yv2, float)
        y2 = np.std(yv2[np.isfinite(yv2)], ddof=1) if np.isfinite(yv2).sum() >= 3 else np.nan
        cells.append(dict(params=p, data=d, dom=dom.replace("eval/","").replace("-validation/Perplexity",""),
                          x=x, x_dt=x_dt, y=y, y_stab=y2, n_ckpt=n_ckpt, n_common=len(common)))

C = pd.DataFrame(cells)
C.to_parquet(f"{OUT}/analysis/t1c_ppl_cells.parquet", index=False)
print("cells:", C.shape)

rows = []
for p in SIZES:
    s = C[(C.params==p)].dropna(subset=["x","y"])
    s = s[(s.x>0)&(s.y>0)]
    if len(s) < 20: continue
    # per-recipe R across 11 domains
    Rs = []
    for d_, sd in s.groupby("data"):
        if len(sd) >= 6:
            Rs.append(stats.pearsonr(np.log10(sd.x), np.log10(sd.y))[0])
    pooled = stats.pearsonr(np.log10(s.x), np.log10(s.y))[0]
    ratio = (s.y/s.x)
    ratio_stab = (s.y_stab/s.x).dropna()
    sdt = s.dropna(subset=["x_dt"]); sdt = sdt[sdt.x_dt>0]
    ratio_dt = (sdt.y/sdt.x_dt)
    pooled_dt = stats.pearsonr(np.log10(sdt.x_dt), np.log10(sdt.y))[0] if len(sdt)>=20 else np.nan
    rows.append(dict(params=p, n_cells=len(s), R_med_recipe=np.nanmedian(Rs) if Rs else np.nan,
                     pooled_R_log=pooled, pooled_R_log_dt=pooled_dt, med_ratio=ratio.median(),
                     med_ratio_stab=ratio_stab.median(), med_ratio_dt=ratio_dt.median(),
                     frac_ratio_in_2x=ratio.between(0.5,2).mean(),
                     frac_ratio_dt_in_2x=ratio_dt.between(0.5,2).mean(),
                     med_x=s.x.median(), med_y=s.y.median()))
S = pd.DataFrame(rows)
S.to_csv(f"{OUT}/tables/t1c_ppl_summary_by_size.csv", index=False)
print(S.round(3).to_string(index=False))
print("\nnote: med_x/med_y are SD of log-ppl ~ relSD of ppl (nats/token equivalent on mean-log scale)")
