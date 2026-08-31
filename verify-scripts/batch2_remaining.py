import pandas as pd, numpy as np
from scipy.special import digamma
from scipy import stats
# ---- (2) y/x median-basis correction
def med_bias(nu):  # median of 0.5*log(chi2_nu/nu)
    return 0.5*(np.log(stats.chi2.median(nu)) - np.log(nu))
c = pd.read_parquet("data/analysis/t1b_cells.parquet")
ORDER = ["4M","6M","8M","10M","14M","16M","20M","60M","90M","150M","300M","530M","750M","1B"]
print("median-basis bias terms: nu=2:", round(med_bias(2),4), " nu=4:", round(med_bias(4),4))
rows=[]
for p in ORDER:
    s = c[c.params==p].dropna(subset=["x","y"]); s=s[(s.x>0)&(s.y>0)]
    per=[]
    for mix,g in s.groupby("data"):
        m = int(g.n_ckpt.median())
        corr = np.exp(med_bias(m-1) - med_bias(2))  # divide out: log y - log x bias
        per.append(((g.y/g.x).median(), ((g.y/g.x)*corr).median()))
    rows.append((p, round(np.median([a for a,_ in per]),2), round(np.median([b for _,b in per]),2)))
print(pd.DataFrame(rows, columns=["size","yx_raw","yx_corr_med"]).set_index("size").to_string())
s = c.dropna(subset=["x","y"]); s=s[(s.x>0)&(s.y>0)]
corr = np.array([np.exp(med_bias(int(mm)-1)-med_bias(2)) for mm in s.n_ckpt])
yc = (s.y/s.x)*corr
print("pooled: raw", round(float((s.y/s.x).median()),3), "-> median-corrected", round(float(yc.median()),3),
      "| within-2x:", round(float((((s.y/s.x)>=0.5)&((s.y/s.x)<=2)).mean()*100),1),
      "->", round(float(((yc>=0.5)&(yc<=2)).mean()*100),1))

# ---- (4) per-domain homogeneity rejection counts (MC test under chi2_2)
t = pd.read_parquet("data/analysis/t1c_ppl_cells.parquet")
rng = np.random.default_rng(5)
def mc_p(logsd, R=2000):
    # observed across-recipe variance of log sd; null: log sd_i = log sigma + 0.5 log(chi2_2/2)
    v = np.var(logsd, ddof=1)
    n = len(logsd)
    sim = np.var(np.log(np.sqrt(rng.chisquare(2, size=(R,n))/2)), axis=1, ddof=1)
    return float((sim >= v).mean()), v
print("\nper-domain homogeneity rejections (p<0.05) per scale:")
for p in ORDER:
    sp = t[t.params==p]
    nrej, ntot = 0, 0
    for dom, g in sp.groupby("dom"):
        y = g.y.dropna().to_numpy(); y = y[y>0]
        if len(y) < 10: continue
        ntot += 1
        pv, _ = mc_p(np.log(y))
        if pv < 0.05: nrej += 1
    print(f"  {p}: {nrej}/{ntot}")

# ---- (5) extreme-small-sigma cell consistency check (c4_en, 150M Falcon+CC (QC 10%); 10M DCLM-Baseline (QC 7%, FW3))
ppl = pd.read_parquet("data/raw_cache/dd_ppl.parquet")
col = "eval/c4_en-validation/Perplexity"
for sz, mix in [("150M","Falcon+CC (QC 10%)"), ("10M","DCLM-Baseline (QC 7%, FW3)")]:
    g = ppl[(ppl.params==sz)&(ppl["data"]==mix)]
    print(f"\n{sz} / {mix}: seeds={sorted(g.seed.unique())}, steps per seed:",
          {int(s_): int((g.seed==s_).sum()) for s_ in g.seed.unique()})
    fin = g[g.step==g.step.max()]
    print("  final-step log-ppl per seed:", {int(r.seed): round(float(np.log(r[col])),6) for r in fin.itertuples()})
    # trajectory diversity: std of per-seed trajectories at 3 points
    for st in sorted(g.step.unique())[-3:]:
        print(f"   step {int(st)}:", {int(r.seed): round(float(np.log(r[col])),6) for r in g[g.step==st].itertuples()})

# ---- (6) first-difference detrending on t1c
print("\n== first-difference variant of the proxy (t1c) ==")
# need per-seed per-step ppl series: rebuild proxy from dd_ppl directly
def fd_relsd(series):
    v = np.asarray(series, float)
    d = np.diff(v)
    if len(d)==0: return np.nan
    sd = np.std(d, ddof=1)/np.sqrt(2)
    return sd/abs(np.mean(v))
rows=[]
for p in ORDER:
    sp = t[t.params==p]
    # y per cell from t1c; x_fd rebuilt per cell from raw ppl
    ymap = {(r.data, r.dom): r.y for r in sp.itertuples()}
    ratios = []
    gp = ppl[ppl.params==p]
    for (mix, dom), g2 in gp.groupby(["data"]):
        pass
    # too heavy; do per (data, dom) using default seed's late window like t1c protocol
    out = []
    for (mix), g2 in gp.groupby("data"):
        seeds = sorted(g2.seed.unique())
        if len(seeds)<3: continue
        step_sets = [set(g2[g2.seed==s].step.unique()) for s in seeds]
        common = sorted(set.intersection(*step_sets))
        if len(common)<3: continue
        n_ckpt = max(3, min(5, len(common)//2)); late = common[-n_ckpt:]
        for domcol in [c_ for c_ in ppl.columns if c_.startswith("eval/")]:
            domname = domcol.split("/")[1].replace("-validation","")
            key = (mix, domname)
            if key not in ymap: continue
            gx = g2[(g2.seed==seeds[0]) & (g2.step.isin(late))].sort_values("step")
            x_fd = fd_relsd(gx[domcol])
            if x_fd and np.isfinite(x_fd) and ymap[key]>0:
                out.append(ymap[key]/x_fd)
    if out:
        print(f"  {p}: median y/x_fd = {np.median(out):.2f}  (n={len(out)})")
