# LOO-seed variant of the T1 porting: for each held-out seed s, x_s = relSD of seed s's
# last-n_ckpt checkpoints; y_-s = relSD of the other two seeds' final-step values.
# R_LOO per cell = mean over the 3 combos of corr(log x_s, log y_-s) across tasks.
# Ceiling: MC perfect-proxy under the same LOO construction (nu_y = 1).
import pandas as pd, numpy as np
from scipy import stats
d = pd.read_parquet("data/analysis/dd_tidy.parquet")
ORDER = ["4M","6M","8M","10M","14M","16M","20M","60M","90M","150M","300M","530M","750M","1B"]
TASKS = sorted(t for t in d.task.unique() if t != "olmes_10_macro_avg")
rng = np.random.default_rng(2)
def relsd(v):
    v = np.asarray(v, float); m = np.abs(np.mean(v))
    return np.nan if m == 0 else float(np.std(v, ddof=1)/m)

# build LOO cells
cells = []
for p in ORDER:
    g0 = d[d.params==p]
    for mix, g in g0.groupby("data"):
        seeds = sorted(g.seed.unique())
        if len(seeds) < 3: continue
        step_sets = [set(g[g.seed==s].step.unique()) for s in seeds]
        common = sorted(set.intersection(*step_sets))
        if len(common) < 3: continue
        final_step = common[-1]
        n_ckpt = max(3, min(5, len(common)//2))
        late = common[-n_ckpt:]
        for t in TASKS:
            gt = g[g.task==t]
            xs, ys = [], []
            for s_ in seeds:
                gx = gt[(gt.seed==s_) & (gt.step.isin(late))].sort_values("step")
                x = relsd(gx["primary_metric"])
                yv = [gt[(gt.seed==o) & (gt.step==final_step)]["primary_metric"].mean() for o in seeds if o != s_]
                y = relsd(yv)
                xs.append(x); ys.append(y)
            cells.append(dict(params=p, data=mix, task=t, x0=xs[0], y0=ys[0], x1=xs[1], y1=ys[1], x2=xs[2], y2=ys[2], n_ckpt=n_ckpt))
C = pd.DataFrame(cells)
print("LOO cells:", C.shape)

def cell_R(sub):
    rs = []
    for k in range(3):
        s = sub[[f"x{k}", f"y{k}"]].dropna(); s = s[(s[f"x{k}"]>0)&(s[f"y{k}"]>0)]
        if len(s) < 5: return np.nan
        rs.append(stats.pearsonr(np.log(s[f"x{k}"]), np.log(s[f"y{k}"]))[0])
    return float(np.mean(rs))

# MC ceiling under LOO protocol: perfect proxy, nu_y=1
def sim_ceiling(tau, n_ckpt, nsim=200):
    m_x = max(int(n_ckpt)-1, 1)
    out = []
    k = len(tau)
    for _ in range(nsim):
        rs = []
        for combo in range(3):
            x = tau*np.sqrt(rng.chisquare(m_x,k)/m_x)
            y = tau*np.sqrt(rng.chisquare(1,k)/1.0)
            with np.errstate(all="ignore"):
                rs.append(stats.pearsonr(np.log(x), np.log(y))[0])
        out.append(np.mean(rs))
    return np.nanmedian(out)

rows=[]
for p in ORDER:
    s = C[C.params==p]
    per = []
    for mix, sub in s.groupby("data"):
        r = cell_R(sub)
        if np.isnan(r): continue
        # ceiling with tau = default-combo x vector of this cell
        tau = sub["x0"].dropna().to_numpy(); tau = tau[tau>0]
        if len(tau) < 5: continue
        cm = sim_ceiling(tau, int(sub.n_ckpt.median()))
        per.append((r, cm))
    R_med = float(np.median([r for r,_ in per])); ceil = float(np.median([cm for _,cm in per]))
    lam = R_med/ceil
    arr = np.array([r for r,_ in per]); n = len(arr)
    bs = [np.median(arr[rng.integers(0,n,n)])/ceil for _ in range(2000)]
    rows.append(dict(size=p, R_loo=round(R_med,2), ceil_loo=round(ceil,2),
        lam_loo=round(lam,2), lo=round(np.percentile(bs,2.5),2), hi=round(np.percentile(bs,97.5),2)))
df = pd.DataFrame(rows).set_index("size")
pd.set_option("display.width",160)
print(df.to_string())
print("\nmedian lam_loo:", df.lam_loo.median(), " | CIs covering 0.9:", int(((df.lo<=0.9)&(df.hi>=0.9)).sum()), "/14")
