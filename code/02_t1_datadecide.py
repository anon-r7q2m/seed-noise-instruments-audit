#!/usr/bin/env python3
"""T1 part B — port the S&N proxy test to DataDecide: 14 sizes x 25 recipes x 3 seeds.

For each (size, recipe):
  per task b in 10 OLMES benchmarks (+ mmlu aggregated, macro excluded from corr):
    x_b = relSD over final n ckpts (default seed, aligned eval grid), n = min(5, floor(grid/2)), >=3
    y_b = relSD across 3 seeds of final score (score = value at last common step; also
          'stab' variant: per-seed mean of last 3 ckpts)
  R(size,recipe)  = Pearson corr across tasks (raw and log10 space)
Also:
  - pooled correlation per size across recipe-task cells
  - calibration ratio y/x per cell
  - Monte-Carlo ceiling: R distribution expected if proxy were PERFECT given
    3-seed / n-ckpt chi^2 estimation error; and null (y independent of x).
Outputs: tables/t1b_*.csv, analysis parquet with cells.
"""
import pandas as pd, numpy as np, os
from scipy import stats

OUT = "data"
rng = np.random.default_rng(20260813)

df = pd.read_parquet(f"{OUT}/analysis/dd_tidy.parquet")
TASKS = ["arc_challenge","arc_easy","boolq","csqa","hellaswag","mmlu",
         "openbookqa","piqa","socialiqa","winogrande"]
SIZES = ["4M","6M","8M","10M","14M","16M","20M","60M","90M","150M","300M","530M","750M","1B"]
METRIC = "primary_metric"

df = df[df.task.isin(TASKS)]

def relsd(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 2 or np.mean(v) == 0: return np.nan
    return np.std(v, ddof=1) / abs(np.mean(v))

cells = []
for (p, d), g in df.groupby(["params","data"]):
    seeds = sorted(g.seed.unique())
    if len(seeds) < 3: continue
    # common step grid across seeds
    step_sets = [set(g[g.seed==s].step.unique()) for s in seeds]
    common = sorted(set.intersection(*step_sets))
    if len(common) < 3: continue
    final_step = common[-1]
    grid_n = len(common)
    n_ckpt = max(3, min(5, grid_n // 2))
    late_steps = common[-n_ckpt:]
    default_seed = "default"
    for t in TASKS:
        gt = g[g.task == t]
        # x: default-seed late-window relSD
        gx = gt[(gt.seed == default_seed) & (gt.step.isin(late_steps))].sort_values("step")
        x = relsd(gx[METRIC])
        # y: cross-seed relSD at final step
        yv = [gt[(gt.seed==s) & (gt.step==final_step)][METRIC].mean() for s in seeds]
        y = relsd(yv)
        # y stabilized: per-seed mean of last 3 common ckpts
        l3 = common[-3:]
        yv2 = [gt[(gt.seed==s) & (gt.step.isin(l3))][METRIC].mean() for s in seeds]
        y_stab = relsd(yv2)
        cells.append(dict(params=p, data=d, task=t, x=x, y=y, y_stab=y_stab,
                          n_ckpt=n_ckpt, grid_n=grid_n, final_step=final_step))

C = pd.DataFrame(cells)
C.to_parquet(f"{OUT}/analysis/t1b_cells.parquet", index=False)
print("cells:", C.shape)

def corr(sub, ycol="y", log=False):
    s = sub[["x", ycol]].dropna()
    s = s[(s.x > 0) & (s[ycol] > 0)]
    if len(s) < 5: return np.nan, len(s)
    a, b = (np.log10(s.x), np.log10(s[ycol])) if log else (s.x, s[ycol])
    return stats.pearsonr(a, b)[0], len(s)

# ---- per (size, recipe) R across 10 tasks
recs = []
for (p, d), sub in C.groupby(["params","data"]):
    r_raw, n1 = corr(sub, "y", log=False)
    r_log, _  = corr(sub, "y", log=True)
    r_stab, _ = corr(sub, "y_stab", log=True)
    ratio = (sub.y / sub.x).median()
    recs.append(dict(params=p, data=d, R_raw=r_raw, R_log=r_log, R_log_stab=r_stab,
                     med_ratio=ratio, n_tasks=n1))
R = pd.DataFrame(recs)
R.to_csv(f"{OUT}/tables/t1b_R_per_size_recipe.csv", index=False)

# ---- Monte-Carlo ceiling & null per (size,recipe)
def simulate_R(tau, n_ckpt, n_seed=3, mode="perfect", nsim=400):
    """tau: vector of true per-task noise. Observed x ~ tau*sqrt(chi2_{m}/m), y likewise.
    mode 'perfect': same tau for x and y. 'null': permuted tau for y."""
    m_x = np.maximum(np.asarray(n_ckpt) - 1, 1)
    m_y = n_seed - 1
    out = []
    k = len(tau)
    for _ in range(nsim):
        ty = rng.permutation(tau) if mode == "null" else tau
        x = tau * np.sqrt(rng.chisquare(m_x, k) / m_x)
        y = ty  * np.sqrt(rng.chisquare(m_y, k) / m_y)
        with np.errstate(all="ignore"):
            r = stats.pearsonr(np.log10(x), np.log10(y))[0]
        out.append(r)
    return np.array(out)

ceil_rows = []
for (p, d), sub in C.groupby(["params","data"]):
    s = sub.dropna(subset=["x","y"])
    s = s[(s.x>0)&(s.y>0)]
    if len(s) < 5: continue
    tau = s.x.to_numpy()
    sims_p = simulate_R(tau, s.n_ckpt.to_numpy(), mode="perfect")
    sims_0 = simulate_R(tau, s.n_ckpt.to_numpy(), mode="null")
    ceil_rows.append(dict(params=p, data=d,
                          ceil_med=np.nanmedian(sims_p), ceil_lo=np.nanquantile(sims_p,0.05),
                          null_hi=np.nanquantile(sims_0,0.95)))
CE = pd.DataFrame(ceil_rows)
R2 = R.merge(CE, on=["params","data"], how="left")
R2.to_csv(f"{OUT}/tables/t1b_R_per_size_recipe.csv", index=False)

# ---- summary by size
lines = []
for p in SIZES:
    s = R2[R2.params == p]
    if s.empty: continue
    lines.append(dict(params=p,
        R_log_med=s.R_log.median(), R_log_q25=s.R_log.quantile(.25), R_log_q75=s.R_log.quantile(.75),
        frac_R_ge_09=(s.R_log >= 0.9).mean(), frac_R_ge_ceil_lo=(s.R_log >= s.ceil_lo).mean(),
        frac_R_le_null=(s.R_log <= s.null_hi).mean(),
        ceil_med=s.ceil_med.median(), null_hi_med=s.null_hi.median(),
        med_ratio=s.med_ratio.median(), n_recipes=len(s)))
S = pd.DataFrame(lines)
S.to_csv(f"{OUT}/tables/t1b_summary_by_size.csv", index=False)
print(S.round(3).to_string())

# ---- pooled per size (cells pooled over recipes)
pool = []
for p in SIZES:
    s = C[(C.params==p)].dropna(subset=["x","y"])
    s = s[(s.x>0)&(s.y>0)]
    if len(s) < 20: continue
    r_log = stats.pearsonr(np.log10(s.x), np.log10(s.y))[0]
    ratio_med = (s.y/s.x).median()
    frac_2x = ((s.y/s.x).between(0.5,2)).mean()
    pool.append(dict(params=p, pooled_R_log=r_log, n_cells=len(s),
                     med_ratio=ratio_med, frac_ratio_in_2x=frac_2x))
P = pd.DataFrame(pool)
P.to_csv(f"{OUT}/tables/t1b_pooled_by_size.csv", index=False)
print(P.round(3).to_string())
