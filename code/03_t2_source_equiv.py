#!/usr/bin/env python3
"""T2 — source equivalence: init-seed-only vs data-order-only variance shares.

Data: allenai/signal-and-noise random_seeds (10 init-seed runs, 9 data-order runs, 1B-5xC).
Published context: 2605.20798 varies init only and implicitly treats it as the full
run-to-run noise floor; S&N A.3.1 reports init/order/ckpt noises all proxied by final-n relSD.
Question: is sigma_init ~= sigma_order per benchmark? And does bundled-seed noise
(DataDecide 3 seeds, both sources varied) match sqrt(sigma_init^2+sigma_order^2)?

Estimators:
  score(run) = mean of metric over last 3 ckpts <= common final step (69000)
  sigma_init = relSD over 10 seed runs; sigma_order = relSD over 9 data runs
  ratio = sigma_init / sigma_order, CI: (a) F-quantile based (normal theory, df 9/8),
  (b) bootstrap over runs (2000 draws).
Also variance shares: share_init = s_i^2/(s_i^2+s_o^2).
DataDecide 1B: bundled 3-seed relSD per OLMES task per recipe -> distribution vs
sqrt(s_i^2+s_o^2) from S&N (different corpus: descriptive comparison only).
"""
import pandas as pd, numpy as np
from scipy import stats

TMP = "./tmp/rank04-zerogpu"
OUT = "data"
rng = np.random.default_rng(20260813)

rs = pd.read_parquet(f"{TMP}/data/sn_random_seeds.parquet")
COMMON_FINAL = 69000

def pick_metric(g):
    ms = set(g["metric"].unique())
    if "acc_per_char" in ms: return "acc_per_char"
    if "acc" in ms: return "acc"
    return None

def run_final_score(g, step_final=COMMON_FINAL, avg_last=3):
    gg = g[g.step <= step_final].sort_values("step")
    if len(gg) == 0: return np.nan
    return gg["value"].tail(avg_last).mean()

def relsd(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 2 or np.mean(v) == 0: return np.nan
    return np.std(v, ddof=1) / abs(np.mean(v))

rows = []
for metric_mode in ["primary_like", "bits_per_byte"]:
    for t in sorted(rs.task_name.unique()):
        m = pick_metric(rs[rs.task_name==t]) if metric_mode == "primary_like" else "bits_per_byte"
        if m is None: continue
        g = rs[(rs.task_name == t) & (rs.metric == m)]
        if g.empty: continue
        si = np.array([run_final_score(gr) for _, gr in g[g.run_type=="seed"].groupby("run_name")])
        so = np.array([run_final_score(gr) for _, gr in g[g.run_type=="data"].groupby("run_name")])
        si, so = si[np.isfinite(si)], so[np.isfinite(so)]
        if len(si) < 5 or len(so) < 5: continue
        s_i, s_o = relsd(si), relsd(so)
        ratio = s_i / s_o
        n_i, n_o = len(si), len(so)
        # F-based CI for sigma ratio
        alpha = 0.05
        Flo = stats.f.ppf(alpha/2, n_i-1, n_o-1); Fhi = stats.f.ppf(1-alpha/2, n_i-1, n_o-1)
        ci_lo, ci_hi = ratio/np.sqrt(Fhi), ratio/np.sqrt(Flo)
        # bootstrap CI
        bs = []
        for _ in range(2000):
            bi = rng.choice(si, n_i, replace=True); bo = rng.choice(so, n_o, replace=True)
            r = relsd(bi) / relsd(bo) if relsd(bo) else np.nan
            bs.append(r)
        bs = np.array(bs); b_lo, b_hi = np.nanquantile(bs, [.025, .975])
        share_init = s_i**2 / (s_i**2 + s_o**2)
        bundle = np.sqrt(s_i**2 + s_o**2)
        rows.append(dict(metric_mode=metric_mode, task=t, metric=m, n_init=n_i, n_order=n_o,
                         sd_init=s_i, sd_order=s_o, ratio=ratio,
                         ci_lo_F=ci_lo, ci_hi_F=ci_hi, ci_lo_bs=b_lo, ci_hi_bs=b_hi,
                         share_init=share_init, bundled_pred=bundle,
                         mean_init=np.mean(si), mean_order=np.mean(so)))

T = pd.DataFrame(rows)
T.to_csv(f"{OUT}/tables/t2_source_equivalence.csv", index=False)

for mm in ["primary_like", "bits_per_byte"]:
    s = T[T.metric_mode == mm]
    n = len(s)
    excl1 = ((s.ci_lo_F > 1) | (s.ci_hi_F < 1)).sum()
    within2 = ((s.ci_lo_F >= 0.5) & (s.ci_hi_F <= 2.0)).sum()
    print(f"\n[{mm}] tasks={n}  ratio init/order: median={s.ratio.median():.2f} "
          f"IQR=({s.ratio.quantile(.25):.2f},{s.ratio.quantile(.75):.2f})")
    print(f"  share_init median={s.share_init.median():.2f}  "
          f"CI-excludes-1: {excl1}/{n}   CI within [0.5,2]: {within2}/{n}")
    print(s[["task","metric","sd_init","sd_order","ratio","ci_lo_F","ci_hi_F","share_init"]]
          .round(4).to_string(index=False))

# ---- DataDecide 1B bundled 3-seed noise vs S&N sqrt-sum prediction (OLMES tasks)
dd = pd.read_parquet(f"{OUT}/analysis/dd_tidy.parquet")
TASK_MAP = {  # dd task -> sn task_name
 "arc_challenge":"arc_challenge","arc_easy":"arc_easy","boolq":"boolq","csqa":"csqa",
 "hellaswag":"hellaswag","mmlu":"mmlu","openbookqa":"openbookqa","piqa":"piqa",
 "socialiqa":"socialiqa","winogrande":"winogrande"}
d1 = dd[(dd.params=="1B") & (dd.task.isin(TASK_MAP))]
rows2 = []
for (t, d_), g in d1.groupby(["task","data"]):
    steps = sorted(set.intersection(*[set(g[g.seed==s].step) for s in g.seed.unique()]))
    if len(steps) < 3: continue
    l3 = steps[-3:]
    vals = [g[(g.seed==s) & (g.step.isin(l3))]["primary_metric"].mean() for s in sorted(g.seed.unique())]
    rows2.append(dict(task=t, data=d_, dd_bundled_relsd=relsd(vals)))
D = pd.DataFrame(rows2)
sn_pl = T[T.metric_mode=="primary_like"][["task","sd_init","sd_order","bundled_pred"]]
M = D.groupby("task")["dd_bundled_relsd"].median().reset_index().merge(sn_pl, on="task", how="inner")
M["dd_over_sqrt_sum"] = M.dd_bundled_relsd / M.bundled_pred
M["dd_over_init_only"] = M.dd_bundled_relsd / M.sd_init
M.to_csv(f"{OUT}/tables/t2_dd1b_vs_sn_bundle.csv", index=False)
print("\nDataDecide-1B bundled (median over 25 recipes) vs S&N components:")
print(M.round(4).to_string(index=False))
print(f"\nmedian dd/sqrt(init^2+order^2) = {M.dd_over_sqrt_sum.median():.2f}; "
      f"median dd/init_only = {M.dd_over_init_only.median():.2f}")
