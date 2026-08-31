#!/usr/bin/env python3
"""T1 part A — replicate Signal&Noise's own proxy claim on their released 1B data.

Published claim (arXiv 2508.13144, S3.1/A.3.1): across ~30 benchmarks at 1B,
init-seed noise, data-order noise, and whole-run checkpoint noise correlate with
the relative SD of the final n checkpoints with R^2 = 0.82, 0.86, 0.95.

Here: allenai/signal-and-noise `random_seeds` split
  - 10 runs run_type='seed'  (init seed varied, data order fixed)
  -  9 runs run_type='data'  (data order varied, init fixed)
  -  1 run  run_type='high-eval' (evals every 10 steps)
Metrics: acc / acc_per_char / bits_per_byte per task (21 tasks).

Estimators (matching the paper's definitions):
  proxy  x_b = mean over seed-arm runs of relSD(final n ckpts of that run), n=30 (500-step grid)
  y_init_b = relSD across 10 seed runs of final score (score = mean of last 3 ckpts <= common step)
  y_data_b = relSD across  9 data runs of final score
Correlate across tasks (Pearson on log10, and raw), compare with published R^2.
"""
import pandas as pd, numpy as np, json, os
from scipy import stats

TMP = "./tmp/rank04-zerogpu"
OUT = "data"

rs = pd.read_parquet(f"{TMP}/data/sn_random_seeds.parquet")

# ------- choose a per-task "primary-like" metric: acc_per_char if present else acc; bpb separately
def pick_metric(g):
    ms = set(g["metric"].unique())
    if "acc_per_char" in ms: return "acc_per_char"
    if "acc" in ms: return "acc"
    return None

task_metric = {}
for t, g in rs.groupby("task_name"):
    task_metric[t] = pick_metric(g)

COMMON_FINAL = 69000  # last 500-multiple step present in both arms & before LR-final (5xC ~ 69369)
N_CKPT = 30           # final n checkpoints on the 500-step grid (paper: final ~30 ckpts)

def relsd(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 2 or np.mean(v) == 0: return np.nan
    return np.std(v, ddof=1) / abs(np.mean(v))

def run_final_score(g, step_final=COMMON_FINAL, avg_last=3):
    gg = g[g.step <= step_final].sort_values("step")
    if len(gg) == 0: return np.nan
    return gg["value"].tail(avg_last).mean()

def run_ckpt_noise(g, step_final=COMMON_FINAL, n=N_CKPT):
    gg = g[g.step <= step_final].sort_values("step")
    return relsd(gg["value"].tail(n))

rows = []
for metric_mode in ["primary_like", "bits_per_byte"]:
    for t in sorted(rs.task_name.unique()):
        m = task_metric[t] if metric_mode == "primary_like" else "bits_per_byte"
        if m is None: continue
        g = rs[(rs.task_name == t) & (rs.metric == m)]
        if g.empty: continue
        seed_arm = g[g.run_type == "seed"]
        data_arm = g[g.run_type == "data"]
        he       = g[g.run_type == "high-eval"]
        # proxy: mean over seed-arm runs of per-run final-n relSD
        xs = [run_ckpt_noise(gr) for _, gr in seed_arm.groupby("run_name")]
        x  = np.nanmean(xs)
        # proxy alt: high-eval run final 300 ckpts (10-step grid ~ last 3000 steps)
        x_he = relsd(he[he.step <= COMMON_FINAL].sort_values("step")["value"].tail(300)) if len(he) else np.nan
        y_init = relsd([run_final_score(gr) for _, gr in seed_arm.groupby("run_name")])
        y_data = relsd([run_final_score(gr) for _, gr in data_arm.groupby("run_name")])
        rows.append(dict(metric_mode=metric_mode, task=t, metric=m, x_proxy=x, x_he=x_he,
                         y_init=y_init, y_data=y_data,
                         n_seed=seed_arm.run_name.nunique(), n_data=data_arm.run_name.nunique()))

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/tables/t1a_sn_replication_per_task.csv", index=False)

def corr_report(sub, xcol, ycol, log=True):
    s = sub[[xcol, ycol]].dropna()
    s = s[(s[xcol] > 0) & (s[ycol] > 0)]
    if len(s) < 4: return dict(n=len(s), R=np.nan, R2=np.nan)
    if log:
        r, p = stats.pearsonr(np.log10(s[xcol]), np.log10(s[ycol]))
    else:
        r, p = stats.pearsonr(s[xcol], s[ycol])
    return dict(n=len(s), R=round(r, 3), R2=round(r * r, 3), p=float(f"{p:.2g}"))

summary = []
for mm in ["primary_like", "bits_per_byte"]:
    sub = df[df.metric_mode == mm]
    for ycol, label in [("y_init", "init_seed_noise"), ("y_data", "data_order_noise")]:
        for xcol in ["x_proxy", "x_he"]:
            for log in [False, True]:
                rep = corr_report(sub, xcol, ycol, log=log)
                summary.append(dict(metric_mode=mm, y=label, x=xcol,
                                    space="log" if log else "raw", **rep))
S = pd.DataFrame(summary)
S.to_csv(f"{OUT}/tables/t1a_sn_replication_summary.csv", index=False)
print(S.to_string())

# calibration ratio: is proxy quantitatively ~= seed noise (ratio ~ 1)?
for mm in ["primary_like", "bits_per_byte"]:
    sub = df[df.metric_mode == mm].dropna(subset=["x_proxy", "y_init", "y_data"])
    ri = sub.y_init / sub.x_proxy
    rd = sub.y_data / sub.x_proxy
    print(f"\n[{mm}] ratio y_init/x median={ri.median():.2f} IQR=({ri.quantile(.25):.2f},{ri.quantile(.75):.2f})  "
          f"ratio y_data/x median={rd.median():.2f} IQR=({rd.quantile(.25):.2f},{rd.quantile(.75):.2f})  n={len(sub)}")
