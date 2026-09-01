#!/usr/bin/env python3
"""R12 major-5: head-to-head of our separability SNR vs Signal & Noise's SNR concept
as predictors of per-task pairwise decision accuracy, same data, same cells.

The original S&N work uses its checkpoint-spread proxy as the noise term:
  SNR_sn(task, scale)  = SD across the 25 recipe means / mean checkpoint-proxy x.
Ours (separability SNR, deconvolved, as in Table 1 but per task):
  SNR_ours(task, scale) = sqrt(max(Var(means) - mean(sd^2)/3, 0)) / RMS(sd),
noise = the true 3-seed within-recipe SD (not the proxy).

Outcome: per-task pairwise decision accuracy vs the 1B target (t3_pairs 'correct').
Predictor comparison: Spearman across all (task, scale) cells (N=140), plus
per-scale medians. If ours does not beat S&N's, the increment is honesty, not
accuracy -- report as is.

Outputs r9_sn_headtohead.json. Pure CPU, deterministic. NFT_R overridable.
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.environ.get("NFT_R", "/home/bingxing2/home/scx7ew2/tanh/Tanhäuser/runs/exp-rank04/zero-gpu/analysis")
d = pd.read_parquet(f"{R}/dd_tidy.parquet")
pairs = pd.read_parquet(f"{R}/t3_pairs.parquet")
cells = pd.read_parquet(f"{R}/t1b_cells.parquet")

TASKS10 = ["arc_challenge","arc_easy","boolq","csqa","hellaswag","mmlu","openbookqa","piqa","socialiqa","winogrande"]
SIZES = ["4M","6M","8M","10M","14M","16M","20M","60M","90M","150M","300M","530M","750M"]

# per-task pairwise accuracy per scale
acc = pairs[pairs.task.isin(TASKS10)].groupby(["task","size"])["correct"].mean().rename("acc")

def recipe_means_sds(sz, task):
    g = d[(d.params == sz) & (d.task == task)]
    out = {}
    for mix, gg in g.groupby("data"):
        c = gg.groupby("step")["seed"].nunique()
        com = c[c >= 3].index
        if len(com) == 0: continue
        v = gg[gg.step == com.max()].groupby("seed")["primary_metric"].mean()
        if len(v) >= 3: out[mix] = (v.mean(), v.std(ddof=1))
    return out

rows = []
for sz in SIZES:
    for t in TASKS10:
        ms = recipe_means_sds(sz, t)
        if len(ms) < 10: continue
        means = np.array([v[0] for v in ms.values()])
        sds = np.array([v[1] for v in ms.values()])
        S2 = means.var(ddof=1)
        sig_obs = np.sqrt(S2)
        sig_dec = np.sqrt(max(S2 - (sds**2).mean()/3, 0.0))
        noise_rms = np.sqrt((sds**2).mean())
        xx = cells[(cells.params==sz)&(cells.task==t)]["x"]
        x_mean = xx[xx>0].mean()
        if not np.isfinite(x_mean) or x_mean == 0: continue
        key = (t, sz)
        if key not in acc.index: continue
        rows.append({"task": t, "size": sz, "acc": float(acc.loc[key]),
                     "snr_ours": float(sig_dec/noise_rms) if noise_rms>0 else np.nan,
                     "snr_sn": float(sig_obs/x_mean)})

df = pd.DataFrame(rows)
print(f"cells: {len(df)}")
rho_ours = stats.spearmanr(df.snr_ours, df.acc)
rho_sn = stats.spearmanr(df.snr_sn, df.acc)
print(f"Spearman ours:  {rho_ours.statistic:.3f} (p={rho_ours.pvalue:.2e})")
print(f"Spearman S&N:   {rho_sn.statistic:.3f} (p={rho_sn.pvalue:.2e})")

# paired comparison: per-cell ranks; also within-scale Spearman medians
per_scale = df.groupby("size").apply(
    lambda g: pd.Series({"ours": stats.spearmanr(g.snr_ours, g.acc).statistic,
                         "sn": stats.spearmanr(g.snr_sn, g.acc).statistic}), include_groups=False)
print(per_scale.round(3).to_string())

# and the paper's task-level held-out variant numbers (90 cells, excl macro):
OUT = {
 "n_cells": int(len(df)),
 "spearman_ours": float(rho_ours.statistic), "p_ours": float(rho_ours.pvalue),
 "spearman_sn": float(rho_sn.statistic), "p_sn": float(rho_sn.pvalue),
 "per_scale": per_scale.round(4).to_dict(),
 "corr_ours_vs_sn": float(stats.spearmanr(df.snr_ours, df.snr_sn).statistic),
}
print("corr(ours, sn) across cells:", round(OUT["corr_ours_vs_sn"],3))
df.to_csv(os.path.join(HERE, "r9_sn_headtohead_cells.csv"), index=False)
with open(os.path.join(HERE, "r9_sn_headtohead.json"), "w") as f:
    json.dump(OUT, f, indent=1)
print("wrote r9_sn_headtohead.json + cells csv")
