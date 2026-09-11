#!/usr/bin/env python3
"""r28_snr_ablation -- EXT M4 (second external round): decompose the SNR head-to-head gain.

The D.5 head-to-head compares our separability SNR (Spearman +0.83) against S&N's (+0.65),
but the two differ in TWO components: the noise denominator (their checkpoint proxy x vs our
measured 3-seed within-recipe SD) and the signal term (their observed recipe-mean SD vs our
deconvolved signal). This script isolates each on the same cells:

  A. theirs, verbatim:        sig_obs / mean(checkpoint proxy x)
  B. swap denominator only:   sig_obs / RMS(true 3-seed SD)
  C. add deconvolution:       sig_dec / RMS(true 3-seed SD)   (= ours)
  D. deconvolution alone with proxy denominator: sig_dec / mean(x)  (for completeness)

Same cells, same target (per-task pairwise accuracy vs 1B), Spearman + a paired bootstrap
interval on the A->C difference. Writes r28_snr_ablation.json. Pure CPU, deterministic.
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.environ.get("NFT_R", os.path.normpath(os.path.join(HERE, "..", "data", "analysis")))
d = pd.read_parquet(f"{R}/dd_tidy.parquet")
pairs = pd.read_parquet(f"{R}/t3_pairs.parquet")
cells = pd.read_parquet(f"{R}/t1b_cells.parquet")

TASKS10 = ["arc_challenge", "arc_easy", "boolq", "csqa", "hellaswag", "mmlu",
           "openbookqa", "piqa", "socialiqa", "winogrande"]
SIZES = ["4M", "6M", "8M", "10M", "14M", "16M", "20M", "60M", "90M", "150M",
         "300M", "530M", "750M"]

acc = pairs[pairs.task.isin(TASKS10)].groupby(["task", "size"])["correct"].mean().rename("acc")


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
        sig_dec = np.sqrt(max(S2 - (sds ** 2).mean() / 3, 0.0))
        noise_rms = np.sqrt((sds ** 2).mean())
        xx = cells[(cells.params == sz) & (cells.task == t)]["x"]
        x_mean = xx[xx > 0].mean()
        if not np.isfinite(x_mean) or x_mean == 0 or noise_rms == 0: continue
        key = (t, sz)
        if key not in acc.index: continue
        rows.append({"task": t, "size": sz, "acc": float(acc.loc[key]),
                     "A_theirs": float(sig_obs / x_mean),
                     "B_swap_denom": float(sig_obs / noise_rms),
                     "C_ours": float(sig_dec / noise_rms),
                     "D_dec_only": float(sig_dec / x_mean)})

df = pd.DataFrame(rows)
out = {"n_cells": int(len(df))}
for c in ["A_theirs", "B_swap_denom", "C_ours", "D_dec_only"]:
    r = stats.spearmanr(df[c], df.acc)
    out[c] = {"spearman": round(float(r.statistic), 4), "p": float(r.pvalue)}
    print(f"{c:<14} Spearman {r.statistic:+.3f} (p={r.pvalue:.1e})")

# paired bootstrap on the per-cell rank difference (A vs C)
rng = np.random.default_rng(0)
n = len(df)
diffs = []
ra = df.A_theirs.rank(); rc = df.C_ours.rank(); ya = df.acc.rank()
for _ in range(2000):
    idx = rng.integers(0, n, n)
    diffs.append(stats.spearmanr(rc.iloc[idx], ya.iloc[idx]).statistic
                 - stats.spearmanr(ra.iloc[idx], ya.iloc[idx]).statistic)
lo, hi = np.percentile(diffs, [2.5, 97.5])
out["C_minus_A"] = {"point": round(out["C_ours"]["spearman"] - out["A_theirs"]["spearman"], 4),
                    "ci95": [round(float(lo), 4), round(float(hi), 4)],
                    "note": "paired bootstrap over cells, B=2000, seed 0"}
print(f"C - A = {out['C_minus_A']['point']:+.3f} [{lo:+.3f}, {hi:+.3f}]")

json.dump(out, open(os.path.join(HERE, "r28_snr_ablation.json"), "w"), indent=1)
print("wrote r28_snr_ablation.json")
