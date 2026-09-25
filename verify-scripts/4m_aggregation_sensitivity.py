#!/usr/bin/env python3
"""Sensitivity of the 4M-vs-1B macro ranking correlation to the aggregation function,
with leave-one-task-out and single-task diagnostics; writes 4m_aggregation_sensitivity.json.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
R = os.environ.get("NFT_R", os.path.join(_ROOT, "data", "analysis"))
d = pd.read_parquet(f"{R}/dd_tidy.parquet")
TASKS10 = ["arc_challenge", "arc_easy", "boolq", "csqa", "hellaswag",
           "mmlu", "openbookqa", "piqa", "socialiqa", "winogrande"]


def fcs(g, need=3):
    c = g.groupby("step")["seed"].nunique()
    com = c[c >= need].index
    return None if len(com) == 0 else com.max()


def cell_stats(sz, task):
    per = {}
    for mix, gg in d[(d.params == sz) & (d.task == task)].groupby("data"):
        s = fcs(gg)
        if s is None:
            continue
        v = gg[gg.step == s].groupby("seed")["primary_metric"].mean()
        if len(v) >= 3:
            per[mix] = v.mean()
    return per


# paper's headline readout (macro_avg column as released)
m4p = cell_stats("4M", "olmes_10_macro_avg")
m1p = cell_stats("1B", "olmes_10_macro_avg")
common = sorted(set(m4p) & set(m1p))
r_paper = stats.pearsonr([m4p[k] for k in common], [m1p[k] for k in common])[0]
rho_paper = stats.spearmanr([m4p[k] for k in common], [m1p[k] for k in common])[0]
print(f"[anchor] released macro_avg column: n={len(common)}  Pearson {r_paper:.3f}  Spearman {rho_paper:.3f}  (paper: -0.56 / -0.52)")

# per-task matrices
T4 = {t: cell_stats("4M", t) for t in TASKS10}
T1 = {t: cell_stats("1B", t) for t in TASKS10}
recipes = sorted(set.intersection(*[set(T4[t]) & set(T1[t]) for t in TASKS10]))
print(f"[anchor] per-task matrices: {len(recipes)} recipes x {len(TASKS10)} tasks")
M4 = np.array([[T4[t][r] for t in TASKS10] for r in recipes])
M1 = np.array([[T1[t][r] for t in TASKS10] for r in recipes])


def pair_acc(v4, v1):
    ii, jj = np.triu_indices(len(v4), k=1)
    dt = v1[ii] - v1[jj]
    keep = dt != 0
    return float(((v4[ii] - v4[jj])[keep] * dt[keep] > 0).mean())


def evaluate(v4, v1):
    return dict(pearson=float(stats.pearsonr(v4, v1)[0]),
                spearman=float(stats.spearmanr(v4, v1)[0]),
                pair_acc=pair_acc(np.asarray(v4), np.asarray(v1)))


AGG = {
    "mean": lambda M: M.mean(axis=1),
    "median": lambda M: np.median(M, axis=1),
    "z_mean": lambda M: ((M - M.mean(axis=0)) / M.std(axis=0, ddof=1)).mean(axis=1),
    "rank_mean": lambda M: np.apply_along_axis(stats.rankdata, 0, M).mean(axis=1),
    "minmax_mean": lambda M: ((M - M.min(axis=0)) / (M.max(axis=0) - M.min(axis=0))).mean(axis=1),
}

out = {"anchor_released_column": {"n": len(common), "pearson": float(r_paper), "spearman": float(rho_paper)},
       "n_recipes_matrix": len(recipes), "variants": {}, "loto": {}, "single_task": {}}

# 1B target: to keep the question "does the 4M readout predict the 1B macro ordering",
# the target side is always aggregated with the SAME variant.
for name, f in AGG.items():
    res = evaluate(f(M4), f(M1))
    out["variants"][name] = res
    print(f"[A {name:<12}] r={res['pearson']:+.3f}  rho={res['spearman']:+.3f}  pair_acc={res['pair_acc']*100:.1f}%", flush=True)

# LOTO per variant
for name, f in AGG.items():
    rs = []
    for k, t in enumerate(TASKS10):
        keep = [j for j in range(len(TASKS10)) if j != k]
        r = stats.pearsonr(f(M4[:, keep]), f(M1[:, keep]))[0]
        rs.append(float(r))
    out["loto"][name] = dict(zip(TASKS10, rs), min=min(rs), max=max(rs),
                             n_negative=int(sum(x < 0 for x in rs)))
    print(f"[B LOTO {name:<10}] range [{min(rs):+.3f}, {max(rs):+.3f}]  negative {sum(x < 0 for x in rs)}/10", flush=True)

# single-task diagnostics
for k, t in enumerate(TASKS10):
    r_own = stats.pearsonr(M4[:, k], M1[:, k])[0]
    keep = [j for j in range(len(TASKS10)) if j != k]
    r_wo = stats.pearsonr(M4[:, keep].mean(axis=1), M1[:, keep].mean(axis=1))[0]
    out["single_task"][t] = {"own_r": float(r_own), "mean_r_without": float(r_wo)}
    print(f"[C {t:<14}] own r={r_own:+.3f}   mean-variant r without it: {r_wo:+.3f}", flush=True)

with open("4m_aggregation_sensitivity.json", "w") as fp:
    json.dump(out, fp, indent=1)
print("\nwrote 4m_aggregation_sensitivity.json")
