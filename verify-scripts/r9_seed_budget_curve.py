#!/usr/bin/env python3
"""Reviewer-A W2: price the 'measure the seeds' advice -- decidable share vs seed budget n.

For each scale: per recipe the 3-seed macro SD sigma_i (measured). With n seeds the Welch
band half-width for pair (i,j) scales as t(0.975, nu) * sqrt((s_i^2+s_j^2)/n). Holding the
measured sigma_i fixed (the atlas's point), the decidable share (BH q=0.05) is computable in
closed form for any n. Reports, per scale, the minimal n where the share leaves 0 and where
it passes 25%/50%. This is a model-based extrapolation (normality, sigma fixed at the 3-seed
estimate) -- stated as such.

Outputs r9_seed_budget_curve.json. Pure CPU, deterministic. NFT_R overridable.
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.environ.get("NFT_R", "/home/bingxing2/home/scx7ew2/tanh/Tanhäuser/runs/exp-rank04/zero-gpu/analysis")
d = pd.read_parquet(f"{R}/dd_tidy.parquet")

def final_scores(params, task="olmes_10_macro_avg"):
    g = d[(d.params == params) & (d.task == task)]
    out = {}
    for mix, gg in g.groupby("data"):
        c = gg.groupby("step")["seed"].nunique()
        com = c[c >= 3].index
        if len(com) == 0: continue
        v = gg[gg.step == com.max()].groupby("seed")["primary_metric"].mean()
        if len(v) >= 3: out[mix] = v.values[:3]
    return out

def decidable_share(mu, sd, n):
    m = len(mu); iu = np.triu_indices(m, 1)
    dm = np.abs(mu[iu[0]] - mu[iu[1]])
    # Welch df with n per arm and the measured variances
    vi, vj = sd[iu[0]]**2, sd[iu[1]]**2
    nu = (vi/n + vj/n)**2 / ((vi/n)**2/(n-1) + (vj/n)**2/(n-1))
    se = np.sqrt(vi/n + vj/n)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = dm/se
    t[~np.isfinite(t)] = 0.0
    p = 2*stats.t.sf(t, nu)
    o = np.argsort(p); mm = len(p)
    k = np.where(p[o] <= 0.05*np.arange(1, mm+1)/mm)[0]
    return (k.max()+1)/mm if len(k) else 0.0

SIZES = ["4M","10M","20M","60M","90M","150M","300M","530M","1B"]
NS = [3, 4, 5, 6, 8, 10, 15, 20, 30, 50]
OUT = {}
for sz in SIZES:
    sc = final_scores(sz)
    rec = sorted(sc)
    mu = np.array([sc[r].mean() for r in rec]); sd = np.array([sc[r].std(ddof=1) for r in rec])
    curve = {n: decidable_share(mu, sd, n) for n in NS}
    n_pos = next((n for n in NS if curve[n] > 0), None)
    n_25 = next((n for n in NS if curve[n] >= 0.25), None)
    n_50 = next((n for n in NS if curve[n] >= 0.50), None)
    OUT[sz] = {"by_n": {str(n): round(curve[n], 3) for n in NS},
               "min_n_nonzero": n_pos, "min_n_25": n_25, "min_n_50": n_50}
    print(f"{sz:>5}: share(n) " + " ".join(f"{n}:{curve[n]:.2f}" for n in NS)
          + f"  | first>0: {n_pos}, >=25%: {n_25}, >=50%: {n_50}")

with open(os.path.join(HERE, "r9_seed_budget_curve.json"), "w") as f:
    json.dump(OUT, f, indent=1)
print("wrote r9_seed_budget_curve.json")
