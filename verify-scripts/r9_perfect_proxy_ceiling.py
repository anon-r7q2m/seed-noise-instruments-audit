#!/usr/bin/env python3
"""R11-Q1/W3a + R12 repro-gap: recipe-noise ranking, observed vs perfect-proxy ceiling.

Paper claim (Sec 3): ranking the 25 recipes by their noise via the checkpoint proxy
is no better than chance (Kendall tau ~ 0, top-5 overlap 16-30%). The original
analysis script was lost (ad-hoc session computation); this script recomputes the
observed numbers AND adds the reviewer-requested ceiling: even a perfect proxy
(proxy = true per-recipe sigma) is ranked against a truth estimated from only
n = 3 seeds (df = 2), so the achievable tau is bounded by chi^2_2 sampling noise.

Two estimands (the paper's claim is per-task; the macro variant is reported too):
  per-task: within each (scale, task) cell, rank 25 recipes by proxy x vs truth y.
  macro:    per recipe, mean of relative SD over the 10 tasks; rank at each scale.

Ceiling design (generous to the proxy): treat observed per-recipe truth values y_r
as the true sigmas (their dispersion thereby includes sampling noise, biasing the
ceiling UP); simulate the 3-seed estimate y_sim = y*sqrt(chi2_2/2); tau(y, y_sim).
Vectorized discordant-pair kernel (no ties: values are continuous). B = 20000 for
the macro rows, B = 4000 per (scale, task) cell, fixed seeds.

Outputs r9_perfect_proxy_ceiling.json. Pure CPU, deterministic. NFT_R overridable.
"""
import os, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.environ.get("NFT_R", "/home/bingxing2/home/scx7ew2/tanh/Tanhäuser/runs/exp-rank04/zero-gpu/analysis")
cells = pd.read_parquet(f"{R}/t1b_cells.parquet")

SIZES = ["4M","6M","8M","10M","14M","16M","20M","60M","90M","150M","300M","530M","750M","1B"]

def tau_batch(y, Y):
    """Kendall tau between fixed y (n,) and each row of Y (B,n); no ties."""
    s = np.sign(y[:, None] - y[None, :])          # n x n
    iu = np.triu_indices(len(y), 1)
    sy = s[iu]                                    # P
    S = np.sign(Y[:, :, None] - Y[:, None, :])    # B x n x n
    disc = (S[:, iu[0], iu[1]] != sy[None, :]).sum(axis=1)
    P = len(sy)
    return 1.0 - 2.0 * disc / P

def top5_batch(y, Y):
    order_true = set(np.argsort(-y)[:5].tolist())
    top = np.argsort(-Y, axis=1)[:, :5]
    return np.array([len(order_true & set(r.tolist())) / 5.0 for r in top])

OUT = {"macro": {}, "per_task": {}, "summary": {}}
rng = np.random.default_rng(20260901)

for sz in SIZES:
    sub = cells[cells.params == sz]
    if sub.empty:
        continue
    # ---- macro estimand
    xr = sub.groupby("data")["x"].mean(); yr = sub.groupby("data")["y"].mean()
    common = xr.dropna().index.intersection(yr.dropna().index)
    x, y = xr[common].values, yr[common].values
    if len(y) < 10:
        continue
    B = 20000
    ysim = y[None, :] * np.sqrt(rng.chisquare(2, size=(B, len(y))) / 2.0)
    ct = tau_batch(y, ysim); c5 = top5_batch(y, ysim)
    OUT["macro"][sz] = {
        "obs_tau": float(1 - 2.0 * ((np.sign(x[:,None]-x[None,:])[np.triu_indices(len(y),1)] != np.sign(y[:,None]-y[None,:])[np.triu_indices(len(y),1)]).sum()) / (len(y)*(len(y)-1)//2)),
        "obs_top5": float(len(set(np.argsort(-x)[:5]) & set(np.argsort(-y)[:5])) / 5.0),
        "ceil_tau_median": float(np.median(ct)), "ceil_tau_p2_5": float(np.percentile(ct, 2.5)),
        "ceil_tau_p97_5": float(np.percentile(ct, 97.5)), "ceil_top5_median": float(np.median(c5)),
    }
    # ---- per-task estimand
    pt_obs, pt_ceil, pt_top5 = [], [], []
    for t, gt in sub.groupby("task"):
        xt = gt.set_index("data")["x"]; yt = gt.set_index("data")["y"]
        cc = xt.dropna().index.intersection(yt.dropna().index)
        if len(cc) < 10:
            continue
        xv, yv = xt[cc].values, yt[cc].values
        syt = np.sign(yv[:, None] - yv[None, :])[np.triu_indices(len(yv), 1)]
        sxt = np.sign(xv[:, None] - xv[None, :])[np.triu_indices(len(yv), 1)]
        pt_obs.append(1 - 2.0 * (sxt != syt).sum() / len(syt))
        pt_top5.append(len(set(np.argsort(-xv)[:5]) & set(np.argsort(-yv)[:5])) / 5.0)
        Bt = 4000
        yst = yv[None, :] * np.sqrt(rng.chisquare(2, size=(Bt, len(yv))) / 2.0)
        pt_ceil.append(float(np.median(tau_batch(yv, yst))))
    OUT["per_task"][sz] = {
        "obs_tau_median": float(np.median(pt_obs)), "obs_tau_min": float(np.min(pt_obs)),
        "obs_tau_max": float(np.max(pt_obs)),
        "obs_top5_median": float(np.median(pt_top5)), "obs_top5_min": float(np.min(pt_top5)),
        "obs_top5_max": float(np.max(pt_top5)),
        "ceil_tau_median_of_tasks": float(np.median(pt_ceil)),
        "ceil_tau_task_range": [float(np.min(pt_ceil)), float(np.max(pt_ceil))],
    }

S = OUT["summary"]
pt = OUT["per_task"]
S["per_task_obs_tau_median_over_scales"] = float(np.median([v["obs_tau_median"] for v in pt.values()]))
S["per_task_obs_tau_scale_range"] = [float(min(v["obs_tau_median"] for v in pt.values())), float(max(v["obs_tau_median"] for v in pt.values()))]
S["per_task_obs_top5_scale_range"] = [float(min(v["obs_top5_median"] for v in pt.values())), float(max(v["obs_top5_median"] for v in pt.values()))]
S["per_task_ceil_tau_median_over_scales"] = float(np.median([v["ceil_tau_median_of_tasks"] for v in pt.values()]))
S["per_task_ceil_tau_scale_range"] = [float(min(v["ceil_tau_median_of_tasks"] for v in pt.values())), float(max(v["ceil_tau_median_of_tasks"] for v in pt.values()))]
S["macro_obs_tau_median"] = float(np.median([v["obs_tau"] for v in OUT["macro"].values()]))
S["macro_ceil_tau_median"] = float(np.median([v["ceil_tau_median"] for v in OUT["macro"].values()]))

print(f"{'scale':>6} | per-task obs tau (med [min,max]) | per-task ceil med | macro obs tau | macro ceil med")
for sz in SIZES:
    if sz in pt:
        p, m = pt[sz], OUT["macro"][sz]
        print(f"{sz:>6} | {p['obs_tau_median']:>7.3f} [{p['obs_tau_min']:>6.3f},{p['obs_tau_max']:>6.3f}] | {p['ceil_tau_median_of_tasks']:>7.3f} | {m['obs_tau']:>7.3f} | {m['ceil_tau_median']:>7.3f}")
print("\nsummary:", json.dumps({k: (round(v,3) if isinstance(v,float) else [round(z,3) for z in v]) for k,v in S.items()}, indent=1))

with open(os.path.join(HERE, "r9_perfect_proxy_ceiling.json"), "w") as f:
    json.dump(OUT, f, indent=1)
print("wrote r9_perfect_proxy_ceiling.json")
