#!/usr/bin/env python3
"""R12-Q4: the BH decidable share under dependence-robust multiplicity control.

Paper (Sec 8 rule 3, r8_prescription3.json): per scale, 300 pairwise Welch tests on
25 recipes (3-seed means); BH at q=0.05 decides the "decidable share". Reviewer:
the 300 tests share 25 recipe-level variance estimates -- BH is not calibrated for
that dependence. Compute the decidable share under:
  (a) BH (reproduce the r8 numbers),
  (b) Benjamini-Yekutieli (valid under arbitrary dependence),
  (c) bootstrap max-T (null imposed by centering each recipe's 3 seeds at the
      recipe mean; resample seeds within recipe; recompute all 300 |t|; the 95th
      percentile of the max calibrates a per-pair threshold -- the standard
      Westfall-Young style single-step adjustment).

Outputs r9_decidable_share.json. Pure CPU, deterministic. NFT_R overridable.
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.environ.get("NFT_R", "/home/bingxing2/home/scx7ew2/tanh/Tanhäuser/runs/exp-rank04/zero-gpu/analysis")
d = pd.read_parquet(f"{R}/dd_tidy.parquet")

SIZES = ["4M","10M","20M","60M","90M","150M","300M","530M","1B"]  # the rule-3 reporting set

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

def pair_stats(scores):
    """Welch t for all pairs. scores: recipe -> 3 values."""
    rec = sorted(scores)
    n = len(rec)
    M = np.array([scores[r].mean() for r in rec])
    V = np.array([scores[r].var(ddof=1) for r in rec])
    iu = np.triu_indices(n, 1)
    dm = M[iu[0]] - M[iu[1]]
    se = np.sqrt(V[iu[0]]/3 + V[iu[1]]/3)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.abs(dm) / se
    # Welch-Satterthwaite df
    num = (V[iu[0]]/3 + V[iu[1]]/3)**2
    den = (V[iu[0]]/3)**2/2 + (V[iu[1]]/3)**2/2
    with np.errstate(divide="ignore", invalid="ignore"):
        nu = num/den
    t[~np.isfinite(t)] = 0.0
    nu[~np.isfinite(nu)] = 2.0
    p = 2*stats.t.sf(t, nu)
    return t, p, iu

rng = np.random.default_rng(20260901)
B = 2000
OUT = {}
for sz in SIZES:
    sc = final_scores(sz)
    if len(sc) < 10: continue
    t_obs, p_obs, iu = pair_stats(sc)
    m = len(t_obs)
    order = np.argsort(p_obs)
    # BH
    bh_k = np.where(p_obs[order] <= 0.05*np.arange(1, m+1)/m)[0]
    share_bh = (bh_k.max()+1)/m if len(bh_k) else 0.0
    # BY
    cm = np.sum(1.0/np.arange(1, m+1))
    by_k = np.where(p_obs[order] <= 0.05*np.arange(1, m+1)/(m*cm))[0]
    share_by = (by_k.max()+1)/m if len(by_k) else 0.0
    # bootstrap max-T under the imposed null (recipe means equalized).
    # Parametric variant: raw with-replacement resampling at n=3 produces
    # zero-variance recipes with prob ~1/9 each, exploding the max statistic;
    # instead draw seed values as N(0, s_i^2) per recipe (the same normal model
    # the Welch test assumes), keeping recipe-specific variances.
    rec = sorted(sc)
    arr = np.array([sc[r] for r in rec])           # 25 x 3
    si = arr.std(axis=1, ddof=1)                   # per-recipe SD
    maxt = np.empty(B)
    for b in range(B):
        boot = rng.normal(0.0, 1.0, size=arr.shape) * si[:, None]
        Mb = boot.mean(axis=1); Vb = boot.var(axis=1, ddof=1)
        dm = Mb[iu[0]] - Mb[iu[1]]
        se = np.sqrt(Vb[iu[0]]/3 + Vb[iu[1]]/3)
        with np.errstate(divide="ignore", invalid="ignore"):
            tb = np.abs(dm)/se
        tb[~np.isfinite(tb)] = 0.0
        maxt[b] = tb.max()
    thr = np.percentile(maxt, 95)
    share_maxt = float((t_obs > thr).mean())
    OUT[sz] = {"m_pairs": int(m), "share_BH": share_bh, "share_BY": share_by,
               "share_maxT": share_maxt, "maxT_thr95": float(thr)}
    print(f"{sz:>5}: BH {share_bh:.3f}  BY {share_by:.3f}  maxT {share_maxt:.3f}  (thr {thr:.2f})")

with open(os.path.join(HERE, "r9_decidable_share.json"), "w") as f:
    json.dump(OUT, f, indent=1)
print("wrote r9_decidable_share.json")
