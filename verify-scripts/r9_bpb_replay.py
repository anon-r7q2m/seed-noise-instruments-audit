#!/usr/bin/env python3
"""Reviewer-A W3: does the 4M inversion survive on a continuous (log-ppl) readout?

The accuracy-readout inversion is an aggregation artifact of near-chance tasks; the
readout panel (App G) shows bpb reads 2-6x higher SNR. This script replays the decision
analysis on DataDecide's own 11-domain perplexity release (full 14 scales x 3 seeds):
per recipe, log-ppl macro score = mean over the 11 domains of log(perplexity) at the final
common step; pairwise decisions vs the 1B log-ppl target; report per-scale pairwise accuracy
(3-seed means) and the 4M macro ranking's correlation with the 1B ranking.

Outputs r9_bpb_replay.json. Pure CPU, deterministic. NFT_PPL overridable.
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
PPL = os.environ.get("NFT_PPL", "/home/bingxing2/home/scx7ew2/tanh/Tanhäuser/tmp/rank04-zerogpu/data/dd_ppl.parquet")
d = pd.read_parquet(PPL)
DOMS = [c for c in d.columns if c.startswith("eval/")]
d = d.dropna(subset=DOMS, how="all")

SIZES = ["4M","6M","8M","10M","14M","16M","20M","60M","90M","150M","300M","530M","750M","1B"]

def final_logppl(sz):
    g = d[d.params == sz]
    out = {}
    for (recipe, seed), gg in g.groupby(["data", "seed"]):
        com = gg["step"].max()  # last step per seed (domains share the grid)
        row = gg[gg.step == com]
        if row.empty: continue
        lp = np.log(row[DOMS].iloc[0].values.astype(float))
        lp = lp[np.isfinite(lp)]
        if len(lp) == len(DOMS):
            out.setdefault(recipe, {})[seed] = lp.mean()   # macro log-ppl per seed
    # keep recipes with >=3 seeds and a common final step
    per = {}
    for r, sd in out.items():
        if len(sd) >= 3:
            steps = []
            per[r] = np.array(list(sd.values())[:3])
    return per

per_scale = {sz: final_logppl(sz) for sz in SIZES}
recipes = sorted(set.intersection(*[set(v) for v in per_scale.values() if v]))
print(f"recipes with full ppl coverage: {len(recipes)}")

tgt = {r: per_scale["1B"][r].mean() for r in recipes}
OUT = {"n_recipes": len(recipes), "per_scale": {}}
for sz in SIZES[:-1]:
    sc = per_scale[sz]
    mu = {r: sc[r].mean() for r in recipes}
    # pairwise accuracy (lower ppl better both sides)
    tot = cor = 0
    rec_l = list(recipes)
    for a in range(len(rec_l)):
        for b in range(a+1, len(rec_l)):
            i, j = rec_l[a], rec_l[b]
            dt = tgt[i] - tgt[j]
            if dt == 0: continue
            tot += 1; cor += ((mu[i] - mu[j]) * dt > 0)
    acc = cor / tot
    r_corr = stats.pearsonr([mu[r] for r in recipes], [tgt[r] for r in recipes])[0]
    sp_corr = stats.spearmanr([mu[r] for r in recipes], [tgt[r] for r in recipes])[0]
    OUT["per_scale"][sz] = {"pairwise_acc": round(acc, 4), "pearson_1B": round(float(r_corr), 4),
                            "spearman_1B": round(float(sp_corr), 4)}
    print(f"{sz:>5}: pairwise acc {acc:.3f}  pearson {r_corr:+.3f}  spearman {sp_corr:+.3f}")

with open(os.path.join(HERE, "r9_bpb_replay.json"), "w") as f:
    json.dump(OUT, f, indent=1)
print("wrote r9_bpb_replay.json")
