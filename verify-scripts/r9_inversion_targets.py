#!/usr/bin/env python3
"""Correlation of the 4M macro ranking with every larger scale's ranking;
writes r9_inversion_targets.json.
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
R = os.environ.get("NFT_R", os.path.join(_ROOT, "data", "analysis"))
d = pd.read_parquet(f"{R}/dd_tidy.parquet")

def fs(params):
    g = d[(d.params == params) & (d.task == "olmes_10_macro_avg")]
    out = {}
    for mix, gg in g.groupby("data"):
        c = gg.groupby("step")["seed"].nunique(); com = c[c >= 3].index
        if len(com) == 0: continue
        v = gg[gg.step == com.max()].groupby("seed")["primary_metric"].mean()
        if len(v) >= 3: out[mix] = float(v.mean())
    return out

m4 = fs("4M")
OUT = {}
for tgt in ["60M", "90M", "150M", "300M", "530M", "1B"]:
    mt = fs(tgt)
    rec = sorted(set(m4) & set(mt))
    a = [m4[r] for r in rec]; b = [mt[r] for r in rec]
    OUT[tgt] = {"pearson": round(float(stats.pearsonr(a, b)[0]), 3),
                "spearman": round(float(stats.spearmanr(a, b)[0]), 3), "n": len(rec)}
    print(f"4M vs {tgt}: pearson {OUT[tgt]['pearson']:+.3f} spearman {OUT[tgt]['spearman']:+.3f}")

with open(os.path.join(HERE, "r9_inversion_targets.json"), "w") as f:
    json.dump(OUT, f, indent=1)
print("wrote r9_inversion_targets.json")
