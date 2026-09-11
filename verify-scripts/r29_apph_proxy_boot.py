#!/usr/bin/env python3
"""r29_apph_proxy_boot -- EXT M3 (second external round): the 20M proxy-calibration
intervals were F-ratio intervals (df 9/4) attached to a *median over runs* of per-run
ratios, whose checkpoints are time-correlated and share runs with the truth. Replace with
a run-level bootstrap: resample the 10 bundled runs, recompute y/x median each time.

Protocol (mirrors stage2_analyze.py): per run, x_s = relSD over the last-5 checkpoints
(steps 17500/18000/18500/19000/final); y = relSD across the 10 runs' final scores;
statistic = median_s(y / x_s). Bootstrap B=2000 over runs, seed 0.

Input: a compact CSV (stage2_ckpt_series.csv) with columns run,step,blimp,lambada_openai,
derived from stage2/runs/*/eval_step*.json + eval_final.json. Writes r29_apph_proxy_boot.json.
"""
import os, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.environ.get("NFT_STAGE2_SERIES", os.path.normpath(
    os.path.join(HERE, "..", "data", "stage_results", "stage2_ckpt_series.csv")))

df = pd.read_csv(CSV)
df["step"] = df["step"].astype(str)
TAIL = ["17500", "18000", "18500", "19000", "final"]

rng = np.random.default_rng(0)
out = {}
for task in ["blimp", "lambada_openai"]:
    xs_per_run, finals = {}, {}
    for run, g in df.groupby("run"):
        v = g.set_index("step")[task]
        if any(st not in v.index for st in TAIL):
            continue
        seq = v.loc[TAIL].to_numpy(float)
        m = np.abs(seq.mean())
        if m == 0: continue
        xs_per_run[run] = float(np.std(seq, ddof=1) / m)
        finals[run] = float(seq[-1])
    runs = sorted(xs_per_run)
    n = len(runs)
    assert n >= 5, f"{task}: only {n} runs"

    def stat(idx):
        sel = [runs[i] for i in idx]
        f = np.array([finals[r] for r in sel])
        y = np.std(f, ddof=1) / abs(f.mean())
        ratios = [y / xs_per_run[r] for r in sel]
        return float(np.median(ratios))

    point = stat(list(range(n)))
    boots = [stat(rng.integers(0, n, n)) for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out[task] = {"n_runs": n, "point": round(point, 3),
                 "boot95": [round(float(lo), 3), round(float(hi), 3)]}
    print(f"{task:<16} n={n}  median y/x {point:.2f}  run-bootstrap 95% [{lo:.2f}, {hi:.2f}]")

out["note"] = ("run-level bootstrap over the 10 bundled runs, B=2000, seed 0; replaces the "
               "F(9,4) intervals ([1.3,8.7]/[0.8,5.2]) which attached a single-ratio reference "
               "to a median-of-ratios over time-correlated shared runs")
out["superseded_F"] = {"blimp": [1.3, 8.7], "lambada_openai": [0.8, 5.2]}
json.dump(out, open(os.path.join(HERE, "r29_apph_proxy_boot.json"), "w"), indent=1)
print("wrote r29_apph_proxy_boot.json")
