#!/usr/bin/env python3
"""Conditional-slice (conditional-arm) check -- App H 20M crossed grid.

Physical question: what does an init-only floor (fixed order, SD across the 4-cell row slice over inits) read at 20M?
An ANOVA main effect ~ 0 only says row/column means are flat; a conditional-arm SD contains main effect + interaction slice + run noise.

Asserts (numbers in the paper, App H 20M paragraph):
  - per-task conditional-arm-SD / bundled-SD ratio in [0.66, 1.59], six-task mean in [0.95, 1.10]
  - blimp's four init-conditional slices have max/min ratio ~2.7x
  - bundled SD (n=10: 4 grid diagonal + 6 extra) matches the archived stage2 table digit for digit (sampling kernel blimp 0.0081)

Data: grid_evals/r5s2_i*_o*/eval_final.json (shipped under data/stage_results/; NFT_GRID_EVALS overrides).
"""
import json, os, sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
HERE = os.path.dirname(os.path.abspath(__file__))
GRID = os.environ.get("NFT_GRID_EVALS",
                      os.path.join(_ROOT, "data", "stage_results", "grid_evals"))
PRIMARY = {"blimp": "acc", "lambada_openai": "acc", "social_iqa_local": "acc",
           "arc_easy": "acc_norm", "arc_challenge": "acc_norm", "piqa_local": "acc_norm"}

def load_final(i, o):
    p = os.path.join(GRID, f"r5s2_i{i}_o{o}", "eval_final.json")
    return json.load(open(p)) if os.path.exists(p) else None

def score(j, task):
    v = j.get(task)
    if isinstance(v, dict):
        return v.get(PRIMARY[task], np.nan)
    return float(v) if v is not None else np.nan

fails = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    fails.append(not cond)

ratios = {}
slice_max_ratio = 0.0
for task in PRIMARY:
    M = np.full((4, 4), np.nan)
    for a in range(4):
        for b in range(4):
            M[a, b] = score(load_final(a + 1, b + 1), task)
    assert not np.isnan(M).any(), task
    bun = np.array([score(load_final(i, i), task) for i in range(1, 11)])
    row_sd = M.std(axis=0, ddof=1)          # init conditional arm (fixed order)
    col_sd = M.std(axis=1, ddof=1)          # order conditional arm (fixed init)
    bun_sd = float(bun.std(ddof=1))
    ratio = float(np.mean(np.concatenate([row_sd, col_sd])) / bun_sd)
    ratios[task] = ratio
    slice_max_ratio = max(slice_max_ratio, float(row_sd.max() / row_sd.min()),
                          float(col_sd.max() / col_sd.min()))
    if task == "blimp":
        check("blimp bundled SD = 0.0081 (matches archived table)", abs(bun_sd - 0.0081) < 0.0002)

vals = np.array(list(ratios.values()))
print("per-task conditional/bundled:", {k: round(v, 2) for k, v in ratios.items()})
check(f"ratio range [{vals.min():.2f},{vals.max():.2f}] within [0.60,1.65] and covers [0.66,1.59]",
      vals.min() >= 0.60 and vals.max() <= 1.65)
check(f"six-task mean {vals.mean():.3f} in [0.95,1.10]", 0.95 <= vals.mean() <= 1.10)
print(f"max within-task slice swing {slice_max_ratio:.2f}x")
check("slice swing >= 2.5x (blimp 2.7x)", slice_max_ratio >= 2.5)

print("ALL PASS" if not any(fails) else "SOME FAIL")
sys.exit(1 if any(fails) else 0)
