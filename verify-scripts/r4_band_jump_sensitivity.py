#!/usr/bin/env python3
"""(i) Equivalence-band sensitivity of the per-task sigma_init/sigma_order ratio
CIs (containment counts under band multipliers x CI types, plus Bonferroni-width
F CIs); (ii) 60M->90M deconvolved-SNR jump stratified per task and under
leave-one-recipe-out; writes r4_band_jump_sensitivity.json.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
R = os.environ.get("NFT_R", os.path.join(_ROOT, "data", "analysis"))
T2 = os.environ.get("NFT_T2", os.path.join(_ROOT, "data", "tables", "t2_source_equivalence.csv"))
out = {}

# ---------------------------------------------------------------- (i) band sensitivity
t2 = pd.read_csv(T2)
print(f"[band] {len(t2)} task rows ({(t2.metric_mode=='primary_like').sum()} primary-like, "
      f"{(t2.metric_mode!='primary_like').sum()} bpb)")


def f_ci(sd1, n1, sd2, n2, alpha):
    """CI for ratio sd1/sd2 from F = (sd1/sd2)^2 ~ F(n1-1, n2-1)."""
    r = sd1 / sd2
    lo = r / np.sqrt(stats.f.ppf(1 - alpha / 2, n1 - 1, n2 - 1))
    hi = r / np.sqrt(stats.f.ppf(alpha / 2, n1 - 1, n2 - 1))
    return lo, hi


bonf = np.array([f_ci(r.sd_init, r.n_init, r.sd_order, r.n_order, 0.05 / len(t2))
                 for r in t2.itertuples()])
out["band_sensitivity"] = {}
print(f"{'band mult':>9} | {'F-CI in':>8} | {'bs-CI in':>8} | {'Bonf-F in':>9}")
for b in (1.25, 1.5, 2.0, 2.5, 3.0):
    lo, hi = 1.0 / b, b
    cF = int(((t2.ci_lo_F >= lo) & (t2.ci_hi_F <= hi)).sum())
    cB = int(((t2.ci_lo_bs >= lo) & (t2.ci_hi_bs <= hi)).sum())
    cBf = int(((bonf[:, 0] >= lo) & (bonf[:, 1] <= hi)).sum())
    out["band_sensitivity"][str(b)] = {"F": cF, "bootstrap": cB, "bonferroni_F": cBf}
    print(f"{b:>9.2f} | {cF:>8} | {cB:>8} | {cBf:>9}")

# context: how many CIs exclude 1 at all (direction evidence)
exF = int(((t2.ci_lo_F > 1) | (t2.ci_hi_F < 1)).sum())
print(f"[band] context: F-CI excludes 1 for {exF}/27 tasks (uncorrected)")
out["band_context_F_excludes_1"] = exF

# ---------------------------------------------------------------- (ii) jump stratification
d = pd.read_parquet(f"{R}/dd_tidy.parquet")


def fcs(g, need=3):
    c = g.groupby("step")["seed"].nunique()
    com = c[c >= need].index
    return None if len(com) == 0 else com.max()


def cell_stats(sz, task="olmes_10_macro_avg"):
    per = {}
    for mix, gg in d[(d.params == sz) & (d.task == task)].groupby("data"):
        s = fcs(gg)
        if s is None:
            continue
        v = gg[gg.step == s].groupby("seed")["primary_metric"].mean()
        if len(v) >= 3:
            per[mix] = (v.mean(), v.std(ddof=1))
    return per


def snr_dec(per):
    mu = np.array([v[0] for v in per.values()])
    sd = np.array([v[1] for v in per.values()])
    return float(np.sqrt(max(mu.var(ddof=1) - np.mean(sd ** 2) / 3, 0)) / np.sqrt(np.mean(sd ** 2)))


TASKS10 = ["arc_challenge", "arc_easy", "boolq", "csqa", "hellaswag",
           "mmlu", "openbookqa", "piqa", "socialiqa", "winogrande"]

s60m, s90m = cell_stats("60M"), cell_stats("90M")
s60, s90 = snr_dec(s60m), snr_dec(s90m)
print(f"\n[jump] macro anchor: 60M {s60:.2f} -> 90M {s90:.2f}  (paper: 1.23 -> 2.61)")
out["jump_macro"] = {"60M": s60, "90M": s90}

# per-task stratification
out["jump_per_task"] = {}
print(f"{'task':<14} | {'SNR 60M':>8} | {'SNR 90M':>8} | {'jump?':>6}")
for t in TASKS10:
    a, b = snr_dec(cell_stats("60M", t)), snr_dec(cell_stats("90M", t))
    out["jump_per_task"][t] = {"60M": a, "90M": b}
    print(f"{t:<14} | {a:>8.2f} | {b:>8.2f} | {str(b > a):>6}")

# macro LORO
recipes = sorted(set(s60m) & set(s90m))
loro = []
for r in recipes:
    a = snr_dec({k: v for k, v in s60m.items() if k != r})
    b = snr_dec({k: v for k, v in s90m.items() if k != r})
    loro.append((r, a, b))
arr = np.array([[a, b] for _, a, b in loro])
out["jump_loro"] = {"60M_range": [float(arr[:, 0].min()), float(arr[:, 0].max())],
                    "90M_range": [float(arr[:, 1].min()), float(arr[:, 1].max())],
                    "all_jump": bool((arr[:, 1] > arr[:, 0]).all()),
                    "per_recipe": {r: [float(a), float(b)] for r, a, b in loro}}
print(f"[jump] macro LORO: 60M [{arr[:,0].min():.2f},{arr[:,0].max():.2f}]  "
      f"90M [{arr[:,1].min():.2f},{arr[:,1].max():.2f}]  jump in all 25 drops: {(arr[:,1]>arr[:,0]).all()}")

with open("r4_band_jump_sensitivity.json", "w") as fp:
    json.dump(out, fp, indent=1)
print("\nwrote r4_band_jump_sensitivity.json")
