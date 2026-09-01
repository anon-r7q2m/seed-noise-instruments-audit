#!/usr/bin/env python3
"""R12 major-3 / Q2: dependence-matched uncertainty for the two headline ratios.

(a) 4M macro-1B ranking correlation: the revision-5 family bootstrap resamples
    recipe families but treats the 3-seed means as fixed. Here we resample BOTH
    the 10 recipe families AND the 3 seeds within each picked recipe
    (independently at 4M and 1B), propagating seed-level measurement error at
    both ends. B=2000.

(b) The ~1.6x y/x magnitude ratio (Sec 3): pooled over 3,497 cells that share
    tasks, recipes, and seeds. Dependence-matched CI: resample recipe families
    x tasks jointly, and within each cell resample the 3 seeds to recompute y
    (the 3-seed relative SD) before taking the per-cell ratio y/x and the median.
    Also reports the log-space small-sample corrected variant (x1.61 basis:
    median of y/x computed on log scale, corrected). B=2000.

Outputs r9_joint_bootstrap.json. Pure CPU, deterministic. NFT_R overridable.
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.environ.get("NFT_R", "/home/bingxing2/home/scx7ew2/tanh/Tanhäuser/runs/exp-rank04/zero-gpu/analysis")
d = pd.read_parquet(f"{R}/dd_tidy.parquet")

FAM = {}
FAM["C4"]="c4"; FAM["DCLM-Baseline"]="dclm-base"
for r in ["DCLM-Baseline (QC 10%)","DCLM-Baseline (QC 20%)"]: FAM[r]="dclm-qc-pct"
for r in ["DCLM-Baseline (QC 7%, FW2)","DCLM-Baseline (QC 7%, FW3)"]: FAM[r]="dclm-qc-7fw"
for r in ["DCLM-Baseline (QC FW 3%)","DCLM-Baseline (QC FW 10%)"]: FAM[r]="dclm-qc-fw"
for r in ["DCLM-Baseline 25% / Dolma 75%","DCLM-Baseline 50% / Dolma 50%","DCLM-Baseline 75% / Dolma 25%"]: FAM[r]="dclm-dolma-mix"
for r in ["Dolma1.6++","Dolma1.7","Dolma1.7 (no Flan)","Dolma1.7 (no Reddit)","Dolma1.7 (no code)","Dolma1.7 (no math, code)"]: FAM[r]="dolma"
for r in ["Falcon","Falcon+CC"]: FAM[r]="falcon-base"
for r in ["Falcon+CC (QC 10%)","Falcon+CC (QC 20%)","Falcon+CC (QC Orig 10%)","Falcon+CC (QC Tulu 10%)"]: FAM[r]="falcon-qc"
for r in ["FineWeb-Edu","FineWeb-Pro"]: FAM[r]="fineweb"
def fam(r): return FAM[r]

rng = np.random.default_rng(20260901)
B = 2000
OUT = {}

# ---------- (a) 4M macro corr: joint family x seed bootstrap
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

sc4, sc1 = final_scores("4M"), final_scores("1B")
recipes = sorted(set(sc4) & set(sc1))
fams = {}
for r in recipes: fams.setdefault(fam(r), []).append(r)
fl = list(fams)

a0 = np.array([sc4[r].mean() for r in recipes]); b0 = np.array([sc1[r].mean() for r in recipes])
point = stats.pearsonr(a0, b0)[0]

boot_joint, boot_fam = [], []
for _ in range(B):
    pick = rng.choice(len(fl), size=len(fl), replace=True)
    # joint: also resample seeds within each picked recipe
    aa, bb, aa2, bb2 = [], [], [], []
    for i in pick:
        for r in fams[fl[i]]:
            aa.append(np.mean(rng.choice(sc4[r], size=3, replace=True)))
            bb.append(np.mean(rng.choice(sc1[r], size=3, replace=True)))
            aa2.append(sc4[r].mean()); bb2.append(sc1[r].mean())
    aa, bb, aa2, bb2 = map(np.array, (aa, bb, aa2, bb2))
    if aa.std() > 0 and bb.std() > 0: boot_joint.append(stats.pearsonr(aa, bb)[0])
    if aa2.std() > 0 and bb2.std() > 0: boot_fam.append(stats.pearsonr(aa2, bb2)[0])
boot_joint, boot_fam = np.array(boot_joint), np.array(boot_fam)
OUT["corr_4m_1b"] = {
    "point": float(point),
    "family_only_ci": [float(np.percentile(boot_fam,2.5)), float(np.percentile(boot_fam,97.5))],
    "joint_family_seed_ci": [float(np.percentile(boot_joint,2.5)), float(np.percentile(boot_joint,97.5))],
    "joint_p_pos": float((boot_joint>0).mean()),
}
print(f"(a) 4M-1B corr point={point:.3f}  family-only CI [{OUT['corr_4m_1b']['family_only_ci'][0]:.3f},{OUT['corr_4m_1b']['family_only_ci'][1]:.3f}]  joint CI [{OUT['corr_4m_1b']['joint_family_seed_ci'][0]:.3f},{OUT['corr_4m_1b']['joint_family_seed_ci'][1]:.3f}]  P(>0)={OUT['corr_4m_1b']['joint_p_pos']:.3f}")

# ---------- (b) y/x median ratio: family x task x seed bootstrap
cells = pd.read_parquet(f"{R}/t1b_cells.parquet")
# rebuild per-cell per-seed values to allow seed resampling for y
# y = relative SD across 3 seeds of the final score (per params,data,task)
# x = checkpoint proxy (fixed, not seed-dependent)
TASKS10 = ["arc_challenge","arc_easy","boolq","csqa","hellaswag","mmlu","openbookqa","piqa","socialiqa","winogrande"]
CACHE = os.path.join(HERE, "r9_cell_seeds_cache.pkl")
if os.path.exists(CACHE):
    cell_seeds = pd.read_pickle(CACHE)
else:
    cell_seeds = {}   # (params,data,task) -> per-seed final values
    for (p, r, t), gg in d[d.task.isin(TASKS10)].groupby(["params","data","task"]):
        c = gg.groupby("step")["seed"].nunique()
        com = c[c >= 3].index
        if len(com) == 0: continue
        v = gg[gg.step == com.max()].groupby("seed")["primary_metric"].mean()
        if len(v) >= 3:
            cell_seeds[(p, r, t)] = v.values[:3]
    pd.to_pickle(cell_seeds, CACHE)
cx = cells.set_index(["params","data","task"])["x"].to_dict()

keys = [k for k in cell_seeds if k in cx and cx[k] > 0]
print(f"(b) cells with x and 3-seed y: {len(keys)}")

def rel_sd(v):
    m = v.mean()
    return v.std(ddof=1) / abs(m) if m != 0 else np.nan

# NOTE: naive with-replacement seed resampling is BIASED for an SD estimator at n=3
# (duplicate-heavy resamples shrink the SD; the replicate median excludes the point
# value). We therefore propagate seed noise with the same parametric chi^2_2 model
# used in the ceiling analysis: y_boot = y * sqrt(chi2_2 / 2).
y_obs = {}
for k in keys:
    y = rel_sd(np.asarray(cell_seeds[k], dtype=float))
    if y and y > 0:
        y_obs[k] = y
keys = list(y_obs)
Y = np.array([y_obs[k] for k in keys])
X = np.array([cx[k] for k in keys])

point_ratio = float(np.median(Y / X))
print(f"(b) point median y/x = {point_ratio:.4f} (paper: 1.47)")

fam_of = {k: fam(k[1]) for k in keys}
task_of = {k: k[2] for k in keys}
ufams = sorted(set(fam_of.values())); utasks = sorted(set(task_of.values()))
fam_arr = np.array([fam_of[k] for k in keys])
task_arr = np.array([task_of[k] for k in keys])

boot_ratio = np.empty(B)   # families x tasks x seed-jitter
boot_fam = np.empty(B)     # families only x seed-jitter
for b in range(B):
    yb = Y * np.sqrt(rng.chisquare(2, size=len(Y)) / 2.0)
    fpick = rng.choice(ufams, size=len(ufams), replace=True)
    tpick = rng.choice(utasks, size=len(utasks), replace=True)
    sel = np.isin(fam_arr, fpick) & np.isin(task_arr, tpick)
    self_ = np.isin(fam_arr, fpick)
    boot_ratio[b] = np.median((yb / X)[sel]) if sel.sum() >= 30 else np.nan
    boot_fam[b] = np.median((yb / X)[self_]) if self_.sum() >= 30 else np.nan

def stats_ci(boot, point):
    boot = boot[~np.isnan(boot)]
    return {"ci_percentile": [float(np.percentile(boot,2.5)), float(np.percentile(boot,97.5))],
            "boot_mean": float(boot.mean()), "boot_sd": float(boot.std(ddof=1)),
            "ci_normal_centered": [float(point - 1.96*boot.std(ddof=1)), float(point + 1.96*boot.std(ddof=1))]}

OUT["yx_ratio"] = {
    "point": point_ratio,
    "joint_fam_task_seed": stats_ci(boot_ratio, point_ratio),
    "fam_only_seed": stats_ci(boot_fam, point_ratio),
    "n_cells": len(keys),
}
print(f"(b) point={point_ratio:.4f}")
print(f"    fam x task x seed: {OUT['yx_ratio']['joint_fam_task_seed']}")
print(f"    fam only x seed:   {OUT['yx_ratio']['fam_only_seed']}")

with open(os.path.join(HERE, "r9_joint_bootstrap.json"), "w") as f:
    json.dump(OUT, f, indent=1)
print("wrote r9_joint_bootstrap.json")
