#!/usr/bin/env python3
"""R10-Q3: is the 4M inversion carried by tasks that are at chance at 4M?

Reviewer: per-task 4M-1B correlations range -0.27..+0.88 and 5/10 tasks are positive;
tasks whose 4M accuracy is indistinguishable from chance may drive the macro
inversion via variance-share weighting. Ablation: drop chance-indistinguishable
tasks wholesale (not leave-one-out), recompute the 4M macro ranking's correlation
with the 1B ranking.

Two chance filters:
  V1 (best-recipe): drop task if the BEST 4M recipe's 3-seed mean accuracy is
     within Welch noise of the task's chance rate (task carries no signal at 4M).
  V2 (pairwise): drop task if the per-task 4M pairwise decision accuracy
     (3-seed means vs 1B target, 300 pairs) bootstrap CI covers 50%.

Recompute macro 4M-1B correlation (Pearson + Spearman) on kept tasks, with
family-cluster bootstrap CI (10 recipe families as in r5_family_bootstrap).
Outputs r9_chance_task_ablation.json. Pure CPU, deterministic. NFT_R overridable.
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.environ.get("NFT_R", "/home/bingxing2/home/scx7ew2/tanh/Tanhäuser/runs/exp-rank04/zero-gpu/analysis")
d = pd.read_parquet(f"{R}/dd_tidy.parquet")

CHANCE = {"arc_challenge":0.25,"arc_easy":0.25,"boolq":0.5,"csqa":0.2,"hellaswag":0.25,
          "mmlu":0.25,"openbookqa":0.25,"piqa":0.5,"socialiqa":1/3,"winogrande":0.5}
TASKS = list(CHANCE)

# recipe families: the 10-family mapping used by the revision-5 family-cluster
# bootstrap (validated below by reproducing its point/CI: -0.559, [-0.82, -0.16]).
FAM = {}
FAM["C4"] = "c4"
FAM["DCLM-Baseline"] = "dclm-base"
for r in ["DCLM-Baseline (QC 10%)", "DCLM-Baseline (QC 20%)"]: FAM[r] = "dclm-qc-pct"
for r in ["DCLM-Baseline (QC 7%, FW2)", "DCLM-Baseline (QC 7%, FW3)"]: FAM[r] = "dclm-qc-7fw"
for r in ["DCLM-Baseline (QC FW 3%)", "DCLM-Baseline (QC FW 10%)"]: FAM[r] = "dclm-qc-fw"
for r in ["DCLM-Baseline 25% / Dolma 75%", "DCLM-Baseline 50% / Dolma 50%",
          "DCLM-Baseline 75% / Dolma 25%"]: FAM[r] = "dclm-dolma-mix"
for r in ["Dolma1.6++", "Dolma1.7", "Dolma1.7 (no Flan)", "Dolma1.7 (no Reddit)",
          "Dolma1.7 (no code)", "Dolma1.7 (no math, code)"]: FAM[r] = "dolma"
for r in ["Falcon", "Falcon+CC"]: FAM[r] = "falcon-base"
for r in ["Falcon+CC (QC 10%)", "Falcon+CC (QC 20%)", "Falcon+CC (QC Orig 10%)",
          "Falcon+CC (QC Tulu 10%)"]: FAM[r] = "falcon-qc"
for r in ["FineWeb-Edu", "FineWeb-Pro"]: FAM[r] = "fineweb"
def fam(r):
    return FAM[r]

def final_scores(params, task):
    """per recipe: 3-seed final scores at the common final step."""
    g = d[(d.params == params) & (d.task == task)]
    out = {}
    for mix, gg in g.groupby("data"):
        c = gg.groupby("step")["seed"].nunique()
        com = c[c >= 3].index
        if len(com) == 0:
            continue
        s = com.max()
        v = gg[gg.step == s].groupby("seed")["primary_metric"].mean()
        if len(v) >= 3:
            out[mix] = v.values[:3]
    return out

rng = np.random.default_rng(20260901)
B = 2000

def macro_scores(sc, tasks):
    """recipe -> 3 seed macro scores (mean over kept tasks)."""
    recipes = set.intersection(*[set(sc[t]) for t in tasks]) if tasks else set()
    m = {}
    for r in recipes:
        m[r] = np.mean([sc[t][r] for t in tasks], axis=0)
    return m

def corr_4m_1b(sc4, sc1, tasks):
    m4, m1 = macro_scores(sc4, tasks), macro_scores(sc1, tasks)
    rec = sorted(set(m4) & set(m1))
    if len(rec) < 8:
        return None
    a = np.array([m4[r].mean() for r in rec]); b = np.array([m1[r].mean() for r in rec])
    pear = stats.pearsonr(a, b)[0]; spear = stats.spearmanr(a, b)[0]
    # family-cluster bootstrap over recipes
    fams = {}
    for r in rec:
        fams.setdefault(fam(r), []).append(r)
    fl = list(fams)
    boot = []
    for _ in range(B):
        pick = rng.choice(len(fl), size=len(fl), replace=True)
        rs = [r for i in pick for r in fams[fl[i]]]
        aa = np.array([m4[r].mean() for r in rs]); bb = np.array([m1[r].mean() for r in rs])
        if aa.std() == 0 or bb.std() == 0:
            continue
        boot.append(stats.pearsonr(aa, bb)[0])
    boot = np.array(boot)
    return {"n_recipes": len(rec), "pearson": float(pear), "spearman": float(spear),
            "boot_lo": float(np.percentile(boot, 2.5)), "boot_hi": float(np.percentile(boot, 97.5)),
            "boot_p_pos": float((boot > 0).mean())}

sc4 = {t: final_scores("4M", t) for t in TASKS}
sc1 = {t: final_scores("1B", t) for t in TASKS}
sc1m = {t: final_scores("1B", t) for t in TASKS}

OUT = {"task_filter": {}, "correlations": {}}

# --- V1: best-recipe vs chance (Welch)
drop_v1, keep_v1 = [], []
for t in TASKS:
    best_r, best_v = None, -1
    for r, v in sc4[t].items():
        if v.mean() > best_v:
            best_r, best_v = r, v.mean()
    v = sc4[t][best_r]
    tstat = (v.mean() - CHANCE[t]) / (v.std(ddof=1) / np.sqrt(len(v)) + 1e-12)
    p = 1 - stats.t.cdf(tstat, df=len(v) - 1)
    rec = {"best_recipe": best_r, "best_mean": float(v.mean()), "chance": CHANCE[t],
           "t": float(tstat), "p_one_sided": float(p)}
    (keep_v1 if p < 0.05 else drop_v1).append(t)
    OUT["task_filter"].setdefault("V1_best_recipe_vs_chance", {})[t] = rec

# --- V2: per-task pairwise accuracy CI vs 50%
pairs_cache = {}
def task_pairacc(t, seed_mean=True):
    m4 = {r: v for r, v in sc4[t].items()}
    m1 = sc1m[t]
    rec = sorted(set(m4) & set(m1))
    def acc(scores4):
        tot = cor = 0
        for i in range(len(rec)):
            for j in range(i + 1, len(rec)):
                ri, rj = rec[i], rec[j]
                d1 = m1[ri].mean() - m1[rj].mean()
                d4 = scores4[ri] - scores4[rj]
                if d1 == 0: continue
                tot += 1; cor += (d4 * d1 > 0)
        return cor / tot
    base = acc({r: v.mean() for r, v in m4.items()})
    # recipe-cluster bootstrap
    fams = {}
    for r in rec: fams.setdefault(fam(r), []).append(r)
    fl = list(fams); boot = []
    for _ in range(B):
        pick = rng.choice(len(fl), size=len(fl), replace=True)
        rs = [r for i in pick for r in fams[fl[i]]]
        # accuracy on resampled multiset of pairs
        tot = cor = 0
        for a in range(len(rs)):
            for b in range(a + 1, len(rs)):
                ri, rj = rs[a], rs[b]
                if ri == rj: continue
                d1 = m1[ri].mean() - m1[rj].mean(); d4 = m4[ri].mean() - m4[rj].mean()
                if d1 == 0: continue
                tot += 1; cor += (d4 * d1 > 0)
        if tot: boot.append(cor / tot)
    boot = np.array(boot)
    return base, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

drop_v2, keep_v2 = [], []
for t in TASKS:
    base, lo, hi = task_pairacc(t)
    OUT["task_filter"].setdefault("V2_pairwise_acc", {})[t] = {"acc": base, "boot_lo": lo, "boot_hi": hi}
    (keep_v2 if lo > 0.5 else drop_v2).append(t)

# sanity gate: reproduce the revision-5 family-bootstrap headline (all 10 tasks)
_gate = corr_4m_1b(sc4, sc1, TASKS)
print(f"GATE all10 pearson={_gate['pearson']:.3f} (paper: -0.559) bootCI=[{_gate['boot_lo']:.3f},{_gate['boot_hi']:.3f}] (paper: [-0.82,-0.16])")
assert abs(_gate["pearson"] - (-0.559)) < 0.01, "family map or final-score construction off"

for tag, tasks in [("all10", TASKS), ("V1_keep", keep_v1), ("V2_keep", keep_v2)]:
    r = corr_4m_1b(sc4, sc1, tasks) if len(tasks) >= 3 else None
    OUT["correlations"][tag] = {"kept_tasks": tasks, "result": r}

print("V1 (best-recipe vs chance): DROP", drop_v1)
print("V2 (pairwise CI covers .5): DROP", drop_v2)
for tag, v in OUT["correlations"].items():
    if v["result"]:
        r = v["result"]
        print(f"{tag:>9}: n_tasks={len(v['kept_tasks'])} pearson={r['pearson']:.3f} spearman={r['spearman']:.3f} bootCI=[{r['boot_lo']:.3f},{r['boot_hi']:.3f}] P(>0)={r['boot_p_pos']:.3f}")

with open(os.path.join(HERE, "r9_chance_task_ablation.json"), "w") as f:
    json.dump(OUT, f, indent=1)
print("wrote r9_chance_task_ablation.json")
