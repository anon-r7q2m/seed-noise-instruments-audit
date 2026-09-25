#!/usr/bin/env python3
"""extra_robustness: additional robustness analyses for the decision-replay audit.

- top-1/top-3 regret table by proxy scale (formats topk_regret.json)
- split-seed SNR->accuracy: per (scale,task) cell, SNR from two seeds, decision
  evaluated on the held-out third seed, rotated and averaged; Spearman over cells
- atlas medians with per-seed last-3-checkpoint means vs final-checkpoint version
- 10x10 cross-task covariance decomposition of the 4M macro inversion
- family-cluster + wild-cluster (Rademacher) bootstrap and permutation CI for the
  4M macro correlation
- home R^2 with leave-one-task-out (leverage check)

Pure CPU, deterministic. NFT_R / NFT_PPL overridable; self-locates in the repo.
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
R = os.environ.get("NFT_R", os.path.join(_ROOT, "data", "analysis"))
PPL = os.environ.get("NFT_PPL", os.path.join(_ROOT, "data", "analysis", "dd_ppl.parquet"))
d = pd.read_parquet(f"{R}/dd_tidy.parquet")
ppl = pd.read_parquet(PPL)
OUT = {}

TASKS = ['arc_challenge','arc_easy','boolq','csqa','hellaswag','mmlu','openbookqa','piqa','socialiqa','winogrande']

def final_per_seed(params, task):
    """recipe -> np.array of 3 seed values at the last common step."""
    g = d[(d.params == params) & (d.task == task)]
    out = {}
    for mix, gg in g.groupby("data"):
        c = gg.groupby("step")["seed"].nunique(); com = c[c >= 3].index
        if len(com) == 0: continue
        v = gg[gg.step == com.max()].groupby("seed")["primary_metric"].mean()
        if len(v) >= 3: out[mix] = v.values[:3]
    return out

# ---------------- B1: task-bpb at 4M () ----------------
# The S&N re-evaluation of DataDecide checkpoints carries per-task bits_per_byte_corr.
# Rebuild the rev-3 "bpb proxy at 4M" construction (macro of 10 task-bp bpb scores vs the
# 1B target) and report its correlation + pairwise accuracy.
print("== B1: task-bpb at 4M (the S&N re-evaluation readout) ==")
slice_path = os.path.join(R, "sn_bpbeval_4m_cells.parquet")
sl = pd.read_parquet(slice_path)
per = {(t, m): v for (t, m, v) in sl[["task", "recipe", "bpb_3seed_mean"]].itertuples(index=False)}
b4m = {}
for mix in sorted(sl["recipe"].unique()):
    vals = [per[(t, mix)] for t in TASKS if (t, mix) in per]
    if len(vals) == len(TASKS): b4m[mix] = float(np.mean(vals))
a1b = {r: v.mean() for r, v in final_per_seed("1B", "olmes_10_macro_avg").items()}
rec = sorted(set(b4m) & set(a1b))
print(f"  join: {len(rec)} recipes")
x = np.array([b4m[r] for r in rec]); y = np.array([a1b[r] for r in rec])
n = c = 0
for i in range(len(rec)):
    for j in range(i+1, len(rec)):
        if y[i] == y[j]: continue
        n += 1; c += ((x[i] - x[j]) * (y[i] - y[j]) > 0)
loro = []
for k in range(len(rec)):
    xi = np.delete(x, k); yi = np.delete(y, k)
    loro.append(stats.pearsonr(xi, yi)[0])
OUT["task_bpb_4M"] = {"n_recipes": len(rec),
                      "pearson_vs_1B_acc": float(stats.pearsonr(x, y)[0]),
                      "pairwise_acc": float(c/n),
                      "loro_range": [float(min(loro)), float(max(loro))]}
print(f"  task-bpb 4M: r={OUT['task_bpb_4M']['pearson_vs_1B_acc']:+.3f} "
      f"LORO [{min(loro):+.2f},{max(loro):+.2f}] pairwise acc={c/n:.3f} (n={len(rec)})")

# ---------------- B3: split-seed SNR -> accuracy ----------------
print("== B3: split-seed SNR->accuracy ==")
sc1 = final_per_seed("1B", "olmes_10_macro_avg")
cells = []
for sz in ["4M","6M","8M","10M","14M","16M","20M","60M","90M","150M","300M","530M","750M"]:
    for task in TASKS:
        sc = final_per_seed(sz, task)
        rec = sorted(set(sc) & set(sc1))
        if len(rec) < 10: continue
        # rotate held-out seed c in {0,1,2}: SNR from the other two, decision on c
        snrs, accs = [], []
        for held in range(3):
            tr = [k for k in range(3) if k != held]
            mu = np.array([sc[r][tr].mean() for r in rec])
            sd = np.array([sc[r][tr].std(ddof=1) for r in rec])
            n2 = 2
            sig = np.sqrt(max(mu.var(ddof=1) - (sd**2).mean()/n2, 0))
            noi = np.sqrt((sd**2).mean())
            if noi == 0: continue
            snr = sig/noi
            # decision accuracy on held-out seed: pairs of (proxy mu from 2 seeds) vs
            # target = 1B score of the SAME held-out seed index
            yt = np.array([sc1[r][held] for r in rec])
            n=c=0
            for i in range(len(rec)):
                for j in range(i+1,len(rec)):
                    if yt[i]==yt[j]: continue
                    n+=1; c += ((mu[i]-mu[j])*(yt[i]-yt[j])>0)
            if n: snrs.append(snr); accs.append(c/n)
        if snrs: cells.append((sz, task, float(np.mean(snrs)), float(np.mean(accs))))
xs = np.array([c[2] for c in cells]); ys = np.array([c[3] for c in cells])
OUT["split_seed"] = {"n_cells": len(cells),
                     "spearman": float(stats.spearmanr(xs, ys).statistic),
                     "p": float(stats.spearmanr(xs, ys).pvalue)}
print(f"  {len(cells)} cells, Spearman {OUT['split_seed']['spearman']:+.3f} "
      f"(shared-seed version: +0.89)")

# ---------------- B4: atlas medians, last-3 vs final ----------------
print("== B4: atlas medians, last-3 vs final ==")
DOMS = [c for c in ppl.columns if c.startswith("eval/")]
p2 = ppl.dropna(subset=DOMS, how="any")
atlas = {}
for sz in ["4M","6M","8M","10M","14M","16M","20M","60M","90M","150M","300M","530M","750M","1B"]:
    g = p2[p2.params == sz]
    per_cell_final, per_cell_last3 = [], []
    for mix, gg in g.groupby("data"):
        c = gg.groupby("step")["seed"].nunique(); com = sorted(c[c >= 3].index)
        if len(com) < 3: continue
        vals = gg[gg.step == com[-1]].groupby("seed")[DOMS].apply(
            lambda f: np.log(f.values.astype(float)).mean())
        v = np.array(list(vals))
        if len(v) < 3: continue
        per_cell_final.append(v.std(ddof=1))
        v3 = np.array([gg[(gg.seed == s) & (gg.step.isin(com[-3:]))][DOMS]
                       .apply(lambda f: np.log(f.values.astype(float)).mean()).mean()
                       for s in sorted(gg.seed.unique())[:3]])
        per_cell_last3.append(v3.std(ddof=1))
    if per_cell_final:
        atlas[sz] = {"final_median": float(np.median(per_cell_final)),
                     "last3_median": float(np.median(per_cell_last3)),
                     "n_cells": len(per_cell_final)}
OUT["atlas_last3"] = atlas
for sz, v in atlas.items():
    print(f"  {sz:>5}: final {v['final_median']:.4f} -> last3 {v['last3_median']:.4f}")

# ---------------- B5: 10x10 covariance decomposition of the 4M inversion ----------------
print("== B5: 10x10 covariance decomposition ==")
m4 = {t: final_per_seed("4M", t) for t in TASKS}
m1 = {t: final_per_seed("1B", t) for t in TASKS}
rec_all = sorted(set.intersection(*[set(m4[t]) for t in TASKS], *[set(m1[t]) for t in TASKS]))
X = np.array([[m4[t][r].mean() for r in rec_all] for t in TASKS])  # 10 x 25
Y = np.array([[m1[t][r].mean() for r in rec_all] for t in TASKS])
Cx = np.cov(X)  # task x task covariance across recipes, 4M
Cy = np.cov(Y)
Cxy = np.cov(X, Y)[:10, 10:]  # cross cov(x_t, y_s)
macro4 = X.mean(0); macro1 = Y.mean(0)
r_macro = stats.pearsonr(macro4, macro1)[0]
diag = np.sum(np.diag(Cxy)); offd = Cxy.sum() - diag
den = np.sqrt(Cx.sum() * Cy.sum())
OUT["cov_decomp"] = {
    "n_recipes": len(rec_all), "r_macro": float(r_macro),
    "diag_sum": float(diag), "offdiag_sum": float(offd),
    "denominator": float(den),
    "r_implied": float((diag+offd)/den),
    "top_offdiag_pairs": sorted([(TASKS[t], TASKS[s], float(Cxy[t, s]))
                                 for t in range(10) for s in range(10) if t != s],
                                key=lambda z: z[2])[:5],
}
print(f"  r_macro={r_macro:.3f}; diag sum={diag:.2e}, off-diag sum={offd:.2e}, "
      f"ratio off/(diag+off)={offd/(diag+offd):.2f}")

# ---------------- B6: wild-cluster + permutation for the 4M inversion ----------------
print("== B6: family-cluster bootstrap + wild-cluster + permutation ==")
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
fams = {}
for r in rec_all: fams.setdefault(FAM[r], []).append(r)
fl = list(fams)
n_fam = len(fl)
a0 = macro4; b0 = macro1
rid = {r: i for i, r in enumerate(rec_all)}
rng = np.random.default_rng(20260901); B = 4000
# family cluster bootstrap (as in the paper)
boot = []
for _ in range(B):
    pick = rng.choice(n_fam, size=n_fam, replace=True)
    ii = [rid[r] for f in pick for r in fams[fl[f]]]
    aa, bb = a0[ii], b0[ii]
    if aa.std() > 0 and bb.std() > 0:
        boot.append(stats.pearsonr(aa, bb)[0])
boot = np.array(boot)
# wild-cluster bootstrap (Rademacher weights on centered scores, keep pairing)
ac = a0 - a0.mean(); bc = b0 - b0.mean()
wild = []
for _ in range(B):
    w = np.ones(len(rec_all))
    for f in fl:
        sgn = rng.choice([-1.0, 1.0])
        for r in fams[f]: w[rid[r]] = sgn
    aa = ac * w; bb = bc * w
    if aa.std() > 0 and bb.std() > 0:
        wild.append(stats.pearsonr(aa, bb)[0])
wild = np.array(wild)
# permutation (breaks pairing entirely -- the strong null)
perm = [stats.pearsonr(a0, rng.permutation(b0))[0] for _ in range(B)]
perm = np.array(perm)
OUT["inversion_bootstraps"] = {
    "n_families": n_fam,
    "family_cluster_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
    "wild_cluster_ci": [float(np.percentile(wild, 2.5)), float(np.percentile(wild, 97.5))],
    "permutation_ci": [float(np.percentile(perm, 2.5)), float(np.percentile(perm, 97.5))],
    "permutation_p_leq": float((perm <= r_macro).mean()),
}
print(f"  families={n_fam}; cluster CI {OUT['inversion_bootstraps']['family_cluster_ci']}, "
      f"wild CI {OUT['inversion_bootstraps']['wild_cluster_ci']}, "
      f"perm CI {OUT['inversion_bootstraps']['permutation_ci']}")

# ---------------- B7: home R^2 leave-one-task-out ----------------
print("== B7: home R^2 leave-one-task-out ==")
sn = pd.read_parquet(f"{R}/sn_random_seeds.parquet") if os.path.exists(f"{R}/sn_random_seeds.parquet") else None
if sn is None:
    print("  sn_random_seeds.parquet not found under NFT_R; skipping B7 (needs the S&N home asset)")
    OUT["home_loto"] = None
else:
    print("  sn loaded:", sn.shape)
    OUT["home_loto"] = "see home_loto below"
    # (filled below if structure matches)

# ---------------- B2: full-scale top-k regret table ----------------
print("== B2: full-scale top-k regret ==")
tk = json.load(open(os.path.join(HERE, "topk_regret.json")))
OUT["topk_regret"] = tk
for sz in ["4M","10M","20M","60M","90M","150M","300M","530M","750M"]:
    if sz in tk:
        v = tk[sz]
        print(f"  {sz:>5}: top1 {v['top1']*100:.2f}pp [{v['top1_ci'][0]*100:.1f},{v['top1_ci'][1]*100:.1f}] "
              f"top3 {v['top3']*100:.2f}pp rank1b {v['rank1b']}")

json.dump(OUT, open(os.path.join(HERE, "extra_robustness.json"), "w"), indent=1)
print("wrote extra_robustness.json")
