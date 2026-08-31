#!/usr/bin/env python3
"""Revision-5 triage verification: check the second external reviewer's factual claims
against the data lake. Each section prints CLAIM vs VERDICT with numbers.
Pure CPU, deterministic. Run from anywhere (paths absolute or NFT_* env).
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats

R = os.environ.get("NFT_R", "data/analysis")
d = pd.read_parquet(f"{R}/dd_tidy.parquet")
t2 = pd.read_csv(f"{R}/../tables/t2_source_equivalence.csv")


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


print("=" * 78)
print("[W7a] '~100x Chinchilla' claim")
# tokens = 100 x params at every size (manifest); Chinchilla = 20 tok/param
print("  paper says ~100x Chinchilla; actual: 100 tok/param / 20 = 5x Chinchilla")
print("  VERDICT: reviewer right, factual error, off by 20x")

print("=" * 78)
print("[W3/Q1] 1.61 (y/x DD) vs 1.62 (additivity inflation S&N) -- seed-semantics explanation?")
prim = t2[t2.metric_mode == "primary_like"]
infl = np.sqrt(prim.sd_init**2 + prim.sd_order**2) / prim.sd_init
print(f"  S&N additivity inflation sqrt(si^2+so^2)/si: median {infl.median():.3f} "
      f"(range {infl.min():.2f}-{infl.max():.2f}, n={len(prim)})")
# DD y/x at what sizes? reproduce 1.61 from t1b path: median corrected y/x pooled cells
# quick: per-size y/x median from cell_stats at final common step needs proxy x=ckpt spread;
# instead read the published table csv
import glob
t1b = pd.read_csv(f"{R}/../tables/t1b_summary_by_size.csv")
print("  t1b_summary_by_size columns:", list(t1b.columns)[:12])
print(t1b.head(16).to_string(max_colwidth=18))

print("=" * 78)
print("[Q2/W4d] argmax pick at 150M: identity + 1B rank; top-k picks' 1B ranks")
m150 = cell_stats("150M"); m1b = cell_stats("1B")
common = sorted(set(m150) & set(m1b))
v150 = np.array([m150[r][0] for r in common]); v1b = np.array([m1b[r][0] for r in common])
order150 = np.argsort(-v150); rank1b = np.argsort(np.argsort(-v1b))
for k in (1, 3, 5):
    idx = order150[:k]
    print(f"  top-{k} at 150M: " + "; ".join(
        f"{common[i][:34]} (1B rank {rank1b[i]+1}/25)" for i in idx))
i0 = order150[0]
print(f"  argmax pick: {common[i0]}; 1B rank {rank1b[i0]+1}/25; "
      f"delta_1B = {v1b.max()-v1b[i0]:+.4f}")
print(f"  paper claim: ranks 21/25, -4.6 macro pts (er_pick_real=0.0459 archived)")

print("=" * 78)
print("[Q4a] 6M/8M/16M noise-median coincidence -- quantization?")
for sz in ("6M", "8M", "16M", "10M", "14M"):
    per = cell_stats(sz)
    sds = np.array([v[1] for v in per.values()])
    print(f"  {sz}: n={len(sds)} median sd={np.median(sds):.6f}  distinct sd values={len(np.unique(np.round(sds,8)))}")
# accuracy grid of macro score: macro = mean of 10 task accs; task acc grid = 1/N_task
g = d[(d.params == "6M") & (d.task != "olmes_10_macro_avg")]
print("  per-task distinct primary_metric values at 6M (sample):")
for t in ("arc_easy", "boolq", "socialiqa"):
    vals = d[(d.params == "6M") & (d.task == t)]["primary_metric"].unique()
    diffs = np.diff(np.sort(vals))
    print(f"    {t}: n_distinct={len(vals)}, min gap={diffs[diffs>0].min() if (diffs>0).any() else 0:.6f}")

print("=" * 78)
print("[Q4b] 20M atlas median 0.0288 vs 16M 0.0111 / 60M 0.0125")
c = pd.read_parquet(f"{R}/t1c_ppl_cells.parquet")
for sz in ("16M", "20M", "60M"):
    cc = c[c.params == sz]
    print(f"  {sz}: cells={len(cc)} median y={cc.y.median():.4f} n_recipes={cc.mix.nunique() if 'mix' in cc else '?'}")
print("  columns:", list(c.columns))

print("=" * 78)
print("[Q5] de-attenuated lambda caps (Table 4/5)")
print(t1b[[c for c in t1b.columns if "lam" in c.lower() or "size" in c.lower() or "R_" in c or "cap" in c.lower()]].to_string())

print("=" * 78)
print("[Q6] 10M: macro vs task-level 3-seed accuracy")
sys_path = "code/enhancement3"
import sys; sys.path.insert(0, sys_path)
import hme
TASKS10 = ["arc_challenge", "arc_easy", "boolq", "csqa", "hellaswag",
           "mmlu", "openbookqa", "piqa", "socialiqa", "winogrande"]
accs = {}
for t in ["olmes_10_macro_avg"] + TASKS10:
    X, Y, _ = hme.extract_cell(d, t, "10M")
    st = hme.descriptive_stats(X, Y)
    accs[t] = st["acc"]
    print(f"  10M {t:<22} pair-acc(3seed)={st['acc']*100:.1f}%")
fit10 = hme.CellFit(*hme.extract_cell(d, "olmes_10_macro_avg", "10M")[:2])
print(f"  10M macro cell: omX={fit10.omX:.5f} (latent signal SD; zero => no macro signal)")

print("=" * 78)
print("[W5] rule-3 blocks: macro vs task-level vs bpb (10M/150M/1B)")
def blocks(per):
    mu = np.array([v[0] for v in per.values()]); sd = np.array([v[1] for v in per.values()])
    n = 3
    ii, jj = np.triu_indices(len(mu), k=1)
    nu = (sd[ii]**2/n + sd[jj]**2/n)**2 / ((sd[ii]**2/n)**2/(n-1) + (sd[jj]**2/n)**2/(n-1))
    band = stats.t.ppf(0.975, nu) * np.sqrt(sd[ii]**2/n + sd[jj]**2/n)
    cov = np.abs(mu[ii]-mu[jj]) < band
    parent = list(range(len(mu)))
    def find(a):
        while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for a, b in zip(ii[cov], jj[cov]): parent[find(a)] = find(b)
    return len({find(i) for i in range(len(mu))})
for sz in ("10M", "150M", "1B"):
    nb = {t: blocks(cell_stats(sz, t)) for t in TASKS10}
    print(f"  {sz}: macro blocks={blocks(cell_stats(sz))}, task-level blocks: "
          + ", ".join(f"{t}:{b}" for t, b in nb.items() if b > 1)
          + (" (all others 1)" if any(b > 1 for b in nb.values()) else " ALL 1"))

print("=" * 78)
print("[W4a] family cluster bootstrap of 4M-1B macro correlation (B=10000)")
FAM = {
    "C4": ["C4"],
    "DCLM": ["DCLM-Baseline"],
    "DCLM-QC": ["DCLM-Baseline (QC 10%)", "DCLM-Baseline (QC 20%)",
                "DCLM-Baseline (QC 7%, FW2)", "DCLM-Baseline (QC 7%, FW3)",
                "DCLM-Baseline (QC FW 10%)", "DCLM-Baseline (QC FW 3%)"],
    "DCLM-Dolma-mix": ["DCLM-Baseline 25% / Dolma 75%", "DCLM-Baseline 50% / Dolma 50%",
                       "DCLM-Baseline 75% / Dolma 25%"],
    "Dolma1.6": ["Dolma1.6++"],
    "Dolma1.7": ["Dolma1.7", "Dolma1.7 (no Flan)", "Dolma1.7 (no Reddit)",
                 "Dolma1.7 (no code)", "Dolma1.7 (no math, code)"],
    "Falcon": ["Falcon"],
    "FalconCC": ["Falcon+CC", "Falcon+CC (QC 10%)", "Falcon+CC (QC 20%)",
                 "Falcon+CC (QC Orig 10%)", "Falcon+CC (QC Tulu 10%)"],
    "FineWeb": ["FineWeb-Edu", "FineWeb-Pro"],
}
m4 = cell_stats("4M"); m1 = cell_stats("1B")
recipes = sorted(set(m4) & set(m1))
assert sum(len(v) for v in FAM.values()) == 25, f"family map covers {sum(len(v) for v in FAM.values())} != 25"
r4v = {r: m4[r][0] for r in recipes}; r1v = {r: m1[r][0] for r in recipes}
rng = np.random.default_rng(20260828)
fams = list(FAM)
B = 10000
boot = np.empty(B)
for b in range(B):
    pick = rng.integers(0, len(fams), len(fams))
    rs = [r for f in pick for r in FAM[fams[f]]]
    boot[b] = stats.pearsonr([r4v[r] for r in rs], [r1v[r] for r in rs])[0]
r_all = stats.pearsonr([r4v[r] for r in recipes], [r1v[r] for r in recipes])[0]
print(f"  point r={r_all:.3f}; family-cluster bootstrap 95% CI "
      f"[{np.percentile(boot,2.5):.3f}, {np.percentile(boot,97.5):.3f}] "
      f"(P(r>0)={float((boot>0).mean()):.4f})")

print("=" * 78)
print("[Q1c] PolyPythias seed semantics: what varies across the 10 seeds?")
import glob as _g
pp = _g.glob("./tmp/rank04-zerogpu/pp_json/*")
print("  pp_json top-level dirs (sample):", sorted(os.path.basename(p) for p in pp)[:8])

print("=" * 78)
print("[Q1c] PolyPythias seed semantics")
import json as _json
from pathlib import Path
pp_root = Path("./tmp/rank04-zerogpu/pp_json")
cands = sorted([p for p in pp_root.glob("pythia-410m-seed*/step*/**/results_*.json")] )[:0]
# 找两个 seed 的 arc_challenge 结果文件对比 config
import glob as _g2
arc = sorted(_g2.glob(str(pp_root/"pythia-410m-seed*"/"step*"/"*"/"results_*.json")))
byseed = {}
for p in arc:
    parts = Path(p).parts
    seed_dir = [x for x in parts if x.startswith("pythia-410m-seed")]
    if not seed_dir: continue
    byseed.setdefault(seed_dir[0], []).append(p)
keys = sorted(byseed)[:3]
cfgs = {}
for k in keys:
    for p in byseed[k]:
        try:
            j = _json.load(open(p))
        except Exception:
            continue
        c = j.get("config", {})
        if "arc_challenge" in p:
            cfgs[k] = {kk: c.get(kk) for kk in ("model_args","seed","random_seed","numpy_seed","torch_seed","fewshot_seed") if kk in c}
            cfgs[k]["_file"] = Path(p).parent.name
            break
for k, v in cfgs.items():
    print(f"  {k}: {v}")

print("=" * 78)
print("[Q4a-bis] which recipe sits at the 6M/8M/16M median SD?")
for sz in ("6M","8M","16M"):
    per = cell_stats(sz)
    items = sorted(per.items(), key=lambda kv: kv[1][1])
    med = items[len(items)//2]
    print(f"  {sz}: median recipe = {med[0]!r} sd={med[1][1]:.6f}")

print("=" * 78)
print("[Q4b-bis] 20M: n_ckpt/n_common vs 16M/60M")
c2 = pd.read_parquet(f"{R}/t1c_ppl_cells.parquet")
for sz in ("16M","20M","60M"):
    cc = c2[c2.params==sz]
    print(f"  {sz}: n_ckpt med={cc.n_ckpt.median():.0f} n_common med={cc.n_common.median():.0f} "
          f"y med={cc.y.median():.4f} y q25-q75=({cc.y.quantile(.25):.4f},{cc.y.quantile(.75):.4f})")
print(c2[(c2.params=='20M')].groupby('dom').y.median().sort_values(ascending=False).head(6))
