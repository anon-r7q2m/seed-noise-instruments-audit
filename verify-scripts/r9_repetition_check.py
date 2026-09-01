#!/usr/bin/env python3
"""R11-Q2: is the 4M inversion explained by data repetition at 1B?

Reviewer: DataDecide fixes 100 tokens/param (1B trains 100B tokens); small filtered
subsets (e.g. QC 3%/7%) would repeat at 1B but not at 4M. If only non-repeating
recipes are kept, does the inversion vanish?

Method:
 1. Pool sizes in TOKENS per recipe pool: summed shard bytes/2 (uint16 memmaps; the
    gpt-neox-olmo-dolma-v1_5 tokenizer, vocab<65536; validated: c4 tree bytes
    276.9GB = 2 x SOURCES_SIZES c4 total_size 138.44B tokens, and fastdclm tree
    7.71TB = 2 x 3.86T tokens ~ DCLM's published 3.8T pool) from
    huggingface allenai/DataDecide-data-recipes tree API, 2026-08-31; recipe->pool
    mapping from allenai/OLMo @ DataDecide branch olmo/data/named_data_mixes.py.
 2. Repetition factor at scale S = 100*S_tokens / pool_tokens; >1 means the 1B run
    wraps the pool (OLMo loader cycles).
 3. Recompute the 4M macro ranking's correlation with the 1B ranking on all 25
    recipes vs dropping the recipes that repeat at 1B (family-cluster bootstrap CI,
    10 families as in r9_chance_task_ablation.py).

Outputs r9_repetition_check.json. Pure CPU, deterministic. NFT_R overridable.
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.environ.get("NFT_R", "/home/bingxing2/home/scx7ew2/tanh/Tanhäuser/runs/exp-rank04/zero-gpu/analysis")
d = pd.read_parquet(f"{R}/dd_tidy.parquet")

# Pool sizes in billions of tokens (provenance: see module docstring).
POOL_BT = {
 "C4": 138.4,
 "DCLM-Baseline": 3857.1,
 "DCLM-Baseline (QC 10%)": 82.1,
 "DCLM-Baseline (QC 20%)": 171.1,
 "DCLM-Baseline (QC 7%, FW2)": 418.2,
 "DCLM-Baseline (QC 7%, FW3)": 418.2,   # dir absent from release; fw2 assumed (same 7th-pct rate)
 "DCLM-Baseline (QC FW 3%)": 118.1,
 "DCLM-Baseline (QC FW 10%)": 98.5,
 "DCLM-Baseline 25% / Dolma 75%": 0.25*3857.1 + 0.75*1715.1,
 "DCLM-Baseline 50% / Dolma 50%": 0.50*3857.1 + 0.50*1715.1,
 "DCLM-Baseline 75% / Dolma 25%": 0.75*3857.1 + 0.25*1715.1,
 "Dolma1.6++": 685.8,                    # olmo-mix v1_6 tree (686B) + extras; lower bound
 "Dolma1.7": 1715.1,                     # SOURCES_SIZES sum
 "Dolma1.7 (no Flan)": 1715.1 - 16.5,
 "Dolma1.7 (no Reddit)": 1715.1 - 79.9,
 "Dolma1.7 (no code)": 1715.1 - 263.8 - 19.6,
 "Dolma1.7 (no math, code)": 1715.1 - 263.8 - 19.6 - 12.7 - 28.0 - 12.6,
 "Falcon": 456.4,
 "Falcon+CC": 456.4 + 598.4,
 "Falcon+CC (QC 10%)": (86.9 + 205.2) / 2,      # tree GB, /2 -> B tokens = 146.05B
 "Falcon+CC (QC 20%)": (238.8 + 332.9) / 2,
 "Falcon+CC (QC Orig 10%)": (99.5 + 196.2) / 2,
 "Falcon+CC (QC Tulu 10%)": (90.5 + 167.4) / 2,
 "FineWeb-Edu": 192.2,
 "FineWeb-Pro": 82.7,
}
# NOTE on the Falcon+CC QC rows: tree bytes in GB summed over the falcon and
# olmo-mix (CC) halves, /2 converts uint16 bytes to B tokens.

PARAMS_B = {"4M":0.004,"6M":0.006,"8M":0.008,"10M":0.010,"14M":0.014,"16M":0.016,
            "20M":0.020,"60M":0.060,"90M":0.090,"150M":0.150,"300M":0.300,
            "530M":0.530,"750M":0.750,"1B":1.0}  # billions of parameters

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

def corr_with_1b(sc4, sc1, recipes, B=2000, seed=20260901):
    rng = np.random.default_rng(seed)
    rec = [r for r in recipes if r in sc4 and r in sc1]
    a = np.array([sc4[r].mean() for r in rec]); b = np.array([sc1[r].mean() for r in rec])
    pear, spear = stats.pearsonr(a,b)[0], stats.spearmanr(a,b)[0]
    fams = {}
    for r in rec: fams.setdefault(fam(r), []).append(r)
    fl = list(fams); boot=[]
    for _ in range(B):
        pick = rng.choice(len(fl), size=len(fl), replace=True)
        rs = [r for i in pick for r in fams[fl[i]]]
        aa = np.array([sc4[r].mean() for r in rs]); bb = np.array([sc1[r].mean() for r in rs])
        if aa.std()==0 or bb.std()==0: continue
        boot.append(stats.pearsonr(aa,bb)[0])
    boot=np.array(boot)
    return {"n":len(rec),"pearson":float(pear),"spearman":float(spear),
            "boot_lo":float(np.percentile(boot,2.5)),"boot_hi":float(np.percentile(boot,97.5)),
            "boot_p_pos":float((boot>0).mean())}

sc4, sc1 = final_scores("4M"), final_scores("1B")
recipes = sorted(set(sc4) & set(sc1) & set(POOL_BT))
assert len(recipes)==25, f"expect 25 recipes, got {len(recipes)}: missing {set(POOL_BT)^set(recipes)}"

OUT = {"pool_tokens_B": POOL_BT, "repetition_factor": {}, "correlations": {}}

# repetition factor per scale (tokens needed = 100 x params)
for sz, pb in PARAMS_B.items():
    need = 100 * pb  # billions of tokens
    OUT["repetition_factor"][sz] = {r: round(need / POOL_BT[r], 3) for r in recipes}

rep_1b = {r: 100.0 / POOL_BT[r] for r in recipes}  # epochs at 1B: 100B tokens needed
repeating = sorted([r for r in recipes if rep_1b[r] > 1.0], key=lambda r: -rep_1b[r])
print("recipes repeating at 1B (epochs = 100B/pool):")
for r in repeating: print(f"  {r}: {rep_1b[r]:.3f} epochs (pool {POOL_BT[r]:.1f}B)")
nonrep = [r for r in recipes if rep_1b[r] <= 1.0]

# also at 750M and 530M for the scale picture
for sz in ["530M","750M"]:
    need = 100*PARAMS_B[sz]
    rep = [r for r in recipes if need/POOL_BT[r] > 1.0]
    print(f"repeating at {sz}: {rep}")

OUT["repeating_at_1B"] = repeating
OUT["correlations"]["all25"] = corr_with_1b(sc4, sc1, recipes)
OUT["correlations"]["drop_repeating"] = corr_with_1b(sc4, sc1, nonrep)

# do the repeating recipes' ranks move differently? descriptive, n=3
r4 = sorted(recipes, key=lambda r: -sc4[r].mean()); r1 = sorted(recipes, key=lambda r: -sc1[r].mean())
rank4 = {r:i+1 for i,r in enumerate(r4)}; rank1 = {r:i+1 for i,r in enumerate(r1)}
OUT["rank_moves"] = {r: {"rank4M": rank4[r], "rank1B": rank1[r]} for r in repeating}
med_abs_move_rep = float(np.median([abs(rank4[r]-rank1[r]) for r in repeating])) if repeating else None
med_abs_move_non = float(np.median([abs(rank4[r]-rank1[r]) for r in nonrep]))
OUT["median_abs_rank_move"] = {"repeating": med_abs_move_rep, "non_repeating": med_abs_move_non}

for tag, v in OUT["correlations"].items():
    print(f"{tag:>16}: n={v['n']} pearson={v['pearson']:.3f} spearman={v['spearman']:.3f} CI=[{v['boot_lo']:.3f},{v['boot_hi']:.3f}] P(>0)={v['boot_p_pos']:.3f}")
print("median |rank move| repeating vs not:", med_abs_move_rep, med_abs_move_non)

with open(os.path.join(HERE, "r9_repetition_check.json"), "w") as f:
    json.dump(OUT, f, indent=1)
print("wrote r9_repetition_check.json")
